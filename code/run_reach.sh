#!/usr/bin/env bash
# run_reach.sh — ทาง 2: หัว reachability แบบ "ถามจุด" (9a′) · ไฟล์ใหม่ทั้งหมด ไม่แตะ base_M3 / sD7 v7 / processed
#   ใช้:  nohup bash run_reach.sh sf lstm > log_reach_sf_lstm.txt 2>&1 &     # Bi-LSTM + หัว reach  → models/sf/reach_lstm
#         nohup bash run_reach.sh sf tfm  > log_reach_sf_tfm.txt  2>&1 &     # Transformer + หัว reach → models/sf/reach_tfm
#         LAM=1.0 bash run_reach.sh sf lstm_lam1                              # ปรับ λ_reach (ชื่อ run = reach_<ENC> · encoder = ส่วนหน้า "_")
#         EXTRA="--reach-only" bash run_reach.sh sf lstm_only                 # อาร์กิวเมนต์เพิ่มให้ sD5_train_v4
#         PROC=processed_f15 DATA=data_f15 EXTRA="--config config/data_sf_occstop.yaml --with-occ" bash run_reach.sh sf lstm_occstop
#              PROC = โฟลเดอร์ D4 ที่ D5 อ่าน (ใต้ data/<city>/) · DATA = data root ที่ D6/D7/reach-eval อ่าน (ใส่เมื่อ window ไม่ align กับ processed/)
#   ขั้น: D5 v4 (เทรน) → D6 (λ_h, T ของ mixture heads) → D7 v7 dev → reach-eval (ทุก run reach_* + base_M3 + abl_tfm บนคู่เดียวกัน)
#   ผล:  reports/reach_eval_<city>.md · models/<city>/reach_*/D7reach_dev/reach_report.md · seed*/reach_head.npz, reach_calib.json
set -euo pipefail
CITY="${1:-sf}"; ENC="${2:-lstm}"; SEEDS="${SEEDS:-0,1,2}"; LAM="${LAM:-2.0}"; K="${K:-32}"; EXTRA="${EXTRA:-}"; PROC="${PROC:-processed}"; DATA="${DATA:-}"
cd "$(dirname "$0")"; source ~/Documents/LSTM/myenv/bin/activate 2>/dev/null || source ~/mcs/venv/bin/activate 2>/dev/null || true
SHA=$(sha256sum sD7_pred_eval.py | cut -c1-8); [ "$SHA" = "0fe82c5d" ] || { echo "sD7_pred_eval.py ไม่ใช่ v7 (sha $SHA)"; exit 1; }
for f in sD5_train.py sD5_train_v4.py sD7_reach_eval.py sD6_calibrate.py; do [ -f $f ] || { echo "ไม่พบ $f"; exit 1; }; done
RUN="reach_$ENC"; RNN="${ENC%%_*}"
case "$RNN" in lstm|gru|rnn|tfm|sttfm) ;; *) echo "encoder $RNN ไม่รู้จัก (ชื่อต้องเริ่มด้วย lstm|gru|rnn|tfm|sttfm)"; exit 1;; esac
[ "$RUN" = "base_M3" ] && exit 1
DARG=(); [ -n "$DATA" ] && DARG=(--data "$DATA")
t0=$(date +%s); stamp() { echo "== [$(date '+%Y-%m-%d %H:%M:%S') +$(( ($(date +%s)-t0)/60 )) min] $*"; }

if [ -f "models/$CITY/$RUN/summary.json" ]; then echo "== $RUN มีแล้ว ข้าม D5"; else
  [ -d "data/$CITY/$PROC" ] || { echo "ไม่พบ data/$CITY/$PROC"; exit 1; }
  stamp "D5 v4 $CITY/$RUN (rnn=$RNN λ_reach=$LAM K=$K proc=$PROC seeds=$SEEDS $EXTRA)"
  python sD5_train_v4.py --city "$CITY" --run "$RUN" --rnn "$RNN" --seeds "$SEEDS" --lam-reach "$LAM" --k-query "$K" --proc "$PROC" $EXTRA
fi
[ -f "models/$CITY/$RUN/seed0/calib.json" ] || { stamp "D6 $CITY/$RUN ${DARG[*]:-}"; python sD6_calibrate.py --city "$CITY" --runs "$RUN" "${DARG[@]}"; }
[ -f "models/$CITY/$RUN/D7_dev/pred_metrics.csv" ] || { stamp "D7 v7 dev $CITY/$RUN ${DARG[*]:-}"; python sD7_pred_eval.py --city "$CITY" --runs "$RUN" --split dev "${DARG[@]}"; }
if [ -n "$DATA" ]; then     # window ไม่ align กับ processed/ → ประเมิน run นี้ตัวเดียวบน data root ของมัน (เทียบข้าม run ด้วยคู่เดียวกันไม่ได้)
  stamp "reach-eval $CITY: $RUN เดี่ยว (--data $DATA)"
  python sD7_reach_eval.py --city "$CITY" --runs "$RUN" --split dev --ref "$RUN" --out "reports/reach_eval_${CITY}_${RUN}.md" "${DARG[@]}"
  stamp "เสร็จ — reports/reach_eval_${CITY}_${RUN}.md"; cat "reports/reach_eval_${CITY}_${RUN}.md"; exit 0
fi
RUNS="$RUN,base_M3"; [ -f "models/$CITY/abl_tfm/seed0/pred_test.npz" ] && RUNS="$RUNS,abl_tfm"
for d in models/$CITY/reach_*/; do r=$(basename "$d"); [ "$r" != "$RUN" ] && [ -f "$d/seed0/pred_test.npz" ] && RUNS="$RUNS,$r"; done
stamp "reach-eval $CITY: $RUNS (คู่เดียวกัน CRN_SEED 12345)"
python sD7_reach_eval.py --city "$CITY" --runs "$RUNS" --split dev --ref base_M3 --out "reports/reach_eval_$CITY.md"
stamp "เสร็จ — reports/reach_eval_$CITY.md"; cat "reports/reach_eval_$CITY.md"
