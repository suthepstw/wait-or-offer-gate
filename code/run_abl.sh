#!/usr/bin/env bash
# run_abl.sh — Bi-LSTM+ ablation ชุดอินพุต (M1 occ · M2 stop-event): D4(v2) → D5(v2) ×3 seeds → D6 → D7 v7 dev → ตารางเทียบ
#   ใช้:   nohup bash run_abl.sh sf occ      > log_abl_sf_occ.txt   2>&1 &     # config/data_sf_occ.yaml   → processed_f11 · models/sf/abl_occ
#          nohup bash run_abl.sh sf stop     > log_abl_sf_stop.txt  2>&1 &     # config/data_sf_stop.yaml  → processed_f13 · models/sf/abl_stop
#          nohup bash run_abl.sh sf occstop  > log_abl_sf_occstop.txt 2>&1 &   # เมื่อ abl_occ ผ่านเกณฑ์ M7 แล้ว
#          nohup bash run_abl.sh rome stop   > log_abl_rome_stop.txt 2>&1 &    # (รันทีละงาน — GPU เดียว)
#          bash run_abl.sh sf occ gru | rnn | tfm | uni | gru_uni | rnn_uni → models/sf/abl_occ_gru … (encoder อื่นบนอินพุตชุดเดียวกัน; ข้าม D4 ถ้ามี processed_f11 แล้ว)
#          SEEDS=0,1,2 (ค่าเริ่มต้น) · EXTRA="--batch 256" ส่งต่อให้ sD5 · SKIP_D4=1 ถ้า processed_<tag> มีแล้ว
#   ตรวจว่าเสร็จ: tail -3 log_abl_sf_occ.txt → "== เสร็จ" · ดู reports/input_ablation_sf.md · models/sf/abl_occ/D7_dev/D7_report.md
#
#   ไม่แตะของเดิม: data/<city>/processed/ · scaler.json · models/<city>/base_M3/ · sD7_pred_eval.py v7 (ตรวจ sha ก่อนรัน)
#   เขียนเฉพาะ data/<city>/processed_<tag>/ · models/<city>/abl_<variant>/ · ต่อแถว models/D5_results.csv, models/D7_results.csv (split=dev)
#   รันซ้ำได้: run ที่มี summary.json แล้วจะข้าม D5
set -euo pipefail
CITY="${1:?ใช้: bash run_abl.sh <sf|rome> <occ|stop|occstop> [lstm|gru|rnn|tfm|uni|gru_uni|rnn_uni]}"; VAR="${2:?variant}"; ENC="${3:-lstm}"
SEEDS="${SEEDS:-0,1,2}"; EXTRA="${EXTRA:-}"; SKIP_D4="${SKIP_D4:-0}"
# encoder ตัวที่ 3 (ไม่บังคับ): ใช้อินพุตชุดเดียวกันกับ Bi-LSTM+ → เทียบ encoder บนฟีเจอร์ใหม่ (M7 ข้อ 4: tfm_plus, gru_plus, …)
case "$ENC" in
  lstm) ENCARG=""; SUF="";;
  gru)  ENCARG="--rnn gru"; SUF="_gru";;
  rnn)  ENCARG="--rnn rnn"; SUF="_rnn";;
  tfm)  ENCARG="--rnn tfm"; SUF="_tfm";;                 # d128×2 เหมือน abl_tfm เดิม
  uni)  ENCARG="--no-bi";   SUF="_uni";;                 # LSTM ทางเดียว
  gru_uni) ENCARG="--rnn gru --no-bi"; SUF="_gru_uni";;  # GRU ทางเดียว (ไม่ Bi)
  rnn_uni) ENCARG="--rnn rnn --no-bi"; SUF="_rnn_uni";;  # vanilla RNN ทางเดียว
  *) echo "encoder ไม่รู้จัก: $ENC"; exit 1;;
esac
cd "$(dirname "$0")"
source ~/Documents/LSTM/myenv/bin/activate 2>/dev/null || source ~/mcs/venv/bin/activate 2>/dev/null || true

CFG="config/data_${CITY}_${VAR}.yaml"; RUN="abl_${VAR}${SUF}"
[ -f "$CFG" ] || { echo "ไม่พบ $CFG"; exit 1; }
[ -f "models/$CITY/base_M3/summary.json" ] || { echo "ไม่พบ models/$CITY/base_M3 — ต้องมี base_M3 เป็นตัวอ้างอิงก่อน"; exit 1; }
SHA=$(sha256sum sD7_pred_eval.py | cut -c1-8); [ "$SHA" = "0fe82c5d" ] || { echo "sD7_pred_eval.py ไม่ใช่ v7 (sha $SHA) — หยุด"; exit 1; }
[ "$RUN" != "base_M3" ] || { echo "ห้ามเขียนทับ base_M3"; exit 1; }

read -r PROC WITHOCC <<EOF
$(python - "$CFG" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
print(c["processed_dir"], "--with-occ" if "occ" in c.get("features", []) else "")
PY
)
EOF
[ "$PROC" != "processed" ] || { echo "processed_dir ใน $CFG ต้องไม่ใช่ processed"; exit 1; }

t0=$(date +%s); stamp() { echo "== [$(date '+%Y-%m-%d %H:%M:%S') +$(( ($(date +%s)-t0)/60 )) min] $*"; }

stamp "D4 v2 $CITY/$VAR → data/$CITY/$PROC"
if [ "$SKIP_D4" = "1" ] && [ -f "data/$CITY/$PROC/train_X.npy" ]; then echo "   มีแล้ว ข้าม"; else
  python sD1_D4_prepare_v2.py --city "$CITY" --config "$CFG"
fi
grep -q '"align_with_processed": {"train": true, "val": true, "test": true}' "data/$CITY/manifest_${VAR}.json" \
  || echo "   ⚠ หน้าต่างไม่ตรงกับ processed/ เดิม — D6/D7 v7 อ่าน Y/meta จาก processed/ จะผิดแถว (ดู data/$CITY/manifest_${VAR}.json)"

stamp "D5 v2 $CITY/$RUN (proc=$PROC $WITHOCC $ENCARG) seeds=$SEEDS"
if [ -f "models/$CITY/$RUN/summary.json" ]; then echo "   มีแล้ว ข้าม"; else
  python sD5_train_v2.py --city "$CITY" --run "$RUN" --proc "$PROC" --config "$CFG" $WITHOCC --seeds "$SEEDS" $ENCARG $EXTRA
fi

stamp "D6 $CITY $RUN";      python sD6_calibrate.py --city "$CITY" --runs "$RUN"
stamp "D7 dev $CITY $RUN";  python sD7_pred_eval.py --city "$CITY" --runs "$RUN" --split dev

stamp "ตารางเทียบ"
ALL="base_M3"; for r in models/$CITY/abl_*/D7_dev/pred_metrics.csv; do [ -f "$r" ] && ALL="$ALL,$(basename "$(dirname "$(dirname "$r")")")"; done
python sX_compare_runs.py --city "$CITY" --runs "$ALL" --out "reports/input_ablation_$CITY.md" || true
stamp "เสร็จ — models/$CITY/$RUN/D7_dev/D7_report.md · reports/input_ablation_$CITY.md · data/$CITY/reports/D1_D4_report_${VAR}.md"
