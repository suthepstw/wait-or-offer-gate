#!/usr/bin/env bash
# run_enc_uni.sh — เทียบ Bi-LSTM (base_M3) กับ uni-GRU · uni-RNN · Transformer บนอินพุต 9 ตัวเดิม (D4 / D6 / D7 v7 ชุดเดียวกัน)
#   ใช้:   nohup bash run_enc_uni.sh sf   > log_enc_uni_sf.txt   2>&1 &
#          nohup bash run_enc_uni.sh rome > log_enc_uni_rome.txt 2>&1 &      (รันทีละเมือง — GPU เดียว)
#          SEEDS=0,1,2 (ค่าเริ่มต้น) · ONLY=abl_gru_uni bash run_enc_uni.sh sf → รันแค่ run เดียว
#   ตรวจว่าเสร็จ: tail -3 log_enc_uni_sf.txt → "== เสร็จ" · ดู reports/encoder_uni_<city>.md
#
#   ไม่แตะของเดิม: base_M3 / abl_tfm / abl_gru / abl_rnn / abl_uni ที่มีแล้วจะไม่ train ซ้ำ (ข้ามถ้ามี summary.json) และเข้าตารางให้
#   เขียนเฉพาะ models/<city>/abl_gru_uni, abl_rnn_uni (+ abl_tfm ถ้ายังไม่มี) · ต่อแถว models/D5_results.csv, models/D7_results.csv (split=dev)
set -euo pipefail
CITY="${1:-sf}"; SEEDS="${SEEDS:-0,1,2}"; ONLY="${ONLY:-}"
cd "$(dirname "$0")"
source ~/Documents/LSTM/myenv/bin/activate 2>/dev/null || source ~/mcs/venv/bin/activate 2>/dev/null || true
[ -f "models/$CITY/base_M3/summary.json" ] || { echo "ไม่พบ models/$CITY/base_M3 — ต้องมี base_M3 (Bi-LSTM) ก่อน"; exit 1; }
SHA=$(sha256sum sD7_pred_eval.py | cut -c1-8); [ "$SHA" = "0fe82c5d" ] || { echo "sD7_pred_eval.py ไม่ใช่ v7 (sha $SHA) — หยุด"; exit 1; }

# run ใหม่ → อาร์กิวเมนต์ที่ต่างจาก base_M3 (อย่างอื่นเท่าเดิม: M=3, skip τ=3, hidden 256, 9 ฟีเจอร์, loss, D6, D7)
declare -A ENC=(
  [abl_gru_uni]="--rnn gru --no-bi"      # GRU ทางเดียว
  [abl_rnn_uni]="--rnn rnn --no-bi"      # vanilla RNN ทางเดียว
  [abl_tfm]="--rnn tfm"                  # Transformer d128×2 (มีแล้วจาก run_encoder_cmp.sh → ข้าม)
)
ORDER=(abl_gru_uni abl_rnn_uni abl_tfm)
[ -n "$ONLY" ] && ORDER=("$ONLY")

t0=$(date +%s); stamp() { echo "== [$(date '+%Y-%m-%d %H:%M:%S') +$(( ($(date +%s)-t0)/60 )) min] $*"; }
NEW=()
for r in "${ORDER[@]}"; do
  [ "$r" != "base_M3" ] || { echo "ห้ามเขียนทับ base_M3"; exit 1; }
  if [ -f "models/$CITY/$r/summary.json" ]; then echo "== $r มีแล้ว ข้าม D5"; else
    stamp "D5 $CITY/$r (${ENC[$r]}) seeds=$SEEDS"
    python sD5_train.py --city "$CITY" --run "$r" --seeds "$SEEDS" ${ENC[$r]}
  fi
  [ -f "models/$CITY/$r/D7_dev/pred_metrics.csv" ] || NEW+=("$r")      # ยังไม่มี D6/D7 dev → ทำให้
done

if [ ${#NEW[@]} -gt 0 ]; then
  RUNS=$(IFS=,; echo "${NEW[*]}")
  stamp "D6 $CITY $RUNS";      python sD6_calibrate.py --city "$CITY" --runs "$RUNS"
  stamp "D7 dev $CITY $RUNS";  python sD7_pred_eval.py --city "$CITY" --runs "$RUNS" --split dev
fi

# ตาราง: ทุกตัวที่มี D7_dev ในชุด {base_M3, abl_uni, abl_gru, abl_gru_uni, abl_rnn, abl_rnn_uni, abl_tfm, abl_sttfm}
ALL=""
for r in base_M3 abl_uni abl_gru abl_gru_uni abl_rnn abl_rnn_uni abl_tfm abl_sttfm; do
  [ -f "models/$CITY/$r/D7_dev/pred_metrics.csv" ] && ALL="${ALL:+$ALL,}$r"
done
stamp "ตาราง $ALL"; python sX_compare_runs.py --city "$CITY" --runs "$ALL" --out "reports/encoder_uni_$CITY.md"
stamp "เสร็จ — reports/encoder_uni_$CITY.md · models/$CITY/abl_*_uni/D7_dev/D7_report.md"
