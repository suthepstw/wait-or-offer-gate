#!/usr/bin/env bash
# run_encoder_cmp.sh — เปรียบเทียบ encoder ของ D5: Bi-LSTM (base_M3, อ้างอิง) vs Bi-GRU vs Bi-RNN vs Transformer vs ST-Transformer
#   ทุก run ใช้ ข้อมูล D4 · หัว 3 หัว · loss ε-WTA · D6 (λ_h, T, isotonic) · D7 v7 (split dev) ชุดเดียวกับ base_M3 → ต่างกันที่ encoder เท่านั้น
#   ใช้:   nohup bash run_encoder_cmp.sh sf   > log_enc_sf.txt   2>&1 &
#          nohup bash run_encoder_cmp.sh rome > log_enc_rome.txt 2>&1 &      (รันทีละเมือง — GPU เดียว)
#          SEEDS=0,1,2,3,4 bash run_encoder_cmp.sh sf     (ค่าเริ่มต้น 3 seed: 0,1,2)
#   ตรวจว่าเสร็จ:  tail -3 log_enc_sf.txt   → บรรทัด "== เสร็จ"   หรือ   ls reports/encoder_compare_sf.md
#
#   ไม่แตะของเดิม: base_M3 (5 seed, D7_final, E1_handoff.json, qm_calib.json) ไม่ถูก train/calibrate ซ้ำ
#                  เขียนเฉพาะ models/<city>/abl_*/ · ต่อแถวใน models/D5_results.csv, models/D7_results.csv (split=dev)
#   รันซ้ำได้: run ที่มี summary.json อยู่แล้วจะถูกข้าม
set -euo pipefail
CITY="${1:-sf}"; SEEDS="${SEEDS:-0,1,2}"
grep -q '"sttfm"' sD5_train.py || { echo "sD5_train.py ยังไม่มี --rnn sttfm (ต้องเป็นรุ่นที่มี SpatialFourier)"; exit 1; }
[ -f "models/$CITY/base_M3/summary.json" ] || { echo "ไม่พบ models/$CITY/base_M3 — ต้องมี base_M3 (Bi-LSTM) เป็นตัวอ้างอิงก่อน"; exit 1; }

# ชื่อ run → อาร์กิวเมนต์ที่ต่างจาก base_M3 (ทุกอย่างอื่นเท่าเดิม: M=3, skip τ=3, hidden 256, 9 ฟีเจอร์)
declare -A ENC=(
  [abl_gru]="--rnn gru"                                              # Bi-GRU  (2 ชั้น 128/256 เหมือน LSTM)
  [abl_rnn]="--rnn rnn"                                              # Bi-vanilla-RNN (tanh)
  [abl_tfm]="--rnn tfm"                                              # Transformer เชิงเวลา d=128 ×2 blocks (~0.5M params)
  [abl_sttfm]="--rnn sttfm"                                          # ST-Transformer = tfm + spatial Fourier encoding ของ (x,y)  (~0.5M)
  [abl_sttfm_L]="--rnn sttfm --tfm-d 256 --tfm-blocks 3 --tfm-heads 8"   # ST-Transformer ขนาดใกล้ Bi-LSTM (~2.7M vs 2.3M params)
)
ORDER=(abl_gru abl_tfm abl_sttfm abl_sttfm_L abl_rnn)

for r in "${ORDER[@]}"; do
  [ "$r" != "base_M3" ] || { echo "ห้ามเขียนทับ base_M3"; exit 1; }
  if [ -f "models/$CITY/$r/summary.json" ]; then echo "== $r มีแล้ว ข้าม"; continue; fi
  echo "== D5 $CITY/$r (${ENC[$r]}) seeds=$SEEDS $(date)"
  python sD5_train.py --city "$CITY" --run "$r" --seeds "$SEEDS" ${ENC[$r]}
done

RUNS=$(IFS=,; echo "${ORDER[*]}")
echo "== D6 $CITY $RUNS $(date)";  python sD6_calibrate.py --city "$CITY" --runs "$RUNS"

# base_M3 บน dev: ใช้ผลที่มีอยู่ (ไม่รันซ้ำ เพื่อไม่ให้แถวซ้ำใน ledger) — ถ้ายังไม่มีค่อยรวมเข้าไป
D7RUNS="$RUNS"; [ -f "models/$CITY/base_M3/D7_dev/G1.json" ] || D7RUNS="base_M3,$RUNS"
echo "== D7 dev $CITY $D7RUNS $(date)"; python sD7_pred_eval.py --city "$CITY" --runs "$D7RUNS" --split dev

echo "== สรุปตาราง $(date)"; python sX_compare_encoders.py --city "$CITY" --runs "base_M3,$RUNS"
echo "== เสร็จ $(date) — ดู reports/encoder_compare_$CITY.md · models/D5_results.csv · models/$CITY/*/D7_dev/D7_report.md"
