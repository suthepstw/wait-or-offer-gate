#!/usr/bin/env bash
# run_hp_sens.sh — hyperparameter sensitivity ของ Bi-LSTM (base_M3) ทีละตัว · 9 ฟีเจอร์ · D6/D7 v7 dev เหมือน base ทุกอย่าง
#   ใช้:   nohup bash run_hp_sens.sh sf   > log_hp_sf.txt   2>&1 &        # 8 run: lr ½× 2× · hidden ½× 2× · batch 256 · dropout 0.30 · Adam · ε-WTA 0.1→0.02
#          FULL=1 nohup bash run_hp_sens.sh sf > log_hp_sf.txt 2>&1 &     # + lr schedule คงที่ · SGD-momentum  (10 run)
#   ต้องมี sD5_train_hp.py (wrapper ของ sD5_train.py: เพิ่ม --dropout --opt --sched; ค่า default = base ทุกประการ)
#          ONLY=hp_lr2e-4 bash run_hp_sens.sh sf                         # รันตัวเดียว
#   ผล:   reports/hp_sens_<city>.md  (ตารางเดียวกับ encoder_uni)   · models/<city>/hp_*/
#   ไม่แตะ base_M3 · run ที่มี summary.json แล้วข้าม · ตรวจ sha sD7 v7 ก่อนรัน
#   อ่านผล: ถ้าทุก run อยู่ใน base ± ½ CI (SF ±0.0055) → "insensitive to hyperparameters" · ห้ามเลือกค่าที่ดีที่สุดจาก dev ไปอ้างว่าเป็นโมเดลใหม่ (optimistic bias)
set -euo pipefail
CITY="${1:-sf}"; SEEDS="${SEEDS:-0,1,2}"; ONLY="${ONLY:-}"; FULL="${FULL:-0}"
cd "$(dirname "$0")"; source ~/Documents/LSTM/myenv/bin/activate 2>/dev/null || source ~/mcs/venv/bin/activate 2>/dev/null || true
[ -f "models/$CITY/base_M3/summary.json" ] || { echo "ไม่พบ models/$CITY/base_M3"; exit 1; }
SHA=$(sha256sum sD7_pred_eval.py | cut -c1-8); [ "$SHA" = "0fe82c5d" ] || { echo "sD7_pred_eval.py ไม่ใช่ v7 (sha $SHA)"; exit 1; }

[ -f sD5_train_hp.py ] || { echo "ไม่พบ sD5_train_hp.py"; exit 1; }
declare -A HP=(
  [hp_lr2e-4]="--lr 2e-4"                            # learning rate ½×   (base 5e-4)
  [hp_lr1e-3]="--lr 1e-3"                            # learning rate 2×
  [hp_h128]="--hidden 128"                           # LSTM ชั้น 2: 256 → 128  (params ≈ 0.85 M)
  [hp_h512]="--hidden 512"                           # 256 → 512               (params ≈ 7 M)
  [hp_b256]="--batch 256"                            # batch 128 → 256
  [hp_do030]="--dropout 0.30"                        # dropout 0.15 → 0.30 (หลัง AttentionPool และ Dense(64))
  [hp_adam]="--opt adam"                             # AdamW (wd 1e-4) → Adam ไม่มี weight decay
  [hp_eps]="--eps-start 0.1 --eps-end 0.02"          # ε-WTA ชันขึ้น (base 0.2 → 0.05)
  [hp_const]="--sched const"                         # warm-up 5 แล้ว lr คงที่ (base cosine)
  [hp_sgd]="--opt sgd --lr 1e-2"                     # SGD momentum 0.9 nesterov (lr ต้องสูงกว่า Adam ~20×)
)
ORDER=(hp_lr2e-4 hp_lr1e-3 hp_h128 hp_h512 hp_b256 hp_do030 hp_adam hp_eps); [ "$FULL" = "1" ] && ORDER+=(hp_const hp_sgd)
[ -n "$ONLY" ] && ORDER=("$ONLY")

t0=$(date +%s); stamp() { echo "== [$(date '+%Y-%m-%d %H:%M:%S') +$(( ($(date +%s)-t0)/60 )) min] $*"; }
NEW=()
for r in "${ORDER[@]}"; do
  [ -n "${HP[$r]:-}" ] || { echo "ไม่รู้จัก $r"; exit 1; }
  if [ -f "models/$CITY/$r/summary.json" ]; then echo "== $r มีแล้ว ข้าม D5"; else
    stamp "D5 $CITY/$r (${HP[$r]}) seeds=$SEEDS"; python sD5_train_hp.py --city "$CITY" --run "$r" --seeds "$SEEDS" ${HP[$r]}; fi
  [ -f "models/$CITY/$r/D7_dev/pred_metrics.csv" ] || NEW+=("$r")
done
if [ ${#NEW[@]} -gt 0 ]; then
  RUNS=$(IFS=,; echo "${NEW[*]}")
  stamp "D6 $CITY $RUNS";     python sD6_calibrate.py --city "$CITY" --runs "$RUNS"
  stamp "D7 dev $CITY $RUNS"; python sD7_pred_eval.py --city "$CITY" --runs "$RUNS" --split dev
fi
ALL="base_M3"; for r in hp_lr2e-4 hp_lr1e-3 hp_h128 hp_h512 hp_b256 hp_do030 hp_adam hp_eps hp_const hp_sgd; do [ -f "models/$CITY/$r/D7_dev/pred_metrics.csv" ] && ALL="$ALL,$r"; done
stamp "ตาราง $ALL"; python sX_compare_runs.py --city "$CITY" --runs "$ALL" --out "reports/hp_sens_$CITY.md"
stamp "เสร็จ — reports/hp_sens_$CITY.md"
