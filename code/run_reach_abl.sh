#!/usr/bin/env bash
# run_reach_abl.sh — ablation ของหัว (9a′) บน SF · 4 run ต่อกัน (≈ 12 ชม. GPU) · ไม่แตะ reach_lstm / base_M3
#   ใช้:  nohup bash run_reach_abl.sh sf > log_reach_abl_sf.txt 2>&1 &
#         ONLY=lam1 bash run_reach_abl.sh sf          # รันตัวเดียว (occstop | lam1 | lam4 | only)
#   run:  reach_lstm_occstop  อินพุต 15 ตัว (occ, occ_dt, stop×4) บนหัวใหม่ → อินพุตอิ่มตัวจริงไหมภายใต้ (9a′)   (ต้องมี data/<city>/processed_f15 จาก sD1_D4_prepare_v2 รอบ abl_occstop)
#         reach_lstm_lam1 / reach_lstm_lam4   λ_reach 1 / 4 (หลัก 2)  → ความไวต่อน้ำหนัก loss ของหัวใหม่
#         reach_lstm_only   --reach-only (ปิด loss mixture)             → ฝึกร่วมกับ mixture จำเป็นไหม
#   ผล:  reports/reach_eval_<city>.md (ทุก run reach_* + base_M3 + abl_tfm บนคู่เดียวกัน) · reports/reach_abl_<city>.md (เทียบกับ reach_lstm/qm_reach_cal แบบ paired)
#   อ่าน: ค่าหลักของ paper คือ λ=2 — รายงาน lam1/lam4 ทั้งคู่ ห้ามหยิบตัวที่ดีที่สุดมาแทน (optimistic bias) · occstop: Δ ≤ 0.003 → ตาราง input ablation เดิมถ่ายทอดมา (9a′) ได้
set -euo pipefail
CITY="${1:-sf}"; ONLY="${ONLY:-}"
cd "$(dirname "$0")"; source ~/Documents/LSTM/myenv/bin/activate 2>/dev/null || source ~/mcs/venv/bin/activate 2>/dev/null || true
[ -f "models/$CITY/reach_lstm/summary.json" ] || { echo "ต้องมี models/$CITY/reach_lstm ก่อน (bash run_reach.sh $CITY lstm)"; exit 1; }
t0=$(date +%s); stamp() { echo "#### [$(date '+%Y-%m-%d %H:%M:%S') +$(( ($(date +%s)-t0)/60 )) min] $*"; }

# occstop: หา data root ที่ D6/D7 ต้องใช้จาก manifest ของ sD1_D4_prepare_v2 (align → ว่าง · ไม่ align → data_f15)
OCC_DATA=""
MF=$(ls data/$CITY/reports/manifest_occstop.json data/$CITY/manifest_occstop.json 2>/dev/null | head -1 || true)
if [ -n "$MF" ]; then
  OCC_DATA=$(python -c "import json,sys;m=json.load(open('$MF'));print('' if m.get('align_with_processed',True) else m.get('data_root_for_D6_D7','data_f15'))")
elif [ -d "data_f15/$CITY/processed" ]; then OCC_DATA="data_f15"; fi

ORDER=(occstop lam1 lam4 only); [ -n "$ONLY" ] && ORDER=("$ONLY")
for a in "${ORDER[@]}"; do
  case "$a" in
    occstop) [ -d "data/$CITY/processed_f15" ] || { echo "ไม่พบ data/$CITY/processed_f15 — ข้าม occstop"; continue; }
             stamp "reach_lstm_occstop (proc=processed_f15 data=${OCC_DATA:-data})"
             PROC=processed_f15 DATA="$OCC_DATA" EXTRA="--config config/data_${CITY}_occstop.yaml --with-occ" bash run_reach.sh "$CITY" lstm_occstop ;;
    lam1)    stamp "reach_lstm_lam1 (λ=1)"; LAM=1.0 bash run_reach.sh "$CITY" lstm_lam1 ;;
    lam4)    stamp "reach_lstm_lam4 (λ=4)"; LAM=4.0 bash run_reach.sh "$CITY" lstm_lam4 ;;
    only)    stamp "reach_lstm_only (--reach-only)"; EXTRA="--reach-only" bash run_reach.sh "$CITY" lstm_only ;;
    *) echo "ไม่รู้จัก $a"; exit 1 ;;
  esac
done

# ตารางเทียบกับ reach_lstm (หัวหลัก) แบบ paired — เฉพาะ run ที่ align กับ processed/
RUNS="reach_lstm"; for r in reach_lstm_lam1 reach_lstm_lam4 reach_lstm_only reach_tfm; do [ -f "models/$CITY/$r/seed0/pred_test.npz" ] && RUNS="$RUNS,$r"; done
[ -z "$OCC_DATA" ] && [ -f "models/$CITY/reach_lstm_occstop/seed0/pred_test.npz" ] && RUNS="$RUNS,reach_lstm_occstop"
stamp "ตาราง ablation: $RUNS"
python sD7_reach_eval.py --city "$CITY" --runs "$RUNS" --split dev --ref reach_lstm --ref-variant qm_reach_cal --out "reports/reach_abl_$CITY.md"
[ -n "$OCC_DATA" ] && echo "หมายเหตุ: reach_lstm_occstop ใช้ window คนละชุด → ดู reports/reach_eval_${CITY}_reach_lstm_occstop.md แยก (เทียบ qm_reach_cal กับ qm_ub_cal ภายใน run เดียวกัน)"
stamp "เสร็จ — reports/reach_abl_$CITY.md"; cat "reports/reach_abl_$CITY.md"
