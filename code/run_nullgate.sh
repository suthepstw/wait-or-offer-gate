#!/usr/bin/env bash
# SPEC #6 · gate + physics null · ใช้: nohup bash run_nullgate.sh dev > log_ng_dev.txt 2>&1 &   แล้วหลังตรวจ dev: nohup bash run_nullgate.sh final > log_ng_final.txt 2>&1 &
set -u; MODE=${1:-dev}; cd "$(dirname "$0")"; mkdir -p logs_ng
run_one() {  # $1 city $2 name
  local c=$1 n=$2 f=""
  [ -f "$ROOT/$n/$c/e5_runs.csv" ] && { echo "ข้าม $n/$c (มีแล้ว)"; return; }
  [ "$n" != null_none ] && f="--shrink e6_rho_null/$c/rho.json --gate ${n#nullg_k} --drop-zero"
  python sE5_replan.py --city $c --inst $INST --e5 $E5 --out $ROOT/$n --algo nsga2 --arms qm_null_dv --adaptive-only --withdraw sticky --gen 480 --e4 none $f > logs_ng/${MODE}_${n}_$c.txt 2>&1 || echo "❌ $n/$c ดู logs_ng/${MODE}_${n}_$c.txt"
  echo "เสร็จ $n/$c $(date +%H:%M)"; }
if [ "$MODE" = dev ]; then
  INST=instances_devsplit; E5=e5inst_devsplit; ROOT=e6ng_dev
  for c in sf rome; do
    [ -f e6_rho_null/$c/rho.json ] || python sE6_rho.py --city $c --inst instances_devsplit --e5 e5inst_devsplit --arm qm_null_dv --out e6_rho_null > logs_ng/rho_null_$c.txt 2>&1 &
  done; wait
  for c in sf rome; do [ -f e6_rho_null/$c/rho.json ] || { echo "❌ ไม่มี e6_rho_null/$c/rho.json ดู logs_ng/rho_null_$c.txt"; exit 1; }; echo "ρ/V ของ null $c พร้อม"; done
  for c in sf rome; do
    (run_one $c null_none; run_one $c nullg_k0.5) &
    (run_one $c nullg_k1.0; run_one $c nullg_k1.5) &
    (run_one $c nullg_k2.0; run_one $c nullg_k3.0) &
  done; wait
  python sE6_nullgate.py --root e6ng_dev --model-gate e6gd_dev/b0dz_k1.5 --model-none e6dev_sticky/nsga2
elif [ "$MODE" = final ]; then
  [ -f e6ng_dev/NULLGATE_select.json ] || { echo "ไม่มี e6ng_dev/NULLGATE_select.json — รัน dev ก่อน"; exit 1; }
  [ -d e6ng_final ] && { echo "e6ng_final มีอยู่แล้ว — final รันครั้งเดียว ไม่รันซ้ำ"; exit 1; }
  INST=instances_v3; E5=e5inst_v3; ROOT=e6ng_final
  N=$(python -c "import json; print(json.load(open('e6ng_dev/NULLGATE_select.json'))['name'])"); echo "ล็อกจาก dev: $N"
  for c in sf rome; do run_one $c null_none & run_one $c $N & done; wait
  python sE6_nullgate.py --final e6ng_final --select e6ng_dev/NULLGATE_select.json --model-gate e6gd_final/b0dz_k1.5 --model-none e6final_sticky/nsga2
else echo "ใช้: bash run_nullgate.sh dev|final"; fi
