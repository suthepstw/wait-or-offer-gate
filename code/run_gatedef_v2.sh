#!/usr/bin/env bash
# ชุดป้องกัน gate (SPEC #4 + #5 greedy) · ใช้: nohup bash run_gatedef_v2.sh dev > log_gd_dev.txt 2>&1 &  ·  greedy บน dev: bash run_gatedef_v2.sh gdev  ·  หลังล็อกค่า: nohup bash run_gatedef_v2.sh final > log_gd_final.txt 2>&1 &
set -u; MODE=${1:-dev}; cd "$(dirname "$0")"
flags() { case "$1" in
  b0_k*)   echo "--gate ${1#b0_k}" ;;
  b0dz_k*) echo "--gate ${1#b0dz_k} --drop-zero" ;;
  g1dz_d*) echo "--delay ${1#g1dz_d} --drop-zero" ;;
  g2dz_t*) echo "--gate-fixed ${1#g2dz_t} --drop-zero" ;;
  g3dz_k*) echo "--gate ${1#g3dz_k} --gate-raw --drop-zero" ;;
  grst_none)   echo "--algo greedy" ;;
  grfree_none) echo "--algo greedy --withdraw free" ;;
  grdz_k*) echo "--algo greedy --gate ${1#grdz_k} --drop-zero" ;;
  *) echo "BAD"; ;; esac; }
run_list() {  # $1 = city, $2.. = names (รันต่อกันในกระบวนการเดียว)
  local c=$1; shift
  for n in "$@"; do
    [ -f "$ROOT/$n/$c/e5_runs.csv" ] && { echo "ข้าม $n/$c (มีแล้ว)"; continue; }
    SHR="--shrink e6_rho/$c/rho.json"; case $n in grst_none|grfree_none) SHR="" ;; esac          # greedy ไม่มี gate = เทียบ NSGA-II เดิม (ไม่มี C̃)
    python sE5_replan.py --city $c --inst $INST --e5 $E5 --out $ROOT/$n --algo nsga2 --arms qm_model --adaptive-only --withdraw sticky --gen 480 --e4 none $SHR $(flags $n) > logs_gd/${MODE}_${n}_$c.txt 2>&1 || echo "❌ $n/$c ดู logs_gd/${MODE}_${n}_$c.txt"
    echo "เสร็จ $n/$c $(date +%H:%M)"
  done; }
mkdir -p logs_gd
if [ "$MODE" = dev ]; then
  INST=instances_devsplit; E5=e5inst_devsplit; ROOT=e6gd_dev
  for c in sf rome; do
    run_list $c b0_k1.5 b0dz_k0.5 b0dz_k1.0 b0dz_k1.5 b0dz_k2.0 b0dz_k3.0 &
    run_list $c g1dz_d10 g1dz_d20 g1dz_d30 &
    run_list $c g2dz_t0.02 g2dz_t0.05 g2dz_t0.10 g2dz_t0.15 g2dz_t0.20 &
    run_list $c g3dz_k0.5 g3dz_k1.0 g3dz_k1.5 g3dz_k2.0 g3dz_k3.0 &
  done; wait
  python sE6_gatedef.py --root e6gd_dev --b0-old e6dev_sticky/nsga2_sh_g1.5
elif [ "$MODE" = g2ext ]; then                                                                     # SPEC #4b: τ ที่ดีที่สุดอยู่ที่ขอบ grid (0.20) → ขยายบน dev ก่อน final
  INST=instances_devsplit; E5=e5inst_devsplit; ROOT=e6gd_dev
  for c in sf rome; do run_list $c g2dz_t0.25 g2dz_t0.30 g2dz_t0.40 & done; wait
  python sE6_gatedef.py --root e6gd_dev --b0-old e6dev_sticky/nsga2_sh_g1.5
elif [ "$MODE" = gdev ]; then
  INST=instances_devsplit; E5=e5inst_devsplit; ROOT=e6gd_dev
  for c in sf rome; do run_list $c grst_none grfree_none grdz_k0.5 grdz_k1.0 grdz_k1.5 grdz_k2.0 grdz_k3.0 & done; wait
  python sE6_gatedef.py --root e6gd_dev --b0-old e6dev_sticky/nsga2_sh_g1.5
elif [ "$MODE" = final ]; then
  [ -f e6gd_dev/GATEDEF_select.json ] || { echo "ไม่มี e6gd_dev/GATEDEF_select.json — รัน dev ก่อน"; exit 1; }
  [ -d e6gd_final ] && { echo "e6gd_final มีอยู่แล้ว — final รันครั้งเดียว ไม่รันซ้ำ"; exit 1; }
  INST=instances_v3; E5=e5inst_v3; ROOT=e6gd_final
  NAMES=$(python -c "import json; s = json.load(open('e6gd_dev/GATEDEF_select.json'))['select']; print(' '.join([v['name'] for v in s.values()] + ['grst_none', 'grfree_none', 'grdz_k%g' % s['b0dz']['param']]))")
  echo "ล็อกจาก dev: $NAMES"
  for c in sf rome; do for n in $NAMES; do run_list $c $n & done; done; wait
  python sE6_gatedef.py --final e6gd_final --select e6gd_dev/GATEDEF_select.json --b0-final e6final_sticky/nsga2_sh_g1.5 --nsga-sticky e6final_sticky/nsga2 --nsga-free e6final_free/nsga2
else echo "ใช้: bash run_gatedef_v2.sh dev|final"; fi
