#!/usr/bin/env bash
# run_abl2.sh — Bi-LSTM+ รอบ 2 (แตะโมเดลเฉพาะชั้นก่อน Dense(64) / หัวเสริม): hist30 · cab · prior · mtl → D6 → D7 v7 dev → ตาราง
#   ใช้:   nohup bash run_abl2.sh sf h30   > log_abl2_sf_h30.txt   2>&1 &   # M3  ประวัติ 30 ก้าว (+ control abl_hist30_k8 อัตโนมัติ) · D6/D7 ใช้ --data data_h30
#          nohup bash run_abl2.sh sf cab   > log_abl2_sf_cab.txt   2>&1 &   # M4  per-cab embedding 16
#          nohup bash run_abl2.sh sf prior > log_abl2_sf_prior.txt 2>&1 &   # M5  fleet transition prior 32
#          nohup bash run_abl2.sh sf mtl   > log_abl2_sf_mtl.txt   2>&1 &   # หัว reachability (auxiliary) W=${MTL_W:-0.1}
#          bash run_abl2.sh sf h30 tfm            → abl_hist30_tfm (Transformer บน 30 ก้าว — คู่ที่ต้องรันกับ h30 เสมอ)
#          BASE=stop bash run_abl2.sh sf cab      → ซ้อนบนชุดอินพุตที่ผ่าน M7 (processed ของ config/data_sf_stop.yaml) → abl_stop_cab
#          SEEDS=0,1,2 · EXTRA="--batch 256" · MTL_W=0.1 · CONTROL=0 (ไม่รัน k8) · รันทีละงาน (GPU เดียว)
#   ไม่แตะของเดิม: processed/ · scaler.json · base_M3 · sD7 v7 (ตรวจ sha) · เขียน data/<city>/processed_<tag>/ · data_h30/ · models/<city>/abl_*/
set -euo pipefail
CITY="${1:?ใช้: bash run_abl2.sh <sf|rome> <h30|cab|prior|mtl> [lstm|gru|rnn|tfm|uni|gru_uni|rnn_uni]}"; VAR="${2:?variant}"; ENC="${3:-lstm}"
SEEDS="${SEEDS:-0,1,2}"; EXTRA="${EXTRA:-}"; MTL_W="${MTL_W:-0.1}"; CONTROL="${CONTROL:-1}"; BASE="${BASE:-}"
cd "$(dirname "$0")"; source ~/Documents/LSTM/myenv/bin/activate 2>/dev/null || source ~/mcs/venv/bin/activate 2>/dev/null || true
[ -f "models/$CITY/base_M3/summary.json" ] || { echo "ไม่พบ models/$CITY/base_M3"; exit 1; }
SHA=$(sha256sum sD7_pred_eval.py | cut -c1-8); [ "$SHA" = "0fe82c5d" ] || { echo "sD7_pred_eval.py ไม่ใช่ v7 (sha $SHA) — หยุด"; exit 1; }
case "$ENC" in
  lstm) ENCARG=""; SUF="";; gru) ENCARG="--rnn gru"; SUF="_gru";; rnn) ENCARG="--rnn rnn"; SUF="_rnn";; tfm) ENCARG="--rnn tfm"; SUF="_tfm";;
  uni) ENCARG="--no-bi"; SUF="_uni";; gru_uni) ENCARG="--rnn gru --no-bi"; SUF="_gru_uni";; rnn_uni) ENCARG="--rnn rnn --no-bi"; SUF="_rnn_uni";;
  *) echo "encoder ไม่รู้จัก: $ENC"; exit 1;;
esac
t0=$(date +%s); stamp() { echo "== [$(date '+%Y-%m-%d %H:%M:%S') +$(( ($(date +%s)-t0)/60 )) min] $*"; }

# ---- อินพุตตั้งต้น: base (processed/, config เดิม) หรือชุดที่ผ่าน M7 (BASE=occ|stop|occstop)
if [ -n "$BASE" ]; then
  BCFG="config/data_${CITY}_${BASE}.yaml"; [ -f "$BCFG" ] || { echo "ไม่พบ $BCFG"; exit 1; }
  read -r BPROC BOCC <<EOF
$(python - "$BCFG" <<'PY'
import sys, yaml; c = yaml.safe_load(open(sys.argv[1])); print(c["processed_dir"], "--with-occ" if "occ" in c.get("features", []) else "")
PY
)
EOF
  [ -f "data/$CITY/$BPROC/train_X.npy" ] || { echo "ไม่พบ data/$CITY/$BPROC — รัน run_abl.sh $CITY $BASE ก่อน"; exit 1; }
  PFX="abl_${BASE}_"
else BCFG="config/data_${CITY}.yaml"; BPROC="processed"; BOCC=""; PFX="abl_"; fi

DATA_ARG=""   # D6/D7 ใช้ data/ เดิม เว้น hist30
case "$VAR" in
  h30)   RUN="${PFX}hist30${SUF}"; CFG="config/data_${CITY}_h30.yaml"; PROC="processed_h30"; D5ARG="--config $CFG"; DATA_ARG="--data data_h30"
         [ -z "$BASE" ] || { echo "h30 ซ้อนบน BASE ยังไม่รองรับ (ต้องทำ config h30 ที่มี features ชุดนั้น)"; exit 1; } ;;
  cab)   RUN="${PFX}cab${SUF}";   CFG="$BCFG"; PROC="$BPROC"; D5ARG="--config $CFG $BOCC --cab-emb ${CAB_DIM:-16}" ;;
  prior) RUN="${PFX}prior${SUF}"; CFG="config/data_${CITY}_prior.yaml"; PROC="processed_prior"; D5ARG="--config $CFG --prior"
         [ -z "$BASE" ] || { echo "prior ซ้อนบน BASE: ทำ config prior ที่มี features ชุดนั้นก่อน"; exit 1; } ;;
  mtl)   RUN="${PFX}mtl${SUF}";   CFG="$BCFG"; PROC="$BPROC"; D5ARG="--config $CFG $BOCC --mtl $MTL_W" ;;
  *) echo "variant ไม่รู้จัก: $VAR (h30|cab|prior|mtl)"; exit 1;;
esac
[ "$RUN" != "base_M3" ] || exit 1

# ---- D4 v2 (เฉพาะ h30 / prior)
if [ "$VAR" = "h30" ] || [ "$VAR" = "prior" ]; then
  stamp "D4 v2 $CITY/$VAR → data/$CITY/$PROC"
  if [ -f "data/$CITY/$PROC/train_X.npy" ]; then echo "   มีแล้ว ข้าม"; else python sD1_D4_prepare_v2.py --city "$CITY" --config "$CFG"; fi
  if [ "$VAR" = "prior" ]; then
    python -c "import json,sys; a=json.load(open('data/$CITY/manifest_prior.json'))['align_with_processed']; sys.exit(0 if a and all(a.values()) else 1)" \
      || { echo "⚠ prior: หน้าต่างไม่ตรง processed/ เดิม — หยุด"; exit 1; }
  fi
fi

# ---- D5 v3
RUNS="$RUN"
run_d5() {  # $1 = run name, $2.. = args
  local r="$1"; shift
  if [ -f "models/$CITY/$r/summary.json" ]; then echo "== $r มีแล้ว ข้าม D5"; else
    stamp "D5 v3 $CITY/$r ($*) seeds=$SEEDS"; python sD5_train_v3.py --city "$CITY" --run "$r" --proc "$PROC" --seeds "$SEEDS" "$@" $ENCARG $EXTRA; fi
}
run_d5 "$RUN" $D5ARG
if [ "$VAR" = "h30" ] && [ "$CONTROL" = "1" ]; then   # control: โมเดล 8 ก้าวบนหน้าต่างชุดเดียวกับ 30 ก้าว
  run_d5 "${PFX}hist30_k8${SUF}" --config "$CFG" --last-k 8; RUNS="$RUNS,${PFX}hist30_k8${SUF}"
fi

# ---- D6 / D7 v7 (hist30 → --data data_h30 ทั้งคู่)
if [ -n "$DATA_ARG" ]; then
  python sD6_calibrate.py -h 2>/dev/null | grep -q -- "--data" || { echo "sD6_calibrate.py ไม่มี --data → เพิ่มก่อน (hist30 ต้องอ่าน data_h30/)"; exit 1; }
fi
stamp "D6 $CITY $RUNS";     python sD6_calibrate.py --city "$CITY" --runs "$RUNS" $DATA_ARG
stamp "D7 dev $CITY $RUNS"; python sD7_pred_eval.py --city "$CITY" --runs "$RUNS" --split dev $DATA_ARG

# ---- ตาราง
ALL="base_M3"; for d in models/$CITY/abl_*/D7_dev/pred_metrics.csv; do [ -f "$d" ] && ALL="$ALL,$(basename "$(dirname "$(dirname "$d")")")"; done
python sX_compare_runs.py --city "$CITY" --runs "$ALL" --out "reports/input_ablation_$CITY.md" || true
stamp "เสร็จ — models/$CITY/$RUN/D7_dev/D7_report.md · reports/input_ablation_$CITY.md"
