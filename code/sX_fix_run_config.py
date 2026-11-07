#!/usr/bin/env python3
"""sX_fix_run_config.py — แก้ run_config.json ของ run ที่ train ด้วย sD5_train_v2 รุ่นแรก (14 ต.ค.) ซึ่งบันทึก config เป็นไฟล์ variant ที่มี `base:`
   → sD7 v7 อ่าน yaml ตรง ๆ ไม่เห็น dates.test_dev → ประเมิน test ทั้งก้อนแทน dev  ·  สคริปต์นี้ชี้ config ไปที่ไฟล์รวม (merged_*.yaml) แล้วต้องรัน D7 dev ใหม่
ใช้:  python sX_fix_run_config.py            (ตรวจทุก models/*/abl_*/run_config.json)
      แล้ว  python sD7_pred_eval.py --city sf --runs <run ที่แก้> --split dev"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sD1_D4_prepare_v2 import merged_config_path
HERE = Path(__file__).resolve().parent
fixed = []
for rc in sorted((HERE / "models").glob("*/abl_*/run_config.json")):
    d = json.load(open(rc, encoding="utf-8")); c = d.get("config") or ""
    if not c or "merged_" in c: continue
    try: raw = open(HERE / c if not Path(c).is_absolute() else c, encoding="utf-8").read()
    except FileNotFoundError: continue
    if "base:" not in raw: continue
    d["config_variant"] = c; d["config"] = merged_config_path(str(HERE / c if not Path(c).is_absolute() else c))
    json.dump(d, open(rc, "w", encoding="utf-8"), indent=2, ensure_ascii=False); fixed.append(f"{rc.parent.parent.name}/{rc.parent.name}")
print("แก้แล้ว:", fixed if fixed else "ไม่มีที่ต้องแก้")
if fixed: print("ต้องรัน D7 dev ใหม่สำหรับ run เหล่านี้ (D5/D6 ไม่ต้อง): python sD7_pred_eval.py --city <city> --runs <run> --split dev")
