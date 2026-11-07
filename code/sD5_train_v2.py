#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sD5_train_v2.py — sD5_train.py เดิมทั้งหมด (encoder · 3 หัว · loss · callbacks · npz) + 2 อย่าง
  --proc <dir>   อ่านหน้าต่างจาก data/<city>/<dir>/ (เช่น processed_f11 ที่ sD1_D4_prepare_v2 สร้าง) และ scaler.json ในโฟลเดอร์นั้น
  --config       รับ config variant ที่มี `base:` (load_config_v2) → cfg["features"] ตรงกับคอลัมน์ X ของ variant
ไม่แก้ sD5_train.py (import แล้ว patch 2 ฟังก์ชัน) · ผลลัพธ์ models/<city>/<run>/ เหมือนเดิม → D6/D7 v7 ใช้ต่อได้ทันที

CLI:
  python sD5_train_v2.py --city sf   --run abl_occ  --proc processed_f11 --config config/data_sf_occ.yaml  --with-occ --seeds 0,1,2
  python sD5_train_v2.py --city sf   --run abl_stop --proc processed_f13 --config config/data_sf_stop.yaml  --seeds 0,1,2
  python sD5_train_v2.py --city rome --run abl_stop --proc processed_f13 --config config/data_rome_stop.yaml --seeds 0,1,2
  (อาร์กิวเมนต์อื่นทั้งหมดของ sD5_train.py ใช้ได้เหมือนเดิม: --M --rnn --epochs --batch …)
หมายเหตุ: ถ้า --proc processed (ค่าเริ่มต้น) และไม่ใส่ --config → พฤติกรรมเท่ากับ sD5_train.py ทุกประการ
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sD5_train as D5                       # noqa: E402
from sD1_D4_prepare_v2 import load_config_v2, merged_config_path  # noqa: E402

# ----------------------------------------------------------------------------- --proc (ดึงออกก่อน argparse ของเดิม)
def _pop_proc(argv: list) -> str:
    proc = "processed"
    if "--proc" in argv:
        k = argv.index("--proc"); proc = argv[k + 1]; del argv[k:k + 2]
    else:
        for k, s in enumerate(argv):
            if s.startswith("--proc="): proc = s.split("=", 1)[1]; del argv[k]; break
    return proc

PROC = _pop_proc(sys.argv)

# ----------------------------------------------------------------------------- patch 1: loader อ่าน processed_<tag>/ + scaler ในนั้น
def load_d4_v2(data_root: Path, city: str, cfg: dict, use_feats: list, subsample=None, seed=0):
    import pandas as pd
    proc = data_root / city / PROC
    if not proc.exists(): sys.exit(f"ไม่พบ {proc} — รัน sD1_D4_prepare_v2.py ก่อน")
    F_all = list(cfg["features"])
    missing = [f for f in use_feats if f not in F_all]
    if missing: sys.exit(f"ฟีเจอร์ {missing} ไม่อยู่ใน config features {F_all} — --config ต้องเป็นไฟล์ variant เดียวกับที่สร้าง {PROC}")
    idx = [F_all.index(f) for f in use_feats]
    sc_path = proc / "scaler.json" if (proc / "scaler.json").exists() else data_root / city / "scaler.json"
    sc = json.load(open(sc_path))
    print(f"  processed_dir {proc.name} · scaler {sc_path.relative_to(data_root)} (z-scored: {sc['names']})")
    d = {}
    for sp in ("train", "val", "test"):
        X = np.load(proc / f"{sp}_X.npy", mmap_mode="r")
        if X.shape[-1] != len(F_all): sys.exit(f"{sp}_X มี {X.shape[-1]} คอลัมน์ แต่ config features มี {len(F_all)} — config กับ {PROC} ไม่ตรงกัน")
        X = X[:, :, idx].astype(np.float32)
        Y = np.load(proc / f"{sp}_Y.npy").astype(np.float32)
        meta = D5.read_meta(proc / f"{sp}_meta")
        if sp == "train" and subsample:
            rng = np.random.RandomState(seed); keep = np.sort(rng.choice(len(X), int(len(X) * subsample), replace=False))
            X, Y, meta = X[keep], Y[keep], meta.iloc[keep].reset_index(drop=True)
        d[sp] = (X, Y, meta)
        print(f"  {sp:<5} X {X.shape}  Y {Y.shape}  cycle-windows {int(meta['is_cycle_start'].sum()) if 'is_cycle_start' in meta else 0}")
    dxdy = {}
    for name in ("dx", "dy"):
        j = sc["names"].index(name); dxdy[name] = (float(sc["mean"][j]), float(sc["std"][j]))
    return d, dxdy

# ----------------------------------------------------------------------------- patch 2: config variant (base:)
def load_cfg_v2(cfg_path: str, city: str) -> dict:
    return load_config_v2(cfg_path, city)

# ----------------------------------------------------------------------------- patch 3: บันทึก proc ลง run_config.json
_parse0 = D5.parse
def parse_v2():
    a = _parse0(); a.proc = PROC
    if a.config:
        a.config_variant = a.config; a.config = merged_config_path(a.config)   # run_config.json["config"] → ไฟล์รวม (sD7 v7 อ่าน dates.test_dev ได้)
    return a

D5.load_d4 = load_d4_v2
D5.load_cfg = load_cfg_v2
D5.parse = parse_v2

if __name__ == "__main__":
    print(f"sD5_train_v2 · --proc {PROC}")
    D5.main()
