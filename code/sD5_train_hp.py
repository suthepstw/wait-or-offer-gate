#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sD5_train_hp.py — sD5_train.py เดิม + 3 ตัวเลือกที่ไม่มีใน CLI เดิม (ไม่แก้ sD5_train.py; import แล้ว patch)
  --dropout P        Dropout หลัง AttentionPool และหลัง Dense(64)  (base 0.15)
  --opt adamw|adam|sgd   optimizer (base AdamW wd 1e-4 clipnorm 1 · adam = ไม่มี weight decay · sgd = momentum 0.9 nesterov)
  --sched cosine|const   lr schedule (base warm-up 5 epoch + cosine · const = warm-up 5 แล้วคงที่)
อาร์กิวเมนต์อื่นทั้งหมดของ sD5_train.py ใช้ได้เหมือนเดิม (--lr --hidden --batch --eps-start …)
CLI:  python sD5_train_hp.py --city sf --run hp_do030 --dropout 0.30 --seeds 0,1,2
      python sD5_train_hp.py --city sf --run hp_adam  --opt adam     --seeds 0,1,2
      python sD5_train_hp.py --city sf --run hp_const --sched const  --seeds 0,1,2
"""
from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))

def _pop(argv, name, default):
    if name in argv:
        k = argv.index(name); v = argv[k + 1]; del argv[k:k + 2]; return v
    for k, s in enumerate(argv):
        if s.startswith(name + "="): del argv[k]; return s.split("=", 1)[1]
    return default

DROPOUT = float(_pop(sys.argv, "--dropout", 0.15)); OPT = _pop(sys.argv, "--opt", "adamw").lower(); SCHED = _pop(sys.argv, "--sched", "cosine").lower()
assert OPT in ("adamw", "adam", "sgd") and SCHED in ("cosine", "const")

import sD5_train as D5                      # noqa: E402
from sD5_train import keras                 # noqa: E402

# --- dropout: ห่อ build_model ให้ส่ง dropout=DROPOUT เสมอ
_build0 = D5.build_model
def build_model_hp(*a, **kw): kw["dropout"] = DROPOUT; return _build0(*a, **kw)
D5.build_model = build_model_hp

# --- optimizer: train_seed เดิมเรียก keras.optimizers.AdamW(learning_rate, weight_decay, clipnorm) → แทนที่ตัวสร้าง
_AdamW0 = keras.optimizers.AdamW
def _make_opt(learning_rate=5e-4, weight_decay=1e-4, clipnorm=1.0, **kw):
    if OPT == "adamw": return _AdamW0(learning_rate=learning_rate, weight_decay=weight_decay, clipnorm=clipnorm, **kw)
    if OPT == "adam":  return keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm)
    return keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True, clipnorm=clipnorm)
keras.optimizers.AdamW = _make_opt

# --- schedule: แทน warmup_cosine ด้วย warm-up แล้วคงที่
if SCHED == "const":
    def warmup_const(base_lr, total, warmup=5, min_lr=1e-6):
        def f(epoch, lr): return float(base_lr * (epoch + 1) / warmup) if epoch < warmup else float(base_lr)
        return f
    D5.warmup_cosine = warmup_const

# --- บันทึกค่าลง run_config.json
_parse0 = D5.parse
def parse_hp():
    a = _parse0(); a.dropout = DROPOUT; a.opt = OPT; a.sched = SCHED; return a
D5.parse = parse_hp

if __name__ == "__main__":
    print(f"sD5_train_hp · dropout {DROPOUT} · opt {OPT} · sched {SCHED}")
    D5.main()
