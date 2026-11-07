#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sD5_train.py — D5: ฝึก Bi-LSTM encoder (ตัวเดิมจาก LSTM_20seed) + 3 หัวใหม่ (Head-P / Head-U / Head-M)
บนหน้าต่างจาก D4 (data/<city>/processed/*)   ·  ค่าเริ่มต้น = 9 ฟีเจอร์ **ไม่มี occ**

สิ่งที่คงไว้จากสคริปต์เดิม (LSTM_20seed original.py)
  encoder : Bidirectional(LSTM 128) → LayerNorm → Bidirectional(LSTM 256) → LayerNorm
            → multi-head attention pool (4 heads) → Dropout(0.15) → Dense(64, tanh) → Dropout
  optimizer: AdamW(lr 5e-4, wd 1e-4, clipnorm 1) + warm-up 5 epoch แล้ว cosine · EarlyStopping (restore best)
  หลาย seed → สรุป mean ± std · เลือก encoder ได้: Bi-LSTM (หลัก) / Bi-GRU / Bi-vanilla-RNN / Transformer (tfm) / ST-Transformer (sttfm)
            ทุกตัวใช้หัว-loss-ข้อมูล-D6-D7 ชุดเดียวกัน → เปรียบเทียบ encoder อย่างเดียว (run_encoder_cmp.sh)
สิ่งที่เปลี่ยน
  ข้อมูล : อ่าน X (N×M_in×F, z-scored แล้ว) / Y (N×H×2 เมตรจาก x_T) / meta จาก D4 — ไม่ทำ feature engineering ซ้ำ
  ฟีเจอร์ : ตัด occ ออกที่ทางเข้าโมเดล (เลือกคอลัมน์จาก config) → 9 ฟีเจอร์  [x y v dx dy dow_sin dow_cos slot_sin slot_cos]
  หัว    : Head-P  Δ̂ ∈ ℝ^{M×H×2} (residual บนเส้นฐาน "ความเร็วหน่วง" v_T·τ(1−e^{−h/τ}), τ=3 = skip)   Head-U  σ̂ ∈ ℝ^{M×H}   Head-M  π ∈ Δ^M
  loss   : ε-WTA mixture Gaussian NLL (หน่วยเมตรบนพิกัด equirectangular ≈ haversine) + cross-entropy ของ π
           + ADE ของ mode ผู้ชนะ (หน่วย 100 m, กัน σ̂ พองก่อน μ̂ จะเรียน) · ε ลดเชิงเส้น 0.2 → 0.05 ใน 20 epoch แรก
  ตัด   : SWA / EMA / Top-K snapshot / TTA (ไม่จำเป็นกับ NLL; ทำให้ σ̂ ผิด calibration)
  ผลลัพธ์: models/<city>/<run>/seed<k>/{best.weights.h5, history.csv, metrics.json, pred_val.npz, pred_test.npz}
           models/<city>/<run>/summary.json · models/D5_results.csv (ledger ทุก run/seed)
           pred_*.npz = mu (N×M×H×2 เมตรจาก x_T) · sigma (N×M×H) · pi (N×M) + คีย์จาก meta (cab, t_T, lat_T, lon_T, date, hour, is_cycle_start)
           แถวเรียงตรงกับ processed/<split>_meta และ _Y.npy ของ D4  → ใช้ต่อใน D6 (calibrate) / D7 (G1) / sai E

รันจาก PyCharm: แก้บล็อก PYCHARM ด้านล่างแล้วกด Run  ·  หรือ CLI:  python sD5_train.py --city sf --run base_M3 --M 3
ทดสอบเร็ว:  python sD5_train.py --demo     (สร้างข้อมูลสังเคราะห์ 3 โหมด ฝึก 3 epoch ตรวจว่าทั้ง pipeline ทำงาน · --epochs 20 จะเห็น minADE ต่ำกว่า const-vel ชัด)
"""
from __future__ import annotations
import argparse, json, math, os, sys, time, random
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# ============================================================================= PyCharm
PYCHARM = dict(
    CITY="sf",                  # "sf" | "rome" | "beijing"
    DATA="data",                # โฟลเดอร์ผลลัพธ์ D4 (data/<city>/processed) ข้างสคริปต์
    CONFIG=r"",                 # เว้นว่าง = config/data_<city>.yaml
    RUN="base_M3",              # ชื่อ run → models/<city>/<run>/
    M=3,                        # จำนวน mode (ablation: 1, 2, 5)
    RNN="lstm",                 # "lstm" | "gru" | "rnn" | "tfm" (Transformer เชิงเวลา) | "sttfm" (ST-Transformer: + spatial Fourier encoding)
    BI=True,                    # False = ablation no-bi
    SKIP=True,                  # False = ablation no-skip (ไม่ใช้เส้นฐาน)
    SKIP_TAU=3.0,               # เส้นฐาน = (dx_T,dy_T)·τ·(1−e^{−h/τ}) "ความเร็วคงที่แบบหน่วง" · τ=3 ก้าว ≈ ไปต่อได้ ~3 นาที
                                #   0 = เส้นฐานอยู่กับที่ (static) · inf = ความเร็วคงที่เต็ม 60 นาที (smoke บน SF: static ชนะ const-vel 3 เท่า → อย่าใช้)
    HIDDEN=256,                 # units ชั้น RNN ที่ 2 (ชั้นแรก 128 เหมือนเดิม)
    WITH_OCC=False,             # True = ablation A6 (SF เท่านั้น) ใส่ occ กลับเข้ามา
    SEEDS=[0, 1, 2, 3, 4],
    EPOCHS=50, BATCH=128, LR=5e-4, PATIENCE=10,
    EPS_START=0.2, EPS_END=0.05, EPS_EPOCHS=20,
    INIT_FROM=r"",              # fine-tune: path ของ best.weights.h5 จาก SF (Rome/Beijing)
    SUBSAMPLE=None,             # เช่น 0.1 = ใช้ train 10% (ลองเร็ว ๆ) · None = ทั้งหมด
    DEMO=False,                 # True = ข้อมูลสังเคราะห์ ทดสอบ pipeline
)

STEP_SCALE_M = 100.0     # Head-P ทำนายก้าวละ (หน่วย 100 m) แล้ว cumsum → Δ̂ เมตร
SIGMA_MIN_M, SIGMA_SCALE_M, SIGMA_MAX_M = 5.0, 100.0, 8000.0   # σ̂_h = min(σ_min + softplus(raw)·scale·√h, σ_max)
LAMBDA_PI = 1.0          # น้ำหนัก cross-entropy ของ π
LAMBDA_ADE = 1.0         # น้ำหนัก ADE ของ mode ผู้ชนะ (หน่วย 100 m) — กัน σ̂ พองก่อนที่ μ̂ จะเรียนรู้ (NLL อย่างเดียวจะติดกับดักนี้)
H_REPORT = [10, 20, 30, 45, 60]

# ----------------------------------------------------------------------------- TF
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers as L  # noqa: E402
KERAS3 = int(str(keras.__version__).split(".")[0]) >= 3

def setup_tf():
    for g in tf.config.list_physical_devices("GPU"):
        try: tf.config.experimental.set_memory_growth(g, True)
        except Exception: pass
    try: tf.keras.mixed_precision.set_global_policy("float32")
    except Exception: pass
    print(f"TensorFlow {tf.__version__} · Keras {keras.__version__} · GPU {[d.name for d in tf.config.list_physical_devices('GPU')] or 'none'}")

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    try: keras.utils.set_random_seed(seed)
    except Exception: tf.random.set_seed(seed)

# ----------------------------------------------------------------------------- config / data
DEFAULT_FEATURES = ["x", "y", "v", "dx", "dy", "dow_sin", "dow_cos", "slot_sin", "slot_cos", "occ"]

def load_cfg(cfg_path: str, city: str) -> dict:
    try:
        sys.path.insert(0, str(HERE)); from sD1_D4_prepare import load_config  # ใช้ตัวเดียวกับ D1–D4
        return load_config(cfg_path, city)
    except Exception:
        import yaml
        cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
        if "profile" in cfg:
            prof = cfg[cfg["profile"]]; cfg = {k: v for k, v in cfg.items() if k not in ("hf", "full")}; cfg.update(prof)
        cfg.setdefault("features", DEFAULT_FEATURES); cfg["city"] = city
        return cfg

def read_meta(path_noext: Path) -> pd.DataFrame:
    for p in (path_noext.with_suffix(".parquet"), path_noext.with_suffix(".pkl")):
        if p.exists():
            return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_pickle(p)
    raise FileNotFoundError(path_noext)

def load_d4(data_root: Path, city: str, cfg: dict, use_feats: list, subsample=None, seed=0):
    proc = data_root / city / "processed"
    F_all = list(cfg["features"]); idx = [F_all.index(f) for f in use_feats]
    sc = json.load(open(data_root / city / "scaler.json"))
    d = {}
    for sp in ("train", "val", "test"):
        X = np.load(proc / f"{sp}_X.npy", mmap_mode="r")[:, :, idx].astype(np.float32)
        Y = np.load(proc / f"{sp}_Y.npy").astype(np.float32)
        meta = read_meta(proc / f"{sp}_meta")
        if sp == "train" and subsample:
            rng = np.random.RandomState(seed); keep = np.sort(rng.choice(len(X), int(len(X) * subsample), replace=False))
            X, Y, meta = X[keep], Y[keep], meta.iloc[keep].reset_index(drop=True)
        d[sp] = (X, Y, meta)
        print(f"  {sp:<5} X {X.shape}  Y {Y.shape}  cycle-windows {int(meta['is_cycle_start'].sum()) if 'is_cycle_start' in meta else 0}")
    # ค่า mean/std ของ dx, dy (เมตร/ก้าว) สำหรับเส้นฐาน skip
    dxdy = {}
    for name in ("dx", "dy"):
        j = sc["names"].index(name); dxdy[name] = (float(sc["mean"][j]), float(sc["std"][j]))
    return d, dxdy

def make_demo(M_in=8, H=60, F=9, n_train=16000, n_val=1500, n_test=1500, seed=0):
    """ข้อมูลสังเคราะห์: เดินแบบสุ่มเลี้ยวได้ 3 แบบ (ตรง/เลี้ยวซ้าย/เลี้ยวขวา) → ทดสอบ multi-modal"""
    rng = np.random.RandomState(seed)
    def gen(n):
        v = rng.uniform(3, 12, n); th0 = rng.uniform(-np.pi, np.pi, n); mode = rng.randint(0, 3, n)
        turn = np.where(mode == 0, 0.0, np.where(mode == 1, 0.03, -0.03))
        t = np.arange(-M_in + 1, H + 1)[None, :]
        th = th0[:, None] + turn[:, None] * np.clip(t, 0, None)
        step = 60.0 * v[:, None]
        dx = step * np.cos(th); dy = step * np.sin(th)
        x = np.cumsum(dx, 1); y = np.cumsum(dy, 1)
        x -= x[:, M_in - 1:M_in]; y -= y[:, M_in - 1:M_in]
        x += rng.normal(0, 8, x.shape); y += rng.normal(0, 8, y.shape)
        Xin = np.stack([x[:, :M_in] / 1000, y[:, :M_in] / 1000, v[:, None].repeat(M_in, 1) / 10,
                        dx[:, :M_in] / 300, dy[:, :M_in] / 300,
                        np.zeros((n, M_in)), np.ones((n, M_in)), np.zeros((n, M_in)), np.ones((n, M_in))], -1)[..., :F]
        Y = np.stack([x[:, M_in:], y[:, M_in:]], -1)
        meta = pd.DataFrame({"cab": np.arange(n), "is_cycle_start": rng.rand(n) < 0.2, "is_cycle_clean": True})
        return Xin.astype(np.float32), Y.astype(np.float32), meta
    d = {"train": gen(n_train), "val": gen(n_val), "test": gen(n_test)}
    return d, {"dx": (0.0, 300.0), "dy": (0.0, 300.0)}

# ----------------------------------------------------------------------------- model
class AttentionPool(L.Layer):
    """multi-head attention pool ตัวเดิม (Dense(1,tanh) → softmax บนเวลา → weighted sum → concat → Dense(d_in,tanh))"""
    def __init__(self, n_heads=4, **kw):
        super().__init__(**kw); self.n_heads = n_heads
    def build(self, shape):
        d_in = int(shape[-1])
        self.scores = [L.Dense(1, activation="tanh", name=f"score_{k}") for k in range(self.n_heads)]
        self.proj = L.Dense(d_in, activation="tanh", name="proj")
        super().build(shape)
    def call(self, h):
        pooled = []
        for sc in self.scores:
            w = tf.nn.softmax(sc(h), axis=1)                 # (B,T,1)
            pooled.append(tf.reduce_sum(h * w, axis=1))      # (B,d)
        return self.proj(tf.concat(pooled, -1))
    def get_config(self): return dict(super().get_config(), n_heads=self.n_heads)

class PosEmbed(L.Layer):
    def build(self, shape):
        self.pos = self.add_weight(name="pos", shape=(int(shape[1]), int(shape[2])), initializer="zeros"); super().build(shape)
    def call(self, x): return x + self.pos[None]

class TransformerBlock(L.Layer):
    def __init__(self, d, n_heads=4, ff=256, drop=0.1, **kw):
        super().__init__(**kw)
        self.att = L.MultiHeadAttention(n_heads, key_dim=d // n_heads, dropout=drop)
        self.ff1 = L.Dense(ff, activation="gelu"); self.ff2 = L.Dense(d)
        self.ln1 = L.LayerNormalization(); self.ln2 = L.LayerNormalization(); self.drop = L.Dropout(drop)
    def call(self, x, training=None):
        x = self.ln1(x + self.drop(self.att(x, x, training=training), training=training))
        return self.ln2(x + self.drop(self.ff2(self.ff1(x)), training=training))

class SpatialFourier(L.Layer):
    """spatial encoding แบบ ST-Transformer (Fourier features ของพิกัด x,y ที่ z-scored แล้ว):
       ต่อ [sin, cos](2^k·π·x), [sin, cos](2^k·π·y), k = 0..n_freq−1 เข้ากับฟีเจอร์เดิม → (B, T, F + 4·n_freq)
       ให้ attention "เห็นตำแหน่งในเมือง" หลายสเกล (2^k) ไม่ใช่แค่ค่า x,y เชิงเส้น · ไม่ต้องใช้ข้อมูลรถคันอื่น (single-agent)"""
    def __init__(self, x_idx=0, y_idx=1, n_freq=8, **kw):
        super().__init__(**kw); self.x_idx, self.y_idx, self.n_freq = int(x_idx), int(y_idx), int(n_freq)
    def call(self, x):
        xy = tf.gather(x, [self.x_idx, self.y_idx], axis=-1)                       # (B,T,2)
        freqs = (2.0 ** tf.range(self.n_freq, dtype=tf.float32)) * math.pi        # (K,)
        ang = xy[..., None] * freqs                                                 # (B,T,2,K)
        ang = tf.reshape(ang, tf.concat([tf.shape(x)[:-1], [2 * self.n_freq]], 0)) # (B,T,2K)
        return tf.concat([x, tf.sin(ang), tf.cos(ang)], axis=-1)
    def compute_output_shape(self, shape): return (*shape[:-1], int(shape[-1]) + 4 * self.n_freq)
    def get_config(self): return dict(super().get_config(), x_idx=self.x_idx, y_idx=self.y_idx, n_freq=self.n_freq)

class MixtureHead(L.Layer):
    """รวม Head-P/U/M → เวกเตอร์เดียว [mu (M·H·2) | sigma (M·H) | logits (M)]
    mu    = cumsum(step·STEP_SCALE) + skip·baseline   (baseline = (dx_T, dy_T)·h  ความเร็วคงที่)
    sigma = min(SIGMA_MIN + softplus(raw)·SIGMA_SCALE·√h, SIGMA_MAX)   (เมตร; prior โตแบบ √h)"""
    def __init__(self, M, H, skip, dx_idx, dy_idx, dx_ms, dy_ms, skip_tau=3.0, **kw):
        super().__init__(**kw)
        self.M, self.H, self.skip, self.skip_tau = M, H, skip, float(skip_tau)
        self.dx_idx, self.dy_idx, self.dx_ms, self.dy_ms = dx_idx, dy_idx, dx_ms, dy_ms
    def call(self, inputs):
        step_raw, sig_raw, logits, x = inputs
        B = tf.shape(step_raw)[0]
        step = tf.reshape(step_raw, (B, self.M, self.H, 2)) * STEP_SCALE_M
        mu = tf.cumsum(step, axis=2)
        if self.skip:
            dx = x[:, -1, self.dx_idx] * self.dx_ms[1] + self.dx_ms[0]
            dy = x[:, -1, self.dy_idx] * self.dy_ms[1] + self.dy_ms[0]
            hs = tf.constant(skip_gain(self.H, self.skip_tau), tf.float32)[None, None, :, None]
            base = tf.stack([dx, dy], -1)[:, None, None, :] * hs           # (B,1,H,2)
            mu = mu + base
        sqrt_h = tf.sqrt(tf.range(1, self.H + 1, dtype=tf.float32))[None, None, :]
        sigma = tf.minimum(SIGMA_MIN_M + tf.nn.softplus(tf.reshape(sig_raw, (B, self.M, self.H))) * SIGMA_SCALE_M * sqrt_h, SIGMA_MAX_M)
        return tf.concat([tf.reshape(mu, (B, -1)), tf.reshape(sigma, (B, -1)), logits], -1)
    def get_config(self):
        return dict(super().get_config(), M=self.M, H=self.H, skip=self.skip, dx_idx=self.dx_idx, dy_idx=self.dy_idx, dx_ms=self.dx_ms, dy_ms=self.dy_ms, skip_tau=self.skip_tau)

def skip_gain(H: int, tau: float) -> np.ndarray:
    """ตัวคูณของ (dx_T,dy_T) ที่ก้าว h=1..H:  τ·(1−e^{−h/τ})  → h เมื่อ τ=∞ (const-vel), 0 เมื่อ τ=0 (static)"""
    h = np.arange(1, H + 1, dtype=np.float64)
    if tau <= 0: return np.zeros(H)
    if not np.isfinite(tau): return h
    return tau * (1.0 - np.exp(-h / tau))

def build_model(M_in, F, H, M, rnn="lstm", bi=True, hidden=256, n_heads=4, dropout=0.15,
                skip=True, skip_tau=3.0, dx_idx=3, dy_idx=4, dx_ms=(0.0, 1.0), dy_ms=(0.0, 1.0), x_idx=0, y_idx=1, n_freq=8,
                tfm_d=128, tfm_blocks=2, tfm_heads=4):
    x = L.Input(shape=(M_in, F), name="x_seq")
    wrap = (lambda layer: L.Bidirectional(layer)) if bi else (lambda layer: layer)
    if rnn == "lstm":
        h = wrap(L.LSTM(128, return_sequences=True, dropout=0.1))(x); h = L.LayerNormalization()(h)
        h = wrap(L.LSTM(hidden, return_sequences=True, dropout=0.1))(h)
    elif rnn == "gru":
        h = wrap(L.GRU(128, return_sequences=True, dropout=0.1))(x); h = L.LayerNormalization()(h)
        h = wrap(L.GRU(hidden, return_sequences=True, dropout=0.1))(h)
    elif rnn == "rnn":   # vanilla RNN (tanh) — เส้นฐาน recurrent อย่างง่าย (เปรียบเทียบ)
        h = wrap(L.SimpleRNN(128, return_sequences=True, dropout=0.1))(x); h = L.LayerNormalization()(h)
        h = wrap(L.SimpleRNN(hidden, return_sequences=True, dropout=0.1))(h)
    elif rnn == "tfm":   # Transformer encoder เชิงเวลา (เปรียบเทียบ): proj + learned positional embedding + 2 blocks
                         # หมายเหตุ: attention บนแกนเวลา 8 ก้าวของคันเดียว (ไม่ใช่ spatio-temporal grid model)
                         # ขนาด: --tfm-d / --tfm-blocks / --tfm-heads (ค่าเริ่มต้น 128/2/4 ≈ 0.38M params · 256/3/8 ≈ 2.4M ≈ Bi-LSTM)
        d = tfm_d
        h = PosEmbed()(L.Dense(d)(x))
        for _ in range(tfm_blocks): h = TransformerBlock(d, n_heads=tfm_heads, ff=4 * d)(h)
    elif rnn == "sttfm":  # ST-Transformer (single-agent): spatial Fourier encoding ของ (x,y) + learned temporal positional embedding
                          # + 2 blocks self-attention บนแกนเวลา · เวลาในวัน/วันในสัปดาห์มาจากฟีเจอร์ slot_*/dow_* อยู่แล้ว
                          # ต่างจาก "tfm" ตรง spatial encoding เท่านั้น (สถาปัตย์ / จำนวน block / heads เท่ากัน) → เทียบกันตรง ๆ ได้
        d = tfm_d
        h = PosEmbed()(L.Dense(d)(SpatialFourier(x_idx, y_idx, n_freq, name="spatial_fourier")(x)))
        for _ in range(tfm_blocks): h = TransformerBlock(d, n_heads=tfm_heads, ff=4 * d)(h)
    else:
        raise ValueError(rnn)
    h = L.LayerNormalization()(h)
    feat = AttentionPool(n_heads, name="mhap")(h)
    feat = L.Dropout(dropout)(feat)
    feat = L.Dense(64, activation="tanh")(feat)
    feat = L.Dropout(dropout)(feat)
    step_raw = L.Dense(M * H * 2, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                       kernel_regularizer=keras.regularizers.l2(1e-5), name="head_P")(feat)
    sig_raw = L.Dense(M * H, bias_initializer=keras.initializers.Constant(1.0), name="head_U")(feat)
    logits = L.Dense(M, name="head_M")(feat)
    out = MixtureHead(M, H, skip, dx_idx, dy_idx, tuple(dx_ms), tuple(dy_ms), skip_tau=skip_tau, name="mixture")([step_raw, sig_raw, logits, x])
    tag = "_noskip" if not skip else ("_cv" if not np.isfinite(skip_tau) else f"_tau{skip_tau:g}")
    if rnn in ("tfm", "sttfm"): tag += f"_d{tfm_d}x{tfm_blocks}"
    return keras.Model(x, out, name=f"D5_{rnn}{'_bi' if bi and rnn not in ('tfm', 'sttfm') else ''}_M{M}{tag}")

# ----------------------------------------------------------------------------- loss / metrics
EPS = tf.Variable(PYCHARM["EPS_START"], trainable=False, dtype=tf.float32, name="eps_wta")

def make_loss_and_metrics(M: int, H: int, wta_by: str = "ade"):
    n_mu, n_sig = M * H * 2, M * H
    def unpack(y_true, y_pred):
        B = tf.shape(y_pred)[0]
        y = tf.reshape(y_true, (B, 1, H, 2))
        mu = tf.reshape(y_pred[:, :n_mu], (B, M, H, 2))
        sig = tf.reshape(y_pred[:, n_mu:n_mu + n_sig], (B, M, H))
        logits = y_pred[:, n_mu + n_sig:]
        d2 = tf.reduce_sum(tf.square(y - mu), -1)                        # (B,M,H)
        return y, mu, sig, logits, d2
    def loss(y_true, y_pred):
        _, _, sig, logits, d2 = unpack(y_true, y_pred)
        nll_m = tf.reduce_mean(d2 / (2.0 * tf.square(sig)) + 2.0 * tf.math.log(sig), -1)   # (B,M) ต่อก้าว (ตัดค่าคงที่)
        ade_m = tf.reduce_mean(tf.sqrt(d2 + 1e-6), -1)
        win = tf.argmin(ade_m if wta_by == "ade" else nll_m, axis=1)
        ade_win = tf.gather(ade_m, win, batch_dims=1) / 100.0
        if M == 1:
            return tf.reduce_mean(nll_m[:, 0] + LAMBDA_ADE * ade_win)
        oh = tf.one_hot(win, M, dtype=tf.float32)
        w = oh * (1.0 - EPS) + (1.0 - oh) * EPS / (M - 1)
        ce = tf.nn.softmax_cross_entropy_with_logits(labels=oh, logits=logits)
        return tf.reduce_mean(tf.reduce_sum(w * nll_m, -1) + LAMBDA_PI * ce + LAMBDA_ADE * ade_win)
    def nll(y_true, y_pred):   # mixture NLL ต่อก้าว (proper score; รวม π) — ใช้ early-stop และ D6
        _, _, sig, logits, d2 = unpack(y_true, y_pred)
        logp = -tf.reduce_sum(d2 / (2.0 * tf.square(sig)) + tf.math.log(2.0 * np.pi * tf.square(sig)), -1)  # (B,M)
        return tf.reduce_mean(-tf.reduce_logsumexp(tf.nn.log_softmax(logits) + logp, -1) / H)
    def minADE(y_true, y_pred):
        _, _, _, _, d2 = unpack(y_true, y_pred)
        return tf.reduce_mean(tf.reduce_min(tf.reduce_mean(tf.sqrt(d2), -1), -1))
    def minFDE(y_true, y_pred):
        _, _, _, _, d2 = unpack(y_true, y_pred)
        return tf.reduce_mean(tf.reduce_min(tf.sqrt(d2[:, :, -1]), -1))
    def ADE_pi(y_true, y_pred):   # mode ที่ π สูงสุด (สิ่งที่ผู้ใช้เห็นจริงถ้าเลือก mode เดียว)
        _, _, _, logits, d2 = unpack(y_true, y_pred)
        k = tf.argmax(logits, -1); ade = tf.reduce_mean(tf.sqrt(d2), -1)
        return tf.reduce_mean(tf.gather(ade, k, batch_dims=1))
    def FDE_pi(y_true, y_pred):
        _, _, _, logits, d2 = unpack(y_true, y_pred)
        k = tf.argmax(logits, -1)
        return tf.reduce_mean(tf.gather(tf.sqrt(d2[:, :, -1]), k, batch_dims=1))
    return loss, [nll, minADE, minFDE, ADE_pi, FDE_pi]

class EpsSchedule(keras.callbacks.Callback):
    def __init__(self, e0, e1, n_ep): super().__init__(); self.e0, self.e1, self.n = e0, e1, max(1, n_ep)
    def on_epoch_begin(self, epoch, logs=None):
        EPS.assign(float(self.e1 + (self.e0 - self.e1) * max(0.0, 1.0 - epoch / self.n)))

def warmup_cosine(base_lr, total, warmup=5, min_lr=1e-6):
    def f(epoch, lr):
        if epoch < warmup: return float(base_lr * (epoch + 1) / warmup)
        p = min(1.0, max(0.0, (epoch - warmup) / max(1, total - warmup)))
        return float(min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * p)))
    return f

# ----------------------------------------------------------------------------- numpy eval
def unpack_np(P: np.ndarray, M: int, H: int):
    n_mu, n_sig = M * H * 2, M * H
    mu = P[:, :n_mu].reshape(-1, M, H, 2); sig = P[:, n_mu:n_mu + n_sig].reshape(-1, M, H)
    lg = P[:, n_mu + n_sig:]; lg = lg - lg.max(1, keepdims=True); pi = np.exp(lg); pi /= pi.sum(1, keepdims=True)
    return mu, sig, pi

def eval_np(Y: np.ndarray, mu: np.ndarray, sig: np.ndarray, pi: np.ndarray, R_m: float, mask=None) -> dict:
    """เมตริกจาก numpy (ทั้งชุด + subset cycle-start) · ADE/FDE ที่ h ต่าง ๆ · hit@R (ปลายทางอยู่ในรัศมี R)"""
    if mask is not None: Y, mu, sig, pi = Y[mask], mu[mask], sig[mask], pi[mask]
    if len(Y) == 0: return {}
    H = Y.shape[1]; d = np.sqrt(((Y[:, None] - mu) ** 2).sum(-1))            # (N,M,H)
    k = pi.argmax(1); d_pi = d[np.arange(len(d)), k]                          # (N,H)
    d_min = d[np.arange(len(d)), d.mean(-1).argmin(1)]                        # mode ที่ ADE ต่ำสุด
    logp = -(((Y[:, None] - mu) ** 2).sum(-1) / (2 * sig ** 2) + np.log(2 * np.pi * sig ** 2)).sum(-1)   # (N,M)
    z = np.log(pi + 1e-12) + logp; nll = -(z.max(1) + np.log(np.exp(z - z.max(1, keepdims=True)).sum(1))) / H
    out = {"n": int(len(Y)), "minADE": float(d_min.mean()), "minFDE": float(d_min[:, -1].mean()),
           "ADE_pi": float(d_pi.mean()), "FDE_pi": float(d_pi[:, -1].mean()), "nll": float(nll.mean()),
           "hitR_pi": float((d_pi[:, -1] <= R_m).mean()), "hitR_min": float((d_min[:, -1] <= R_m).mean()),
           "pi_max_mean": float(pi.max(1).mean())}
    for h in [x for x in H_REPORT if x <= H]:
        out[f"DE_pi@{h}"] = float(d_pi[:, h - 1].mean()); out[f"minDE@{h}"] = float(d_min[:, h - 1].mean())
        # calibration เบื้องต้น: สัดส่วนที่ระยะจริง ≤ σ̂·√(−2 ln 0.1) (วงกลม 90%) ของ mode π สูงสุด
        r90 = sig[np.arange(len(d)), k, h - 1] * math.sqrt(-2 * math.log(0.1))
        out[f"cov90@{h}"] = float((d_pi[:, h - 1] <= r90).mean())
    return out

def baselines_np(X: np.ndarray, Y: np.ndarray, dx_idx, dy_idx, dx_ms, dy_ms, R_m, skip_tau=3.0) -> dict:
    """เส้นฐาน: อยู่กับที่ (Δ̂=0) · ความเร็วคงที่ (dx_T,dy_T)·h · ความเร็วหน่วง τ (= เส้นฐาน skip ที่โมเดลใช้) — ไว้เทียบใน G1"""
    H = Y.shape[1]
    if dx_idx is None or dy_idx is None:   # ฟีเจอร์ความเร็วถูกตัด (abl_novel) → เส้นฐานความเร็วคำนวณไม่ได้ รายงาน NaN
        d0 = np.sqrt((Y ** 2).sum(-1)); st = {"ADE": float(d0.mean()), "FDE": float(d0[:, -1].mean()), "hitR": float((d0[:, -1] <= R_m).mean()),
                                              **{f"DE@{h}": float(d0[:, h - 1].mean()) for h in H_REPORT if h <= H}}
        nan = {k: float("nan") for k in st}
        return {"static": st, "const_vel": nan, "damped_vel": nan}
    dx = X[:, -1, dx_idx] * dx_ms[1] + dx_ms[0]; dy = X[:, -1, dy_idx] * dy_ms[1] + dy_ms[0]
    v = np.stack([dx, dy], -1)[:, None, :]
    cv = v * np.arange(1, H + 1)[None, :, None]; dv = v * skip_gain(H, skip_tau)[None, :, None]
    res = {}
    for name, P in (("static", np.zeros_like(Y)), ("const_vel", cv), ("damped_vel", dv)):
        d = np.sqrt(((Y - P) ** 2).sum(-1))
        res[name] = {"ADE": float(d.mean()), "FDE": float(d[:, -1].mean()), "hitR": float((d[:, -1] <= R_m).mean()),
                     **{f"DE@{h}": float(d[:, h - 1].mean()) for h in H_REPORT if h <= H}}
    return res

# ----------------------------------------------------------------------------- train one seed
def train_seed(args, data, dxdy, feats, seed: int, out_dir: Path, log) -> dict:
    set_seed(seed)
    Xtr, Ytr, _ = data["train"]; Xva, Yva, mva = data["val"]; Xte, Yte, mte = data["test"]
    M_in, F = Xtr.shape[1:]; H = Ytr.shape[1]; M = args.M
    has_v = "dx" in feats and "dy" in feats
    if not has_v and args.skip:
        log("  ฟีเจอร์ไม่มี dx,dy → ปิด skip อัตโนมัติ (เส้นฐาน damped-vel ต้องใช้ความเร็ว)"); args.skip = False
    dx_idx, dy_idx = (feats.index("dx"), feats.index("dy")) if has_v else (None, None)
    x_idx, y_idx = (feats.index("x"), feats.index("y")) if ("x" in feats and "y" in feats) else (0, 1)
    model = build_model(M_in, F, H, M, rnn=args.rnn, bi=args.bi, hidden=args.hidden, skip=args.skip, skip_tau=args.skip_tau,
                        dx_idx=(dx_idx if has_v else 0), dy_idx=(dy_idx if has_v else 1), dx_ms=dxdy["dx"], dy_ms=dxdy["dy"], x_idx=x_idx, y_idx=y_idx,
                        tfm_d=args.tfm_d, tfm_blocks=args.tfm_blocks, tfm_heads=args.tfm_heads)
    if args.init_from:
        try: model.load_weights(args.init_from, skip_mismatch=True)     # หัวที่ขนาดต่าง (M ต่างกัน) จะสุ่มใหม่ ส่วน encoder ใช้ต่อ
        except TypeError: model.load_weights(args.init_from)
        log(f"  โหลดน้ำหนักเริ่มต้น (fine-tune) จาก {args.init_from}")
    try: opt = keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=1e-4, clipnorm=1.0)
    except Exception: opt = keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0)
    loss, metrics = make_loss_and_metrics(M, H, args.wta_by)
    model.compile(optimizer=opt, loss=loss, metrics=metrics, jit_compile=False)
    if seed == args.seeds[0]:
        with open(out_dir.parent / "model_summary.txt", "w", encoding="utf-8") as f: model.summary(print_fn=lambda s: f.write(s + "\n"))
    n_params = int(model.count_params()); log(f"  {model.name} · params {n_params:,} · train {len(Xtr):,} · batch {args.batch}")

    cbs = [EpsSchedule(args.eps_start, args.eps_end, args.eps_epochs),
           keras.callbacks.LearningRateScheduler(warmup_cosine(args.lr, args.epochs, warmup=min(5, args.epochs))),
           keras.callbacks.EarlyStopping(monitor="val_nll", mode="min", patience=args.patience, restore_best_weights=True, verbose=0),
           keras.callbacks.CSVLogger(str(out_dir / "history.csv"))]
    t0 = time.time()
    hist = model.fit(Xtr, Ytr.reshape(len(Ytr), -1), validation_data=(Xva, Yva.reshape(len(Yva), -1)),
                     epochs=args.epochs, batch_size=args.batch, shuffle=True, callbacks=cbs, verbose=args.verbose)
    train_s = time.time() - t0
    h = hist.history; n_ep = len(h["loss"]); best_ep = int(np.argmin(h["val_nll"])) + 1
    model.save_weights(str(out_dir / "best.weights.h5"))
    log(f"  ฝึก {n_ep} epoch ({train_s/60:.1f} นาที · {train_s/n_ep:.0f} s/epoch) · best epoch {best_ep} · val_nll {min(h['val_nll']):.3f}")

    # ---- predictions → npz (ใช้ต่อ D6/D7/sai E)
    res = {"city": args.city, "run": args.run, "seed": seed, "model": model.name, "M": M, "rnn": args.rnn, "bi": args.bi, "skip": args.skip, "skip_tau": args.skip_tau,
           "hidden": args.hidden, "features": ",".join(feats), "n_features": F, "M_in": int(M_in), "H": int(H), "params": n_params,
           "n_train": int(len(Xtr)), "n_val": int(len(Xva)), "n_test": int(len(Xte)), "epochs_run": n_ep, "best_epoch": best_ep,
           "train_min": round(train_s / 60, 2), "s_per_epoch": round(train_s / n_ep, 1), "init_from": args.init_from or "",
           "val_nll_best": float(min(h["val_nll"]))}
    for sp, (X, Y, meta) in (("val", (Xva, Yva, mva)), ("test", (Xte, Yte, mte))):
        t1 = time.time(); P = model.predict(X, batch_size=2048, verbose=0); infer_ms = (time.time() - t1) * 1000 / max(1, len(X)) * 1000
        mu, sig, pi = unpack_np(P, M, H)
        # คีย์อ้างอิงจาก meta ติดไปด้วย (แถวเรียงเหมือน <sp>_meta ของ D4) → sai E ใช้ npz ไฟล์เดียวหา cab/เวลา/จุดตั้งต้น lat_T,lon_T ได้
        keys = {k: meta[k].values for k in ("cab", "seg", "t_T", "lat_T", "lon_T", "date", "hour", "is_cycle_start", "is_cycle_clean") if k in meta}
        with np.errstate(over="ignore"):
            np.savez_compressed(out_dir / f"pred_{sp}.npz", mu=np.clip(mu, -6e4, 6e4).astype(args.pred_dtype), sigma=sig.astype(args.pred_dtype),
                                pi=pi.astype(np.float32), row=np.arange(len(X)), **keys)
        if sp == "test": res["infer_ms_per_1k_windows"] = round(infer_ms, 1)
        ev = eval_np(Y, mu, sig, pi, args.R_m); res.update({f"{sp}_{k}": v for k, v in ev.items()})
        if sp == "test":
            mask = meta["is_cycle_start"].values.astype(bool) if "is_cycle_start" in meta else None
            if mask is not None and mask.any():
                res.update({f"cyc_{k}": v for k, v in eval_np(Y, mu, sig, pi, args.R_m, mask).items()})
            bl = baselines_np(X, Y, dx_idx, dy_idx, dxdy["dx"], dxdy["dy"], args.R_m, args.skip_tau)
            for b, dd in bl.items(): res.update({f"bl_{b}_{k}": v for k, v in dd.items()})
    json.dump(res, open(out_dir / "metrics.json", "w"), indent=2, ensure_ascii=False)
    log(f"  test  minADE {res['test_minADE']:.1f} m · ADE_π {res['test_ADE_pi']:.1f} m · FDE_π {res['test_FDE_pi']:.1f} m · hit@R(π) {res['test_hitR_pi']:.3f} · nll {res['test_nll']:.3f}"
        f"   | baseline static ADE {res['bl_static_ADE']:.1f} m · damped_vel {res['bl_damped_vel_ADE']:.1f} m · const_vel {res['bl_const_vel_ADE']:.1f} m")
    return res

# ----------------------------------------------------------------------------- main
def parse():
    P = PYCHARM
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", default=P["CITY"]); ap.add_argument("--data", default=P["DATA"]); ap.add_argument("--config", default=P["CONFIG"])
    ap.add_argument("--run", default=P["RUN"]); ap.add_argument("--M", type=int, default=P["M"])
    ap.add_argument("--rnn", choices=["lstm", "gru", "rnn", "tfm", "sttfm"], default=P["RNN"], help="encoder: lstm | gru | rnn (vanilla) | tfm (Transformer เชิงเวลา) | sttfm (ST-Transformer: + spatial Fourier encoding)")
    ap.add_argument("--no-bi", dest="bi", action="store_false", default=P["BI"])
    ap.add_argument("--tfm-d", type=int, default=128, help="tfm/sttfm: ความกว้าง d_model (ff = 4d)")
    ap.add_argument("--tfm-blocks", type=int, default=2, help="tfm/sttfm: จำนวน encoder blocks")
    ap.add_argument("--tfm-heads", type=int, default=4, help="tfm/sttfm: จำนวน attention heads")
    ap.add_argument("--no-skip", dest="skip", action="store_false", default=P["SKIP"])
    ap.add_argument("--skip-tau", type=float, default=P["SKIP_TAU"], help="τ ของเส้นฐานความเร็วหน่วง (ก้าว) · 0 = static · inf = const-vel")
    ap.add_argument("--hidden", type=int, default=P["HIDDEN"]); ap.add_argument("--with-occ", action="store_true", default=P["WITH_OCC"])
    ap.add_argument("--features", default="", help="รายชื่อฟีเจอร์คั่นด้วย , (ค่าเริ่มต้น = config ตัด occ)")
    ap.add_argument("--seeds", default=",".join(map(str, P["SEEDS"])))
    ap.add_argument("--epochs", type=int, default=P["EPOCHS"]); ap.add_argument("--batch", type=int, default=P["BATCH"])
    ap.add_argument("--lr", type=float, default=P["LR"]); ap.add_argument("--patience", type=int, default=P["PATIENCE"])
    ap.add_argument("--eps-start", type=float, default=P["EPS_START"]); ap.add_argument("--eps-end", type=float, default=P["EPS_END"])
    ap.add_argument("--eps-epochs", type=int, default=P["EPS_EPOCHS"]); ap.add_argument("--wta-by", choices=["ade", "nll"], default="ade")
    ap.add_argument("--init-from", default=P["INIT_FROM"]); ap.add_argument("--subsample", type=float, default=P["SUBSAMPLE"])
    ap.add_argument("--pred-dtype", default="float32", help="dtype ของ mu/sigma ใน pred_*.npz (float32 เต็ม · float16 ประหยัดที่ครึ่งหนึ่ง, ละเอียด ~1 m ที่ 2 km)")
    ap.add_argument("--verbose", type=int, default=2); ap.add_argument("--demo", action="store_true", default=P["DEMO"])
    a = ap.parse_args(); a.seeds = [int(s) for s in str(a.seeds).split(",") if s != ""]
    return a

def main():
    a = parse(); setup_tf()
    t_start = time.time(); log = lambda *m: print(f"[{time.time()-t_start:7.1f}s]", *m, flush=True)
    data_root = Path(a.data) if Path(a.data).is_absolute() else HERE / a.data
    if a.demo:
        a.city, a.run = "demo", a.run + "_demo"; a.epochs = 3 if a.epochs == PYCHARM["EPOCHS"] else a.epochs; a.eps_epochs = min(a.eps_epochs, a.epochs // 2)
        feats = [f for f in DEFAULT_FEATURES if f != "occ"]; a.R_m = 200.0
        data, dxdy = make_demo(F=len(feats)); log("DEMO: ข้อมูลสังเคราะห์ 3 โหมด (ตรง/ซ้าย/ขวา)")
        # เขียน val/test ของ demo ไว้ใน data/demo/ ให้ sD6/sD7 --demo ใช้ต่อได้
        dd = data_root / "demo"; (dd / "processed").mkdir(parents=True, exist_ok=True)
        for sp in ("val", "test"):
            np.save(dd / "processed" / f"{sp}_X.npy", data[sp][0]); np.save(dd / "processed" / f"{sp}_Y.npy", data[sp][1]); data[sp][2].to_pickle(dd / "processed" / f"{sp}_meta.pkl")
        json.dump({"cols": [0, 1, 2, 3, 4], "names": ["x", "y", "v", "dx", "dy"], "mean": [0, 0, 0, dxdy["dx"][0], dxdy["dy"][0]], "std": [1, 1, 1, dxdy["dx"][1], dxdy["dy"][1]]}, open(dd / "scaler.json", "w"))
    else:
        cfg_path = a.config or str(HERE / "config" / f"data_{a.city}.yaml")
        cfg = load_cfg(cfg_path, a.city); a.R_m = float(cfg.get("R_m", 200))
        if a.features: feats = [f.strip() for f in a.features.split(",")]
        else: feats = [f for f in cfg["features"] if f != "occ" or a.with_occ]
        log(f"config {cfg_path} · ฟีเจอร์ {len(feats)}: {feats}")
        data, dxdy = load_d4(data_root, a.city, cfg, feats, a.subsample, a.seeds[0])
    out_root = HERE / "models" / a.city / a.run; out_root.mkdir(parents=True, exist_ok=True)
    json.dump({**{k: (v if isinstance(v, (int, float, str, bool, list)) or v is None else str(v)) for k, v in vars(a).items()}, "features": feats},
              open(out_root / "run_config.json", "w"), indent=2, ensure_ascii=False)
    rows = []
    for seed in a.seeds:
        out_dir = out_root / f"seed{seed}"; out_dir.mkdir(exist_ok=True)
        log(f"=== {a.city} · {a.run} · seed {seed} ===")
        rows.append(train_seed(a, data, dxdy, feats, seed, out_dir, log))
        keras.backend.clear_session()
    df = pd.DataFrame(rows)
    ledger = HERE / "models" / "D5_results.csv"
    df.to_csv(ledger, mode="a", header=not ledger.exists(), index=False)
    num = df.select_dtypes("number").drop(columns=["seed"], errors="ignore")
    summ = {"city": a.city, "run": a.run, "seeds": a.seeds, "features": feats,
            "mean": {k: float(v) for k, v in num.mean().items()}, "std": {k: float(v) for k, v in num.std(ddof=0).items()}}
    json.dump(summ, open(out_root / "summary.json", "w"), indent=2, ensure_ascii=False)
    m, s = summ["mean"], summ["std"]
    log(f"สรุป {a.city}/{a.run} ({len(rows)} seeds): test minADE {m['test_minADE']:.1f}±{s['test_minADE']:.1f} m · ADE_π {m['test_ADE_pi']:.1f}±{s['test_ADE_pi']:.1f} m"
        f" · FDE_π {m['test_FDE_pi']:.1f} m · hit@R(π) {m['test_hitR_pi']:.3f} · nll {m['test_nll']:.3f} · static ADE {m['bl_static_ADE']:.1f} m · damped-vel {m['bl_damped_vel_ADE']:.1f} m")
    log(f"ผลลัพธ์ → {out_root}  · ledger → {ledger}")

if __name__ == "__main__":
    main()
