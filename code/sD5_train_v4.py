#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sD5_train_v4.py — D5 + หัว reachability แบบ "ถามจุด" (9a′):  C_ij = σ( g(h_j, l_i − x_T, st, et, R) )
ไม่แก้ sD5_train.py / v3 (import ส่วนที่ไม่เปลี่ยน) · encoder + 3 หัว mixture เดิมยังอยู่ (ADE/NLL/qm_ub ยังคำนวณได้เพื่อเทียบบนโมเดลเดียวกัน)

หัวใหม่ (reach head)
  อินพุตต่อ query: [Δx/5000, Δy/5000, r/5000, st/H, et/H, (et−st)/H, R/250]  concat กับ feat64 (ชั้นหลัง Dense(64) ที่หัว mixture ใช้ร่วม)
  → Dense(128, relu) → Dense(64, relu) → Dense(1)  = logit ของ Pr[∃ h∈[st,et]: ‖Y_h − l‖ ≤ R]   (นิยามเดียวกับ reach_labels ของ sD7 v7, K=1)
  train: ต่อหน้าต่างสุ่ม K query ต่อ batch (สดทุก batch): 40% ใกล้เส้นทางจริง (≤ 2R) · 30% วงแหวน 2R–6R (ขอบ, ส่วนใหญ่ลบ) · 30% สม่ำเสมอในวงรัศมี p75|Y_et| (เหมือน sample_pairs ของ D7)
         dur ∈ {5,10,15,20,30}, R ∈ r_list → หัวเดียวรองรับทุก R · label คำนวณตรงจาก Y จริง
  loss = loss เดิม (ε-WTA mixture NLL + CE + ADE) + λ_reach · BCE(mean over K)  · early-stop บน val_reach_bce (--monitor nll เพื่อใช้แบบเดิม)
ผลลัพธ์เพิ่ม: pred_{val,test}.npz มี key feat64 (N×64 float16) · seed*/reach_head.npz (น้ำหนัก MLP + สเปก) → sD7_pred_eval_v8 คำนวณ C_ij ด้วย numpy ไม่ต้องใช้ TF
CLI:  python sD5_train_v4.py --city sf --run reach_lstm --seeds 0,1,2
      python sD5_train_v4.py --city sf --run reach_tfm  --rnn tfm --seeds 0,1,2
      --lam-reach 2.0 · --k-query 32 · --r-list 100,150,200,250 · --monitor reach|nll · --proc processed_xxx · --reach-only (ปิด loss mixture: ablation)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sD5_train as D5                                                                   # noqa: E402
from sD5_train import (tf, keras, L, AttentionPool, PosEmbed, TransformerBlock, SpatialFourier, MixtureHead,  # noqa: E402
                       EpsSchedule, warmup_cosine, unpack_np, eval_np, baselines_np, setup_tf, set_seed, read_meta, EPS, LAMBDA_PI, LAMBDA_ADE)
try:
    from sD1_D4_prepare_v2 import load_config_v2, merged_config_path                     # noqa: E402
except Exception:
    load_config_v2 = D5.load_cfg; merged_config_path = lambda p: p

DURATIONS = [5, 10, 15, 20, 30]          # เท่ากับ sD7 v7
Q_SCALE_M = 5000.0; R_SCALE_M = 250.0; N_QF = 7

# ----------------------------------------------------------------------------- query sampler (numpy, เหมือน sample_pairs ของ D7 + วงแหวนขอบ)
def query_feats(l, st, et, R, H):
    r = np.sqrt((l ** 2).sum(-1, keepdims=True))
    return np.concatenate([l / Q_SCALE_M, r / Q_SCALE_M, (st / H)[..., None], (et / H)[..., None], ((et - st) / H)[..., None], (R / R_SCALE_M)[..., None]], -1).astype(np.float32)

def reach_label_np(Y, l, st, et, R):
    """Y (B,H,2) · l (B,K,2) · st/et (B,K) 1-based inclusive · R (B,K) → (B,K) bool: ∃h∈[st,et] ‖Y_h − l‖ ≤ R"""
    H = Y.shape[1]; hs = np.arange(1, H + 1)[None, None, :]
    d = np.sqrt(((Y[:, None, :, :] - l[:, :, None, :]) ** 2).sum(-1))          # (B,K,H)
    inwin = (hs >= st[..., None]) & (hs <= et[..., None])
    return ((d <= R[..., None]) & inwin).any(-1)

class QuerySampler:
    def __init__(self, p75_h, r_list, K, H, seed=0, mix=(0.4, 0.3, 0.3)):
        self.p75_h = np.asarray(p75_h, np.float32); self.r_list = np.asarray(r_list, np.float32); self.K = int(K); self.H = int(H); self.mix = mix
        self.rng = np.random.RandomState(seed)
    def __call__(self, Y, rng=None):
        rng = rng or self.rng; B, H, K = len(Y), self.H, self.K
        dur = rng.choice(DURATIONS, (B, K)); st = 1 + (rng.rand(B, K) * np.maximum(1, H - dur)).astype(int); et = np.minimum(H, st + dur)
        R = rng.choice(self.r_list, (B, K))
        hstar = np.minimum(st + (rng.rand(B, K) * (et - st + 1)).astype(int), H)
        base = np.take_along_axis(Y, (hstar - 1)[..., None].repeat(2, -1), axis=1)       # (B,K,2) จุดจริงในหน้าต่างเวลา
        ang = rng.rand(B, K) * 2 * np.pi; u = rng.rand(B, K); dirv = np.stack([np.cos(ang), np.sin(ang)], -1)
        kind = rng.rand(B, K)
        rad_near = np.sqrt(u) * 2 * R                                                   # ≤ 2R (ส่วนใหญ่บวก)
        rad_ring = (2 + 4 * u) * R                                                      # 2R–6R (ขอบ)
        scale = np.maximum(self.p75_h[et - 1], 4 * self.r_list.max()); rad_unif = np.sqrt(u) * scale
        l = np.where((kind < self.mix[0])[..., None], base + dirv * rad_near[..., None],
            np.where((kind < self.mix[0] + self.mix[1])[..., None], base + dirv * rad_ring[..., None], dirv * rad_unif[..., None])).astype(np.float32)
        lab = reach_label_np(Y, l, st, et, R).astype(np.float32)
        return query_feats(l, st, et, R, H), lab

class ReachBatches(keras.utils.PyDataset):
    """ป้อน ((X[,extras], Q), [Y_flat | labels]) ทีละ batch · train: query สดทุก batch · val: query คงที่ (seed ตายตัว) เพื่อให้ val_reach_bce เทียบข้าม epoch ได้"""
    def __init__(self, X, Y, sampler, batch, shuffle, seed=0, fixed=False, **kw):
        super().__init__(**kw); self.X, self.Y, self.s, self.batch, self.shuffle, self.fixed = X, Y, sampler, int(batch), shuffle, fixed
        self.n = len(Y); self.rng = np.random.RandomState(seed); self.idx = np.arange(self.n); self.seed = seed
        if shuffle: self.rng.shuffle(self.idx)
        self.cache = {}
    def __len__(self): return int(np.ceil(self.n / self.batch))
    def __getitem__(self, i):
        j = self.idx[i * self.batch:(i + 1) * self.batch]; Y = self.Y[j]
        if self.fixed:
            if i not in self.cache: self.cache[i] = self.s(Y, np.random.RandomState(self.seed * 100003 + i))
            Q, lab = self.cache[i]
        else: Q, lab = self.s(Y)
        return (self.X[j], Q), np.concatenate([Y.reshape(len(Y), -1), lab], 1)
    def on_epoch_end(self):
        if self.shuffle: self.rng.shuffle(self.idx)

# ----------------------------------------------------------------------------- model
class TileConcat(L.Layer):
    """feat (B,d) + Q (B,K,q) → (B,K,d+q)"""
    def call(self, inp):
        f, q = inp; K = tf.shape(q)[1]
        return tf.concat([tf.repeat(f[:, None, :], K, axis=1), q], -1)
    def compute_output_shape(self, s): return (s[1][0], s[1][1], s[0][-1] + s[1][-1])

def build_model_v4(M_in, F, H, M, K, rnn="lstm", bi=True, hidden=256, n_heads=4, dropout=0.15, skip=True, skip_tau=3.0,
                   dx_idx=3, dy_idx=4, dx_ms=(0.0, 1.0), dy_ms=(0.0, 1.0), x_idx=0, y_idx=1, n_freq=8, tfm_d=128, tfm_blocks=2, tfm_heads=4):
    x = L.Input(shape=(M_in, F), name="x_seq"); q = L.Input(shape=(None, N_QF), name="query")
    wrap = (lambda layer: L.Bidirectional(layer)) if bi else (lambda layer: layer)
    if rnn == "lstm":
        h = wrap(L.LSTM(128, return_sequences=True, dropout=0.1))(x); h = L.LayerNormalization()(h); h = wrap(L.LSTM(hidden, return_sequences=True, dropout=0.1))(h)
    elif rnn == "gru":
        h = wrap(L.GRU(128, return_sequences=True, dropout=0.1))(x); h = L.LayerNormalization()(h); h = wrap(L.GRU(hidden, return_sequences=True, dropout=0.1))(h)
    elif rnn == "rnn":
        h = wrap(L.SimpleRNN(128, return_sequences=True, dropout=0.1))(x); h = L.LayerNormalization()(h); h = wrap(L.SimpleRNN(hidden, return_sequences=True, dropout=0.1))(h)
    elif rnn == "tfm":
        h = PosEmbed()(L.Dense(tfm_d)(x))
        for _ in range(tfm_blocks): h = TransformerBlock(tfm_d, n_heads=tfm_heads, ff=4 * tfm_d)(h)
    elif rnn == "sttfm":
        h = PosEmbed()(L.Dense(tfm_d)(SpatialFourier(x_idx, y_idx, n_freq, name="spatial_fourier")(x)))
        for _ in range(tfm_blocks): h = TransformerBlock(tfm_d, n_heads=tfm_heads, ff=4 * tfm_d)(h)
    else: raise ValueError(rnn)
    h = L.LayerNormalization()(h)
    feat = AttentionPool(n_heads, name="mhap")(h); feat = L.Dropout(dropout)(feat)
    feat64 = L.Dense(64, activation="tanh", name="feat64")(feat); fd = L.Dropout(dropout)(feat64)
    step_raw = L.Dense(M * H * 2, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02), kernel_regularizer=keras.regularizers.l2(1e-5), name="head_P")(fd)
    sig_raw = L.Dense(M * H, bias_initializer=keras.initializers.Constant(1.0), name="head_U")(fd)
    logits = L.Dense(M, name="head_M")(fd)
    mix = MixtureHead(M, H, skip, dx_idx, dy_idx, tuple(dx_ms), tuple(dy_ms), skip_tau=skip_tau, name="mixture")([step_raw, sig_raw, logits, x])
    z = TileConcat(name="tile")([feat64, q])
    z = L.Dense(128, activation="relu", name="reach_1")(z); z = L.Dense(64, activation="relu", name="reach_2")(z)
    reach = L.Reshape((-1,), name="reach_logit")(L.Dense(1, name="reach_3")(z))          # (B,K)
    out = L.Concatenate(name="out_all")([mix, reach])
    tag = ("" if skip else "_noskip") + (f"_d{tfm_d}x{tfm_blocks}" if rnn in ("tfm", "sttfm") else "") + "_reach"
    return keras.Model([x, q], out, name=f"D5_{rnn}{'_bi' if bi and rnn not in ('tfm', 'sttfm') else ''}_M{M}{tag}")

def export_reach_head(model, path: Path):
    w = {}
    for i, name in enumerate(("reach_1", "reach_2", "reach_3"), 1):
        Wt, b = model.get_layer(name).get_weights(); w[f"W{i}"] = Wt.astype(np.float32); w[f"b{i}"] = b.astype(np.float32)
    np.savez(path, **w, q_scale_m=Q_SCALE_M, r_scale_m=R_SCALE_M, n_qf=N_QF, feat_dim=w["W1"].shape[0] - N_QF)

def reach_head_np(head, feat64, Q):
    """numpy forward ของหัว (ใช้ใน v8): feat64 (n,d) · Q (n,K,7) → logits (n,K)"""
    z = np.concatenate([np.repeat(feat64[:, None, :], Q.shape[1], 1), Q], -1)
    z = np.maximum(z @ head["W1"] + head["b1"], 0); z = np.maximum(z @ head["W2"] + head["b2"], 0)
    return (z @ head["W3"] + head["b3"])[..., 0]

# ----------------------------------------------------------------------------- loss
def make_loss_v4(M, H, K, wta_by, lam_reach, reach_only=False):
    n_mu, n_sig = M * H * 2, M * H; n_mix = n_mu + n_sig + M; n_y = H * 2
    def split(y_true, y_pred): return y_true[:, :n_y], y_pred[:, :n_mix], y_true[:, n_y:], y_pred[:, n_mix:]
    def unpack(y, p):
        B = tf.shape(p)[0]; y = tf.reshape(y, (B, 1, H, 2)); mu = tf.reshape(p[:, :n_mu], (B, M, H, 2)); sig = tf.reshape(p[:, n_mu:n_mu + n_sig], (B, M, H))
        return y, mu, sig, p[:, n_mu + n_sig:], tf.reduce_sum(tf.square(y - mu), -1)
    def mix_loss(y, p):
        _, _, sig, logits, d2 = unpack(y, p)
        nll_m = tf.reduce_mean(d2 / (2.0 * tf.square(sig)) + 2.0 * tf.math.log(sig), -1); ade_m = tf.reduce_mean(tf.sqrt(d2 + 1e-6), -1)
        win = tf.argmin(ade_m if wta_by == "ade" else nll_m, axis=1); ade_win = tf.gather(ade_m, win, batch_dims=1) / 100.0
        if M == 1: return nll_m[:, 0] + LAMBDA_ADE * ade_win
        oh = tf.one_hot(win, M, dtype=tf.float32); w = oh * (1.0 - EPS) + (1.0 - oh) * EPS / (M - 1)
        return tf.reduce_sum(w * nll_m, -1) + LAMBDA_PI * tf.nn.softmax_cross_entropy_with_logits(labels=oh, logits=logits) + LAMBDA_ADE * ade_win
    def loss(y_true, y_pred):
        y, p, yr, pr = split(y_true, y_pred)
        bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=yr, logits=pr), -1)
        return tf.reduce_mean((0.0 if reach_only else 1.0) * mix_loss(y, p) + lam_reach * bce)
    def nll(y_true, y_pred):
        y, p, _, _ = split(y_true, y_pred); _, _, sig, logits, d2 = unpack(y, p)
        logp = -tf.reduce_sum(d2 / (2.0 * tf.square(sig)) + tf.math.log(2.0 * np.pi * tf.square(sig)), -1)
        return tf.reduce_mean(-tf.reduce_logsumexp(tf.nn.log_softmax(logits) + logp, -1) / H)
    def minADE(y_true, y_pred):
        y, p, _, _ = split(y_true, y_pred); _, _, _, _, d2 = unpack(y, p); return tf.reduce_mean(tf.reduce_min(tf.reduce_mean(tf.sqrt(d2), -1), -1))
    def reach_bce(y_true, y_pred):
        _, _, yr, pr = split(y_true, y_pred); return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=yr, logits=pr))
    def reach_acc(y_true, y_pred):
        _, _, yr, pr = split(y_true, y_pred); return tf.reduce_mean(tf.cast(tf.equal(tf.cast(pr > 0, tf.float32), yr), tf.float32))
    return loss, [nll, minADE, reach_bce, reach_acc]

# ----------------------------------------------------------------------------- data
def load_d4(data_root, city, cfg, use_feats, proc_name, subsample=None, seed=0):
    proc = data_root / city / proc_name
    F_all = list(cfg["features"]); idx = [F_all.index(f) for f in use_feats]
    sc_path = proc / "scaler.json" if (proc / "scaler.json").exists() else data_root / city / "scaler.json"; sc = json.load(open(sc_path))
    d = {}
    for sp in ("train", "val", "test"):
        X = np.load(proc / f"{sp}_X.npy", mmap_mode="r")[:, :, idx].astype(np.float32); Y = np.load(proc / f"{sp}_Y.npy").astype(np.float32); meta = read_meta(proc / f"{sp}_meta")
        if sp == "train" and subsample:
            rng = np.random.RandomState(seed); keep = np.sort(rng.choice(len(X), int(len(X) * subsample), replace=False)); X, Y, meta = X[keep], Y[keep], meta.iloc[keep].reset_index(drop=True)
        d[sp] = {"X": X, "Y": Y, "meta": meta}; print(f"  {sp:<5} X {X.shape}  Y {Y.shape}")
    dxdy = {n: (float(sc["mean"][sc["names"].index(n)]), float(sc["std"][sc["names"].index(n)])) for n in ("dx", "dy")}
    return d, dxdy

# ----------------------------------------------------------------------------- train one seed
def train_seed(a, data, dxdy, feats, seed, out_dir: Path, log, p75_h):
    set_seed(seed); tr, va, te = data["train"], data["val"], data["test"]
    M_in, F = tr["X"].shape[1:]; H = tr["Y"].shape[1]; M = a.M; K = a.k_query
    has_v = "dx" in feats and "dy" in feats
    if not has_v and a.skip: a.skip = False
    dx_idx, dy_idx = (feats.index("dx"), feats.index("dy")) if has_v else (0, 1); x_idx, y_idx = (feats.index("x"), feats.index("y")) if ("x" in feats and "y" in feats) else (0, 1)
    model = build_model_v4(M_in, F, H, M, K, rnn=a.rnn, bi=a.bi, hidden=a.hidden, skip=a.skip, skip_tau=a.skip_tau, dx_idx=dx_idx, dy_idx=dy_idx, dx_ms=dxdy["dx"], dy_ms=dxdy["dy"],
                           x_idx=x_idx, y_idx=y_idx, tfm_d=a.tfm_d, tfm_blocks=a.tfm_blocks, tfm_heads=a.tfm_heads)
    try: opt = keras.optimizers.AdamW(learning_rate=a.lr, weight_decay=1e-4, clipnorm=1.0)
    except Exception: opt = keras.optimizers.Adam(learning_rate=a.lr, clipnorm=1.0)
    loss, metrics = make_loss_v4(M, H, K, a.wta_by, a.lam_reach, a.reach_only)
    model.compile(optimizer=opt, loss=loss, metrics=metrics, jit_compile=False)
    if seed == a.seeds[0]:
        with open(out_dir.parent / "model_summary.txt", "w", encoding="utf-8") as f: model.summary(print_fn=lambda s: f.write(s + "\n"))
    n_params = int(model.count_params()); log(f"  {model.name} · params {n_params:,} · train {len(tr['X']):,} · batch {a.batch} · K {K} · λ_reach {a.lam_reach} · monitor {a.monitor}")
    sampler = QuerySampler(p75_h, a.r_list, K, H, seed=seed)
    ds_tr = ReachBatches(tr["X"], tr["Y"], sampler, a.batch, shuffle=True, seed=seed)
    ds_va = ReachBatches(va["X"], va["Y"], sampler, 1024, shuffle=False, seed=777, fixed=True)
    monitor = "val_reach_bce" if a.monitor == "reach" else "val_nll"
    cbs = [EpsSchedule(a.eps_start, a.eps_end, a.eps_epochs), keras.callbacks.LearningRateScheduler(warmup_cosine(a.lr, a.epochs, warmup=min(5, a.epochs))),
           keras.callbacks.EarlyStopping(monitor=monitor, mode="min", patience=a.patience, restore_best_weights=True, verbose=0), keras.callbacks.CSVLogger(str(out_dir / "history.csv"))]
    t0 = time.time(); hist = model.fit(ds_tr, validation_data=ds_va, epochs=a.epochs, callbacks=cbs, verbose=a.verbose); train_s = time.time() - t0
    h = hist.history; n_ep = len(h["loss"]); best_ep = int(np.argmin(h[monitor])) + 1
    model.save_weights(str(out_dir / "best.weights.h5")); export_reach_head(model, out_dir / "reach_head.npz")
    log(f"  ฝึก {n_ep} epoch ({train_s/60:.1f} นาที · {train_s/n_ep:.0f} s/epoch) · best {best_ep} · val_nll {min(h['val_nll']):.3f} · val_reach_bce {min(h['val_reach_bce']):.4f} · val_reach_acc {max(h['val_reach_acc']):.3f}")

    # ---- inference: mixture (เหมือนเดิม) + feat64 สำหรับ v8
    n_mix = M * H * 2 + M * H + M
    feat_model = keras.Model(model.inputs[0], model.get_layer("feat64").output)
    res = {"city": a.city, "run": a.run, "seed": seed, "model": model.name, "M": M, "rnn": a.rnn, "bi": a.bi, "skip": a.skip, "skip_tau": a.skip_tau, "hidden": a.hidden,
           "features": ",".join(feats), "n_features": int(F), "M_in": int(M_in), "H": int(H), "params": n_params, "k_query": K, "lam_reach": a.lam_reach, "reach_only": a.reach_only,
           "monitor": monitor, "proc": a.proc, "n_train": int(len(tr["X"])), "n_val": int(len(va["X"])), "n_test": int(len(te["X"])), "epochs_run": n_ep, "best_epoch": best_ep,
           "train_min": round(train_s / 60, 2), "s_per_epoch": round(train_s / n_ep, 1), "val_nll_best": float(min(h["val_nll"])), "val_reach_bce_best": float(min(h["val_reach_bce"]))}
    dummyQ = np.zeros((1, 1, N_QF), np.float32)
    for sp, d in (("val", va), ("test", te)):
        t1 = time.time(); X = d["X"]; Y = d["Y"]; meta = d["meta"]
        P = model.predict(ReachBatches(X, Y, QuerySampler(p75_h, a.r_list, 1, H, seed=0), 2048, shuffle=False, seed=0, fixed=True), verbose=0)[:, :n_mix]
        feat = feat_model.predict(X, batch_size=2048, verbose=0); infer_ms = (time.time() - t1) * 1e6 / max(1, len(X))
        mu, sig, pi = unpack_np(P, M, H)
        keys = {k: meta[k].values for k in ("cab", "seg", "t_T", "lat_T", "lon_T", "date", "hour", "is_cycle_start", "is_cycle_clean") if k in meta}
        with np.errstate(over="ignore"):
            np.savez_compressed(out_dir / f"pred_{sp}.npz", mu=np.clip(mu, -6e4, 6e4).astype(a.pred_dtype), sigma=sig.astype(a.pred_dtype), pi=pi.astype(np.float32),
                                feat64=feat.astype(np.float16), row=np.arange(len(X)), **keys)
        if sp == "test": res["infer_ms_per_1k_windows"] = round(infer_ms, 1)
        res.update({f"{sp}_{k}": v for k, v in eval_np(Y, mu, sig, pi, a.R_m).items()})
        # reach AUC อย่างเร็วบน query ชุดเดียวกับ val (fixed) — ตัวเลขอ้างอิงภายใน ไม่ใช่ D7
        Q, lab = QuerySampler(p75_h, a.r_list, 16, H, seed=0)(Y, np.random.RandomState(4242)); head = dict(np.load(out_dir / "reach_head.npz"))
        lg = reach_head_np(head, feat.astype(np.float32), Q); sc_ = 1 / (1 + np.exp(-lg))
        try:
            from sklearn.metrics import roc_auc_score; res[f"{sp}_reach_auc_internal"] = float(roc_auc_score(lab.ravel(), sc_.ravel()))
        except Exception: pass
        if sp == "test":
            bl = baselines_np(X, Y, dx_idx if has_v else None, dy_idx if has_v else None, dxdy["dx"], dxdy["dy"], a.R_m, a.skip_tau)
            for b, dd in bl.items(): res.update({f"bl_{b}_{k}": v for k, v in dd.items()})
    json.dump(res, open(out_dir / "metrics.json", "w"), indent=2, ensure_ascii=False)
    log(f"  test  minADE {res['test_minADE']:.1f} m · ADE_π {res['test_ADE_pi']:.1f} m · nll {res['test_nll']:.3f} · reach AUC(internal) {res.get('test_reach_auc_internal', float('nan')):.3f}")
    return res

# ----------------------------------------------------------------------------- main
def parse():
    P = D5.PYCHARM
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", default=P["CITY"]); ap.add_argument("--data", default=P["DATA"]); ap.add_argument("--config", default=""); ap.add_argument("--proc", default="processed")
    ap.add_argument("--run", required=True); ap.add_argument("--M", type=int, default=P["M"])
    ap.add_argument("--rnn", choices=["lstm", "gru", "rnn", "tfm", "sttfm"], default="lstm"); ap.add_argument("--no-bi", dest="bi", action="store_false", default=True)
    ap.add_argument("--tfm-d", type=int, default=128); ap.add_argument("--tfm-blocks", type=int, default=2); ap.add_argument("--tfm-heads", type=int, default=4)
    ap.add_argument("--no-skip", dest="skip", action="store_false", default=True); ap.add_argument("--skip-tau", type=float, default=P["SKIP_TAU"]); ap.add_argument("--hidden", type=int, default=P["HIDDEN"])
    ap.add_argument("--with-occ", action="store_true"); ap.add_argument("--features", default="")
    ap.add_argument("--k-query", type=int, default=32); ap.add_argument("--lam-reach", type=float, default=2.0); ap.add_argument("--reach-only", action="store_true")
    ap.add_argument("--r-list", default="100,150,200,250"); ap.add_argument("--monitor", choices=["reach", "nll"], default="reach")
    ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--epochs", type=int, default=P["EPOCHS"]); ap.add_argument("--batch", type=int, default=P["BATCH"])
    ap.add_argument("--lr", type=float, default=P["LR"]); ap.add_argument("--patience", type=int, default=P["PATIENCE"])
    ap.add_argument("--eps-start", type=float, default=P["EPS_START"]); ap.add_argument("--eps-end", type=float, default=P["EPS_END"]); ap.add_argument("--eps-epochs", type=int, default=P["EPS_EPOCHS"])
    ap.add_argument("--wta-by", choices=["ade", "nll"], default="ade"); ap.add_argument("--subsample", type=float, default=None); ap.add_argument("--pred-dtype", default="float32"); ap.add_argument("--verbose", type=int, default=2)
    a = ap.parse_args(); a.seeds = [int(s) for s in a.seeds.split(",") if s]; a.r_list = [float(x) for x in a.r_list.split(",")]; return a

def main():
    a = parse(); setup_tf(); t_start = time.time(); log = lambda *m: print(f"[{time.time()-t_start:7.1f}s]", *m, flush=True)
    data_root = Path(a.data) if Path(a.data).is_absolute() else HERE / a.data
    cfg_path = a.config or str(HERE / "config" / f"data_{a.city}.yaml"); a.config_variant = cfg_path; a.config = merged_config_path(cfg_path)
    cfg = load_config_v2(cfg_path, a.city); a.R_m = float(cfg.get("R_m", 200))
    feats = [f.strip() for f in a.features.split(",")] if a.features else [f for f in cfg["features"] if f != "occ" or a.with_occ]
    log(f"config {cfg_path} · proc {a.proc} · ฟีเจอร์ {len(feats)}: {feats} · reach head K={a.k_query} λ={a.lam_reach} R={a.r_list}")
    data, dxdy = load_d4(data_root, a.city, cfg, feats, a.proc, a.subsample, a.seeds[0])
    p75_h = np.percentile(np.sqrt((data["train"]["Y"] ** 2).sum(-1)), 75, axis=0)          # เหมือน sample_pairs ของ D7 (คิดจาก train)
    if a.run == "base_M3": sys.exit("ห้ามเขียนทับ base_M3")
    out_root = HERE / "models" / a.city / a.run; out_root.mkdir(parents=True, exist_ok=True)
    json.dump({**{k: (v if isinstance(v, (int, float, str, bool, list)) or v is None else str(v)) for k, v in vars(a).items()}, "features": feats, "script": "sD5_train_v4", "p75_h": [float(x) for x in p75_h]},
              open(out_root / "run_config.json", "w"), indent=2, ensure_ascii=False)
    rows = []
    for seed in a.seeds:
        out_dir = out_root / f"seed{seed}"; out_dir.mkdir(exist_ok=True); log(f"=== {a.city} · {a.run} · seed {seed} ===")
        rows.append(train_seed(a, data, dxdy, feats, seed, out_dir, log, p75_h)); keras.backend.clear_session()
    df = pd.DataFrame(rows); ledger = HERE / "models" / "D5_results.csv"; df.to_csv(ledger, mode="a", header=not ledger.exists(), index=False)
    num = df.select_dtypes("number").drop(columns=["seed"], errors="ignore")
    summ = {"city": a.city, "run": a.run, "seeds": a.seeds, "features": feats, "mean": {k: float(v) for k, v in num.mean().items()}, "std": {k: float(v) for k, v in num.std(ddof=0).items()}}
    json.dump(summ, open(out_root / "summary.json", "w"), indent=2, ensure_ascii=False)
    m = summ["mean"]; log(f"สรุป {a.city}/{a.run}: minADE {m['test_minADE']:.1f} m · nll {m['test_nll']:.3f} · reach AUC(internal) {m.get('test_reach_auc_internal', float('nan')):.3f} → {out_root}")

if __name__ == "__main__":
    main()
