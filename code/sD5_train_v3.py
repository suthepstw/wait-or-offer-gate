#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sD5_train_v3.py — Bi-LSTM+ รอบ 2: แตะโมเดล "เฉพาะชั้นก่อน Dense(64)" และหัวเสริม (M6) · ไม่แก้ sD5_train.py (import ส่วนที่ไม่เปลี่ยน)
  --proc <dir>      อ่าน data/<city>/<dir>/ (เหมือน v2)
  --last-k K        ใช้เฉพาะ K ก้าวสุดท้ายของอินพุต (control ของ hist30: โมเดล 8 ก้าวบนหน้าต่างชุดเดียวกับ 30 ก้าว)
  --cab-emb D       M4: Embedding(cab_id, D) (dropout 0.3) concat กับ output ของ AttentionPool ก่อน Dense(64) · คันที่ไม่อยู่ใน train → เวกเตอร์ 0
  --prior           M5: concat {split}_prior.npy (32 มิติ z-scored จาก sD1_D4_prepare_v2 config prior:) ก่อน Dense(64)
  --mtl W           หัวที่ 4 reachability: logits ของ cell 13×13 (500 m, ±3 km รอบ x_T) ที่เส้นทางจริง 60 นาทีผ่าน · Σ_cell BCE × W เพิ่มใน loss (แนะนำ W = 0.1)
                    (ใช้เป็น auxiliary loss เท่านั้น — C_ij ยังมาจาก mixture → D6/D7 v7 ไม่เปลี่ยน · logits เก็บใน pred_*.npz key "reach" ไว้ให้ D7 v8)
  อย่างอื่น (encoder, 3 หัว, ε-WTA loss, optimizer, callbacks, npz, ledger) เหมือน sD5_train.py ทุกประการ
CLI:
  python sD5_train_v3.py --city sf --run abl_hist30    --proc processed_h30   --config config/data_sf_h30.yaml   --seeds 0,1,2
  python sD5_train_v3.py --city sf --run abl_hist30_k8 --proc processed_h30   --config config/data_sf_h30.yaml   --last-k 8 --seeds 0,1,2
  python sD5_train_v3.py --city sf --run abl_cab       --proc processed       --cab-emb 16 --seeds 0,1,2
  python sD5_train_v3.py --city sf --run abl_prior     --proc processed_prior --config config/data_sf_prior.yaml --prior --seeds 0,1,2
  python sD5_train_v3.py --city sf --run abl_mtl       --proc processed       --mtl 1.0 --seeds 0,1,2
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sD5_train as D5                                                   # noqa: E402
from sD5_train import (tf, keras, L, AttentionPool, PosEmbed, TransformerBlock, SpatialFourier, MixtureHead, skip_gain,  # noqa: E402
                       EpsSchedule, warmup_cosine, unpack_np, eval_np, baselines_np, setup_tf, set_seed, read_meta, EPS,
                       LAMBDA_PI, LAMBDA_ADE)
from sD1_D4_prepare_v2 import load_config_v2, merged_config_path       # noqa: E402

REACH_CELL_M, REACH_K = 500.0, 6                                        # 13×13 = 169 cells
N_REACH = (2 * REACH_K + 1) ** 2

# ----------------------------------------------------------------------------- data
def load_d4_v3(data_root: Path, city: str, cfg: dict, use_feats: list, proc_name: str, subsample=None, seed=0, last_k=0, want_prior=False):
    proc = data_root / city / proc_name
    if not proc.exists(): sys.exit(f"ไม่พบ {proc}")
    F_all = list(cfg["features"]); idx = [F_all.index(f) for f in use_feats]
    sc_path = proc / "scaler.json" if (proc / "scaler.json").exists() else data_root / city / "scaler.json"
    sc = json.load(open(sc_path)); print(f"  processed_dir {proc.name} · scaler {sc_path.relative_to(data_root)}")
    d = {}
    for sp in ("train", "val", "test"):
        X = np.load(proc / f"{sp}_X.npy", mmap_mode="r")
        if X.shape[-1] != len(F_all): sys.exit(f"{sp}_X มี {X.shape[-1]} คอลัมน์ ≠ config features {len(F_all)}")
        X = X[:, :, idx].astype(np.float32)
        if last_k and last_k < X.shape[1]: X = X[:, -last_k:, :]
        Y = np.load(proc / f"{sp}_Y.npy").astype(np.float32); meta = read_meta(proc / f"{sp}_meta")
        pr = np.load(proc / f"{sp}_prior.npy").astype(np.float32) if want_prior else None
        if want_prior and pr is None: sys.exit(f"ไม่พบ {sp}_prior.npy — รัน sD1_D4_prepare_v2 ด้วย config ที่มี prior:")
        if sp == "train" and subsample:
            rng = np.random.RandomState(seed); keep = np.sort(rng.choice(len(X), int(len(X) * subsample), replace=False))
            X, Y, meta = X[keep], Y[keep], meta.iloc[keep].reset_index(drop=True); pr = pr[keep] if pr is not None else None
        d[sp] = {"X": X, "Y": Y, "meta": meta, "prior": pr}
        print(f"  {sp:<5} X {X.shape}  Y {Y.shape}  prior {None if pr is None else pr.shape}  cycle-windows {int(meta['is_cycle_start'].sum()) if 'is_cycle_start' in meta else 0}")
    dxdy = {n: (float(sc["mean"][sc["names"].index(n)]), float(sc["std"][sc["names"].index(n)])) for n in ("dx", "dy")}
    return d, dxdy

def cab_vocab(meta_train: pd.DataFrame) -> dict:
    cabs = sorted(meta_train["cab"].astype(str).unique()); return {c: i + 1 for i, c in enumerate(cabs)}   # 0 = ไม่รู้จัก

def cab_ids(meta: pd.DataFrame, vocab: dict) -> np.ndarray:
    return np.array([vocab.get(c, 0) for c in meta["cab"].astype(str).values], np.int32)

def reach_targets(Y: np.ndarray) -> np.ndarray:
    """multi-hot (N, 169): cell 500 m ±6 รอบ x_T ที่ Y (60 ก้าว) ผ่าน · นอกช่วง → clip ขอบ"""
    nb = 2 * REACH_K + 1
    ix = np.clip(np.floor(Y[..., 0] / REACH_CELL_M).astype(int) + REACH_K, 0, nb - 1); iy = np.clip(np.floor(Y[..., 1] / REACH_CELL_M).astype(int) + REACH_K, 0, nb - 1)
    cell = ix * nb + iy; out = np.zeros((len(Y), nb * nb), np.float32)
    np.put_along_axis(out, cell, 1.0, axis=1); return out

# ----------------------------------------------------------------------------- batch feeder (กัน GPU OOM เมื่ออินพุตเป็น list: Keras จะ copy ทั้งก้อนขึ้น GPU)
class ArrayBatches(keras.utils.PyDataset):
    """ป้อน (xs, y) ทีละ batch จาก numpy บน CPU · shuffle ทุก epoch เมื่อ shuffle=True (เทียบเท่า model.fit(..., shuffle=True))"""
    def __init__(self, xs, y, batch, shuffle, seed=0, **kw):
        super().__init__(**kw); self.xs = xs if isinstance(xs, list) else [xs]; self.y = y; self.batch = int(batch); self.shuffle = shuffle
        self.n = len(y); self.rng = np.random.RandomState(seed); self.idx = np.arange(self.n)
        if shuffle: self.rng.shuffle(self.idx)
    def __len__(self): return int(np.ceil(self.n / self.batch))
    def __getitem__(self, i):
        j = self.idx[i * self.batch:(i + 1) * self.batch]
        xs = [x[j] for x in self.xs]
        return (tuple(xs) if len(xs) > 1 else xs[0]), self.y[j]
    def on_epoch_end(self):
        if self.shuffle: self.rng.shuffle(self.idx)

# ----------------------------------------------------------------------------- model
def build_model_v3(M_in, F, H, M, rnn="lstm", bi=True, hidden=256, n_heads=4, dropout=0.15, skip=True, skip_tau=3.0,
                   dx_idx=3, dy_idx=4, dx_ms=(0.0, 1.0), dy_ms=(0.0, 1.0), x_idx=0, y_idx=1, n_freq=8, tfm_d=128, tfm_blocks=2, tfm_heads=4,
                   n_cabs=0, cab_dim=0, prior_dim=0, mtl=False):
    x = L.Input(shape=(M_in, F), name="x_seq"); inputs = [x]
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
    feat = AttentionPool(n_heads, name="mhap")(h)
    feat = L.Dropout(dropout)(feat)
    extras = []
    if cab_dim > 0:                                             # M4
        cab_in = L.Input(shape=(), dtype="int32", name="cab_id"); inputs.append(cab_in)
        emb = L.Embedding(n_cabs + 1, cab_dim, name="cab_emb")(cab_in)                       # index 0 = ไม่รู้จัก
        emb = L.Lambda(lambda t: t[0] * tf.cast(tf.not_equal(t[1], 0), tf.float32)[:, None], output_shape=(cab_dim,), name="cab_mask")([emb, cab_in])   # → เวกเตอร์ 0
        extras.append(L.Dropout(0.3)(emb))
    if prior_dim > 0:                                           # M5
        pr_in = L.Input(shape=(prior_dim,), name="prior"); inputs.append(pr_in); extras.append(pr_in)
    if extras: feat = L.Concatenate(name="concat_extra")([feat] + extras)
    feat = L.Dense(64, activation="tanh")(feat)
    feat = L.Dropout(dropout)(feat)
    step_raw = L.Dense(M * H * 2, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02), kernel_regularizer=keras.regularizers.l2(1e-5), name="head_P")(feat)
    sig_raw = L.Dense(M * H, bias_initializer=keras.initializers.Constant(1.0), name="head_U")(feat)
    logits = L.Dense(M, name="head_M")(feat)
    out = MixtureHead(M, H, skip, dx_idx, dy_idx, tuple(dx_ms), tuple(dy_ms), skip_tau=skip_tau, name="mixture")([step_raw, sig_raw, logits, x])
    if mtl:                                                     # หัวที่ 4
        reach = L.Dense(N_REACH, name="head_R")(feat); out = L.Concatenate(name="out_all")([out, reach])
    tag = "_noskip" if not skip else ("_cv" if not np.isfinite(skip_tau) else f"_tau{skip_tau:g}")
    if rnn in ("tfm", "sttfm"): tag += f"_d{tfm_d}x{tfm_blocks}"
    tag += (f"_cab{cab_dim}" if cab_dim else "") + (f"_prior{prior_dim}" if prior_dim else "") + ("_mtl" if mtl else "") + (f"_in{M_in}" if M_in != 8 else "")
    return keras.Model(inputs, out, name=f"D5_{rnn}{'_bi' if bi and rnn not in ('tfm', 'sttfm') else ''}_M{M}{tag}")

# ----------------------------------------------------------------------------- loss (เดิม + BCE ของหัว reach)
def make_loss_v3(M: int, H: int, wta_by: str, mtl_w: float):
    n_mu, n_sig = M * H * 2, M * H; n_mix = n_mu + n_sig + M; n_y = H * 2
    def split(y_true, y_pred):
        return y_true[:, :n_y], y_pred[:, :n_mix], (y_true[:, n_y:] if mtl_w > 0 else None), (y_pred[:, n_mix:] if mtl_w > 0 else None)
    def unpack(y, p):
        B = tf.shape(p)[0]; y = tf.reshape(y, (B, 1, H, 2)); mu = tf.reshape(p[:, :n_mu], (B, M, H, 2)); sig = tf.reshape(p[:, n_mu:n_mu + n_sig], (B, M, H))
        return y, mu, sig, p[:, n_mu + n_sig:], tf.reduce_sum(tf.square(y - mu), -1)
    def loss(y_true, y_pred):
        y, p, yr, pr = split(y_true, y_pred); _, _, sig, logits, d2 = unpack(y, p)
        nll_m = tf.reduce_mean(d2 / (2.0 * tf.square(sig)) + 2.0 * tf.math.log(sig), -1); ade_m = tf.reduce_mean(tf.sqrt(d2 + 1e-6), -1)
        win = tf.argmin(ade_m if wta_by == "ade" else nll_m, axis=1); ade_win = tf.gather(ade_m, win, batch_dims=1) / 100.0
        if M == 1: base = nll_m[:, 0] + LAMBDA_ADE * ade_win
        else:
            oh = tf.one_hot(win, M, dtype=tf.float32); w = oh * (1.0 - EPS) + (1.0 - oh) * EPS / (M - 1)
            base = tf.reduce_sum(w * nll_m, -1) + LAMBDA_PI * tf.nn.softmax_cross_entropy_with_logits(labels=oh, logits=logits) + LAMBDA_ADE * ade_win
        if mtl_w > 0: base = base + mtl_w * tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=yr, logits=pr), -1)   # BCE รวม 169 cell (≈ 17 ตอนเริ่ม) × W
        return tf.reduce_mean(base)
    def nll(y_true, y_pred):
        y, p, _, _ = split(y_true, y_pred); _, _, sig, logits, d2 = unpack(y, p)
        logp = -tf.reduce_sum(d2 / (2.0 * tf.square(sig)) + tf.math.log(2.0 * np.pi * tf.square(sig)), -1)
        return tf.reduce_mean(-tf.reduce_logsumexp(tf.nn.log_softmax(logits) + logp, -1) / H)
    def minADE(y_true, y_pred):
        y, p, _, _ = split(y_true, y_pred); _, _, _, _, d2 = unpack(y, p); return tf.reduce_mean(tf.reduce_min(tf.reduce_mean(tf.sqrt(d2), -1), -1))
    def ADE_pi(y_true, y_pred):
        y, p, _, _ = split(y_true, y_pred); _, _, _, logits, d2 = unpack(y, p); k = tf.argmax(logits, -1)
        return tf.reduce_mean(tf.gather(tf.reduce_mean(tf.sqrt(d2), -1), k, batch_dims=1))
    def reach_bce(y_true, y_pred):
        _, _, yr, pr = split(y_true, y_pred)
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=yr, logits=pr)) if mtl_w > 0 else tf.constant(0.0)
    return loss, [nll, minADE, ADE_pi] + ([reach_bce] if mtl_w > 0 else [])

# ----------------------------------------------------------------------------- train one seed
def train_seed_v3(a, data, dxdy, feats, seed, out_dir: Path, log, vocab):
    set_seed(seed); tr, va, te = data["train"], data["val"], data["test"]
    M_in, F = tr["X"].shape[1:]; H = tr["Y"].shape[1]; M = a.M
    has_v = "dx" in feats and "dy" in feats
    if not has_v and a.skip: log("  ไม่มี dx,dy → ปิด skip"); a.skip = False
    dx_idx, dy_idx = (feats.index("dx"), feats.index("dy")) if has_v else (0, 1)
    x_idx, y_idx = (feats.index("x"), feats.index("y")) if ("x" in feats and "y" in feats) else (0, 1)
    prior_dim = tr["prior"].shape[1] if a.prior else 0
    model = build_model_v3(M_in, F, H, M, rnn=a.rnn, bi=a.bi, hidden=a.hidden, skip=a.skip, skip_tau=a.skip_tau, dx_idx=dx_idx, dy_idx=dy_idx,
                           dx_ms=dxdy["dx"], dy_ms=dxdy["dy"], x_idx=x_idx, y_idx=y_idx, tfm_d=a.tfm_d, tfm_blocks=a.tfm_blocks, tfm_heads=a.tfm_heads,
                           n_cabs=len(vocab) if a.cab_emb else 0, cab_dim=a.cab_emb, prior_dim=prior_dim, mtl=a.mtl > 0)
    if a.init_from:
        try: model.load_weights(a.init_from, skip_mismatch=True)
        except TypeError: model.load_weights(a.init_from)
    try: opt = keras.optimizers.AdamW(learning_rate=a.lr, weight_decay=1e-4, clipnorm=1.0)
    except Exception: opt = keras.optimizers.Adam(learning_rate=a.lr, clipnorm=1.0)
    loss, metrics = make_loss_v3(M, H, a.wta_by, a.mtl)
    model.compile(optimizer=opt, loss=loss, metrics=metrics, jit_compile=False)
    if seed == a.seeds[0]:
        with open(out_dir.parent / "model_summary.txt", "w", encoding="utf-8") as f: model.summary(print_fn=lambda s: f.write(s + "\n"))
    n_params = int(model.count_params()); log(f"  {model.name} · params {n_params:,} · train {len(tr['X']):,} · batch {a.batch} · inputs {[i.name for i in model.inputs]}")

    def pack_x(d):
        xs = [d["X"]]
        if a.cab_emb: xs.append(cab_ids(d["meta"], vocab))
        if a.prior: xs.append(d["prior"])
        return xs if len(xs) > 1 else xs[0]
    def pack_y(d):
        y = d["Y"].reshape(len(d["Y"]), -1)
        return np.concatenate([y, reach_targets(d["Y"])], 1) if a.mtl > 0 else y
    cbs = [EpsSchedule(a.eps_start, a.eps_end, a.eps_epochs), keras.callbacks.LearningRateScheduler(warmup_cosine(a.lr, a.epochs, warmup=min(5, a.epochs))),
           keras.callbacks.EarlyStopping(monitor="val_nll", mode="min", patience=a.patience, restore_best_weights=True, verbose=0),
           keras.callbacks.CSVLogger(str(out_dir / "history.csv"))]
    t0 = time.time()
    if a.cab_emb or a.prior or a.mtl > 0:   # list input / y กว้าง → ป้อนทีละ batch (Keras ไม่ copy ทั้งชุดขึ้น GPU)
        ds_tr = ArrayBatches(pack_x(tr), pack_y(tr), a.batch, shuffle=True, seed=seed); ds_va = ArrayBatches(pack_x(va), pack_y(va), 2048, shuffle=False)
        hist = model.fit(ds_tr, validation_data=ds_va, epochs=a.epochs, callbacks=cbs, verbose=a.verbose)
    else:
        hist = model.fit(pack_x(tr), pack_y(tr), validation_data=(pack_x(va), pack_y(va)), epochs=a.epochs, batch_size=a.batch, shuffle=True, callbacks=cbs, verbose=a.verbose)
    train_s = time.time() - t0; h = hist.history; n_ep = len(h["loss"]); best_ep = int(np.argmin(h["val_nll"])) + 1
    model.save_weights(str(out_dir / "best.weights.h5"))
    log(f"  ฝึก {n_ep} epoch ({train_s/60:.1f} นาที · {train_s/n_ep:.0f} s/epoch) · best epoch {best_ep} · val_nll {min(h['val_nll']):.3f}")

    n_mix = M * H * 2 + M * H + M
    res = {"city": a.city, "run": a.run, "seed": seed, "model": model.name, "M": M, "rnn": a.rnn, "bi": a.bi, "skip": a.skip, "skip_tau": a.skip_tau, "hidden": a.hidden,
           "features": ",".join(feats), "n_features": int(F), "M_in": int(M_in), "H": int(H), "params": n_params, "cab_emb": a.cab_emb, "prior_dim": prior_dim, "mtl_w": a.mtl,
           "last_k": a.last_k, "proc": a.proc, "n_train": int(len(tr["X"])), "n_val": int(len(va["X"])), "n_test": int(len(te["X"])), "epochs_run": n_ep, "best_epoch": best_ep,
           "train_min": round(train_s / 60, 2), "s_per_epoch": round(train_s / n_ep, 1), "init_from": a.init_from or "", "val_nll_best": float(min(h["val_nll"]))}
    for sp, d in (("val", va), ("test", te)):
        t1 = time.time()
        P = model.predict(ArrayBatches(pack_x(d), d["Y"].reshape(len(d["Y"]), -1), 2048, shuffle=False), verbose=0) if (a.cab_emb or a.prior or a.mtl > 0) else model.predict(pack_x(d), batch_size=2048, verbose=0)
        infer_ms = (time.time() - t1) * 1e6 / max(1, len(d["X"]))
        mu, sig, pi = unpack_np(P[:, :n_mix], M, H); meta = d["meta"]
        keys = {k: meta[k].values for k in ("cab", "seg", "t_T", "lat_T", "lon_T", "date", "hour", "is_cycle_start", "is_cycle_clean") if k in meta}
        extra = {"reach": P[:, n_mix:].astype(np.float16)} if a.mtl > 0 else {}
        with np.errstate(over="ignore"):
            np.savez_compressed(out_dir / f"pred_{sp}.npz", mu=np.clip(mu, -6e4, 6e4).astype(a.pred_dtype), sigma=sig.astype(a.pred_dtype), pi=pi.astype(np.float32),
                                row=np.arange(len(d["X"])), **keys, **extra)
        if sp == "test": res["infer_ms_per_1k_windows"] = round(infer_ms, 1)
        res.update({f"{sp}_{k}": v for k, v in eval_np(d["Y"], mu, sig, pi, a.R_m).items()})
        if a.mtl > 0:
            yr = reach_targets(d["Y"]); pr = 1 / (1 + np.exp(-P[:, n_mix:].astype(np.float64)))
            try:
                from sklearn.metrics import roc_auc_score; res[f"{sp}_reach_auc"] = float(roc_auc_score(yr.ravel(), pr.ravel()))
            except Exception: pass
        if sp == "test":
            mask = meta["is_cycle_start"].values.astype(bool) if "is_cycle_start" in meta else None
            if mask is not None and mask.any(): res.update({f"cyc_{k}": v for k, v in eval_np(d["Y"], mu, sig, pi, a.R_m, mask).items()})
            bl = baselines_np(d["X"], d["Y"], dx_idx if has_v else None, dy_idx if has_v else None, dxdy["dx"], dxdy["dy"], a.R_m, a.skip_tau)
            for b, dd in bl.items(): res.update({f"bl_{b}_{k}": v for k, v in dd.items()})
    json.dump(res, open(out_dir / "metrics.json", "w"), indent=2, ensure_ascii=False)
    log(f"  test  minADE {res['test_minADE']:.1f} m · ADE_π {res['test_ADE_pi']:.1f} m · FDE_π {res['test_FDE_pi']:.1f} m · hit@R(π) {res['test_hitR_pi']:.3f} · nll {res['test_nll']:.3f}"
        + (f" · reach AUC {res.get('test_reach_auc', float('nan')):.3f}" if a.mtl > 0 else "") + f"   | static {res['bl_static_ADE']:.1f} m · damped_vel {res['bl_damped_vel_ADE']:.1f} m")
    return res

# ----------------------------------------------------------------------------- main
def parse():
    P = D5.PYCHARM
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", default=P["CITY"]); ap.add_argument("--data", default=P["DATA"]); ap.add_argument("--config", default="")
    ap.add_argument("--run", required=True); ap.add_argument("--proc", default="processed"); ap.add_argument("--M", type=int, default=P["M"])
    ap.add_argument("--rnn", choices=["lstm", "gru", "rnn", "tfm", "sttfm"], default="lstm"); ap.add_argument("--no-bi", dest="bi", action="store_false", default=True)
    ap.add_argument("--tfm-d", type=int, default=128); ap.add_argument("--tfm-blocks", type=int, default=2); ap.add_argument("--tfm-heads", type=int, default=4)
    ap.add_argument("--no-skip", dest="skip", action="store_false", default=True); ap.add_argument("--skip-tau", type=float, default=P["SKIP_TAU"])
    ap.add_argument("--hidden", type=int, default=P["HIDDEN"]); ap.add_argument("--with-occ", action="store_true"); ap.add_argument("--features", default="")
    ap.add_argument("--last-k", type=int, default=0); ap.add_argument("--cab-emb", type=int, default=0); ap.add_argument("--prior", action="store_true"); ap.add_argument("--mtl", type=float, default=0.0)
    ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--epochs", type=int, default=P["EPOCHS"]); ap.add_argument("--batch", type=int, default=P["BATCH"])
    ap.add_argument("--lr", type=float, default=P["LR"]); ap.add_argument("--patience", type=int, default=P["PATIENCE"])
    ap.add_argument("--eps-start", type=float, default=P["EPS_START"]); ap.add_argument("--eps-end", type=float, default=P["EPS_END"]); ap.add_argument("--eps-epochs", type=int, default=P["EPS_EPOCHS"])
    ap.add_argument("--wta-by", choices=["ade", "nll"], default="ade"); ap.add_argument("--init-from", default=""); ap.add_argument("--subsample", type=float, default=None)
    ap.add_argument("--pred-dtype", default="float32"); ap.add_argument("--verbose", type=int, default=2)
    a = ap.parse_args(); a.seeds = [int(s) for s in str(a.seeds).split(",") if s != ""]; return a

def main():
    a = parse(); setup_tf(); t_start = time.time(); log = lambda *m: print(f"[{time.time()-t_start:7.1f}s]", *m, flush=True)
    data_root = Path(a.data) if Path(a.data).is_absolute() else HERE / a.data
    cfg_path = a.config or str(HERE / "config" / f"data_{a.city}.yaml")
    a.config_variant = cfg_path; a.config = merged_config_path(cfg_path)          # run_config.json["config"] = ไฟล์รวม (sD7 v7 อ่าน dates ได้)
    cfg = load_config_v2(cfg_path, a.city); a.R_m = float(cfg.get("R_m", 200))
    feats = [f.strip() for f in a.features.split(",")] if a.features else [f for f in cfg["features"] if f != "occ" or a.with_occ]
    log(f"config {cfg_path} · proc {a.proc} · ฟีเจอร์ {len(feats)}: {feats} · last_k {a.last_k} · cab_emb {a.cab_emb} · prior {a.prior} · mtl {a.mtl}")
    data, dxdy = load_d4_v3(data_root, a.city, cfg, feats, a.proc, a.subsample, a.seeds[0], a.last_k, a.prior)
    vocab = cab_vocab(data["train"]["meta"]) if a.cab_emb else {}
    if a.cab_emb:
        unk = {sp: float((cab_ids(data[sp]["meta"], vocab) == 0).mean()) for sp in ("val", "test")}
        log(f"  cab vocab {len(vocab)} คัน (train) · ไม่รู้จักใน val {unk['val']*100:.1f}% · test {unk['test']*100:.1f}%")
    out_root = HERE / "models" / a.city / a.run; out_root.mkdir(parents=True, exist_ok=True)
    if a.run == "base_M3": sys.exit("ห้ามเขียนทับ base_M3")
    json.dump({**{k: (v if isinstance(v, (int, float, str, bool, list)) or v is None else str(v)) for k, v in vars(a).items()}, "features": feats, "script": "sD5_train_v3"},
              open(out_root / "run_config.json", "w"), indent=2, ensure_ascii=False)
    if a.cab_emb: json.dump(vocab, open(out_root / "cab_vocab.json", "w"))
    rows = []
    for seed in a.seeds:
        out_dir = out_root / f"seed{seed}"; out_dir.mkdir(exist_ok=True); log(f"=== {a.city} · {a.run} · seed {seed} ===")
        rows.append(train_seed_v3(a, data, dxdy, feats, seed, out_dir, log, vocab)); keras.backend.clear_session()
    df = pd.DataFrame(rows); ledger = HERE / "models" / "D5_results.csv"; df.to_csv(ledger, mode="a", header=not ledger.exists(), index=False)
    num = df.select_dtypes("number").drop(columns=["seed"], errors="ignore")
    summ = {"city": a.city, "run": a.run, "seeds": a.seeds, "features": feats, "mean": {k: float(v) for k, v in num.mean().items()}, "std": {k: float(v) for k, v in num.std(ddof=0).items()}}
    json.dump(summ, open(out_root / "summary.json", "w"), indent=2, ensure_ascii=False)
    m, s = summ["mean"], summ["std"]
    log(f"สรุป {a.city}/{a.run} ({len(rows)} seeds): test minADE {m['test_minADE']:.1f}±{s['test_minADE']:.1f} m · ADE_π {m['test_ADE_pi']:.1f} m · hit@R(π) {m['test_hitR_pi']:.3f} · nll {m['test_nll']:.3f}")
    log(f"ผลลัพธ์ → {out_root} · ledger → {ledger}")

if __name__ == "__main__":
    main()
