#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sD7_pred_eval.py (v7) — D7: ประเมินคำทำนายบนชุด test หลัง calibration จาก D6  → ⛔ gate G1
v7 = ตอบข้อสังเกตจาก council review (2026-09-20) โดย **ไม่แตะ D5/D6** (แก้เฉพาะขั้นวัดผล):
  1) null ที่ยุติธรรมกว่า persistence แบบ 0/1: gauss_pers = Gaussian รอบตำแหน่งปัจจุบัน · gauss_dv = Gaussian รอบเส้นฐาน damped-velocity
     σ_h ของ null fit จาก val (RMS residual รายขอบฟ้า) → ผ่านสูตร p_h/qm_ub และ isotonic **เดียวกัน** กับโมเดล → บอกว่า Bi-LSTM เพิ่มอะไรจาก "ยิ่งไกลยิ่งไปไม่ถึง"
  2) ชุด hard (งานใกล้เส้นทางจริง ≤ 2R): รายงาน base rate, AUC, AUPRC ของทุก variant + reliability
  3) isotonic ของ QM: รวม block ท้ายจนมีตัวอย่าง ≥ N_MIN_TAIL (กัน over-predict ที่ bins > 0.5 ซึ่ง optimizer จะไล่ล่า) · reliability ระบุ n ต่อ bin
  4) E1_handoff.json: เลือก seed ที่ AUC(QM_cal) ใกล้ median ที่สุด + sha256 ของ pred_test.npz / calib.json / qm_calib.json → sai E ใช้ seed เดียวนี้
  5) --sub-steps K (sensitivity เท่านั้น, ค่าหลัก K=1): แทรกจุด K−1 จุดระหว่างนาที **ทั้ง label และโมเดล** → QM เปลี่ยนความหมายจาก "อยู่ในวง ≥ pt" เป็น "วิ่งผ่านวง"
  (จาก v6: split dev/final · QM = qm_lb/qm_ub/qm_mc · cluster bootstrap · --clean-only · isotonic fit บน val — คงเดิม, gate G1 คงเกณฑ์เดิมตาม EXPERIMENT_SCOPE §9.4)

นิยาม QM (คงเดิม): QM_ij = P(เส้นทางของ i "มีจุดตัวอย่าง" ใน disc(l_j, R) ระหว่าง [st_j, et_j]) — ที่ Δτ = pt = 60 s "มีจุดตัวอย่างในวง" ≈ dwell ≥ pt
  label ใช้กติกาเดียวกับโมเดล (ดูทีละ Δτ) → สมมาตร; p_h = Σ_m π_m P(‖N(μ_mh, (λ_hσ_mh)²I) − l‖ ≤ R) · qm_lb = max_h p_h · qm_ub = 1−Π(1−p_h)

อ่าน  models/<city>/<run>/seed<k>/{pred_val.npz, pred_test.npz, calib.json, metrics.json} · data/<city>/processed/{val,test}_{X,Y}.npy, *_meta · scaler.json · config
เขียน models/<city>/<run>/D7_<split>[_clean][_kK]/  pred_metrics.csv · D7_report.md · G1.json · E1_handoff.json · qm_reliability.csv · fig_P1..P4.png
      models/<city>/<run>/seed<k>/qm_calib.json   isotonic map ของ QM (fit บน val) ต่อ R + σ_h ของ null  (--sub-steps K>1 → qm_calib_kK.json แยกไฟล์)
      models/D7_results.csv                          ledger (คอลัมน์ split, clean, sub_steps) — schema เปลี่ยนจาก v6 → rm ก่อนรัน

Gate G1 (ประกาศล่วงหน้าสำหรับ final test — เกณฑ์ไม่เปลี่ยนใน v7):
  (a) model_pi ชนะ persistence และ const_vel ทุก h ≥ 5: Holm p < 0.05 (ข้าม seed) และ cluster-bootstrap 95% CI ของผลต่าง > 0
  (b) calibration ratio ∈ [0.8, 1.25] ทุก h
  (c) Reachability @R หลัก: AUC(QM = qm_ub) > 0.8 (รายงาน cluster-CI) และ AUPRC(QM) ≥ 2 × AUPRC(persistence)
  (d) QM calibrated: ECE ≤ 0.01 หลัง isotonic (fit บน val) และ Brier skill score > 0
  + รายงานประกอบ (ไม่ใช่เกณฑ์): AUC/AUPRC ของ null gauss_dv_cal, lift เทียบ null, ผลบนชุด hard

ไม่ต้องใช้ GPU · numpy/pandas/scipy (+matplotlib) · รันซ้ำได้ · Python ≥ 3.8
PyCharm: แก้บล็อก PYCHARM แล้ว Run · CLI: python sD7_pred_eval.py --city sf,rome --runs base_M3 --split dev · ทดสอบ: --demo
"""
from __future__ import annotations
import argparse, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

PYCHARM = dict(
    CITY="sf", RUNS="base_M3", DATA="data", MODELS="models",
    SPLIT="dev",                                 # "dev" (ออกแบบ/ตรวจ) · "final" (เปิดครั้งเดียวหลัง freeze) · "all" (ชุด test ทั้งก้อนของ D4)
    CLEAN_ONLY=False,                            # True = เฉพาะหน้าต่างที่ gap ดิบ ≤ clean_gap_mult·Δτ ตลอด 68 นาที (sensitivity ต่อ interpolation)
    R_LIST=[100, 150, 200, 250], R_MAIN=200,
    N_PAIRS=60000, N_HARD=5000,                  # คู่หลัก (จุดสุ่มทั่วบริเวณที่ไปถึงได้) · คู่ hard (งานใกล้เส้นทางจริง ≤ 2R — รายงานแยก ไม่ใช้ตัดสิน)
    N_VAL_PAIRS=30000,                           # คู่จาก val สำหรับ fit isotonic ของ QM และ σ_h ของ null
    S=30, NOISE="brownian",                      # scenario ใน optimizer (เทียบ)
    S_EVAL=200, N_MC=8000,                       # Monte-Carlo QM (S_EVAL scenario) บนชุดย่อย N_MC คู่แรก
    QM="ub",                                     # QM ที่ใช้ตัดสิน G1 และ sai E ใช้: "ub" (ตัดสินใจบน dev 2026-09-12) | "lb" | "mc"
    SUB_STEPS=1,                                 # v7: 1 = ความหมายเดิม (dwell ≥ pt) · 4 = แทรกจุดทุก 15 s ทั้ง label และโมเดล (sensitivity "วิ่งผ่าน")
    N_MIN_TAIL=50,                               # v7: block ท้ายของ isotonic ต้องมีตัวอย่าง ≥ เท่านี้ (กัน over-predict ที่ QM สูง)
    N_BOOT=300, DEMO=False,
)
H_REPORT = [5, 10, 20, 30, 45, 60]
DURATIONS = [5, 10, 15, 20, 30]
CRN_SEED = 12345
SIGMA0_M = 5.0                                                   # σ ที่ h=0 เมื่อแทรกจุด (เท่ากับ SIGMA_MIN ของ D5)
REL_EDGES = [0, .01, .02, .05, .1, .2, .3, .5, .7, 1.0001]   # bins ของ reliability สำหรับ QM (positive rate ต่ำ → bins ละเอียดใกล้ 0)
NULLS = ("gauss_pers", "gauss_dv")                               # v7: null แบบ Gaussian (M=1) รอบ persistence / damped-velocity
SUBSET_VARS = ("qm_lb", "qm_ub", "qm_ub_cal", "scenario", "persistence", "gauss_pers_cal", "gauss_dv_cal")

# ----------------------------------------------------------------------------- utils
def load_npz(p: Path):
    z = np.load(p, allow_pickle=True); d = {k: z[k] for k in z.files}
    for k in ("mu", "sigma", "pi"): d[k] = d[k].astype(np.float32)
    return d

def softmax_T(pi, T):
    lg = np.log(np.clip(pi, 1e-12, None)) / T; lg -= lg.max(1, keepdims=True); e = np.exp(lg); return e / e.sum(1, keepdims=True)

def skip_gain(H, tau):
    h = np.arange(1, H + 1, dtype=np.float64)
    if tau <= 0: return np.zeros(H)
    if not np.isfinite(tau): return h
    return tau * (1 - np.exp(-h / tau))

def upsample_h(arr, K, axis, first):
    """v7: แทรก K−1 จุดเชิงเส้นระหว่างแต่ละก้าวตามแกน h (ความยาว H → H·K) · first = ค่าที่ h=0 (สเกลาร์ หรือ array รูป arr โดยไม่มีแกน h)"""
    if K == 1: return arr
    a = np.moveaxis(np.asarray(arr), axis, -1); H = a.shape[-1]
    f = np.broadcast_to(np.asarray(first, a.dtype)[..., None] if np.ndim(first) else np.asarray(first, a.dtype), a.shape[:-1] + (1,))
    prev = np.concatenate([f, a[..., :-1]], -1)
    w = (np.arange(1, K + 1) / K).astype(a.dtype)
    out = (prev[..., None] + (a - prev)[..., None] * w).reshape(a.shape[:-1] + (H * K,))
    return np.moveaxis(out, -1, axis)

def auc_rank(score, label) -> float:
    from scipy.stats import rankdata
    r = rankdata(score); n1 = label.sum(); n0 = len(label) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def auprc(score, label) -> float:
    if label.sum() == 0: return float("nan")
    o = np.argsort(-score, kind="stable"); l = label[o]
    tp = np.cumsum(l); prec = tp / np.arange(1, len(l) + 1)
    return float((prec * l).sum() / l.sum())

def ttest_1samp(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2 or x.std(ddof=1) == 0: return float("nan")
    t = x.mean() / (x.std(ddof=1) / math.sqrt(n))
    from scipy.stats import t as tdist; return float(2 * tdist.sf(abs(t), n - 1))

def holm(pvals: dict) -> dict:
    items = sorted(((v, k) for k, v in pvals.items() if not np.isnan(v))); m = len(items); out = {}; prev = 0.0
    for i, (p, k) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p)); out[k] = adj; prev = adj
    for k in pvals: out.setdefault(k, float("nan"))
    return out

def pct(x, q): return float(np.percentile(x, q))

def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""): h.update(blk)
    return h.hexdigest()

class ClusterBoot:
    """bootstrap ตาม cluster: สุ่ม cluster ซ้ำแล้วต่อ index ของสมาชิก → สถิติใด ๆ ที่รับ index"""
    def __init__(self, clusters, n_boot=300, seed=0):
        uniq, inv = np.unique(clusters, return_inverse=True)
        order = np.argsort(inv, kind="stable"); counts = np.bincount(inv, minlength=len(uniq)); bounds = np.r_[0, np.cumsum(counts)]
        self.groups = [order[bounds[g]:bounds[g + 1]] for g in range(len(uniq))]
        self.n_boot = n_boot; self.rng = np.random.RandomState(seed); self.n_clusters = len(uniq)
    def ci(self, stat_fn):
        vals = []
        for _ in range(self.n_boot):
            pick = self.rng.randint(0, self.n_clusters, self.n_clusters)
            idx = np.concatenate([self.groups[g] for g in pick]); vals.append(stat_fn(idx))
        vals = np.array(vals, float); vals = vals[~np.isnan(vals)]
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) else (float("nan"), float("nan"))

def pav_fit(x, y, n_min_tail=0):
    """isotonic regression (pool-adjacent-violators) → จุดต่อ (xs, ys, ns) สำหรับ np.interp
    v7: n_min_tail > 0 → รวม block ท้ายเข้าด้วยกันจนมีตัวอย่าง ≥ n_min_tail (ยังคง monotone) กันค่าสูงที่ประมาณจากตัวอย่างไม่กี่คู่"""
    o = np.argsort(x, kind="stable"); xs = x[o].astype(float); ys = y[o].astype(float)
    S, N, X = [], [], []                                         # blocks: [sum_y, n, sum_x]
    for xi, yi in zip(xs, ys):
        S.append(yi); N.append(1); X.append(xi)
        while len(S) > 1 and S[-2] / N[-2] > S[-1] / N[-1]:
            s, n, xx = S.pop(), N.pop(), X.pop(); S[-1] += s; N[-1] += n; X[-1] += xx
    n_merged = 0
    while n_min_tail > 0 and len(S) > 1 and N[-1] < n_min_tail:
        s, n, xx = S.pop(), N.pop(), X.pop(); S[-1] += s; N[-1] += n; X[-1] += xx; n_merged += 1
    bx = np.array(X) / np.array(N); by = np.array(S) / np.array(N)
    return bx, by, np.array(N), n_merged

def pav_apply(bx, by, x): return np.interp(x, bx, by, left=by[0], right=by[-1]) if len(bx) else x

def reliability(score, label, edges=REL_EDGES):
    rows = []; ece = 0.0; n = len(score)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (score >= a) & (score < b)
        if m.sum():
            rows.append({"lo": a, "hi": min(b, 1.0), "n": int(m.sum()), "qm_mean": float(score[m].mean()), "freq": float(label[m].mean())})
            ece += m.sum() / n * abs(score[m].mean() - label[m].mean())
    return rows, float(ece)

def brier(score, label):
    b = float(((score - label) ** 2).mean()); p = label.mean(); b0 = float(p * (1 - p))
    return b, (1 - b / b0) if b0 > 0 else float("nan")

# ----------------------------------------------------------------------------- data
def load_cfg(city, data_root, rd):
    rc = json.load(open(rd / "run_config.json")); feats = rc["features"]; tau = float(rc.get("skip_tau", 3.0))
    cfg_path = rc.get("config") or ""
    cfg = {}
    for cand in ([Path(cfg_path)] if cfg_path else []) + [HERE / "config" / f"data_{city}.yaml"]:
        if cand and Path(cand).exists():
            import yaml; c = yaml.safe_load(open(cand, encoding="utf-8"))
            if "profile" in c: prof = c[c["profile"]]; c = {k: v for k, v in c.items() if k not in ("hf", "full")}; c.update(prof)
            cfg = c; break
    return feats, tau, cfg

def read_meta(proc: Path, sp: str):
    for p in (proc / f"{sp}_meta.parquet", proc / f"{sp}_meta.pkl", proc / f"{sp}_meta.csv"):
        if p.exists():
            return pd.read_parquet(p) if p.suffix == ".parquet" else (pd.read_pickle(p) if p.suffix == ".pkl" else pd.read_csv(p))
    return None

def load_split(data_root, city, rd, sp, a, log):
    """คืน Y, v (dx,dy ก้าวสุดท้าย m/ก้าว), meta, mask (split+clean), clusters, tau, cfg"""
    feats, tau, cfg = load_cfg(city, data_root, rd)
    proc = data_root / city / "processed"
    Y = np.load(proc / f"{sp}_Y.npy").astype(np.float32); N, H = Y.shape[:2]
    X = np.load(proc / f"{sp}_X.npy", mmap_mode="r"); sc = json.load(open(data_root / city / "scaler.json"))
    di, dj = (feats.index("dx"), feats.index("dy")) if X.shape[-1] == len(feats) else (3, 4)
    mdx, sdx = sc["mean"][sc["names"].index("dx")], sc["std"][sc["names"].index("dx")]; mdy, sdy = sc["mean"][sc["names"].index("dy")], sc["std"][sc["names"].index("dy")]
    v = np.stack([np.asarray(X[:, -1, di]) * sdx + mdx, np.asarray(X[:, -1, dj]) * sdy + mdy], -1).astype(np.float32)
    meta = read_meta(proc, sp)
    mask = np.ones(N, bool); note = []
    if sp == "test" and a.split != "all":
        key = f"test_{a.split}"; rng_ = (cfg.get("dates") or {}).get(key)
        if rng_ is None:
            log(f"  config ไม่มี dates.{key} → ใช้ test ทั้งก้อน (ประกาศเป็น dev)"); note.append(f"{key}: not defined → all")
        elif meta is None or "date" not in meta:
            sys.exit("ต้องมี test_meta (คอลัมน์ date) เพื่อแบ่ง dev/final")
        else:
            d0, d1 = [str(x) for x in rng_]; dates = meta["date"].astype(str).values; mask &= (dates >= d0) & (dates <= d1); note.append(f"{key} {d0}..{d1}")
    if sp == "test" and a.clean_only and meta is not None and "max_gap_s" in meta:
        thr = float(cfg.get("clean_gap_mult", 2)) * float(cfg.get("dt_s", 60)); mask &= meta["max_gap_s"].values <= thr; note.append(f"clean gap≤{thr:g}s")
    clusters = (meta["cab"].astype(str) + "_" + meta["date"].astype(str)).values if meta is not None and "cab" in meta and "date" in meta else np.arange(N).astype(str)
    return Y, v, meta, mask, clusters, tau, cfg, note

# ----------------------------------------------------------------------------- 1) error รายขอบฟ้า
def horizon_errors(Y, mu, pi, v, tau) -> dict:
    N, H = Y.shape[:2]; hs = np.arange(1, H + 1)
    d = np.sqrt(((Y[:, None] - mu) ** 2).sum(-1)); k = pi.argmax(1)
    return {"model_pi": d[np.arange(N), k], "model_min": d[np.arange(N), d.mean(-1).argmin(1)], "persistence": np.sqrt((Y ** 2).sum(-1)),
            "const_vel": np.sqrt(((Y - v[:, None, :] * hs[None, :, None]) ** 2).sum(-1)),
            "damped_vel": np.sqrt(((Y - v[:, None, :] * skip_gain(H, tau)[None, :, None]) ** 2).sum(-1))}

# ----------------------------------------------------------------------------- 2) probabilistic
def prob_metrics(Y, mu, sig, pi) -> dict:
    d2 = ((Y[:, None] - mu) ** 2).sum(-1); d = np.sqrt(d2)
    s_emp = np.sqrt((pi[:, :, None] * d2).sum(1).mean(0) / 2); s_pred = np.sqrt((pi[:, :, None] * sig ** 2).sum(1).mean(0))
    cov = {k: (pi[:, :, None] * (d <= k * sig)).sum(1).mean(0) for k in (1.0, 2.0, 2.146)}
    z = np.log(np.clip(pi, 1e-12, None))[:, :, None] - d2 / (2 * sig ** 2) - np.log(2 * np.pi * sig ** 2)
    zm = z.max(1, keepdims=True); nll_h = -(zm[:, 0] + np.log(np.exp(z - zm).sum(1))).mean(0)
    win = d.mean(-1).argmin(1); conf = pi.max(1); hit = pi.argmax(1) == win
    edges = np.linspace(1 / pi.shape[1], 1, 11); ece = 0.0
    for a_, b_ in zip(edges[:-1], edges[1:]):
        m = (conf >= a_) & (conf < b_) if b_ < 1 else conf >= a_
        if m.sum(): ece += m.mean() * abs(conf[m].mean() - hit[m].mean())
    return {"ratio": s_emp / s_pred, "nll_h": nll_h, "cov1": cov[1.0], "cov2": cov[2.0], "cov90": cov[2.146], "ece": float(ece), "pi_acc": float(hit.mean())}

# ----------------------------------------------------------------------------- 3) reachability
def sample_pairs(Y, n_pairs, R_max, rng, n_hard=0):
    """คู่หลัก: l สุ่มสม่ำเสมอในวงรัศมี p75 ของ |y_et| (ค่ารวมทั้งชุด ไม่ใช่รายคัน; ≥ 4R) · คู่ hard: l ห่างเส้นทางจริง ≤ 2R"""
    N, H = Y.shape[:2]; n_tot = n_pairs + n_hard
    i = rng.randint(0, N, n_tot)
    dur = rng.choice(DURATIONS, n_tot); st = rng.randint(1, np.maximum(2, H - dur + 1)); et = np.minimum(H, st + dur)
    near = np.r_[np.zeros(n_pairs, bool), np.ones(n_hard, bool)]
    hstar = np.minimum(st + (rng.rand(n_tot) * (et - st + 1)).astype(int), H)
    ang = rng.rand(n_tot) * 2 * np.pi; rad = np.sqrt(rng.rand(n_tot))
    l_near = Y[i, hstar - 1] + np.stack([np.cos(ang), np.sin(ang)], -1) * (rad * 2 * R_max)[:, None]
    p75_h = np.percentile(np.sqrt((Y ** 2).sum(-1)), 75, axis=0); scale = np.maximum(p75_h[et - 1], 4 * R_max)
    l_rand = np.stack([np.cos(ang), np.sin(ang)], -1) * (rad * scale)[:, None]
    return i, st, et, np.where(near[:, None], l_near, l_rand).astype(np.float32), near

def hit_mask(st, et, H, K=1):
    hs = (np.arange(1, H * K + 1) / K)[None, :]; return (hs >= st[:, None]) & (hs <= et[:, None])

def reach_labels(Y, i, st, et, l, R, K=1):
    """label = มีจุดตัวอย่าง (ทุก Δτ/K) ใน disc ระหว่าง [st, et] — กติกาเดียวกับ p_h ของโมเดล"""
    Yi = upsample_h(Y[i], K, 1, 0.0)
    d = np.sqrt(((Yi - l[:, None, :]) ** 2).sum(-1)); return ((d <= R) & hit_mask(st, et, Y.shape[1], K)).any(1)

def reach_score_mc(mu, sig, pi, lam, T, i, st, et, l, R_list, S, noise, seed, K=1, chunk=4000):
    """Monte-Carlo: สัดส่วน scenario ที่เข้าวงในช่วง · แบ่ง chunk กัน RAM"""
    H = mu.shape[2]; out = {R: np.zeros(len(i), np.float32) for R in R_list}; chunk = max(500, chunk // K)
    for c0 in range(0, len(i), chunk):
        sl = slice(c0, c0 + chunk); ii = i[sl]; n = len(ii); rng = np.random.RandomState(seed + c0)
        piT = softmax_T(pi[ii], T); u = rng.rand(n, S, 1); m = (u > np.cumsum(piT, 1)[:, None, :]).sum(-1)
        mu_s = upsample_h(mu[ii[:, None], m], K, 2, 0.0); sg_s = upsample_h(sig[ii[:, None], m] * lam[None, None, :], K, 2, SIGMA0_M); HK = H * K
        if noise == "common": z = np.repeat(rng.randn(n, S, 1, 2), HK, 2)
        elif noise == "indep": z = rng.randn(n, S, HK, 2)
        else: z = np.cumsum(rng.randn(n, S, HK, 2), 2) / np.sqrt(np.arange(1, HK + 1))[None, None, :, None]
        d = np.sqrt(((mu_s + sg_s[..., None] * z - l[sl][:, None, None, :]) ** 2).sum(-1)); hm = hit_mask(st[sl], et[sl], H, K)[:, None, :]
        for R in R_list: out[R][sl] = ((d <= R) & hm).any(2).mean(1)
    return out

def _analytic_core(mu_c, sg_c, piT, st, et, l, R_list, H, K):
    """mu_c (n,M,HK,2) · sg_c (n,M,HK) · piT (n,M) → lb, ub ต่อ R"""
    from scipy.stats import ncx2
    a2 = ((mu_c - l[:, None, None, :]) ** 2).sum(-1) / sg_c ** 2; hm = hit_mask(st, et, H, K)
    lb, ub = {}, {}
    for R in R_list:
        p = (piT[:, :, None] * ncx2.cdf((R / sg_c) ** 2, 2, a2)).sum(1)              # (n,HK)
        p = np.where(hm, p, 0.0)
        lb[R] = p.max(1); ub[R] = 1 - np.exp(np.log1p(-np.clip(p, 0, 1 - 1e-9)).sum(1))
    return lb, ub

def reach_score_analytic(mu, sig, pi, lam, T, i, st, et, l, R_list, K=1, chunk=10000):
    """p_h = Σ_m π_m P(‖N(μ_mh, σ_mh²I) − l‖ ≤ R) (noncentral χ², 2 dof) → lb = max_h p_h · ub = 1 − Π_h (1 − p_h)"""
    H = mu.shape[2]; lb = {R: np.zeros(len(i), np.float32) for R in R_list}; ub = {R: np.zeros(len(i), np.float32) for R in R_list}
    for c0 in range(0, len(i), chunk):
        sl = slice(c0, c0 + chunk); ii = i[sl]
        mu_c = upsample_h(mu[ii], K, 2, 0.0); sg_c = upsample_h(sig[ii] * lam[None, None, :], K, 2, SIGMA0_M)
        lb_c, ub_c = _analytic_core(mu_c, sg_c, softmax_T(pi[ii], T), st[sl], et[sl], l[sl], R_list, H, K)
        for R in R_list: lb[R][sl] = lb_c[R]; ub[R][sl] = ub_c[R]
    return lb, ub

def null_path_gain(kind, H, tau):
    return np.zeros(H) if kind == "gauss_pers" else skip_gain(H, tau)

def fit_null_sigma(Y, v, tau, kind):
    """v7: σ_h ของ null = RMS residual ต่อแกน รายขอบฟ้า รอบเส้นฐาน (persistence: μ=0 · damped: μ=v·g)"""
    H = Y.shape[1]; g = null_path_gain(kind, H, tau)
    res = Y - v[:, None, :] * g[None, :, None]
    return np.sqrt((res ** 2).sum(-1).mean(0) / 2).astype(np.float32)

def reach_score_null(v, tau, H, sigma_h, i, st, et, l, R_list, kind, K=1, chunk=20000):
    """v7: Gaussian เดี่ยว (M=1) รอบเส้นฐาน กับ σ_h จาก val → ผ่าน p_h/lb/ub เดียวกับโมเดล = "distance-decay kernel" ที่มีหลักฟิสิกส์"""
    g = null_path_gain(kind, H, tau); lb = {R: np.zeros(len(i), np.float32) for R in R_list}; ub = {R: np.zeros(len(i), np.float32) for R in R_list}
    sg_up = upsample_h(sigma_h.astype(np.float32), K, 0, SIGMA0_M); g_up = upsample_h(g.astype(np.float32), K, 0, 0.0)
    for c0 in range(0, len(i), chunk):
        sl = slice(c0, c0 + chunk); ii = i[sl]; n = len(ii)
        mu_c = (v[ii][:, None, :] * g_up[None, :, None])[:, None]                    # (n,1,HK,2)
        sg_c = np.broadcast_to(sg_up[None, None, :], (n, 1, H * K)); piT = np.ones((n, 1), np.float32)
        lb_c, ub_c = _analytic_core(mu_c, sg_c, piT, st[sl], et[sl], l[sl], R_list, H, K)
        for R in R_list: lb[R][sl] = lb_c[R]; ub[R][sl] = ub_c[R]
    return lb, ub

def reach_score_baseline(v, tau, H, i, st, et, l, R_list, kind, K=1):
    hs = np.arange(1, H + 1); g = np.zeros(H) if kind == "persistence" else (hs if kind == "const_vel" else skip_gain(H, tau))
    g_up = upsample_h(g.astype(np.float32), K, 0, 0.0)
    d = np.sqrt(((v[i][:, None, :] * g_up[None, :, None] - l[:, None, :]) ** 2).sum(-1)); hm = hit_mask(st, et, H, K)
    return {R: ((d <= R) & hm).any(1).astype(float) for R in R_list}

def fit_qm_calibration(sd, data_root, city, rd, a, lam, T, tau, log):
    """fit isotonic ของ QM (lb, ub และ null) บน val → qm_calib.json ต่อ seed (ไม่แตะ test) · v7: + σ_h ของ null, รวม block ท้าย ≥ n_min_tail"""
    K = a.sub_steps; p = sd / ("qm_calib.json" if K == 1 else f"qm_calib_k{K}.json")     # sensitivity run ไม่ทับไฟล์หลักที่ sai E ใช้
    Yv, vv, _, _, _, _, _, _ = load_split(data_root, city, rd, "val", a, log)
    Pv = load_npz(sd / "pred_val.npz")
    rng = np.random.RandomState(CRN_SEED + 7); i, st, et, l, _ = sample_pairs(Yv, a.n_val_pairs, max(a.r_list), rng, 0)
    scores = {}
    scores["lb"], scores["ub"] = reach_score_analytic(Pv["mu"], Pv["sigma"], Pv["pi"], lam, T, i, st, et, l, a.r_list, K)
    null_sigma = {kind: fit_null_sigma(Yv, vv, tau, kind) for kind in NULLS}
    for kind in NULLS: scores[kind] = reach_score_null(vv, tau, Yv.shape[1], null_sigma[kind], i, st, et, l, a.r_list, kind, K)[1]
    calib = {"version": 7, "n_pairs": a.n_val_pairs, "sub_steps": K, "n_min_tail": a.n_min_tail, "qm_main": a.qm,
             "null_sigma_h": {k: [float(x) for x in s] for k, s in null_sigma.items()}, "R": {}}
    for R in a.r_list:
        lab = reach_labels(Yv, i, st, et, l, R, K).astype(float); calib["R"][str(int(R))] = {}
        for name, sc_ in scores.items():
            bx, by, bn, nm = pav_fit(sc_[R], lab, a.n_min_tail)
            calib["R"][str(int(R))][name] = {"x": [float(x) for x in bx], "y": [float(y) for y in by], "n": [int(x) for x in bn], "tail_merged": int(nm),
                                            "y_max": float(by[-1]), "pos_rate_val": float(lab.mean())}
    json.dump(calib, open(p, "w"))
    return calib, null_sigma

def apply_qm_calib(calib, R, name, score):
    c = calib["R"][str(int(R))][name]; return pav_apply(np.array(c["x"]), np.array(c["y"]), score)

# ----------------------------------------------------------------------------- main per run
def evaluate_run(city, run_name, data_root, models_root, a, log):
    rd = models_root / city / run_name; K = a.sub_steps
    tag = f"D7_{a.split}" + ("_clean" if a.clean_only else "") + (f"_k{K}" if K > 1 else ""); out = rd / tag; out.mkdir(parents=True, exist_ok=True)
    seeds = sorted(p for p in rd.glob("seed*") if (p / "pred_test.npz").exists())
    if not seeds: log(f"ข้าม {rd}: ไม่พบ pred_test.npz"); return None
    Y_all, v_all, meta, mask, clusters_all, tau, cfg, note = load_split(data_root, city, rd, "test", a, log)
    W_min = int((cfg.get("usable_cycle_rule") or {}).get("min_workers_ready", cfg.get("W", 60)))
    Y, v, clusters = Y_all[mask], v_all[mask], clusters_all[mask]; idx_keep = np.flatnonzero(mask); N, H = Y.shape[:2]
    cyc = meta["is_cycle_start"].values[mask].astype(bool) if meta is not None and "is_cycle_start" in meta else np.zeros(N, bool)
    n_cycles = 0
    if meta is not None and cyc.any() and all(c in meta for c in ("date", "hour", "cab")):
        mm = meta.iloc[idx_keep][cyc]; per = mm.groupby(["date", "hour"])["cab"].nunique(); n_cycles = int((per >= W_min).sum())
    log(f"=== D7 v7 {city}/{run_name} · split={a.split}{' clean' if a.clean_only else ''}{f' · sub-steps {K}' if K > 1 else ''} ({'; '.join(note) or 'all'}) · {len(seeds)} seeds · หน้าต่าง {N:,}/{len(mask):,} · cycles ≥{W_min}: {n_cycles} · clusters {len(np.unique(clusters)):,} ===")
    if N == 0: log("  ไม่มีหน้าต่างใน split นี้"); return None
    cb_win = ClusterBoot(clusters, a.n_boot, seed=1)
    rng_pairs = np.random.RandomState(CRN_SEED); i, st, et, l, near = sample_pairs(Y, a.n_pairs, max(a.r_list), rng_pairs, a.n_hard)
    main = ~near; pair_clusters = clusters[i]
    cb_pairs = ClusterBoot(pair_clusters[main], a.n_boot, seed=2); cb_hard = ClusterBoot(pair_clusters[near], a.n_boot, seed=3) if near.any() else None
    subsets = {"hard": near, "st<=15": main & (st <= 15), "st16-30": main & (st > 15) & (st <= 30), "st31-45": main & (st > 30) & (st <= 45), "st>45": main & (st > 45)}
    labels = {R: reach_labels(Y, i, st, et, l, R, K) for R in a.r_list}
    rows = []; per_seed_err = {}; ratio_all = []; calib_ok_all = []; qm_scores_all = {R: {} for R in a.r_list}; rel_rows = []; rel_hard_rows = []; seed_auc = {}
    qm_name = {"lb": "qm_lb", "ub": "qm_ub", "mc": "qm_mc"}[a.qm]; qm_cal = qm_name + "_cal" if qm_name != "qm_mc" else qm_name
    for sd in seeds:
        P = load_npz(sd / "pred_test.npz"); mu, sig, pi = P["mu"][mask], P["sigma"][mask], P["pi"][mask]
        if (sd / "calib.json").exists():
            c = json.load(open(sd / "calib.json")); lam = np.asarray(c["lambda_h"], np.float32); T = float(c["T"])
        else:
            lam = np.ones(H, np.float32); T = 1.0; log(f"  {sd.name}: ไม่มี calib.json → λ=1, T=1")
        piT = softmax_T(pi, T); sigL = sig * lam[None, None, :]
        add = lambda variant, metric, val, h=None, R=None: rows.append({"city": city, "run": run_name, "split": a.split, "clean": a.clean_only, "sub_steps": K, "seed": sd.name, "variant": variant, "metric": metric, "h": h, "R": R, "value": float(val)})
        # ---- 1) error รายขอบฟ้า
        errs = horizon_errors(Y, mu, piT, v, tau)
        for var, d in errs.items():
            for h in range(H):
                add(var, "mean", d[:, h].mean(), h + 1); add(var, "median", np.median(d[:, h]), h + 1); add(var, "p90", pct(d[:, h], 90), h + 1); add(var, "p95", pct(d[:, h], 95), h + 1)
            add(var, "ADE", d.mean()); add(var, "FDE", d[:, -1].mean())
            if cyc.any(): add(var, "ADE_cycle", d[cyc].mean()); add(var, "FDE_cycle", d[cyc, -1].mean())
            per_seed_err.setdefault(var, []).append(d)
        k = piT.argmax(1); fin = mu[np.arange(N), k, -1]
        add("model_pi", "grid_acc_500m", (np.floor(fin / 500) == np.floor(Y[:, -1] / 500)).all(-1).mean())
        # ---- 2) probabilistic
        for tg, (s_, p_) in (("raw", (sig, pi)), ("cal", (sigL, piT))):
            pm = prob_metrics(Y, mu, s_, p_)
            for h in range(H):
                add(f"model_{tg}", "ratio", pm["ratio"][h], h + 1); add(f"model_{tg}", "nll", pm["nll_h"][h], h + 1)
                add(f"model_{tg}", "cov1s", pm["cov1"][h], h + 1); add(f"model_{tg}", "cov2s", pm["cov2"][h], h + 1); add(f"model_{tg}", "cov90", pm["cov90"][h], h + 1)
            add(f"model_{tg}", "NLL", pm["nll_h"].mean()); add(f"model_{tg}", "ECE_pi", pm["ece"]); add(f"model_{tg}", "pi_acc", pm["pi_acc"])
            if tg == "cal": ratio_all.append(pm["ratio"]); calib_ok_all.append(bool(((pm["ratio"] >= 0.8) & (pm["ratio"] <= 1.25)).all()))
        # ---- 3) reachability scores
        qc, null_sigma = fit_qm_calibration(sd, data_root, city, rd, a, lam, T, tau, log)        # fit บน val ก่อน (ไม่แตะ test)
        lb, ub = reach_score_analytic(mu, sig, pi, lam, T, i, st, et, l, a.r_list, K)
        scores = {"qm_lb": lb, "qm_ub": ub,
                  "scenario": reach_score_mc(mu, sig, pi, lam, T, i, st, et, l, a.r_list, a.s, a.noise, CRN_SEED + 1, K),
                  "scenario_raw": reach_score_mc(mu, sig, pi, np.ones(H, np.float32), 1.0, i, st, et, l, a.r_list, a.s, a.noise, CRN_SEED + 1, K)}
        for kind in ("persistence", "const_vel", "damped_vel"): scores[kind] = reach_score_baseline(v, tau, H, i, st, et, l, a.r_list, kind, K)
        for kind in NULLS:                                                                        # v7: null Gaussian + isotonic เดียวกัน
            scores[kind] = reach_score_null(v, tau, H, null_sigma[kind], i, st, et, l, a.r_list, kind, K)[1]
            scores[kind + "_cal"] = {R: apply_qm_calib(qc, R, kind, scores[kind][R]) for R in a.r_list}
        n_mc = min(a.n_mc, int(main.sum())); mc_idx = np.flatnonzero(main)[:n_mc]
        mc = reach_score_mc(mu, sig, pi, lam, T, i[mc_idx], st[mc_idx], et[mc_idx], l[mc_idx], a.r_list, a.s_eval, a.noise, CRN_SEED + 3, K)
        qm_full = {R: np.full(len(i), np.nan, np.float32) for R in a.r_list}
        for R in a.r_list: qm_full[R][mc_idx] = mc[R]
        scores["qm_mc"] = qm_full
        scores["qm_lb_cal"] = {R: apply_qm_calib(qc, R, "lb", lb[R]) for R in a.r_list}; scores["qm_ub_cal"] = {R: apply_qm_calib(qc, R, "ub", ub[R]) for R in a.r_list}
        for R in a.r_list:
            lab = labels[R]; add("label", "pos_rate", lab[main].mean(), R=R); add("label", "n_pos", lab[main].sum(), R=R)
            if near.any(): add("label", "pos_rate_hard", lab[near].mean(), R=R); add("label", "n_pos_hard", lab[near].sum(), R=R)
            for var, sc_ in scores.items():
                s_main = sc_[R][main]; ok = ~np.isnan(s_main)
                add(var, "AUC", auc_rank(s_main[ok], lab[main][ok]), R=R); add(var, "AUPRC", auprc(s_main[ok], lab[main][ok]), R=R)
                if var.startswith("qm") or var == "scenario" or var.startswith("gauss"):
                    b, bss = brier(s_main[ok], lab[main][ok].astype(float)); rr, ece = reliability(s_main[ok], lab[main][ok].astype(float))
                    add(var, "Brier", b, R=R); add(var, "BSS", bss, R=R); add(var, "ECE_qm", ece, R=R)
                    if R == a.r_main and var in (qm_name, qm_cal):
                        for r_ in rr: rel_rows.append({"seed": sd.name, "variant": var, **r_})
                if var in SUBSET_VARS and R == a.r_main:
                    for nm, msk in subsets.items():
                        sm = sc_[R][msk]; ok2 = ~np.isnan(sm)
                        if ok2.sum() == 0: continue
                        add(var, f"AUC_{nm}", auc_rank(sm[ok2], lab[msk][ok2]), R=R)
                        if nm == "hard":
                            add(var, "AUPRC_hard", auprc(sm[ok2], lab[msk][ok2]), R=R)
                            if var == qm_cal:
                                rr_h, ece_h = reliability(sm[ok2], lab[msk][ok2].astype(float)); add(var, "ECE_qm_hard", ece_h, R=R)
                                for r_ in rr_h: rel_hard_rows.append({"seed": sd.name, "variant": var, **r_})
                qm_scores_all[R].setdefault(var, []).append(sc_[R])
        mj = sd / "metrics.json"
        if mj.exists(): add("model_pi", "latency_ms_per_1k", json.load(open(mj)).get("infer_ms_per_1k_windows", float("nan")))
        labm = labels[a.r_main][main]; sc_qc = scores[qm_cal][a.r_main][main]; okq = ~np.isnan(sc_qc)
        seed_auc[sd.name] = auc_rank(sc_qc[okq], labm[okq])
        log(f"  {sd.name}: ADE π {errs['model_pi'].mean():.0f} m (pers {errs['persistence'].mean():.0f}) · ratio(cal) {ratio_all[-1].min():.2f}–{ratio_all[-1].max():.2f} · "
            f"AUC@{a.r_main:g} lb {auc_rank(lb[a.r_main][main], labm):.3f} / ub {auc_rank(ub[a.r_main][main], labm):.3f} / mc{a.s_eval} {auc_rank(mc[a.r_main], labels[a.r_main][mc_idx]):.3f} / S{a.s} {auc_rank(scores['scenario'][a.r_main][main], labm):.3f} · "
            f"null gauss_dv {auc_rank(scores['gauss_dv'][a.r_main][main], labm):.3f} · hard {qm_cal} {auc_rank(scores[qm_cal][a.r_main][near], labels[a.r_main][near]):.3f}")

    # ---- cluster-bootstrap CI: (a) paired difference ของ error รายขอบฟ้า (เฉลี่ยข้าม seed ต่อหน้าต่าง) · (c) AUC/AUPRC ของ QM (เฉลี่ยคะแนนข้าม seed)
    Rm = a.r_main; hs = [x for x in H_REPORT if x <= H]
    row_all = lambda var, met, val, h=None, R=None: rows.append({"city": city, "run": run_name, "split": a.split, "clean": a.clean_only, "sub_steps": K, "seed": "all", "variant": var, "metric": met, "h": h, "R": R, "value": float(val)})
    e_model = np.mean(per_seed_err["model_pi"], 0); diff_ci = {}
    for base in ("persistence", "const_vel"):
        e_base = np.mean(per_seed_err[base], 0)
        for h in hs:
            dd = e_base[:, h - 1] - e_model[:, h - 1]; lo, hi = cb_win.ci(lambda idx: dd[idx].mean()); diff_ci[f"{base}@{h}"] = (float(dd.mean()), lo, hi)
            row_all(f"diff_{base}", "ci_lo", lo, h=h); row_all(f"diff_{base}", "ci_hi", hi, h=h)
    auc_ci = {}; lab_m = labels[Rm][main]; lab_h = labels[Rm][near]
    for var in (qm_name, qm_cal, "qm_lb", "qm_ub", "scenario", "persistence", "gauss_pers_cal", "gauss_dv_cal"):
        if var not in qm_scores_all[Rm]: continue
        s_all = np.nanmean(qm_scores_all[Rm][var], 0); s_mean = s_all[main]; ok = ~np.isnan(s_mean)
        if ok.sum() < len(ok): continue   # qm_mc มีเฉพาะชุดย่อย
        auc_ci[var] = {"AUC": cb_pairs.ci(lambda idx: auc_rank(s_mean[idx], lab_m[idx])), "AUPRC": cb_pairs.ci(lambda idx: auprc(s_mean[idx], lab_m[idx]))}
        if cb_hard is not None:
            s_h = s_all[near]; auc_ci[var]["AUC_hard"] = cb_hard.ci(lambda idx: auc_rank(s_h[idx], lab_h[idx]))
        for met, (lo, hi) in auc_ci[var].items(): row_all(var, f"{met}_ci_lo", lo, R=Rm); row_all(var, f"{met}_ci_hi", hi, R=Rm)
    df = pd.DataFrame(rows); df.to_csv(out / "pred_metrics.csv", index=False)
    ledger = models_root / "D7_results.csv"; df.to_csv(ledger, mode="a", header=not ledger.exists(), index=False)
    pd.DataFrame(rel_rows).to_csv(out / "qm_reliability.csv", index=False); pd.DataFrame(rel_hard_rows).to_csv(out / "qm_reliability_hard.csv", index=False)

    # ---- G1
    g = lambda var, met, R=None, h=None: df[(df.variant == var) & (df.metric == met) & ((df.R == R) if R is not None else df.R.isna()) & ((df.h == h) if h is not None else df.h.isna())].value
    pv = {}
    for base in ("persistence", "const_vel"):
        for h in hs:
            d_seeds = [b[:, h - 1].mean() - m[:, h - 1].mean() for m, b in zip(per_seed_err["model_pi"], per_seed_err[base])]
            pv[f"{base}@{h}"] = ttest_1samp(d_seeds) if np.mean(d_seeds) > 0 else 1.0
    padj = holm(pv)
    g1a_stat = all(padj[k] < 0.05 for k in padj); g1a_ci = all(diff_ci[k][1] > 0 for k in diff_ci); g1a = g1a_stat and g1a_ci
    ratio_mean = np.mean(ratio_all, 0); g1b = all(calib_ok_all)
    auc_qm = float(g(qm_name, "AUC", Rm).mean()); aup_qm = float(g(qm_name, "AUPRC", Rm).mean()); aup_pers = float(g("persistence", "AUPRC", Rm).mean())
    lift = aup_qm / aup_pers if aup_pers > 0 else float("nan"); ci_auc = auc_ci.get(qm_name, {}).get("AUC", (float("nan"), float("nan"))); ci_aup = auc_ci.get(qm_name, {}).get("AUPRC", (float("nan"), float("nan")))
    g1c = bool(auc_qm > 0.8 and lift >= 2.0)
    ece_raw = float(g(qm_name, "ECE_qm", Rm).mean()); ece_cal = float(g(qm_cal, "ECE_qm", Rm).mean())
    bss_cal = float(g(qm_cal, "BSS", Rm).mean())
    g1d = bool(ece_cal <= 0.01 and bss_cal > 0)
    n_pos = float(g("label", "n_pos", Rm).mean()); pos_rate = float(g("label", "pos_rate", Rm).mean())
    # v7: ข้อมูลประกอบ (ไม่ใช่เกณฑ์) — null และ hard set
    mean_or_nan = lambda q: float(q.mean()) if len(q) else float("nan")
    info = {"null_AUC_gauss_pers_cal": mean_or_nan(g("gauss_pers_cal", "AUC", Rm)), "null_AUC_gauss_dv_cal": mean_or_nan(g("gauss_dv_cal", "AUC", Rm)),
            "null_AUPRC_gauss_dv_cal": mean_or_nan(g("gauss_dv_cal", "AUPRC", Rm)), "null_AUC_ci_gauss_dv_cal": list(auc_ci.get("gauss_dv_cal", {}).get("AUC", (float("nan"), float("nan")))),
            "hard_pos_rate": mean_or_nan(g("label", "pos_rate_hard", Rm)), "hard_n_pos": mean_or_nan(g("label", "n_pos_hard", Rm)),
            f"hard_AUC_{qm_cal}": mean_or_nan(g(qm_cal, "AUC_hard", Rm)), f"hard_AUC_ci_{qm_cal}": list(auc_ci.get(qm_cal, {}).get("AUC_hard", (float("nan"), float("nan")))),
            "hard_AUC_gauss_dv_cal": mean_or_nan(g("gauss_dv_cal", "AUC_hard", Rm)), f"hard_AUPRC_{qm_cal}": mean_or_nan(g(qm_cal, "AUPRC_hard", Rm)), "hard_AUPRC_gauss_dv_cal": mean_or_nan(g("gauss_dv_cal", "AUPRC_hard", Rm)),
            f"hard_ECE_{qm_cal}": mean_or_nan(g(qm_cal, "ECE_qm_hard", Rm))}
    info["lift_vs_null_gauss_dv"] = aup_qm / info["null_AUPRC_gauss_dv_cal"] if info["null_AUPRC_gauss_dv_cal"] > 0 else float("nan")
    info["AUC_gain_vs_null_gauss_dv"] = float(g(qm_cal, "AUC", Rm).mean()) - info["null_AUC_gauss_dv_cal"]
    G1 = {"version": 7, "city": city, "run": run_name, "split": a.split, "clean_only": a.clean_only, "sub_steps": K, "n_seeds": len(seeds), "n_windows": int(N), "n_cycles_ge_W": n_cycles, "W_min": W_min,
          "a_pass": g1a, "a_holm_p": padj, "a_diff_ci": {k: list(v) for k, v in diff_ci.items()},
          "b_pass": g1b, "b_ratio_range": [float(ratio_mean.min()), float(ratio_mean.max())],
          "c_pass": g1c, "c_qm": qm_name, f"c_AUC@{Rm:g}": auc_qm, "c_AUC_cluster_ci": list(ci_auc), f"c_AUPRC@{Rm:g}": aup_qm, "c_AUPRC_cluster_ci": list(ci_aup), "c_AUPRC_persistence": aup_pers, "c_lift": lift, "c_n_pos": n_pos, "c_pos_rate": pos_rate,
          "d_pass": g1d, "d_ECE_qm_raw": ece_raw, "d_ECE_qm_cal": ece_cal, "d_BSS_cal": bss_cal,
          "PASS": bool(g1a and g1b and g1c and g1d), "info_v7": info}
    json.dump(G1, open(out / "G1.json", "w"), indent=2)

    # ---- v7: E1 handoff — seed ที่ AUC(QM_cal) ใกล้ median ที่สุด + hash
    aucs = {k: v_ for k, v_ in seed_auc.items() if not np.isnan(v_)}
    if aucs:
        med = float(np.median(list(aucs.values()))); pick = min(aucs, key=lambda k_: (abs(aucs[k_] - med), k_))
        sdp = rd / pick; qc_name = "qm_calib.json" if K == 1 else f"qm_calib_k{K}.json"
        files = {f: str((sdp / f).relative_to(models_root)) for f in ("pred_test.npz", "pred_val.npz", "calib.json", qc_name) if (sdp / f).exists()}
        hand = {"version": 7, "city": city, "run": run_name, "split": a.split, "clean_only": a.clean_only, "sub_steps": K, "qm": a.qm, "qm_variant_for_E1": qm_cal, "R_main": Rm,
                "seed": pick, f"AUC_{qm_cal}@{Rm:g}_per_seed": {k: float(v_) for k, v_ in aucs.items()}, "median_AUC": med,
                "files": files, "sha256": {f: sha256_of(sdp / f) for f in files}, "y_max_isotonic": json.load(open(sdp / qc_name))["R"][str(int(Rm))][a.qm if a.qm != "mc" else "ub"]["y_max"] if (sdp / qc_name).exists() else None,
                "note": "sai E ใช้ seed นี้ตัวเดียว: QM_ij = isotonic(qm_%s) จาก qm_calib.json กับ λ_h, T จาก calib.json · ตรวจ sha256 ก่อนใช้" % a.qm}
        json.dump(hand, open(out / "E1_handoff.json", "w"), indent=2, ensure_ascii=False)
    else: pick = None

    # ---- report
    md = lambda d: "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n" + "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in d.values)
    def tbl(metric, variants, cols, byR=False, fmt="{:.3f}"):
        rows_ = []
        for var in variants:
            r = {"variant": var}
            for c in cols:
                q = g(var, metric, R=c) if byR else g(var, metric, h=c)
                r[f"{c:g}"] = (fmt.format(q.mean()) + f" ± {q.std(ddof=0):.0f}" if fmt == "{:.0f}" else fmt.format(q.mean()) + f" ± {q.std(ddof=0):.3f}") if len(q) else "—"
            rows_.append(r)
        return pd.DataFrame(rows_)
    fm = lambda q, f="{:.3f}": f.format(q.mean()) if len(q) else "—"
    P1 = tbl("mean", ["model_pi", "model_min", "persistence", "const_vel", "damped_vel"], hs, fmt="{:.0f}")
    P1c = pd.DataFrame([{"baseline − model_pi (m)": k, "mean": f"{v_[0]:.0f}", "cluster 95% CI": f"[{v_[1]:.0f}, {v_[2]:.0f}]", "Holm p": f"{padj[k]:.2g}"} for k, v_ in diff_ci.items()])
    P4 = pd.DataFrame([{"h": h, "ratio_raw": fm(g('model_raw','ratio',h=h)), "ratio_cal": fm(g('model_cal','ratio',h=h)), "cov90_raw": fm(g('model_raw','cov90',h=h)), "cov90_cal": fm(g('model_cal','cov90',h=h)), "nll_raw": fm(g('model_raw','nll',h=h), "{:.2f}"), "nll_cal": fm(g('model_cal','nll',h=h), "{:.2f}")} for h in hs])
    vars3 = ["qm_lb", "qm_ub", "qm_ub_cal", "qm_mc", "scenario", "scenario_raw", "gauss_pers", "gauss_pers_cal", "gauss_dv", "gauss_dv_cal", "persistence", "const_vel", "damped_vel"]
    P3 = tbl("AUC", vars3, a.r_list, byR=True); P3b = tbl("AUPRC", vars3, a.r_list, byR=True)
    P3c = pd.DataFrame([{"variant": var, **{nm: fm(g(var, f'AUC_{nm}', R=Rm)) for nm in subsets}, "AUPRC_hard": fm(g(var, "AUPRC_hard", R=Rm))} for var in SUBSET_VARS if len(g(var, "AUC_hard", R=Rm))])
    Pq = pd.DataFrame([{"QM": var, "Brier": fm(g(var,'Brier',R=Rm), "{:.4f}"), "BSS": fm(g(var,'BSS',R=Rm)), "ECE": fm(g(var,'ECE_qm',R=Rm), "{:.4f}"), "AUC": fm(g(var,'AUC',R=Rm)), "AUPRC": fm(g(var,'AUPRC',R=Rm))}
                       for var in ("qm_lb", "qm_lb_cal", "qm_ub", "qm_ub_cal", "qm_mc", "scenario", "gauss_pers", "gauss_pers_cal", "gauss_dv", "gauss_dv_cal") if len(g(var, "Brier", R=Rm))])
    rel = pd.DataFrame(rel_rows); relh = pd.DataFrame(rel_hard_rows)
    agg = lambda d: d.groupby(["lo", "hi"]).agg(n=("n", "sum"), qm_mean=("qm_mean", "mean"), freq=("freq", "mean")).reset_index().round(4)
    Prel = agg(rel[rel.variant == qm_cal]) if len(rel) else pd.DataFrame(); Prelh = agg(relh) if len(relh) else pd.DataFrame()
    if len(Prel): Prel["flag"] = np.where(Prel["n"] / max(len(seeds), 1) < a.n_min_tail, f"n<{a.n_min_tail}/seed", "")
    ci_txt = lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]" if not np.isnan(c[0]) else "—"
    hand_txt = (f"seed สำหรับ E1 = **{pick}** (AUC {qm_cal} {aucs[pick]:.3f}, median {np.median(list(aucs.values())):.3f}; ต่อ seed: " + ", ".join(f"{k_} {v_:.3f}" for k_, v_ in aucs.items()) + ") → E1_handoff.json") if pick else "—"
    text = [f"# D7 (v7) — {city} / {run_name} · split = **{a.split}**{' · clean-only' if a.clean_only else ''}{f' · sub-steps {K} (sensitivity: QM = วิ่งผ่าน)' if K > 1 else ''}  ({'; '.join(note) or 'ชุด test ทั้งก้อน'})",
            f"{len(seeds)} seeds · หน้าต่าง {N:,} · cycles ที่มีคันพร้อม ≥ {W_min}: **{n_cycles}** · clusters (cab×วัน) {len(np.unique(clusters)):,} · reachability {a.n_pairs:,} คู่หลัก (positive rate @R{Rm:g} {pos_rate:.4f}, บวก {n_pos:.0f}) + {a.n_hard:,} คู่ hard (positive rate {info['hard_pos_rate']:.3f}, บวก {info['hard_n_pos']:.0f}) · CI = cluster bootstrap {a.n_boot} รอบ",
            f"นิยาม QM: มีจุดตัวอย่าง (ทุก {60 // K if 60 % K == 0 else 60 / K:g} s) ใน disc ระหว่าง [st, et] — label และโมเดลใช้กติกาเดียวกัน · isotonic รวม block ท้าย ≥ {a.n_min_tail} ตัวอย่าง", "",
            "## Table P-1 · mean error (m) รายขอบฟ้า (mean ± std ข้าม seed)", md(P1), "", "ผลต่างเทียบเส้นฐาน (บวก = โมเดลดีกว่า)", md(P1c), "",
            "## Table P-4 · calibration ของ σ̂ (ก่อน → หลัง λ_h, T)", md(P4), "",
            f"## Table P-3 · Reachability @R ∈ {{{', '.join(f'{r:g}' for r in a.r_list)}}} — ชุดหลัก (จุดสุ่มทั่วบริเวณที่ไปถึงได้)",
            f"qm_lb = max_h p_h · qm_ub = 1−Π(1−p_h) · _cal = หลัง isotonic (fit บน val) · qm_mc = Monte-Carlo S={a.s_eval} บน {min(a.n_mc, int(main.sum())):,} คู่แรก · scenario = S={a.s} ({a.noise}) · "
            f"**gauss_pers / gauss_dv = null Gaussian (M=1) รอบ persistence / damped-velocity, σ_h จาก val, ผ่านสูตรและ isotonic เดียวกัน** · persistence/const_vel/damped_vel = 0/1 · QM ที่ใช้ตัดสิน = **{qm_name}**",
            "AUC", md(P3), "", "AUPRC", md(P3b), "",
            f"## Table P-3h · ชุด hard @R={Rm:g} (งานห่างเส้นทางจริง ≤ 2R — สิ่งที่ optimizer ต้องแยกจริง) และแยกตามเวลาเริ่มช่วง", md(P3c), "",
            f"{qm_cal}: AUC hard {info[f'hard_AUC_{qm_cal}']:.3f} cluster-CI {ci_txt(tuple(info[f'hard_AUC_ci_{qm_cal}']))} · null gauss_dv_cal {info['hard_AUC_gauss_dv_cal']:.3f} · AUPRC hard {info[f'hard_AUPRC_{qm_cal}']:.3f} vs null {info['hard_AUPRC_gauss_dv_cal']:.3f} (base rate {info['hard_pos_rate']:.3f}) · ECE hard {info[f'hard_ECE_{qm_cal}']:.4f}",
            f"reliability ของ {qm_cal} บนชุด hard", md(Prelh) if len(Prelh) else "—", "",
            f"## Table P-5 · calibration ของ QM @R={Rm:g} (isotonic fit บน val {a.n_val_pairs:,} คู่ → apply บน test)", md(Pq), "",
            f"reliability ของ {qm_cal} (bins ของค่า QM → ความถี่จริง; flag = bin ที่ตัวอย่างน้อย)", md(Prel) if len(Prel) else "—", "",
            f"grid acc 500 m: {fm(g('model_pi','grid_acc_500m'))} · latency {fm(g('model_pi','latency_ms_per_1k'), '{:.1f}')} ms / 1k หน้าต่าง", "",
            "## Gate G1 (เกณฑ์ประกาศล่วงหน้า — ไม่เปลี่ยนใน v7)",
            f"- (a) ชนะ persistence & const_vel ทุก h ≥ 5 — Holm p<0.05 {'✔' if g1a_stat else '✘'} · cluster-CI ของผลต่าง > 0 {'✔' if g1a_ci else '✘'} → **{'ผ่าน' if g1a else 'ไม่ผ่าน'}**",
            f"- (b) ratio ∈ [0.8, 1.25] ทุก h → **{'ผ่าน' if g1b else 'ไม่ผ่าน'}** (ช่วง {ratio_mean.min():.2f}–{ratio_mean.max():.2f})",
            f"- (c) AUC({qm_name}) @R={Rm:g} > 0.8: {auc_qm:.3f} cluster-CI {ci_txt(ci_auc)} · AUPRC {aup_qm:.3f} {ci_txt(ci_aup)} vs persistence {aup_pers:.3f} → lift {lift:.1f}× (ต้อง ≥ 2) → **{'ผ่าน' if g1c else 'ไม่ผ่าน'}**",
            f"- (d) QM calibrated: ECE {ece_raw:.4f} → {ece_cal:.4f} หลัง isotonic (ต้อง ≤ 0.01) · BSS {bss_cal:.3f} (ต้อง > 0) → **{'ผ่าน' if g1d else 'ไม่ผ่าน'}**", "",
            f"**G1: {'PASS ✔' if G1['PASS'] else 'FAIL ✘'}**", "",
            "## ข้อมูลประกอบ v7 (ไม่ใช่เกณฑ์ตัดสิน — สำหรับเขียนบทความอย่างซื่อสัตย์)",
            f"- null gauss_dv_cal: AUC {info['null_AUC_gauss_dv_cal']:.3f} {ci_txt(tuple(info['null_AUC_ci_gauss_dv_cal']))} · AUPRC {info['null_AUPRC_gauss_dv_cal']:.3f} → โมเดลเพิ่ม AUC **{info['AUC_gain_vs_null_gauss_dv']:+.3f}** และ lift เทียบ null **{info['lift_vs_null_gauss_dv']:.2f}×** (เทียบ lift vs persistence 0/1 {lift:.1f}×)",
            f"- null gauss_pers_cal: AUC {info['null_AUC_gauss_pers_cal']:.3f} (= distance-decay kernel ล้วน ไม่ใช้ทิศทาง)",
            f"- ชุด hard: {qm_cal} AUC {info[f'hard_AUC_{qm_cal}']:.3f} vs null {info['hard_AUC_gauss_dv_cal']:.3f} — ถ้าทั้งคู่ ≈ 0.5 ให้ตีความเป็นเพดานของปัญหา (resolution limit) ไม่ใช่โมเดลล้มเหลว",
            f"- {hand_txt}"]
    (out / "D7_report.md").write_text("\n".join(text), encoding="utf-8")
    make_figs(df, out, city, run_name, H, a, Prel, qm_name)
    log(f"  G1 {city}/{run_name} [{a.split}{f' k{K}' if K > 1 else ''}]: (a){'✔' if g1a else '✘'} (b){'✔' if g1b else '✘'} (c){'✔' if g1c else '✘'} AUC {auc_qm:.3f} {ci_txt(ci_auc)} lift {lift:.1f}× (d){'✔' if g1d else '✘'} ECE {ece_cal:.4f} → {'PASS' if G1['PASS'] else 'FAIL'}"
        f" · null gauss_dv {info['null_AUC_gauss_dv_cal']:.3f} (gain {info['AUC_gain_vs_null_gauss_dv']:+.3f}) · hard {info[f'hard_AUC_{qm_cal}']:.3f} vs null {info['hard_AUC_gauss_dv_cal']:.3f} · E1 seed {pick}   → {out / 'D7_report.md'}")
    return G1

def make_figs(df, out, city, run_name, H, a, Prel, qm_name):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        print("  (ไม่มี matplotlib — ข้ามรูป)"); return
    hs = np.arange(1, H + 1)
    def series(var, metric):
        gg = df[(df.variant == var) & (df.metric == metric)].groupby("h").value; return gg.mean().reindex(hs).values, gg.std(ddof=0).reindex(hs).values
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for var, lab, ls in (("model_pi", "OWMP (mode π max)", "-"), ("model_min", "OWMP (best of M)", "--"), ("persistence", "persistence", ":"), ("const_vel", "const-velocity", "-."), ("damped_vel", "damped-velocity", ":")):
        m, s = series(var, "mean"); ax.plot(hs, m, ls, label=lab); ax.fill_between(hs, m - s, m + s, alpha=.15)
    ax.axhline(a.r_main, color="k", lw=.8, alpha=.5); ax.set_xlabel("horizon h (min)"); ax.set_ylabel("mean error (m)")
    ax.set_ylim(0, 1.3 * np.nanmax(series("persistence", "mean")[0])); ax.legend(fontsize=8); ax.set_title(f"P1 · {city}/{run_name} · {a.split}")
    fig.tight_layout(); fig.savefig(out / "fig_P1_error_vs_h.png", dpi=150); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for tg, lab in (("model_raw", "before"), ("model_cal", "after λ_h, T")):
        r, _ = series(tg, "ratio"); axs[0].plot(hs, r, label=lab); c, _ = series(tg, "cov90"); axs[1].plot(hs, c, label=lab)
    axs[0].axhspan(0.8, 1.25, color="g", alpha=.1); axs[0].axhline(1, color="k", lw=.8); axs[0].set_ylabel("calibration ratio"); axs[0].set_xlabel("h (min)"); axs[0].legend()
    axs[1].axhline(0.9, color="k", lw=.8); axs[1].set_ylabel("coverage of 90% circle"); axs[1].set_xlabel("h (min)"); axs[1].set_ylim(0, 1); axs[1].legend()
    fig.suptitle(f"P2 · σ̂ calibration · {city}/{run_name} · {a.split}"); fig.tight_layout(); fig.savefig(out / "fig_P2_calibration.png", dpi=150); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for met, ax in zip(("AUC", "AUPRC"), axs):
        for var, lab in (("qm_lb", "QM lower bound"), ("qm_ub", "QM upper bound"), ("qm_mc", f"QM MC S={a.s_eval}"), ("scenario", f"scenario S={a.s}"), ("gauss_dv_cal", "null: Gaussian damped-vel"), ("gauss_pers_cal", "null: Gaussian persistence"), ("persistence", "persistence 0/1")):
            gg = df[(df.variant == var) & (df.metric == met)].groupby("R").value
            if len(gg): ax.errorbar(gg.mean().index, gg.mean().values, yerr=gg.std(ddof=0).values, marker="o", capsize=3, label=lab)
        ax.set_xlabel("R (m)"); ax.set_ylabel(met); ax.legend(fontsize=7)
    axs[0].axhline(0.8, color="k", lw=.8, alpha=.6); axs[0].set_ylim(0.4, 1)
    fig.suptitle(f"P3 · reachability · {city}/{run_name} · {a.split}"); fig.tight_layout(); fig.savefig(out / "fig_P3_reachability.png", dpi=150); plt.close(fig)
    if len(Prel):
        fig, ax = plt.subplots(figsize=(4.8, 4.4))
        ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.plot(Prel["qm_mean"], Prel["freq"], "o-", label=f"{qm_name} (calibrated)")
        for _, r in Prel.iterrows(): ax.annotate(f"n={int(r['n'])}", (r["qm_mean"], r["freq"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("symlog", linthresh=0.01); ax.set_yscale("symlog", linthresh=0.01); ax.set_xlabel("QM (predicted probability)"); ax.set_ylabel("observed frequency"); ax.legend(fontsize=8)
        ax.set_title(f"P4 · QM reliability @R={a.r_main:g} · {a.split}"); fig.tight_layout(); fig.savefig(out / "fig_P4_qm_reliability.png", dpi=150); plt.close(fig)

def main():
    P = PYCHARM
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", default=P["CITY"]); ap.add_argument("--runs", default=P["RUNS"]); ap.add_argument("--data", default=P["DATA"]); ap.add_argument("--models", default=P["MODELS"])
    ap.add_argument("--split", choices=["dev", "final", "all"], default=P["SPLIT"]); ap.add_argument("--clean-only", action="store_true", default=P["CLEAN_ONLY"])
    ap.add_argument("--r-list", default=",".join(map(str, P["R_LIST"]))); ap.add_argument("--r-main", type=float, default=P["R_MAIN"])
    ap.add_argument("--n-pairs", type=int, default=P["N_PAIRS"]); ap.add_argument("--n-hard", type=int, default=P["N_HARD"]); ap.add_argument("--n-val-pairs", type=int, default=P["N_VAL_PAIRS"])
    ap.add_argument("--s", type=int, default=P["S"]); ap.add_argument("--s-eval", type=int, default=P["S_EVAL"]); ap.add_argument("--n-mc", type=int, default=P["N_MC"])
    ap.add_argument("--noise", choices=["brownian", "common", "indep"], default=P["NOISE"]); ap.add_argument("--qm", choices=["lb", "ub", "mc"], default=P["QM"])
    ap.add_argument("--sub-steps", type=int, default=P["SUB_STEPS"], help="1 = ความหมายเดิม · K>1 = sensitivity (แทรกจุดทั้ง label และโมเดล)")
    ap.add_argument("--n-min-tail", type=int, default=P["N_MIN_TAIL"], help="block ท้ายของ isotonic ต้องมีตัวอย่าง ≥ เท่านี้ (0 = ปิด)")
    ap.add_argument("--n-boot", type=int, default=P["N_BOOT"]); ap.add_argument("--demo", action="store_true", default=P["DEMO"])
    a = ap.parse_args(); a.r_list = [float(x) for x in a.r_list.split(",")]
    if a.sub_steps < 1: sys.exit("--sub-steps ต้อง ≥ 1")
    data_root = Path(a.data) if Path(a.data).is_absolute() else HERE / a.data; models_root = Path(a.models) if Path(a.models).is_absolute() else HERE / a.models
    t0 = time.time(); log = lambda *m: print(f"[{time.time()-t0:6.1f}s]", *m, flush=True)
    if a.demo:
        runs = [p.name for p in (models_root / "demo").glob("*_demo") if p.is_dir()]
        if not runs: sys.exit("ยังไม่มี models/demo — รัน sD5 --demo แล้ว sD6 --demo ก่อน")
        a.split = "all"
        for r in runs: evaluate_run("demo", r, data_root, models_root, a, log)
        return
    for city in [c.strip() for c in a.city.split(",") if c.strip()]:
        for r in [x.strip() for x in a.runs.split(",") if x.strip()]:
            evaluate_run(city, r, data_root, models_root, a, log)

if __name__ == "__main__":
    main()
