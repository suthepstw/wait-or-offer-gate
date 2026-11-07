#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sD6_calibrate.py — D6: ปรับ calibration ของ σ̂ (λ_h รายขอบฟ้า) และ π (temperature T) บนชุด **val**  → ไม่แตะ test
อ่าน  models/<city>/<run>/seed<k>/pred_val.npz  +  data/<city>/processed/val_Y.npy
เขียน models/<city>/<run>/seed<k>/calib.json          {lambda_h[H], T, ตัวเลขก่อน/หลัง}
      models/<city>/<run>/D6_calibration.csv           รายขอบฟ้า ต่อ seed (ratio/coverage ก่อน–หลัง)  → Fig P2
      models/<city>/<run>/D6_summary.json              mean ± std ข้าม seed

นิยาม (ต่อขอบฟ้า h)
  s_emp²(h)  = E_i Σ_m π_im ‖y_ih − μ_imh‖² / 2      ความแปรปรวนจริงต่อมิติ ถ่วงด้วย π (นับความผิดจากการเลือก mode ด้วย — ตรงกับที่ scenario sampling จะเจอ)
  s_pred²(h) = E_i Σ_m π_im σ̂_imh²                      ความแปรปรวนที่โมเดลอ้าง
  ratio(h)   = s_emp / s_pred        → λ_h = ratio(h) บน val (moment matching) แล้วทำให้เรียบด้วย moving average
  σ̂' = λ_h · σ̂                       → หลังปรับ ratio ≈ 1 · coverage@kσ ≈ 1 − e^{−k²/2} (2 มิติ: 1σ→39.3%, 2σ→86.5%)
  T   = argmin_T CE( softmax(log π / T), one-hot(mode ที่ ADE ต่ำสุด) )   → π' ∝ π^{1/T}

ไม่ต้องใช้ GPU · numpy/pandas เท่านั้น · รันซ้ำได้ (เขียนทับ)
PyCharm: แก้บล็อก PYCHARM แล้ว Run · CLI: python sD6_calibrate.py --city sf --runs base_M3      · ทดสอบ: python sD6_calibrate.py --demo
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

PYCHARM = dict(
    CITY="sf",                  # "sf" | "rome" | "beijing"  (หรือ "sf,rome,beijing")
    RUNS="base_M3",             # run ที่จะ calibrate คั่นด้วย ,  เช่น "base_M3,ft_sf"
    DATA="data", MODELS="models",
    SMOOTH=5,                   # moving-average window ของ λ_h (ก้าว) · 1 = ไม่ทำให้เรียบ
    METHOD="moment",            # "moment" = λ_h = s_emp/s_pred (ตาม Master Plan; ratio→1 แน่นอน) · "nll" = λ_h ที่ทำให้ mixture NLL ต่ำสุด
    DEMO=False,
)
COV_K = [1.0, 2.0, 2.146]       # 1σ, 2σ, วง 90% (√(2 ln 10))
H_REPORT = [5, 10, 20, 30, 45, 60]

# ----------------------------------------------------------------------------- helpers
def load_pred(p: Path):
    z = np.load(p, allow_pickle=True)
    return z["mu"].astype(np.float32), z["sigma"].astype(np.float32), z["pi"].astype(np.float32)

def softmax_T(pi: np.ndarray, T: float) -> np.ndarray:
    lg = np.log(np.clip(pi, 1e-12, None)) / T; lg -= lg.max(1, keepdims=True)
    e = np.exp(lg); return e / e.sum(1, keepdims=True)

def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1: return x
    pad = w // 2; xp = np.pad(x, (pad, w - 1 - pad), mode="edge")
    return np.convolve(xp, np.ones(w) / w, mode="valid")

def calib_stats(Y, mu, sig, pi) -> dict:
    """ต่อขอบฟ้า: s_emp, s_pred, ratio, coverage@k (π-weighted), NLL mixture ต่อก้าว"""
    d2 = ((Y[:, None] - mu) ** 2).sum(-1)                      # (N,M,H)
    s_emp = np.sqrt((pi[:, :, None] * d2).sum(1).mean(0) / 2)   # (H,)
    s_pred = np.sqrt((pi[:, :, None] * sig ** 2).sum(1).mean(0))
    d = np.sqrt(d2)
    cov = {k: (pi[:, :, None] * (d <= k * sig)).sum(1).mean(0) for k in COV_K}
    logp = -(d2 / (2 * sig ** 2) + np.log(2 * np.pi * sig ** 2))          # (N,M,H) log N ต่อก้าว
    z = np.log(np.clip(pi, 1e-12, None))[:, :, None] + logp
    zmax = z.max(1, keepdims=True); nll_h = -(zmax[:, 0] + np.log(np.exp(z - zmax).sum(1))).mean(0)   # (H,)
    return {"s_emp": s_emp, "s_pred": s_pred, "ratio": s_emp / s_pred, "cov": cov, "nll_h": nll_h}

def fit_lambda_nll(Y, mu, sig, pi) -> np.ndarray:
    """λ_h ต่อขอบฟ้าที่ทำให้ mixture NLL ของก้าว h ต่ำสุด (grid search)"""
    d2 = ((Y[:, None] - mu) ** 2).sum(-1); logpi = np.log(np.clip(pi, 1e-12, None))[:, :, None]
    grid = np.exp(np.linspace(np.log(0.3), np.log(5.0), 50)); out = np.ones(Y.shape[1])
    for h in range(Y.shape[1]):
        best = (np.inf, 1.0)
        for lam in grid:
            s2 = (sig[:, :, h] * lam) ** 2
            z = logpi[:, :, 0] - d2[:, :, h] / (2 * s2) - np.log(2 * np.pi * s2); zm = z.max(1, keepdims=True)
            nll = float(-(zm[:, 0] + np.log(np.exp(z - zm).sum(1))).mean())
            if nll < best[0]: best = (nll, lam)
        out[h] = best[1]
    return out

def fit_temperature(Y, mu, pi) -> tuple:
    ade = np.sqrt(((Y[:, None] - mu) ** 2).sum(-1)).mean(-1)      # (N,M)
    win = ade.argmin(1); M = pi.shape[1]
    def ce(T):
        p = softmax_T(pi, T); return float(-np.log(np.clip(p[np.arange(len(p)), win], 1e-12, None)).mean())
    Ts = np.exp(np.linspace(np.log(0.2), np.log(5.0), 61))
    ces = [ce(T) for T in Ts]; T = float(Ts[int(np.argmin(ces))])
    acc = float((pi.argmax(1) == win).mean())
    usage = np.bincount(win, minlength=M) / len(win)               # สัดส่วนที่แต่ละ mode ชนะ (mode-collapse check)
    return T, ce(1.0), min(ces), acc, usage

def reliability(pi, win, bins=10):
    """reliability ของ π_max: bin ตามความมั่นใจ → accuracy จริง"""
    conf = pi.max(1); hit = (pi.argmax(1) == win)
    edges = np.linspace(1 / pi.shape[1], 1, bins + 1)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf >= a) & (conf < b) if b < 1 else (conf >= a)
        if m.sum(): rows.append({"conf_lo": a, "conf_hi": b, "n": int(m.sum()), "conf_mean": float(conf[m].mean()), "acc": float(hit[m].mean())})
    r = pd.DataFrame(rows); ece = float((r["n"] / r["n"].sum() * (r["conf_mean"] - r["acc"]).abs()).sum()) if len(r) else float("nan")
    return r, ece

# ----------------------------------------------------------------------------- main per seed
def calibrate_seed(seed_dir: Path, Yv: np.ndarray, smooth: int, log, method: str = "moment") -> dict:
    mu, sig, pi = load_pred(seed_dir / "pred_val.npz"); H = Yv.shape[1]
    before = calib_stats(Yv, mu, sig, pi)
    lam_raw = before["ratio"] if method == "moment" else fit_lambda_nll(Yv, mu, sig, pi)
    lam = moving_average(lam_raw, smooth)
    T, ce0, ce1, acc, usage = fit_temperature(Yv, mu, pi)
    pi_T = softmax_T(pi, T)
    after = calib_stats(Yv, mu, sig * lam[None, None, :], pi_T)
    rel0, ece0 = reliability(pi, np.sqrt(((Yv[:, None] - mu) ** 2).sum(-1)).mean(-1).argmin(1))
    rel1, ece1 = reliability(pi_T, np.sqrt(((Yv[:, None] - mu) ** 2).sum(-1)).mean(-1).argmin(1))
    calib = {"H": H, "method": method, "lambda_h": [float(x) for x in lam], "lambda_raw": [float(x) for x in lam_raw], "smooth": smooth, "T": T,
             "val": {"ratio_before": [float(x) for x in before["ratio"]], "ratio_after": [float(x) for x in after["ratio"]],
                     "nll_before": float(before["nll_h"].mean()), "nll_after": float(after["nll_h"].mean()),
                     "cov90_before": [float(x) for x in before["cov"][2.146]], "cov90_after": [float(x) for x in after["cov"][2.146]],
                     "pi_ce_before": ce0, "pi_ce_after": ce1, "pi_acc": acc, "pi_ece_before": ece0, "pi_ece_after": ece1,
                     "mode_usage": [float(x) for x in usage]},
             "reliability_before": rel0.to_dict("records"), "reliability_after": rel1.to_dict("records")}
    json.dump(calib, open(seed_dir / "calib.json", "w"), indent=1)
    rows = []
    for h in range(H):
        rows.append({"seed": seed_dir.name, "h": h + 1, "lambda": lam[h], "ratio_before": before["ratio"][h], "ratio_after": after["ratio"][h],
                     "cov1s_before": before["cov"][1.0][h], "cov1s_after": after["cov"][1.0][h], "cov2s_before": before["cov"][2.0][h], "cov2s_after": after["cov"][2.0][h],
                     "cov90_before": before["cov"][2.146][h], "cov90_after": after["cov"][2.146][h], "nll_before": before["nll_h"][h], "nll_after": after["nll_h"][h]})
    log(f"  {seed_dir.name}: λ_h {lam.min():.2f}–{lam.max():.2f} (h=10: {lam[min(9,H-1)]:.2f}, h=60: {lam[-1]:.2f}) · T={T:.2f} · "
        f"ratio ก่อน {before['ratio'].min():.2f}–{before['ratio'].max():.2f} → หลัง {after['ratio'].min():.2f}–{after['ratio'].max():.2f} · "
        f"cov90 ก่อน {before['cov'][2.146].mean():.3f} → {after['cov'][2.146].mean():.3f} · NLL {before['nll_h'].mean():.3f} → {after['nll_h'].mean():.3f} · "
        f"π acc {acc:.3f} ECE {ece0:.3f}→{ece1:.3f} · mode usage {np.round(usage, 2)}")
    return {"rows": rows, "T": T, "lam": lam, "nll_before": float(before["nll_h"].mean()), "nll_after": float(after["nll_h"].mean()),
            "cov90_before": float(before["cov"][2.146].mean()), "cov90_after": float(after["cov"][2.146].mean()), "ece_before": ece0, "ece_after": ece1, "usage": usage}

def run(city: str, runs: list, data_root: Path, models_root: Path, smooth: int, method: str = "moment"):
    t0 = time.time(); log = lambda *m: print(f"[{time.time()-t0:6.1f}s]", *m, flush=True)
    Yv = np.load(data_root / city / "processed" / "val_Y.npy").astype(np.float32)
    for run_name in runs:
        rd = models_root / city / run_name
        seeds = sorted(p for p in rd.glob("seed*") if (p / "pred_val.npz").exists())
        if not seeds: log(f"ข้าม {rd}: ไม่พบ seed*/pred_val.npz"); continue
        log(f"=== D6 {city}/{run_name} · {len(seeds)} seeds · val {len(Yv):,} หน้าต่าง ===")
        res = [calibrate_seed(sd, Yv, smooth, log, method) for sd in seeds]
        tbl = pd.DataFrame([r for x in res for r in x["rows"]]); tbl.to_csv(rd / "D6_calibration.csv", index=False)
        agg = {k: {"mean": float(np.mean([x[k] for x in res])), "std": float(np.std([x[k] for x in res]))} for k in ("T", "nll_before", "nll_after", "cov90_before", "cov90_after", "ece_before", "ece_after")}
        lam_mean = np.mean([x["lam"] for x in res], 0)
        summ = {"city": city, "run": run_name, "n_seeds": len(res), "method": method, "lambda_h_mean": [float(v) for v in lam_mean],
                "lambda_at": {str(h): float(lam_mean[h - 1]) for h in H_REPORT if h <= len(lam_mean)},
                "mode_usage_mean": [float(v) for v in np.mean([x["usage"] for x in res], 0)], **agg}
        json.dump(summ, open(rd / "D6_summary.json", "w"), indent=2)
        log(f"สรุป {city}/{run_name}: λ_h เฉลี่ย " + " ".join(f"h{h}={summ['lambda_at'][str(h)]:.2f}" for h in H_REPORT if str(h) in summ["lambda_at"]) +
            f" · T {agg['T']['mean']:.2f}±{agg['T']['std']:.2f} · cov90 {agg['cov90_before']['mean']:.3f}→{agg['cov90_after']['mean']:.3f} · NLL {agg['nll_before']['mean']:.3f}→{agg['nll_after']['mean']:.3f}")
        log(f"  → {rd / 'D6_calibration.csv'} · seed*/calib.json")

def main():
    P = PYCHARM
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", default=P["CITY"]); ap.add_argument("--runs", default=P["RUNS"])
    ap.add_argument("--data", default=P["DATA"]); ap.add_argument("--models", default=P["MODELS"])
    ap.add_argument("--smooth", type=int, default=P["SMOOTH"]); ap.add_argument("--method", choices=["moment", "nll"], default=P["METHOD"])
    ap.add_argument("--demo", action="store_true", default=P["DEMO"])
    a = ap.parse_args()
    data_root = Path(a.data) if Path(a.data).is_absolute() else HERE / a.data
    models_root = Path(a.models) if Path(a.models).is_absolute() else HERE / a.models
    if a.demo:
        runs = [p.name for p in (models_root / "demo").glob("*_demo") if p.is_dir()]
        if not runs: sys.exit("ยังไม่มี models/demo — รัน python sD5_train.py --demo --epochs 20 ก่อน")
        run("demo", runs, data_root, models_root, a.smooth, a.method); return
    for city in [c.strip() for c in a.city.split(",") if c.strip()]:
        run(city, [r.strip() for r in a.runs.split(",") if r.strip()], data_root, models_root, a.smooth, a.method)

if __name__ == "__main__":
    main()
