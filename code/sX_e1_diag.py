#!/usr/bin/env python3
"""
sX_e1_diag.py — วินิจฉัยว่าทำไม (9a′) แพ้ (9a) บนคู่ของ E1 ทั้งที่ชนะบน D7
อ่าน instances_v3/<city>/pilot/inst_*.npz + oracle_*.npz (ไม่แตะอะไร) → AUC ของทุก arm แยกตาม
  (1) ระยะ |l| = ‖task − worker‖ (bins km)   (2) ความยาวหน้าต่าง et−st   (3) เวลาเริ่ม st
และเทียบการกระจายระยะของคู่ E1 กับคู่ของ D7 (sample_pairs v7 บน test_Y ของ split เดียวกัน ถ้าให้ --data)
ใช้: python sX_e1_diag.py --city sf [--inst instances_v3/sf/pilot] [--data data] [--models models]
"""
import argparse, glob, json, sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent

def auc_rank(s, y):
    from scipy.stats import rankdata
    y = y.astype(bool); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s); return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--city", default="sf"); ap.add_argument("--inst", default=""); ap.add_argument("--R", type=int, default=200)
    ap.add_argument("--data", default=""); ap.add_argument("--models", default="models"); ap.add_argument("--out", default="")
    a = ap.parse_args(); inst = Path(a.inst or f"instances_v3/{a.city}/pilot")
    files = sorted(glob.glob(str(inst / "inst_*.npz")))
    if not files: sys.exit(f"ไม่พบ inst_*.npz ใน {inst}")
    arms = ["qm_model", "qm_model_uncal", "qm_lstm_reach", "qm_tfm_ub", "qm_trajgeom", "qm_null_dv", "qm_persist_cont"]
    rows = []
    for f in files:
        z = np.load(f, allow_pickle=True); o = np.load(f.replace("inst_", "oracle_"))
        k = int(Path(f).stem.split("_")[1]); nW, nT = z["w_xy"].shape[0], z["task_xy"].shape[0]
        d = np.linalg.norm(z["task_xy"][None] - z["w_xy"][:, None], axis=-1).ravel() / 1000
        st = np.tile(z["task_st"], nW).astype(int); et = np.tile(z["task_et"], nW).astype(int); y = o[f"qm_oracle_R{a.R}"].ravel()
        rec = dict(k=k, d_km=d, dur=et - st, st=st, y=y, **{m: z[f"{m}_R{a.R}"].ravel() for m in arms if f"{m}_R{a.R}" in z})
        rows.append(rec)
    cat = lambda key: np.concatenate([r[key] for r in rows])
    d, dur, st, y = cat("d_km"), cat("dur"), cat("st"), cat("y"); S = {m: cat(m) for m in arms if m in rows[0]}
    lines = [f"# E1 diag — {a.city} · {len(files)} instance · {len(y):,} คู่ · pos rate {y.mean():.4f} · R = {a.R}", ""]
    lines += [f"ระยะ |l| ของคู่ E1 (km): p10 {np.percentile(d,10):.2f} · p25 {np.percentile(d,25):.2f} · p50 {np.percentile(d,50):.2f} · p75 {np.percentile(d,75):.2f} · p90 {np.percentile(d,90):.2f} · max {d.max():.1f}", ""]
    def table(name, bins, val):
        hdr = ["bin", "n", "pos rate"] + list(S); out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (val >= lo) & (val < hi)
            if m.sum() < 50: continue
            out.append("| " + " | ".join([f"{lo:g}–{hi:g}", f"{m.sum():,}", f"{y[m].mean():.3f}"] + [f"{auc_rank(S[s][m], y[m]):.3f}" if y[m].sum() > 0 and (~y[m].astype(bool)).sum() > 0 else "—" for s in S]) + " |")
        return [f"## AUC แยกตาม {name}", ""] + out + [""]
    lines += table("ระยะ |l| (km)", [0, 0.25, 0.5, 1, 2, 3, 5, 8, 100], d)
    lines += table("ความยาวหน้าต่าง et−st (นาที)", [0, 6, 11, 16, 21, 31], dur)
    lines += table("เวลาเริ่ม st (นาที)", [0, 11, 21, 31, 41, 61], st)
    # monotonic check: ค่าเฉลี่ยของแต่ละ arm ตามระยะ (หัวควรลดตามระยะ)
    bins = [0, 0.25, 0.5, 1, 2, 3, 5, 8, 100]; hdr = ["bin"] + list(S) + ["oracle"]
    mono = ["## ค่าเฉลี่ย C_ij ตามระยะ (ตรวจ monotone)", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum() < 50: continue
        mono.append("| " + " | ".join([f"{lo:g}–{hi:g}"] + [f"{S[s][m].mean():.4f}" for s in S] + [f"{y[m].mean():.4f}"]) + " |")
    lines += mono + [""]
    # AUC บนคู่ที่อยู่ใน support ของ D7 (|l| ≤ p75 ของ D7) ถ้ามีข้อมูล
    if a.data:
        try:
            p = HERE / "sD7_pred_eval.py"; spec = importlib.util.spec_from_file_location("v7", p); V7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(V7)
            Y = np.load(Path(a.data) / a.city / "processed" / "test_Y.npy", mmap_mode="r")
            rng = np.random.RandomState(V7.CRN_SEED); i, st7, et7, l7, _ = V7.sample_pairs(np.asarray(Y[:20000]), 20000, float(a.R) if False else 250.0, rng, 2000)
            d7 = np.linalg.norm(l7, axis=1) / 1000
            lines += [f"## ระยะ |l| ของคู่ D7 (sample_pairs v7, 20k+2k hard): p10 {np.percentile(d7,10):.2f} · p25 {np.percentile(d7,25):.2f} · p50 {np.percentile(d7,50):.2f} · p75 {np.percentile(d7,75):.2f} · p90 {np.percentile(d7,90):.2f} · max {d7.max():.1f} km", ""]
            dmax = float(np.percentile(d7, 95)); m = d <= dmax
            lines += [f"**AUC บนคู่ E1 ที่ |l| ≤ p95 ของ D7 ({dmax:.2f} km): n = {m.sum():,} ({m.mean()*100:.0f} % ของคู่), pos rate {y[m].mean():.3f}**", ""]
            lines += ["| arm | AUC ทุกคู่ | AUC ใน support D7 | AUC นอก support |", "|---|---|---|---|"]
            for s in S: lines.append(f"| {s} | {auc_rank(S[s], y):.3f} | {auc_rank(S[s][m], y[m]):.3f} | {auc_rank(S[s][~m], y[~m]) if y[~m].sum() > 0 else float('nan'):.3f} |")
            lines.append("")
        except Exception as e: lines += [f"(D7 sample_pairs ไม่สำเร็จ: {e})", ""]
    txt = "\n".join(lines); out = Path(a.out or (inst / "E1_diag.md")); out.write_text(txt, encoding="utf-8"); print(txt); print(f"→ {out}")

if __name__ == "__main__": main()
