#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sE6_rho.py (E6 v2 ส่วน A · v0.1, 2026-10-23) — ประมาณ ρ(h) และ V_j(t) จากชุด **dev** (SPEC_MOPSO_IQ_v2 §1 A1/A2, B0)
  ρ(h)   = Σ y_ij / Σ C_ij บนคู่ที่ X*(t) ของ NSGA-II เลือก (แก้สดที่ทุก t บนงานเปิด, โควตาเต็ม) · bin h = หน้าต่างที่เหลือของงาน (et_i − t): ≤10, 11–20, 21–40, >40 นาที
           + proxy ไม่ใช้ optimizer: top-1 ต่องาน · ρ > 1 → 1 (ไม่ขยาย)
  V(t)   = E_j[ k-th largest_i C̃_ij(t+Δ) ] บนงานเปิดที่ t+Δ, k = 1..3 (มูลค่าของการเก็บช่องโควตา 1 ช่องไว้ 1 ขั้น) · C̃ = ρ·C
  V∞(t)  = E_j[ max_{t'>t} max_i C̃_ij(t') ] (เก็บไว้จนจบรอบ)
  inf_frac = 0.30 × min(1, (AUC_dev(t=0) − 0.5)/0.35)  (A2; ค่าสัมประสิทธิ์ประกาศในสเปก)
ใช้:  python sE6_rho.py --city sf --inst instances_devsplit --e5 e5inst_devsplit --out e6_rho [--gen 480] [--limit 5]
ผล:   <out>/<city>/rho.json · RHO_report.md
ห้าม: ใช้กับ instances_v3 (ชุดทดสอบ)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import sE4_lite as E4, sE5_replan as E5

VERSION = "E6-rho v0.2 (2026-10-23: --rho-mode full|clip — ค่าหลัก full: ρ ไม่ตัดที่ 1, C̃ = min(1, ρ·C) · แก้ไขสเปก A1 หลังเห็น dev ว่าหน้าต่างสั้นถูกประมาณต่ำ ×1.7–2.4)"
BINS = [10, 20, 40]                         # h ≤ 10 · 11–20 · 21–40 · > 40
BIN_LABEL = ["≤10", "11–20", "21–40", ">40"]


def hbin(h):
    return np.searchsorted(np.array(BINS), np.asarray(h), side="left")      # 0..3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", default="sf"); ap.add_argument("--inst", default="instances_devsplit"); ap.add_argument("--e5", default="e5inst_devsplit"); ap.add_argument("--out", default="e6_rho")
    ap.add_argument("--arm", default="qm_model"); ap.add_argument("--t", default="0,10,20,30,40,50"); ap.add_argument("--pop", type=int, default=80); ap.add_argument("--gen", type=int, default=480)
    ap.add_argument("--seed", type=int, default=12345); ap.add_argument("--q-max", type=int, default=3); ap.add_argument("--theta-T", type=float, default=0.05); ap.add_argument("--theta-W", type=float, default=0.05); ap.add_argument("--n-cand", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--allow-test", action="store_true", help="ปลดล็อกการใช้ instances_v3 (ห้ามใช้ยกเว้นทดสอบโค้ด)"); ap.add_argument("--rho-mode", choices=["full", "clip"], default="full", help="full: ρ ตามที่วัด (C̃ ตัดที่ 1) · clip: ρ ≤ 1 (สเปกเดิม A1)")
    a = ap.parse_args(); t0 = time.time(); log = lambda *m: print(f"[{time.time()-t0:7.1f}s]", *m, flush=True); log(VERSION)
    if "instances_v3" in a.inst and not a.allow_test: sys.exit("ห้ามประมาณ ρ/V จาก instances_v3 (ชุดทดสอบ) — ใช้ instances_devsplit")
    city = a.city; inst_dir = Path(a.inst) / city; e5_dir = Path(a.e5) / city; out = Path(a.out) / city; out.mkdir(parents=True, exist_ok=True)
    t_list = [int(x) for x in a.t.split(",")]; dt = t_list[1] - t_list[0] if len(t_list) > 1 else 10
    files = sorted(inst_dir.glob("inst_*.npz")); files = files[:a.limit] if a.limit else files; ks = [int(f.stem.split("_")[1]) for f in files]
    log(f"{city}: {len(ks)} dev instance · arm {a.arm} · t {t_list} · NSGA-II pop {a.pop} × gen {a.gen} ต่อจุด (แก้สด โควตาเต็ม)")
    sumC = np.zeros((2, 4)); sumY = np.zeros((2, 4)); nP = np.zeros((2, 4), int)            # [X*, top1] × bin
    per_t = {tm: dict(C=0.0, Y=0.0, n=0) for tm in t_list}
    auc0 = []
    Cs_t = {tm: [] for tm in t_list}                                                        # สำหรับ V: (W, open) C ต่อ instance ต่อ t
    for k in ks:
        z = np.load(inst_dir / f"inst_{k:03d}.npz", allow_pickle=True)
        K = z["task_K"].astype(int); L = np.ones_like(K); r = z["task_r"].astype(float); eps = z["task_eps"].astype(float); e0 = z["w_e"].astype(float); W = len(e0)
        Z = {tm: np.load(e5_dir / f"inst_{k:03d}_t{tm:02d}.npz") for tm in t_list}
        for tm in t_list:
            open_ = Z[tm]["open"].astype(bool); idx = np.flatnonzero(open_)
            if len(idx) == 0: continue
            C = np.nan_to_num(Z[tm][a.arm].astype(float))[:, idx]; Y = Z[tm]["oracle"].astype(float)[:, idx]; h = Z[tm]["et_rel_all"].astype(int)[idx]
            Cs_t[tm].append(C)
            if tm == 0: auc0.append(_auc(C.ravel(), Y.ravel()))
            # --- X* ของ NSGA-II (แก้สด)
            prob = E5.ProblemT(C, K[idx], L[idx], r[idx], eps[idx], np.full(W, a.q_max), e0, np.zeros(W), False, a.theta_T, a.theta_W, a.n_cand)
            G, F = E5.run_nsga2_warm(prob, a.pop, a.gen, a.seed + k, None); x = prob.decode(G); i_sel = int(np.argmin(np.linalg.norm(1.0 - F, axis=1))); xs = x[i_sel]      # (n_open, Smax)
            ti, si = np.nonzero(xs >= 0); wj = xs[ti, si]
            b = hbin(h[ti]); np.add.at(sumC[0], b, C[wj, ti]); np.add.at(sumY[0], b, Y[wj, ti]); np.add.at(nP[0], b, 1)
            per_t[tm]["C"] += C[wj, ti].sum(); per_t[tm]["Y"] += Y[wj, ti].sum(); per_t[tm]["n"] += len(ti)
            # --- proxy top-1 ต่องาน
            j1 = C.argmax(0); b1 = hbin(h); np.add.at(sumC[1], b1, C[j1, np.arange(len(idx))]); np.add.at(sumY[1], b1, Y[j1, np.arange(len(idx))]); np.add.at(nP[1], b1, 1)
        log(f"  inst {k:03d} done · X* pairs so far {int(nP[0].sum())}")
    rho_x = np.where(sumC[0] > 0, sumY[0] / np.maximum(sumC[0], 1e-9), 1.0); rho_1 = np.where(sumC[1] > 0, sumY[1] / np.maximum(sumC[1], 1e-9), 1.0)
    rho_x_c = np.minimum(rho_x, 1.0); rho_1_c = np.minimum(rho_1, 1.0)
    rho_used = rho_x if a.rho_mode == "full" else rho_x_c
    auc_dev = float(np.mean(auc0)) if auc0 else float("nan"); inf_frac = float(0.30 * min(1.0, max(0.0, (auc_dev - 0.5) / 0.35)))
    # --- V(t): ใช้ C̃ = ρ_x(h)·C ของขั้นถัดไป · ต้องการ h ที่ t+Δ → อ่านจาก Z อีกครั้ง (เก็บเฉพาะ C ไว้; h คำนวณใหม่)
    V = {}; Vfut = {}
    for tm in t_list[:-1]:
        vals = {1: [], 2: [], 3: []}; vf = []
        for k in ks:
            Zn = {t2: np.load(e5_dir / f"inst_{k:03d}_t{t2:02d}.npz") for t2 in t_list if t2 > tm}
            mats = []
            for t2, Zt in Zn.items():
                op = Zt["open"].astype(bool); idx = np.flatnonzero(op)
                if len(idx) == 0: continue
                C2 = np.nan_to_num(Zt[a.arm].astype(float))[:, idx]; h2 = Zt["et_rel_all"].astype(int)[idx]; Ct = np.minimum(1.0, C2 * rho_used[hbin(h2)][None, :]); mats.append((t2, Ct))
            if not mats: continue
            t_next, Cn = mats[0]
            if t_next != tm + dt: continue
            srt = -np.sort(-Cn, axis=1)                                                     # (W, open) มาก→น้อย
            for kk in (1, 2, 3): vals[kk].append(srt[:, kk - 1].mean() if srt.shape[1] >= kk else 0.0)
            vf.append(np.max(np.stack([Ct.max(1) for _, Ct in mats], 1), 1).mean())
        V[tm] = [float(np.mean(vals[kk])) if vals[kk] else 0.0 for kk in (1, 2, 3)]; Vfut[tm] = float(np.mean(vf)) if vf else 0.0
    V[t_list[-1]] = [0.0, 0.0, 0.0]; Vfut[t_list[-1]] = 0.0
    res = dict(version=VERSION, city=city, arm=a.arm, n_inst=len(ks), bins=BINS, bin_label=BIN_LABEL, rho_mode=a.rho_mode, rho_used=rho_used.tolist(), rho_xstar=rho_x.tolist(), rho_xstar_clipped=rho_x_c.tolist(), rho_top1=rho_1.tolist(), rho_top1_clipped=rho_1_c.tolist(),
               n_pairs_xstar=nP[0].tolist(), n_pairs_top1=nP[1].tolist(), per_t={str(tm): dict(ratio=(d["Y"] / d["C"] if d["C"] > 0 else None), n=d["n"]) for tm, d in per_t.items()},
               auc_dev_t0=auc_dev, inf_frac=inf_frac, V={str(tm): v for tm, v in V.items()}, Vfut={str(tm): v for tm, v in Vfut.items()}, dt=dt, pop=a.pop, gen=a.gen)
    json.dump(res, open(out / "rho.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    md = lambda d: "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n" + "\n".join("| " + " | ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r) + " |" for r in d.values)
    T1 = pd.DataFrame(dict(h_bin=BIN_LABEL, n_xstar=nP[0], rho_xstar=rho_x, rho_used=rho_used, n_top1=nP[1], rho_top1=rho_1))
    T2 = pd.DataFrame([dict(t=tm, ratio_real_over_plan=(d["Y"] / d["C"] if d["C"] > 0 else float("nan")), n_pairs=d["n"]) for tm, d in per_t.items()])
    T3 = pd.DataFrame([dict(t=tm, V1=V[tm][0], V2=V[tm][1], V3=V[tm][2], V_future=Vfut[tm]) for tm in t_list])
    lines = [f"# ρ(h) และ V(t) จาก dev — {city} · {VERSION} · {time.strftime('%Y-%m-%d %H:%M')}", "", f"dev instances {len(ks)} ({inst_dir}) · arm {a.arm} · NSGA-II pop {a.pop} × gen {a.gen} แก้สดทุก t · AUC_dev(t=0) {auc_dev:.3f} → inf_frac (A2) = {inf_frac:.3f}", "",
             "## ρ(h) = Σy/ΣC บนคู่ที่เลือก ตาม bin หน้าต่างที่เหลือ (rho_used ตาม --rho-mode " + a.rho_mode + "; C̃ = min(1, ρ·C))", "", md(T1.round(4)), "",
             "## อัตราส่วน realised/planned ของ X* ตาม t (เทียบกับ 1/2.2 ≈ .45 ที่พบในชุดทดสอบ SF ที่ t=50)", "", md(T2.round(4)), "",
             "## V(t): มูลค่าคาดหวังของช่องโควตาที่เก็บไว้ (k-th largest C̃ ขั้นถัดไป · และดีสุดจนจบรอบ)", "", md(T3.round(4)), "",
             "## ใช้ต่อ", f"- sE5_replan --shrink {out / 'rho.json'} → C̃_ij(t) = ρ(h_i)·C_ij(t) (ทุก arm ยกเว้น oracle) · --inf-frac auto อ่าน inf_frac จากไฟล์เดียวกัน", "- B0: gate C̃_ij ≥ κ·V_k(t) (k = ช่องโควตาที่เหลือของ j) · κ ∈ {0.5, 1.0, 1.5}", ""]
    (out / "RHO_report.md").write_text("\n".join(lines), encoding="utf-8"); log(f"→ {out / 'RHO_report.md'}")


def _auc(s, y):
    from scipy.stats import rankdata
    y = np.asarray(y) > 0.5; n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    rk = rankdata(s); return float((rk[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


if __name__ == "__main__":
    main()
