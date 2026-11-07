#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sE6_figs.py (E6 v0.1, 2026-10-22) — ตาราง/รูปเทียบอัลกอริทึมจากผล sE5_replan v0.6+ (EVAL_PLAN_E6.md §2)
อ่าน:  <root>/<algo>/<city>/{e5_runs.csv, e5_timeline.csv, e5_trace.csv, e5_fronts.npz}   (root เช่น e6_sticky · เพิ่ม --root2 e6_free ได้)
ผล:   <out>/E6_report.md · tables (csv) · fig_A1_f1_box.png · fig_A2_forest.png · fig_A3_hv_conv.png · fig_A4_fronts.png · fig_A5_timeline.png · fig_A6_quota.png · fig_A7_churn.png · fig_A8_runtime.png
สถิติ: Δ paired ต่อ instance vs --ref (nsga2) · bootstrap 2,000 · Wilcoxon · Holm ข้ามอัลกอริทึม (ต่อเมือง ต่อตัวชี้วัด) · wins/losses · IGD⁺ ต่อ reference front รวมทุกอัลกอริทึม (plan, t=0)
ใช้:  python sE6_figs.py --root e6_sticky --out e6_figs            # หลัก
      python sE6_figs.py --root e6_sticky --root2 e6_free --root3 e6_static --out e6_figs
"""
from __future__ import annotations
import argparse, glob, sys, time
from pathlib import Path
import numpy as np, pandas as pd

VERSION = "E6-figs v0.1.1 (2026-10-23: รับชื่อ algo นอกรายการ เช่น nsga2_sh, mopso_iq_rk_v2)"
ORDER = ["nsga2", "mopso", "mopso_q", "mopso_iq", "mopso_rk", "mopso_q_rk", "mopso_iq_rk"]
NAME = {"nsga2": "NSGA-II", "mopso": "MOPSO", "mopso_q": "MOPSO-Q", "mopso_iq": "MOPSO-IQ", "mopso_rk": "MOPSO (rk)", "mopso_q_rk": "MOPSO-Q (rk)", "mopso_iq_rk": "MOPSO-IQ (rk)"}
COL = {"nsga2": "#4c4c4c", "mopso": "#1f77b4", "mopso_q": "#ff7f0e", "mopso_iq": "#d62728", "mopso_rk": "#9ecae1", "mopso_q_rk": "#ffbb78", "mopso_iq_rk": "#ff9896"}


def load_root(root, tag):
    runs, tl, tr, fronts = [], [], [], {}
    for algo_dir in sorted(Path(root).glob("*")):
        algo = algo_dir.name
        for city_dir in sorted(algo_dir.glob("*")):
            f = city_dir / "e5_runs.csv"
            if not f.exists(): continue
            d = pd.read_csv(f); d["algo"] = algo; d["city"] = city_dir.name; d["setting"] = tag; runs.append(d)
            t = pd.read_csv(city_dir / "e5_timeline.csv"); t["algo"] = algo; t["city"] = city_dir.name; t["setting"] = tag; tl.append(t)
            if (city_dir / "e5_trace.csv").exists():
                x = pd.read_csv(city_dir / "e5_trace.csv")
                if len(x): x["algo"] = algo; x["city"] = city_dir.name; x["setting"] = tag; tr.append(x)
            if (city_dir / "e5_fronts.npz").exists(): fronts[(tag, algo, city_dir.name)] = np.load(city_dir / "e5_fronts.npz")
    cat = lambda L: pd.concat(L, ignore_index=True) if L else pd.DataFrame()
    return cat(runs), cat(tl), cat(tr), fronts


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def paired(df, ref, col, by=("city",), n_boot=2000, seed=0):
    """Δ = algo − ref ต่อ instance k (เฉพาะแถวที่ k มีทั้งคู่)"""
    rng = np.random.RandomState(seed); out = []
    for keys, g in df.groupby(list(by)):
        piv = g.pivot_table(index="k", columns="algo", values=col)
        if ref not in piv: continue
        rows = []
        for algo in [a for a in ORDER if a in piv.columns and a != ref] + sorted(set(piv.columns) - set(ORDER) - {ref}):
            d = (piv[algo] - piv[ref]).dropna().to_numpy()
            if not len(d): continue
            bs = np.array([rng.choice(d, len(d)).mean() for _ in range(n_boot)])
            try:
                from scipy.stats import wilcoxon; p = float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0
            except Exception: p = float("nan")
            rows.append(dict(**dict(zip(by, keys if isinstance(keys, tuple) else (keys,))), metric=col, algo=algo, n=len(d), ref_mean=float(piv[ref].mean()), mean=float(piv[algo].mean()), delta=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), p=p, wins=int((d > 0).sum()), losses=int((d < 0).sum())))
        if rows:
            R = pd.DataFrame(rows); R["p_holm"] = holm(R.p.fillna(1.0)); out.append(R)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def igd_plus(F, R):
    """IGD⁺ (Ishibuchi 2015) สำหรับ max: d⁺(r, a) = ‖max(r − a, 0)‖ · เฉลี่ยบน reference R"""
    if not len(F) or not len(R): return float("nan")
    d = np.sqrt((np.maximum(R[:, None, :] - F[None, :, :], 0) ** 2).sum(2)).min(1)
    return float(d.mean())


def md(d):
    return "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n" + "\n".join("| " + " | ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r) + " |" for r in d.values)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="e6_sticky"); ap.add_argument("--root2", default=""); ap.add_argument("--root3", default=""); ap.add_argument("--out", default="e6_figs")
    ap.add_argument("--ref", default="nsga2"); ap.add_argument("--arm", default="qm_model"); ap.add_argument("--inst-fig", type=int, default=0, help="instance สำหรับ F-A4")
    a = ap.parse_args(); t0 = time.time(); log = lambda *m: print(f"[{time.time()-t0:6.1f}s]", *m, flush=True); log(VERSION)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    roots = [(a.root, Path(a.root).name.replace("e6_", ""))] + [(r, Path(r).name.replace("e6_", "")) for r in (a.root2, a.root3) if r]
    R, TL, TR, FR = [], [], [], {}
    for r, tag in roots:
        d, t, x, f = load_root(r, tag); R.append(d); TL.append(t); TR.append(x); FR.update(f); log(f"{r}: {len(d)} rows · algos {sorted(d.algo.unique()) if len(d) else []}")
    R = pd.concat(R, ignore_index=True); TL = pd.concat(TL, ignore_index=True); TR = pd.concat([x for x in TR if len(x)], ignore_index=True) if any(len(x) for x in TR) else pd.DataFrame()
    main_tag = roots[0][1]; present = set(R.algo); algos = [x for x in ORDER if x in present] + sorted(present - set(ORDER)); cities = sorted(R.city.unique())
    for x in algos: NAME.setdefault(x, x); COL.setdefault(x, "#2ca02c" if "v2" in x else ("#8c564b" if "sh" in x else "#7f7f7f"))
    M = R[(R.arm == a.arm) & (R["mode"] == "adaptive")]; S = R[(R.arm == a.arm) & (R["mode"] == "static")]
    O = R[(R.arm == "oracle") & (R["mode"] == "adaptive")]
    # ---------------------------------------------------------------- Table 4: mean ± sd
    def msd(g, c): return f"{g[c].mean():.4f} ± {g[c].std(ddof=1):.4f}" if len(g) > 1 else f"{g[c].mean():.4f}"
    t4 = []
    for tag in R.setting.unique():
        for city in cities:
            for algo in algos:
                g = M[(M.setting == tag) & (M.city == city) & (M.algo == algo)]; gs = S[(S.setting == tag) & (S.city == city) & (S.algo == algo)]; go = O[(O.setting == tag) & (O.city == city) & (O.algo == algo)]
                if not len(g): continue
                t4.append(dict(setting=tag, city=city, algo=NAME.get(algo, algo), n=len(g), f1_adaptive=msd(g, "f1_real"), f1_static=msd(gs, "f1_real") if len(gs) else "—", f2=msd(g, "f2_real"), f3=msd(g, "f3_real"),
                               hv_plan_t0=msd(g, "hv_plan_t0") if "hv_plan_t0" in g else "—", hv_real_t0=msd(g, "hv_real_t0") if "hv_real_t0" in g else "—", gap_plan_real_t0=f"{(g.f1_plan_t0 / g.f1_exp_t0.replace(0, np.nan)).median():.2f}" if "f1_plan_t0" in g else "—",
                               churn=msd(g, "n_changed_mean"), quota_used=msd(g, "quota_used_final") if "quota_used_final" in g else "—", sec_per_solve=msd(g, "sec_per_solve") if "sec_per_solve" in g else "—", f1_oracle=msd(go, "f1_real") if len(go) else "—"))
    T4 = pd.DataFrame(t4); T4.to_csv(out / "table4_summary.csv", index=False)
    # ---------------------------------------------------------------- Table 5: paired Δ vs ref
    T5 = []
    for tag in R.setting.unique():
        for col in ["f1_real", "f2_real", "f3_real", "hv_plan_t0", "hv_real_t0", "n_changed_mean"]:
            if col not in M: continue
            P = paired(M[M.setting == tag], a.ref, col)
            if len(P): P.insert(0, "setting", tag); T5.append(P)
        Po = paired(O[O.setting == tag], a.ref, "f1_real")
        if len(Po): Po.insert(0, "setting", tag); Po["metric"] = "f1_real (oracle arm)"; T5.append(Po)
    T5 = pd.concat(T5, ignore_index=True) if T5 else pd.DataFrame(); T5.to_csv(out / "table5_paired.csv", index=False)
    # ---------------------------------------------------------------- IGD⁺ (plan front t=0, adaptive) ต่อ reference รวม
    igd_rows = []
    for city in cities:
        ks = sorted(M[(M.setting == main_tag) & (M.city == city)].k.unique())
        for k in ks:
            fronts = {}
            for algo in algos:
                z = FR.get((main_tag, algo, city)); key = f"{k}_{a.arm}_adaptive_t00_plan"
                if z is not None and key in z: fronts[algo] = z[key].astype(float)
            if len(fronts) < 2: continue
            allF = np.vstack(list(fronts.values())); nd = np.ones(len(allF), bool)
            for i in range(len(allF)):
                if ((allF >= allF[i]).all(1) & (allF > allF[i]).any(1)).any(): nd[i] = False
            ref_front = allF[nd]
            for algo, F in fronts.items(): igd_rows.append(dict(city=city, k=k, algo=algo, igd_plus=igd_plus(F, ref_front), n_front=len(F), hv_plan=float(np.prod(np.max(F, 0))) if False else np.nan))
    IG = pd.DataFrame(igd_rows)
    if len(IG):
        IG.to_csv(out / "igd_plus_t0.csv", index=False)
        IGs = IG.groupby(["city", "algo"]).agg(igd_plus_mean=("igd_plus", "mean"), igd_plus_median=("igd_plus", "median"), n=("k", "count")).reset_index()
        IGp = paired(IG, a.ref, "igd_plus")
    # ---------------------------------------------------------------- figures
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e: log(f"ไม่มี matplotlib ({e}) — ข้ามรูป"); plt = None
    if plt is not None:
        Mm = M[M.setting == main_tag]
        # F-A1 box + paired lines
        fig, axes = plt.subplots(1, len(cities), figsize=(5.2 * len(cities), 4.2), squeeze=False)
        for ax, city in zip(axes[0], cities):
            piv = Mm[Mm.city == city].pivot_table(index="k", columns="algo", values="f1_real")[[x for x in algos if x in Mm.algo.unique()]]
            ax.boxplot([piv[c].dropna() for c in piv.columns], tick_labels=[NAME.get(c, c) for c in piv.columns], showmeans=True, widths=.5)
            xs = np.arange(1, len(piv.columns) + 1)
            for _, row in piv.iterrows(): ax.plot(xs, row.values, color="#888", lw=.4, alpha=.5)
            oc = O[(O.setting == main_tag) & (O.city == city)].f1_real.mean()
            if np.isfinite(oc): ax.axhline(oc, ls=":", color="k", lw=1); ax.text(0.6, oc, "oracle", va="bottom", fontsize=8)
            ax.set_title(f"{city} · realised f₁* ({main_tag}, adaptive, arm {a.arm})"); ax.tick_params(axis="x", rotation=25, labelsize=8); ax.set_ylabel("f₁*")
        fig.tight_layout(); fig.savefig(out / "fig_A1_f1_box.png", dpi=150); plt.close(fig)
        # F-A2 forest Δ vs ref × setting × city
        if len(T5):
            D = T5[T5.metric == "f1_real"]; sets = list(D.setting.unique())
            fig, axes = plt.subplots(1, len(cities), figsize=(5.2 * len(cities), 0.45 * len(D) / max(1, len(cities)) + 2), squeeze=False)
            for ax, city in zip(axes[0], cities):
                d = D[D.city == city]; y = 0; labels = []
                for algo in [x for x in algos if x != a.ref]:
                    for si, tag in enumerate(sets):
                        r = d[(d.algo == algo) & (d.setting == tag)]
                        if not len(r): continue
                        r = r.iloc[0]; ax.errorbar(r.delta, y, xerr=[[r.delta - r.lo], [r.hi - r.delta]], fmt="o", color=COL.get(algo, "k"), alpha=1 - 0.3 * si, capsize=3)
                        labels.append(f"{NAME.get(algo, algo)} · {tag}" + (" *" if r.p_holm < .05 else "")); y += 1
                ax.axvline(0, color="#888", ls=":"); ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8); ax.invert_yaxis(); ax.set_xlabel(f"Δf₁* vs {NAME.get(a.ref, a.ref)} (paired, 95 % CI) · * p_Holm < .05"); ax.set_title(city)
            fig.tight_layout(); fig.savefig(out / "fig_A2_forest.png", dpi=150); plt.close(fig)
        # F-A3 HV convergence (static solve t=0, arm main)
        if len(TR):
            Tr = TR[(TR.setting == main_tag) & (TR.arm == a.arm) & (TR["mode"] == "static") & (TR.t == 0)]
            fig, axes = plt.subplots(1, len(cities), figsize=(5.2 * len(cities), 4), squeeze=False)
            for ax, city in zip(axes[0], cities):
                for algo in algos:
                    g = Tr[(Tr.city == city) & (Tr.algo == algo)]
                    if not len(g): continue
                    q = g.groupby("evals").hv.quantile([.25, .5, .75]).unstack()
                    ax.plot(q.index, q[.5], color=COL.get(algo, "k"), label=NAME.get(algo, algo)); ax.fill_between(q.index, q[.25], q[.75], color=COL.get(algo, "k"), alpha=.12)
                ax.set_xlabel("evaluations"); ax.set_ylabel("HV_plan (median, IQR)"); ax.set_title(f"{city} · convergence (static solve, t=0)"); ax.legend(fontsize=7)
            fig.tight_layout(); fig.savefig(out / "fig_A3_hv_conv.png", dpi=150); plt.close(fig)
        # F-A4 fronts one instance: plan hollow, real filled
        fig, axes = plt.subplots(len(cities), 2, figsize=(10, 4.2 * len(cities)), squeeze=False)
        for ci, city in enumerate(cities):
            for algo in algos:
                z = FR.get((main_tag, algo, city))
                if z is None: continue
                kp, kr = f"{a.inst_fig}_{a.arm}_adaptive_t00_plan", f"{a.inst_fig}_{a.arm}_adaptive_t00_real"
                if kp not in z: continue
                Fp, Fr = z[kp], z[kr]
                for j, (yi, lab) in enumerate([(1, "f₂"), (2, "f₃")]):
                    ax = axes[ci, j]; ax.scatter(Fp[:, 0], Fp[:, yi], facecolors="none", edgecolors=COL.get(algo, "k"), s=18, label=f"{NAME.get(algo, algo)} plan"); ax.scatter(Fr[:, 0], Fr[:, yi], color=COL.get(algo, "k"), s=10, alpha=.6, label=f"{NAME.get(algo, algo)} realised")
                    ax.set_xlabel("f₁"); ax.set_ylabel(lab); ax.set_title(f"{city} inst {a.inst_fig:03d} · front t=0 (hollow = plan, filled = realised)")
            axes[ci, 0].legend(fontsize=6, ncol=2)
        fig.tight_layout(); fig.savefig(out / "fig_A4_fronts.png", dpi=150); plt.close(fig)
        # F-A5 timeline done(t) + F-A6 quota_used(t)
        Ta = TL[(TL.setting == main_tag) & (TL["mode"] == "adaptive") & (TL.arm == a.arm)]
        for col, fn, ylab in [("done", "fig_A5_timeline.png", "tasks fulfilled (cumulative, mean)"), ("quota_used", "fig_A6_quota.png", "fraction of worker quota consumed (mean)")]:
            if col not in Ta: continue
            fig, axes = plt.subplots(1, len(cities), figsize=(5.2 * len(cities), 4), squeeze=False)
            for ax, city in zip(axes[0], cities):
                for algo in algos:
                    g = Ta[(Ta.city == city) & (Ta.algo == algo)].groupby("t")[col].mean()
                    if len(g): ax.plot(g.index, g.values, marker="o", ms=3, color=COL.get(algo, "k"), label=NAME.get(algo, algo))
                if col == "done":
                    go = TL[(TL.setting == main_tag) & (TL["mode"] == "adaptive") & (TL.arm == "oracle") & (TL.city == city) & (TL.algo == a.ref)].groupby("t")[col].mean()
                    if len(go): ax.plot(go.index, go.values, ls=":", color="k", label="oracle (ref algo)")
                ax.set_xlabel("decision time t (min)"); ax.set_ylabel(ylab); ax.set_title(f"{city} · {main_tag}"); ax.legend(fontsize=7)
            fig.tight_layout(); fig.savefig(out / fn, dpi=150); plt.close(fig)
        # F-A7 churn vs f1*  · F-A8 runtime
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
        for algo in algos:
            g = Mm[Mm.algo == algo]; axes[0].scatter(g.n_changed_mean, g.f1_real, s=14, color=COL.get(algo, "k"), label=NAME.get(algo, algo), alpha=.7)
        axes[0].set_xlabel("offers changed per decision point (churn)"); axes[0].set_ylabel("f₁*"); axes[0].legend(fontsize=7); axes[0].set_title("churn vs realised f₁*")
        if "sec_per_solve" in Mm:
            g = Mm.groupby("algo").sec_per_solve.agg(["mean", "std"]).reindex(algos)
            axes[1].bar([NAME.get(x, x) for x in g.index], g["mean"], yerr=g["std"], color=[COL.get(x, "k") for x in g.index]); axes[1].axhline(600, ls=":", color="k"); axes[1].text(0, 600, " Δ = 10 min", va="bottom", fontsize=8)
            axes[1].set_ylabel("s / solve"); axes[1].tick_params(axis="x", rotation=25, labelsize=8); axes[1].set_title("runtime per solve (38,400 evals)"); axes[1].set_yscale("log")
        fig.tight_layout(); fig.savefig(out / "fig_A8_runtime.png", dpi=150); plt.close(fig)
    # ---------------------------------------------------------------- report
    lines = [f"# E6 — เทียบอัลกอริทึม · {VERSION} · {time.strftime('%Y-%m-%d %H:%M')}", "", f"roots: {[r for r, _ in roots]} · ref {a.ref} · arm {a.arm} · algos {[NAME.get(x, x) for x in algos]} · cities {cities}", "",
             "## Table 4 — mean ± sd ต่ออัลกอริทึม (adaptive = โหมดหลัก · static เพื่อเทียบ · oracle = เพดานของ optimizer ตัวนั้นเมื่อ C = ความจริง)", "", md(T4) if len(T4) else "—", "",
             f"## Table 5 — Δ paired vs {NAME.get(a.ref, a.ref)} ต่อ instance (bootstrap 95 % CI · Wilcoxon · Holm ต่อเมือง/ตัวชี้วัด)", ""]
    if len(T5):
        for metric in T5.metric.unique():
            d = T5[T5.metric == metric][["setting", "city", "algo", "n", "ref_mean", "mean", "delta", "lo", "hi", "p", "p_holm", "wins", "losses"]].copy(); d["algo"] = d.algo.map(NAME)
            lines += [f"### {metric}", "", md(d.round(4)), ""]
    if len(IG):
        lines += ["## IGD⁺ (plan front ที่ t=0, adaptive) ต่อ reference front รวมทุกอัลกอริทึม — ต่ำ = ดี", "", md(IGs.round(4)), ""]
        if len(IGp): d = IGp[["city", "algo", "n", "ref_mean", "mean", "delta", "lo", "hi", "p", "p_holm", "wins", "losses"]].copy(); d["algo"] = d.algo.map(NAME); lines += ["### Δ IGD⁺ vs ref (ลบ = ดีกว่า ref)", "", md(d.round(4)), ""]
    lines += ["## อ่าน", f"- ตัวชี้วัดหลัก: f₁* adaptive ({main_tag}) · ผู้ชนะต้อง Δ > 0 และ CI ไม่คร่อม 0 ทั้งสองเมือง (p_Holm < .05)", "- oracle arm: ทุกอัลกอริทึมเห็น C = ความจริง → Δ ที่นี่คือความแข็งของ optimizer ล้วน ๆ ไม่ปนความคลาดเคลื่อนของตัวทำนาย",
              "- gap_plan_real_t0 = f₁_plan(X*) / f₁_exp(X*) ที่ t=0 (winner's curse; >1 = optimizer เชื่อเกินจริง)", "- รูป: fig_A1 box+paired · A2 forest · A3 HV convergence · A4 fronts · A5 timeline · A6 quota · A8 churn+runtime", ""]
    (out / "E6_report.md").write_text("\n".join(lines), encoding="utf-8"); log(f"→ {out / 'E6_report.md'}")


if __name__ == "__main__":
    main()
