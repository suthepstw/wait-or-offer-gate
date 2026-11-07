#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sE6_paperfigs.py (v0.3, 2026-10-24 · สร้างครบทั้ง 9 รูปของบทความจากไฟล์ผลดิบ) — รูปผลการทดลองสำหรับบทความ จากไฟล์ผล final ที่มีอยู่แล้ว (อ่านอย่างเดียว ไม่รันอะไรใหม่)
ผล: paper_figs/fig_auc_t.pdf · fig_cumulative.pdf · fig_main.pdf · fig_sensitivity.pdf · fig_ablation.pdf (ใหม่ v0.3)
    paper_figs/fig_timeline.pdf   — โควตาที่ใช้สะสม + งานสำเร็จสะสมตาม t (ไม่มี/มี gate × learned/physics + oracle) · 2 เมือง
    paper_figs/fig_paired.pdf     — ต่อ instance: f1* ไม่มี gate vs มี gate (learned, physics, greedy)
    paper_figs/fig_dist.pdf       — การกระจาย f1* f2* f3* ต่อแขน (box + จุด)
    paper_figs/fig_hvconv.pdf     — HV ของแผนตามรอบ (sticky, NSGA-II vs MOPSO family)
    paper_figs/paperfigs_check.md — ค่าเฉลี่ยที่ใช้ (ไว้เทียบกับตารางในบทความ)
ใช้: python sE6_paperfigs.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
OUT = Path("paper_figs"); OUT.mkdir(exist_ok=True)
CITIES = [("sf", "San Francisco"), ("rome", "Rome")]
ARMS = {  # key: (folder, arm, label, colour, style)
    "lr0": ("e6final_sticky/nsga2", "qm_model", "Learned, no gate", "#7f7f7f", "-"),
    "lr1": ("e6gd_final/b0dz_k1.5", "qm_model", "Learned + gate", "#1f4e79", "-"),
    "ph0": ("e6ng_final/null_none", "qm_null_dv", "Physics, no gate", "#bfbfbf", "--"),
    "ph1": ("e6ng_final/nullg_k1.5", "qm_null_dv", "Physics + gate", "#c55a11", "--"),
    "gr0": ("e6gd_final/grst_none", "qm_model", "Greedy, no gate", "#a9d18e", ":"),
    "gr1": ("e6gd_final/grdz_k1.5", "qm_model", "Greedy + gate", "#548235", ":"),
    "orc": ("e6final_sticky/nsga2", "oracle", "Oracle", "#000000", "-"),
}
log = []


def runs(key, city):
    fo, arm = ARMS[key][:2]; f = Path(fo) / city / "e5_runs.csv"
    if not f.exists(): log.append(f"- ไม่มี {f}"); return None
    d = pd.read_csv(f); d = d[(d.arm == arm) & (d["mode"] == "adaptive")].set_index("k").sort_index()
    return d if len(d) else None


def timeline(key, city):
    fo, arm = ARMS[key][:2]; f = Path(fo) / city / "e5_timeline.csv"
    if not f.exists(): log.append(f"- ไม่มี {f}"); return None
    d = pd.read_csv(f); d = d[(d.arm == arm) & (d["mode"] == "adaptive")]
    return d if len(d) else None


# ---- Fig: timeline (quota + completion)
fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True)
for ci, (c, cname) in enumerate(CITIES):
    for key in ["lr0", "lr1", "ph1", "orc"]:
        tl = timeline(key, c)
        if tl is None: continue
        _, _, lab, col, ls = ARMS[key]; g = tl.groupby("t")
        q = g.quota_used.mean(); n_tasks = 200.0
        axs[0, ci].plot(q.index, q.values, ls, color=col, marker="o", ms=3, label=lab)
        dn = g.done.mean() / n_tasks; axs[1, ci].plot(dn.index, dn.values, ls, color=col, marker="o", ms=3, label=lab)
        log.append(f"- {cname} {lab}: quota@t {dict(zip(q.index, q.round(3)))} · done/200 end {dn.values[-1]:.3f}")
    axs[0, ci].set_title(cname, fontsize=9); axs[1, ci].set_xlabel("Decision time t (min)")
axs[0, 0].set_ylabel("Capacity used (fraction)"); axs[1, 0].set_ylabel("Tasks completed (fraction)")
axs[0, 0].legend(frameon=False, fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "fig_timeline.pdf"); plt.close(fig)

# ---- Fig: paired per-instance
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.5))
for ax, (k0, k1, title) in zip(axs, [("lr0", "lr1", "Learned predictor, NSGA-II"), ("ph0", "ph1", "Physics predictor, NSGA-II"), ("gr0", "gr1", "Learned predictor, greedy")]):
    lim = 0
    for c, cname in CITIES:
        a, b = runs(k0, c), runs(k1, c)
        if a is None or b is None: continue
        ks = a.index.intersection(b.index); x, y = a.loc[ks, "f1_real"], b.loc[ks, "f1_real"]
        ax.scatter(x, y, s=12, marker="o" if c == "sf" else "s", color="#1f4e79" if c == "sf" else "#c55a11", label=cname, alpha=0.8)
        lim = max(lim, x.max(), y.max()); log.append(f"- {title} {cname}: gate wins {(y > x).sum()}/{len(ks)} · mean {x.mean():.3f} → {y.mean():.3f}")
    lim = lim * 1.08 if lim else 0.3; ax.plot([0, lim], [0, lim], "k:", lw=0.8); ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_title(title, fontsize=8); ax.set_xlabel("$f_1^*$ without gate")
axs[0].set_ylabel("$f_1^*$ with gate"); axs[0].legend(frameon=False, fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "fig_paired.pdf"); plt.close(fig)

# ---- Fig: distributions of f1*, f2*, f3*
keys = ["lr0", "lr1", "ph0", "ph1", "gr0", "gr1"]
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.8))
for ax, (m, mname) in zip(axs, [("f1_real", "$f_1^*$ completion"), ("f2_real", "$f_2^*$ worker fairness"), ("f3_real", "$f_3^*$ task equity")]):
    data, pos, cols = [], [], []
    for i, key in enumerate(keys):
        for j, (c, _) in enumerate(CITIES):
            d = runs(key, c)
            if d is None: continue
            data.append(d[m].dropna().values); pos.append(i * 3 + j); cols.append(ARMS[key][3])
    if data:
        bp = ax.boxplot(data, positions=pos, widths=0.8, patch_artist=True, showfliers=False)
        for patch, col in zip(bp["boxes"], cols): patch.set_facecolor(col); patch.set_alpha(0.75)
        for med in bp["medians"]: med.set_color("k")
    ax.set_xticks([i * 3 + 0.5 for i in range(len(keys))]); ax.set_xticklabels(["L", "L+G", "P", "P+G", "Gr", "Gr+G"], fontsize=7)
    ax.set_title(mname, fontsize=8)
fig.text(0.5, 0.01, "L = learned, P = physics, Gr = greedy, +G = with gate · left/right box of each pair = San Francisco / Rome", ha="center", fontsize=7)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(OUT / "fig_dist.pdf"); plt.close(fig)

# ---- Fig: HV convergence of the planner (sticky)
ALG = [("nsga2", "NSGA-II", "#000000"), ("mopso", "MOPSO", "#9ecae1"), ("mopso_q", "MOPSO-Q", "#fdae6b"), ("mopso_iq", "MOPSO-IQ", "#d62728"),
       ("mopso_rk", "MOPSO (rk)", "#3182bd"), ("mopso_q_rk", "MOPSO-Q (rk)", "#e6550d"), ("mopso_iq_rk", "MOPSO-IQ (rk)", "#a50f15")]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=False)
for ax, (c, cname) in zip(axs, CITIES):
    for a, lab, col in ALG:
        f = Path("e6_sticky") / a / c / "e5_trace.csv"
        if not f.exists(): log.append(f"- ไม่มี {f}"); continue
        d = pd.read_csv(f); d = d[(d.arm == "qm_model") & (d["mode"] != "static")]
        g = d.groupby("gen").hv.median()
        ax.plot(g.index, g.values, color=col, lw=1.4 if a == "nsga2" else 1.0, label=lab)
    ax.set_xscale("log"); ax.set_title(cname, fontsize=9); ax.set_xlabel("Generation / iteration")
axs[0].set_ylabel("Median planned hypervolume")
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=7, ncol=7, loc="lower center")
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(OUT / "fig_hvconv.pdf"); plt.close(fig)

# ======================= v0.3: รูปที่เหลือ (เดิมคัดตัวเลขจากรายงานใน make_figs.py) =======================
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sE6_gatedef import paired                                   # bootstrap 2,000 รอบ seed 0 — ชุดเดียวกับรายงาน GATEDEF/NULLGATE
C = dict(ours="#1f4e79", phys="#c55a11", base="#7f7f7f", free="#548235", orc="#000000")
T_ = [0, 10, 20, 30, 40, 50]


def runs_at(folder, city, arm="qm_model"):
    f = Path(folder) / city / "e5_runs.csv"
    if not f.exists(): log.append(f"- ไม่มี {f}"); return None
    d = pd.read_csv(f); d = d[(d.arm == arm) & (d["mode"] == "adaptive")].set_index("k").sort_index()
    return d if len(d) else None


# ---- Fig 1a: AUC ของ C_ij(t) ตามจุดตัดสินใจ (e5inst_v3/<city>/e5inst_table.csv = AUC ต่อ instance ต่อ t, งานที่ยังเปิด)
fig, ax = plt.subplots(figsize=(3.3, 2.3))
for c, cname, mk in [("sf", "San Francisco", "o"), ("rome", "Rome", "s")]:
    f = Path("e5inst_v3") / c / "e5inst_table.csv"
    if not f.exists(): log.append(f"- ไม่มี {f}"); continue
    g = pd.read_csv(f).groupby("t_min").mean(numeric_only=True)
    ax.plot(g.index, g["AUC_qm_model"], "-" + mk, color=C["ours"] if c == "sf" else C["phys"], ms=3, label=f"Learned, {cname}")
    if "AUC_qm_persist_cont" in g: ax.plot(g.index, g["AUC_qm_persist_cont"], ":" + mk, color=C["ours"] if c == "sf" else C["phys"], ms=3, mfc="white", label=f"Persistence, {cname}")
    log.append(f"- AUC {cname}: learned {g['AUC_qm_model'].round(3).to_dict()} · persistence {g.get('AUC_qm_persist_cont', pd.Series(dtype=float)).round(3).to_dict()}")
ax.set_xlabel("Decision time t (min)"); ax.set_ylabel("AUC of $C_{ij}(t)$"); ax.set_xticks(T_); ax.set_ylim(0.6, 1.0); ax.legend(frameon=False, fontsize=6, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "fig_auc_t.pdf"); plt.close(fig)

# ---- Fig 1b: งานสำเร็จสะสม (SF, ไม่มี gate) free vs sticky จากผล E5
def first_dir(cands):
    for d in cands:
        if (Path(d) / "sf" / "e5_timeline.csv").exists(): return d
    log.append(f"- ไม่พบ {cands}"); return None
fdir, sdir = first_dir(["e5replan", "e5replan_free"]), first_dir(["e5replan_sticky3", "e5replan_sticky"])
fig, ax = plt.subplots(figsize=(3.5, 2.6))
for d, reg in [(fdir, "free withdrawal"), (sdir, "sticky")]:
    if d is None: continue
    tl = pd.read_csv(Path(d) / "sf" / "e5_timeline.csv"); tl = tl[tl["mode"] == "adaptive"]
    for arm, lab, col, ls, mfc in [("qm_model", "Learned", C["free"] if reg.startswith("free") else C["ours"], "-", None), ("qm_null_dv", "Physics", C["free"] if reg.startswith("free") else C["ours"], "--", "white"), ("oracle", "Oracle", C["orc"], "-" if reg == "sticky" else ":", None)]:
        if arm == "oracle" and reg.startswith("free"): continue
        g = tl[tl.arm == arm].groupby("t").done.mean()
        if len(g) == 0: log.append(f"- {d}: ไม่มี arm {arm}"); continue
        ax.plot(g.index, g.values, ls, color=col, marker="o", ms=3, mfc=mfc, label=f"{lab}, {reg}")
        log.append(f"- cumulative SF {lab} {reg} ({d}): {g.round(1).to_dict()}")
ax.set_xlabel("Decision time t (min)"); ax.set_ylabel("Tasks completed (of 200)"); ax.set_xticks(T_); ax.legend(frameon=False, fontsize=6.5, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig_cumulative.pdf"); plt.close(fig)

# ---- Fig 3: ผลหลัก (test, sticky)
labels = ["No gate\n(learned)", "No gate\n(physics)", "Gate\n(learned)", "Gate\n(physics)"]
src = [("e6final_sticky/nsga2", "qm_model"), ("e6ng_final/null_none", "qm_null_dv"), ("e6gd_final/b0dz_k1.5", "qm_model"), ("e6ng_final/nullg_k1.5", "qm_null_dv")]
cols = [C["base"], "#bfbfbf", C["ours"], C["phys"]]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
for ax, (c, cname) in zip(axs, CITIES):
    v = []
    for fo, arm in src:
        d = runs_at(fo, c, arm); v.append(d.f1_real.mean() if d is not None else np.nan)
    fr = runs_at("e6final_free/nsga2", c); free = fr.f1_real.mean() if fr is not None else np.nan
    ax.bar(range(4), v, color=cols, width=0.65)
    for i, x in enumerate(v): ax.text(i, x + 0.005, f"{x:.3f}", ha="center", fontsize=7)
    ax.axhline(free, ls=":", color=C["free"]); ax.text(3.45, free + 0.006, f"free-withdrawal reference {free:.3f}", ha="right", fontsize=7, color=C["free"])
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=7.5); ax.set_title(cname, fontsize=9)
    log.append(f"- main {cname}: {[round(x, 3) for x in v]} · free {free:.3f}")
axs[0].set_ylabel("Realised completion $f_1^*$"); axs[0].set_ylim(0, 0.30)
fig.tight_layout(); fig.savefig(OUT / "fig_main.pdf"); plt.close(fig)

# ---- Fig 4: ความไวบน dev (e6gd_dev)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
for ax, (pref, grid, fmt, xl, sel) in zip(axs, [("b0dz_k", [0.5, 1.0, 1.5, 2.0, 3.0], "{:.1f}", r"$\kappa$ (gate $\tilde C \geq \kappa V_q(t)$)", 1.5),
                                                ("g2dz_t", [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40], "{:.2f}", r"$\tau$ (fixed threshold $\tilde C \geq \tau$)", 0.25)]):
    for c, cname, mk, col in [("sf", "San Francisco", "o", C["ours"]), ("rome", "Rome", "s", C["phys"])]:
        y = []
        for x in grid:
            d = runs_at(f"e6gd_dev/{pref}{fmt.format(x)}", c); y.append(d.f1_real.mean() if d is not None else np.nan)
        ax.plot(grid, y, "-" + mk, color=col, ms=3, label=cname); log.append(f"- dev {pref} {cname}: {[round(v, 3) for v in y]}")
    ax.axvline(sel, ls=":", color="k"); ax.set_xlabel(xl); ax.set_ylim(0.05, 0.20)
axs[0].set_ylabel("$f_1^*$ (development set)"); axs[0].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "fig_sensitivity.pdf"); plt.close(fig)

# ---- Fig 8: ablation (paired Δ vs gate ฉบับสุดท้าย, bootstrap CI)
ref = "e6gd_final/b0dz_k1.5"
rows = [("Waiting only (no threshold)", "e6gd_final/g1dz_d20"), ("Uncalibrated $C$", "e6gd_final/g3dz_k1.0"), ("Greedy instead of NSGA-II", "e6gd_final/grdz_k1.5"),
        ("Zero-probability offers kept", "e6final_sticky/nsga2_sh_g1.5"), ("Fixed threshold (no $V_q(t)$)", "e6gd_final/g2dz_t0.25")]
fig, ax = plt.subplots(figsize=(4.6, 2.6)); yy = np.arange(len(rows))[::-1]
for y0, (lab, fo) in zip(yy, rows):
    for c, cname, off, mk, col in [("sf", "San Francisco", 0.12, "o", C["ours"]), ("rome", "Rome", -0.12, "s", C["phys"])]:
        a, b = runs_at(fo, c), runs_at(ref, c)
        if a is None or b is None: continue
        r = paired(a.f1_real, b.f1_real)
        ax.errorbar(r["delta"], y0 + off, xerr=[[r["delta"] - r["lo"]], [r["hi"] - r["delta"]]], fmt=mk, color=col, ms=4, capsize=2, label=cname if y0 == yy[0] else None)
        log.append(f"- ablation {lab} {cname}: Δ {r['delta']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] W/L {r['wins']}/{r['losses']}")
ax.axvline(0, color="k", lw=0.8); ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
ax.set_xlabel(r"$\Delta f_1^*$ relative to the full gate (95% CI)"); ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
fig.tight_layout(); fig.savefig(OUT / "fig_ablation.pdf"); plt.close(fig)

(OUT / "paperfigs_check.md").write_text("# ค่าที่ใช้ในรูป (ทั้ง 9 รูป)\n\n" + "\n".join(log) + "\n", encoding="utf-8")
print("\n".join(log[-40:])); print(f"เสร็จ v0.3 → {OUT}/ (9 รูป + paperfigs_check.md)")
