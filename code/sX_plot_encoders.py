#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sX_plot_encoders.py — รูปเปรียบเทียบ encoder สำหรับ paper (จากผล D5 + D7 dev เดียวกับ sX_compare_encoders.py)
  (a) Reachability AUC @R = 200 m ของ qm_ub_cal: จุด = mean ข้าม seed · เส้น = 95 % cluster-bootstrap CI · จุดเล็ก = ราย seed
  (b) Mean error ของ mode-π รายขอบฟ้า (เมตร) ทุก encoder + persistence (เทา)
  (c) ต้นทุน–คุณภาพ: เวลาฝึกต่อ epoch (เทียบ Bi-LSTM = 1) vs AUC · ขนาดจุด ∝ params
  ใช้:  python sX_plot_encoders.py --city sf                 → reports/fig_encoder_compare_sf.{png,pdf}
        python sX_plot_encoders.py --city sf,rome            → แถวละเมือง (2×3) ไฟล์เดียว reports/fig_encoder_compare_sf_rome.{png,pdf}
        --runs base_M3,abl_gru,...  --no-c  (ตัด panel c)  --split dev
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
R_MAIN = 200.0; H_PLOT = [5, 10, 20, 30, 45, 60]
# ลำดับ run คงที่ → สี/marker ผูกกับ encoder ไม่ใช่ลำดับในตาราง (categorical palette ผ่าน CVD validator; secondary encoding = marker)
ENC = {  # run: (label, color, marker)
    "base_M3":     ("Bi-LSTM (main)",            "#2a78d6", "o"),
    "abl_gru":     ("Bi-GRU",                    "#eb6834", "s"),
    "abl_rnn":     ("Bi-RNN (vanilla)",          "#1baf7a", "D"),
    "abl_tfm":     ("Transformer (temporal)",    "#eda100", "^"),
    "abl_sttfm":   ("Transformer + spatial enc.","#e87ba4", "v"),
    "abl_sttfm_L": ("Transformer-L (2.7 M)",     "#4a3aa7", "P"),
}
GRAY = "#8a8985"; INK = "#0b0b0b"; INK2 = "#52514e"

def load(city, run, models, split):
    rd = models / city / run; d7 = rd / f"D7_{split}"
    if not (rd / "summary.json").exists() or not (d7 / "pred_metrics.csv").exists(): return None
    S = json.load(open(rd / "summary.json", encoding="utf-8"))["mean"]
    df = pd.read_csv(d7 / "pred_metrics.csv"); sd = df[df.seed != "all"]; al = df[df.seed == "all"]
    q = sd[(sd.variant == "qm_ub_cal") & (sd.metric == "AUC") & np.isclose(sd.R.astype(float), R_MAIN)]
    auc_seed = q.groupby("seed")["value"].mean().values
    ci = [al[(al.variant == "qm_ub_cal") & (al.metric == m) & np.isclose(al.R.astype(float), R_MAIN)].value for m in ("AUC_ci_lo", "AUC_ci_hi")]
    ci = [float(c.iloc[0]) if len(c) else np.nan for c in ci]
    null = sd[(sd.variant == "gauss_dv_cal") & (sd.metric == "AUC") & np.isclose(sd.R.astype(float), R_MAIN)].value.mean()
    def per_h(var):
        t = sd[(sd.variant == var) & (sd.metric == "mean")]
        return np.array([t[t.h == h].groupby("seed")["value"].mean().mean() if (t.h == h).any() else np.nan for h in H_PLOT])
    return dict(run=run, auc=auc_seed.mean(), auc_seed=auc_seed, ci=ci, null=null, ade_h=per_h("model_pi"), pers_h=per_h("persistence"),
                params=S.get("params", np.nan) / 1e6, s_ep=S.get("s_per_epoch", np.nan))

def panel_a(ax, rows, city):
    rows = rows[::-1]; y = np.arange(len(rows))
    for i, r in enumerate(rows):
        lab, c, mk = ENC.get(r["run"], (r["run"], GRAY, "o"))
        ax.plot(r["ci"], [i, i], color=c, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter(r["auc_seed"], np.full(len(r["auc_seed"]), i), s=12, color=c, alpha=.45, zorder=3, edgecolor="none")
        ax.scatter([r["auc"]], [i], s=70, color=c, marker=mk, edgecolor="white", lw=1.2, zorder=4)
        ax.text(r["ci"][1] + 0.002, i, f"{r['auc']:.3f}", va="center", ha="left", fontsize=8, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([ENC.get(r["run"], (r["run"],))[0] for r in rows], fontsize=8.5)
    base = next((r for r in rows if r["run"] == "base_M3"), None)
    if base is not None: ax.axvspan(*base["ci"], color=ENC["base_M3"][1], alpha=.07, lw=0, zorder=1)
    lo = min(r["ci"][0] for r in rows); hi = max(r["ci"][1] for r in rows); pad = (hi - lo) * .25
    ax.set_xlim(lo - pad, hi + pad * 2.2)
    ax.set_xlabel(f"AUC @R = {R_MAIN:g} m (mean, 95 % cluster CI, per-seed dots)\nphysics-null gauss_dv_cal = {rows[0]['null']:.3f} (off-scale)", fontsize=7.8)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_title(f"(a) {city.upper()} · reachability AUC", loc="left", fontsize=8.5, color=INK)

def panel_b(ax, rows, city):
    pers = next((r["pers_h"] for r in rows if np.isfinite(r["pers_h"]).any()), None)
    if pers is not None:
        ax.plot(H_PLOT, pers, color=GRAY, ls="--", lw=1.6, label="persistence"); ax.text(H_PLOT[-1] - 1, pers[-1] - 60, "persistence", fontsize=7.5, color=INK2, va="top", ha="right")
    for r in rows:
        lab, c, mk = ENC.get(r["run"], (r["run"], GRAY, "o"))
        ax.plot(H_PLOT, r["ade_h"], color=c, lw=1.8, marker=mk, ms=5, mec="white", mew=.8, label=lab)
    ax.set_xlabel("horizon h (min)", fontsize=8.5); ax.set_ylabel("mean error of mode-π (m)", fontsize=8.5); ax.set_xticks([10, 20, 30, 45, 60])
    ax.set_xlim(0, 65); ax.set_title("(b) error by horizon (mode π)", loc="left", fontsize=8.5, color=INK)

def panel_c(ax, rows):
    base = next((r for r in rows if r["run"] == "base_M3"), rows[0])
    for r in rows:
        lab, c, mk = ENC.get(r["run"], (r["run"], GRAY, "o"))
        x = r["s_ep"] / base["s_ep"]
        ax.errorbar([x], [r["auc"]], yerr=[[r["auc"] - r["ci"][0]], [r["ci"][1] - r["auc"]]], fmt="none", ecolor=c, elinewidth=1.4, capsize=2.5, zorder=2)
        ax.scatter([x], [r["auc"]], s=40 + 60 * r["params"], color=c, marker=mk, edgecolor="white", lw=1.2, zorder=3)
    ax.axvline(1, color=GRAY, lw=.8, ls=":"); ax.set_xlabel("training time per epoch (× Bi-LSTM)", fontsize=8.5)
    ax.set_ylabel(f"AUC @R = {R_MAIN:g} m", fontsize=8.5); ax.set_title("(c) cost vs. quality (dot size ∝ params)", loc="left", fontsize=8.5, color=INK)
    ax.set_xlim(0.3, max(r["s_ep"] / base["s_ep"] for r in rows) + .4)

def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#c8c7c2")
    ax.tick_params(colors=INK2, labelsize=8, length=3); ax.grid(axis="both", color="#ebeae6", lw=.6); ax.set_axisbelow(True)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", default="sf"); ap.add_argument("--runs", default=",".join(ENC)); ap.add_argument("--models", default="models")
    ap.add_argument("--split", default="dev"); ap.add_argument("--no-c", action="store_true"); ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args(); models = Path(a.models) if Path(a.models).is_absolute() else HERE / a.models
    cities = [c.strip() for c in a.city.split(",") if c.strip()]; runs = [r.strip() for r in a.runs.split(",") if r.strip()]
    ncol = 2 if a.no_c else 3
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "medium"})
    fig, axes = plt.subplots(len(cities), ncol, figsize=(7.4 if ncol == 3 else 5.4, 2.75 * len(cities) + .55), squeeze=False,
                             gridspec_kw=dict(width_ratios=[1.45, 1.15, 1.0][:ncol]))
    for i, city in enumerate(cities):
        rows = [r for r in (load(city, run, models, a.split) for run in runs) if r is not None]
        if not rows: print(f"{city}: ไม่พบผล"); continue
        panel_a(axes[i, 0], rows, city); panel_b(axes[i, 1], rows, city)
        if not a.no_c: panel_c(axes[i, 2], rows)
        for ax in axes[i]: style(ax)
    h, l = axes[0, 1].get_legend_handles_labels(); h, l = h[1:] + h[:1], l[1:] + l[:1]   # persistence ท้ายสุด
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.01), handlelength=1.8, columnspacing=1.4)
    fig.tight_layout(w_pad=1.4, h_pad=1.0, rect=(0, 0.09 / len(cities), 1, 1))
    (HERE / "reports").mkdir(exist_ok=True); stem = HERE / "reports" / f"fig_encoder_compare_{'_'.join(cities)}"
    fig.savefig(f"{stem}.png", dpi=a.dpi, bbox_inches="tight"); fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    print(f"→ {stem}.png / .pdf")

if __name__ == "__main__":
    main()
