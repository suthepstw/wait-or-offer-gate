#!/usr/bin/env python3
"""
sX_plot_encoders_reach.py — รูป encoder × estimator บน (9a′)   (แทนรูปเดิมที่วัดบน (9a) และใช้เวลา/epoch เป็นแกน x)

อ่านเฉพาะไฟล์ที่ sD7_reach_eval.py / sD5_train_v4.py เขียนไว้แล้ว — ไม่คำนวณอะไรใหม่ ไม่แตะ frozen
  · models/<city>/<run>/D7reach_dev/reach_metrics.csv   → AUC ต่อ seed (qm_reach_cal, qm_ub_cal, gauss_dv_cal), cluster-CI, Δ(reach−ub) ภายใน run
  · models/<city>/<run>/seed*/metrics.json               → params, s_per_epoch, infer_ms_per_1k_windows
  · reports/<cross-run md> (เช่น reach_enc6_sf.md)       → paired ΔAUC vs ref [CI] ของแต่ละ run (ค่านี้มีเฉพาะใน md)

panel (a) AUC@R (9a′) ต่อ encoder: จุดต่อ seed + mean + cluster-CI · ตัวกลวง = (9a) ของ run เดียวกัน (ให้เห็นว่า estimator gain » encoder gain)
panel (b) paired ΔAUC vs reference (Bi-LSTM) บน (9a′) [cluster-CI] — ✔ เมื่อ CI ไม่คร่อม 0
panel (c) AUC (9a′) vs params (log) · error bar = paired CI เลื่อนไปรอบค่า reference (ไม่ใช่ CI เดี่ยว)

ใช้:
  python sX_plot_encoders_reach.py --city sf --md reports/reach_enc6_sf.md \
      --runs reach_lstm,reach_gru,reach_rnn,reach_tfm,reach_sttfm,reach_tfm_L --ref reach_lstm \
      --out reports/fig_encoders_reach_sf
  (--city rome --md reports/reach_enc_rome.md --runs reach_lstm,reach_tfm … ก็ได้ ตัวที่ไม่มีจะถูกข้ามพร้อมแจ้ง)
ผล: <out>.pdf + <out>.png + <out>.csv (ตัวเลขที่ใช้วาด — แนบ appendix ได้)
"""
import argparse, glob, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ชื่อ/สี/marker คงที่ต่อ encoder (ไม่หมุนตามลำดับ) — Okabe-Ito, ผ่าน CVD check พร้อม marker ต่างกันเป็น secondary encoding
SPEC = {
    "reach_lstm":  ("Bi-LSTM (reference)",        "#0072B2", "o"),
    "reach_gru":   ("Bi-GRU",                     "#D55E00", "s"),
    "reach_rnn":   ("Bi-RNN (vanilla)",           "#009E73", "D"),
    "reach_tfm":   ("Transformer (temporal, main)", "#E69F00", "^"),
    "reach_sttfm": ("Transformer + spatial enc.", "#CC79A7", "v"),
    "reach_tfm_L": ("Transformer-L (d256×3)",     "#56B4E9", "P"),
}
def spec(run): return SPEC.get(run, (run, "#7f7f7f", "x"))

def read_run(models, city, run, R, V="qm_reach_cal_g"):
    d = Path(models) / city / run
    f = d / "D7reach_dev" / "reach_metrics.csv"
    if not f.exists(): print(f"  ข้าม {run}: ไม่พบ {f}"); return None
    df = pd.read_csv(f); df = df[df.R == R]
    def per_seed(var, met="AUC"):
        q = df[(df.seed != "all") & (df.variant == var) & (df.metric == met) & df.subset.isna()]
        return q.value.to_numpy()
    def one(var, met):
        q = df[(df.seed == "all") & (df.variant == var) & (df.metric == met)]
        return float(q.value.iloc[0]) if len(q) else np.nan
    if not len(per_seed(V)): print(f"  ข้าม {run}: ไม่มี {V} ใน {f} (รัน sD7_reach_eval ตัวที่มี _g ก่อน หรือใช้ --variant qm_reach_cal)"); return None
    mj = [json.load(open(p)) for p in sorted(glob.glob(str(d / "seed*" / "metrics.json")))]
    g = lambda k: float(np.mean([m[k] for m in mj if k in m])) if any(k in m for m in mj) else np.nan
    return dict(run=run, label=spec(run)[0], color=spec(run)[1], marker=spec(run)[2],
                auc_reach_seeds=per_seed(V), auc_ub_seeds=per_seed("qm_ub_cal"), auc_null=per_seed("gauss_dv_cal").mean(),
                auc_reach=per_seed(V).mean(), auc_ub=per_seed("qm_ub_cal").mean(),
                ci_lo=one(V, "AUC_ci_lo"), ci_hi=one(V, "AUC_ci_hi"),
                hard_reach=df[(df.seed != "all") & (df.variant == V) & (df.metric == "AUC") & (df.subset == "hard")].value.mean(),
                ece_reach=per_seed(V, "ECE").mean(), variant=V,
                d_in=one("delta_reach_minus_ub", "dAUC"), d_in_lo=one("delta_reach_minus_ub", "dAUC_ci_lo"), d_in_hi=one("delta_reach_minus_ub", "dAUC_ci_hi"),
                params_M=g("params") / 1e6, s_epoch=g("s_per_epoch"), infer_ms=g("infer_ms_per_1k_windows"), n_seeds=len(per_seed("qm_reach_cal")))

PAT = re.compile(r"([+-]\d+\.\d+)\s*\[\s*([+-]\d+\.\d+)\s*,\s*([+-]\d+\.\d+)\s*\]")
def read_paired(md_path, variant="qm_reach_cal"):
    """คืน {run: (d, lo, hi)} จากคอลัมน์ 'ΔAUC vs … (paired)' ของตารางรวม — แถว variant ที่ระบุ"""
    out = {}
    if not md_path or not Path(md_path).exists(): print(f"  ไม่มี md {md_path} → panel (b)/(c) ไม่มี paired CI"); return out
    for line in Path(md_path).read_text(encoding="utf-8").splitlines():
        if not line.startswith("| reach_") and not line.startswith("| base_"): continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 9 or cells[1] != variant: continue
        m = PAT.search(cells[-1])
        if m: out[cells[0]] = tuple(float(x) for x in m.groups())
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", default="sf"); ap.add_argument("--models", default="models"); ap.add_argument("--R", type=float, default=200)
    ap.add_argument("--runs", default="reach_lstm,reach_gru,reach_rnn,reach_tfm,reach_sttfm,reach_tfm_L"); ap.add_argument("--ref", default="reach_lstm")
    ap.add_argument("--md", default="", help="ตารางรวมจาก sD7_reach_eval (--ref เดียวกัน, --ref-variant qm_reach_cal)")
    ap.add_argument("--out", default=""); ap.add_argument("--title", default="")
    ap.add_argument("--variant", default="qm_reach_cal_g", help="คอลัมน์ (9a′) ที่ใช้: qm_reach_cal_g (ฉบับสมบูรณ์ มี domain guard, ค่าหลัก) | qm_reach_cal (หัวล้วน)")
    a = ap.parse_args()
    runs = [r for r in a.runs.split(",") if r]
    rows = [x for x in (read_run(a.models, a.city, r, a.R, a.variant) for r in runs) if x]
    if not rows: sys.exit("ไม่มี run ให้วาด")
    paired = read_paired(a.md, a.variant)
    ref = next((x for x in rows if x["run"] == a.ref), rows[0])
    for x in rows:
        d = paired.get(x["run"]); x["d_ref"], x["d_lo"], x["d_hi"] = d if d else ((0.0, 0.0, 0.0) if x["run"] == ref["run"] else (np.nan,) * 3)
    out = Path(a.out or f"reports/fig_encoders_reach_{a.city}"); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: v for k, v in x.items() if not isinstance(v, np.ndarray)} for x in rows]).to_csv(out.with_suffix(".csv"), index=False)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .6})
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.35, 1.0, 1.05]})
    n = len(rows); ys = np.arange(n)[::-1]

    # ---- (a) AUC (9a′) forest + (9a) hollow
    for y, x in zip(ys, rows):
        a1.plot([x["ci_lo"], x["ci_hi"]], [y, y], color=x["color"], lw=2, solid_capstyle="round", zorder=2)
        a1.scatter(x["auc_reach_seeds"], np.full(len(x["auc_reach_seeds"]), y), s=14, color=x["color"], alpha=.55, zorder=3, linewidths=0)
        a1.scatter([x["auc_reach"]], [y], s=110, color=x["color"], marker=x["marker"], edgecolor="white", linewidth=1.2, zorder=4)
        a1.scatter([x["auc_ub"]], [y], s=70, facecolor="none", edgecolor=x["color"], marker=x["marker"], linewidth=1.3, zorder=4, alpha=.9)
        a1.annotate(f"{x['auc_reach']:.3f}", (x["ci_hi"], y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color="#333")
    a1.set_yticks(ys); a1.set_yticklabels([x["label"] for x in rows]); a1.set_ylim(-.7, n - .3)
    a1.axvspan(ref["ci_lo"], ref["ci_hi"], color=ref["color"], alpha=.07, zorder=0)
    a1.set_xlabel(f"AUC @R = {a.R:g} m · filled = (9a′){' (with domain guard)' if a.variant.endswith('_g') else ' (learned head only)'}, hollow = (9a) union bound (same run)\n"
                  f"mean, 95 % cluster CI, per-seed dots · physics null gauss_dv_cal = {np.nanmean([x['auc_null'] for x in rows]):.3f} (off-scale)", fontsize=8.5)
    a1.set_title(f"(a) {a.city.upper()} · reachability AUC by encoder and estimator", fontsize=10, loc="left")
    lo = min(min(x["auc_ub"] for x in rows), min(x["ci_lo"] for x in rows)) - .004; hi = max(x["ci_hi"] for x in rows) + .012; a1.set_xlim(lo, hi)

    # ---- (b) paired Δ vs ref on (9a′)
    for y, x in zip(ys, rows):
        if np.isnan(x["d_ref"]): a2.text(0, y, "n/a", va="center", ha="center", fontsize=8, color="#888"); continue
        sig = (x["d_lo"] > 0) or (x["d_hi"] < 0)
        a2.plot([x["d_lo"], x["d_hi"]], [y, y], color=x["color"], lw=2.2 if sig else 1.4, alpha=1 if sig else .6, solid_capstyle="round", zorder=2)
        a2.scatter([x["d_ref"]], [y], s=90, color=x["color"], marker=x["marker"], edgecolor="white", linewidth=1.2, zorder=4)
        if x["run"] != ref["run"]:
            a2.annotate(f"{x['d_ref']:+.4f}" + (" ✔" if sig else ""), (x["d_hi"], y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color="#333")
    a2.axvline(0, color="#555", lw=1, ls=":"); a2.set_yticks(ys); a2.set_yticklabels([]); a2.set_ylim(-.7, n - .3)
    a2.set_xlabel(f"paired ΔAUC vs {ref['label'].split(' (')[0]} on (9a′) · same {('pairs' if paired else '—')}, cluster CI\n✔ = CI excludes 0", fontsize=8.5)
    a2.set_title("(b) encoder effect, paired on identical pairs", fontsize=10, loc="left")
    dl = [x["d_lo"] for x in rows if not np.isnan(x["d_lo"])] + [-.002]; dh = [x["d_hi"] for x in rows if not np.isnan(x["d_hi"])] + [.002]
    a2.set_xlim(min(dl) - .003, max(dh) + .006)

    # ---- (c) AUC vs params
    # ป้ายชื่อ: เรียงตาม params แล้วสลับ บน/ล่าง เพื่อเลี่ยงทับกันเมื่อ params ใกล้กัน
    order = sorted([x for x in rows if not np.isnan(x["params_M"])], key=lambda x: x["params_M"])
    for k, x in enumerate(order):
        yc = ref["auc_reach"] + (0 if np.isnan(x["d_ref"]) else x["d_ref"]); err = None if np.isnan(x["d_lo"]) else [[yc - (ref["auc_reach"] + x["d_lo"])], [(ref["auc_reach"] + x["d_hi"]) - yc]]
        a3.errorbar([x["params_M"]], [yc], yerr=err, fmt=x["marker"], color=x["color"], ms=10, mec="white", mew=1.2, capsize=3, lw=1.5, zorder=4)
        close = k > 0 and x["params_M"] / order[k - 1]["params_M"] < 1.6
        dx, dy, ha = ((-8, -4, "right") if close and k % 2 else (8, 4, "left"))
        a3.annotate(x["label"].split(" (")[0].replace("Transformer + spatial enc.", "Tfm + spatial"), (x["params_M"], yc), xytext=(dx, dy), textcoords="offset points", fontsize=8, color="#333", ha=ha)
    a3.set_xscale("log"); a3.axhline(ref["auc_reach"], color=ref["color"], lw=1, ls=":", alpha=.6)
    from matplotlib.ticker import FixedLocator, FuncFormatter
    tk = [t for t in (0.25, 0.5, 1, 2, 4, 8) if (pm := [x["params_M"] for x in rows if not np.isnan(x["params_M"])]) and min(pm) / 1.8 <= t <= max(pm) * 2.2]
    a3.xaxis.set_major_locator(FixedLocator(tk)); a3.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); a3.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    a3.set_xlabel("parameters (M, log scale)", fontsize=9); a3.set_ylabel(f"AUC (9a′) @R = {a.R:g} m\n(reference + paired Δ, cluster CI)", fontsize=9)
    a3.set_title("(c) quality vs. model size", fontsize=10, loc="left")
    pm = [x["params_M"] for x in rows if not np.isnan(x["params_M"])]
    if pm: a3.set_xlim(min(pm) / 2.6, max(pm) * 2.2)

    # legend (ชื่อ + marker — identity ไม่ใช่สีอย่างเดียว)
    hs = [plt.Line2D([], [], color=x["color"], marker=x["marker"], ls="", ms=8, label=x["label"]) for x in rows]
    hs += [plt.Line2D([], [], color="#555", marker="o", ls="", ms=8, label="(9a′) learned head", mfc="#555"), plt.Line2D([], [], color="#555", marker="o", ls="", ms=8, label="(9a) union bound", mfc="none")]
    fig.legend(handles=hs, loc="lower center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(.5, -.02))
    if a.title: fig.suptitle(a.title, fontsize=11)
    fig.tight_layout(rect=(0, .07, 1, 1 if not a.title else .95))
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"→ {out.with_suffix('.pdf')} · {out.with_suffix('.png')} · {out.with_suffix('.csv')}")
    for x in rows: print(f"  {x['run']:14s} (9a′) {x['auc_reach']:.4f} [{x['ci_lo']:.4f},{x['ci_hi']:.4f}] · (9a) {x['auc_ub']:.4f} · Δin {x['d_in']:+.4f} · Δvs ref {x['d_ref']:+.4f} [{x['d_lo']:+.4f},{x['d_hi']:+.4f}] · hard {x['hard_reach']:.3f} · ECE {x['ece_reach']:.4f} · {x['params_M']:.2f} M")

if __name__ == "__main__": main()
