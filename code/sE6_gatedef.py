#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sE6_gatedef.py (SPEC #4 · v0.1, 2026-10-24) — วิเคราะห์ชุดป้องกัน gate (G1 รอเฉยๆ · G2 threshold คงที่ · G3 gate บน C ดิบ) เทียบ B0
=================================================================================================================
โครงโฟลเดอร์ : <root>/<name>/<city>/e5_runs.csv + e5_timeline.csv   (จาก sE5_replan v1.0 --arms qm_model --adaptive-only --withdraw sticky)
ชื่อ <name>  : <fam>_<param> · fam ∈ b0 (B0 เดิม ไม่ drop-zero, repro) · b0dz (B0 + drop-zero) · g1dz (delay t_d) · g2dz (τ) · g3dz (κ บน C ดิบ)
dev   : ตาราง f1*/f2*/f3*/โควตา/สัดส่วน offer ใหม่ที่ C = 0 ต่อ name × เมือง · เลือกพารามิเตอร์ต่อ fam = f1* เฉลี่ยสองเมืองสูงสุด (เสมอ → ค่าแรกตามลำดับ)
        · repro: b0_k1.5 ต้องเท่า --b0-old (ผล dev เดิม) ทุก instance
        → <root>/GATEDEF_select.json · GATEDEF_dev.md
final : อ่าน --select · เทียบ g1dz/g2dz/g3dz ที่ค่าที่เลือก กับ b0dz ที่ค่าที่เลือก (paired ต่อ k, bootstrap 2000, Wilcoxon, Holm 3 การเทียบต่อเมือง)
        · รายงาน b0dz vs B0 final เดิม (--b0-final) แยก · → <final>/GATEDEF_final.md
ใช้:  python sE6_gatedef.py --root e6gd_dev --b0-old e6dev_sticky/nsga2_sh_g1.5
      python sE6_gatedef.py --final e6gd_final --select e6gd_dev/GATEDEF_select.json --b0-final e6final_sticky/nsga2_sh_g1.5
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

VERSION = "E6-gatedef v0.2 (2026-10-24: + greedy baseline)"
CITIES = ["sf", "rome"]
FAMS = ["b0dz", "g1dz", "g2dz", "g3dz"]
LABEL = dict(b0="B0 (เดิม, ไม่ drop-zero)", b0dz="B0: C̃ ≥ κ·V_q(t)", g1dz="G1 รอเฉยๆ (t_d)", g2dz="G2 C̃ ≥ τ คงที่", g3dz="G3 C ดิบ ≥ κ·V_q(t)", grst="Greedy (sticky, ไม่มี gate)", grfree="Greedy (free, ไม่มี gate)", grdz="Greedy + C̃ + gate")


def load(folder: Path, city: str):
    r = folder / city / "e5_runs.csv"
    if not r.exists(): return None, None
    d = pd.read_csv(r); d = d[(d.arm == "qm_model") & (d["mode"] == "adaptive")].set_index("k").sort_index()
    t = folder / city / "e5_timeline.csv"; tl = pd.read_csv(t) if t.exists() else None
    if tl is not None: tl = tl[(tl.arm == "qm_model") & (tl["mode"] == "adaptive")]
    return d, tl


def zshare(tl):
    if tl is None or "n_zero_new" not in tl or tl.n_new.sum() == 0: return float("nan")
    return float(tl.n_zero_new.sum() / tl.n_new.sum())


def paired(a, b, seed=0):
    """Δ = a − b ต่อ k ที่มีทั้งคู่"""
    from scipy.stats import wilcoxon
    ks = a.index.intersection(b.index); d = (a.loc[ks] - b.loc[ks]).to_numpy()
    rng = np.random.RandomState(seed); bs = np.array([rng.choice(d, len(d)).mean() for _ in range(2000)])
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return dict(n=len(d), delta=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), p=p, wins=int((d > 0).sum()), losses=int((d < 0).sum()))


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj


def parse(name):
    fam, _, par = name.partition("_")
    if par == "none": return fam, 0.0
    try: return fam, float(par.lstrip("kdt"))
    except ValueError: return fam, None


def do_dev(a):
    root = Path(a.root); names = sorted(p.name for p in root.iterdir() if p.is_dir())
    rows = []
    for n in names:
        fam, par = parse(n)
        for c in CITIES:
            d, tl = load(root / n, c)
            if d is None: continue
            rows.append(dict(name=n, fam=fam, param=par, city=c, n=len(d), f1=d.f1_real.mean(), f2=d.f2_real.mean(), f3=d.f3_real.mean(), quota=d.quota_used_final.mean(), zero_share=zshare(tl)))
    T = pd.DataFrame(rows)
    if T.empty: sys.exit(f"ไม่พบผลใน {root}")
    lines = [f"# ชุดป้องกัน gate — dev · {VERSION}", "", "sticky · NSGA-II pop 80 × gen 480 · arm qm_model · ค่าเฉลี่ยต่อ instance (SF 30, Rome 20) · zero = สัดส่วน offer ใหม่ในแผนที่ C = 0 (ก่อน drop)", ""]
    W = T.pivot_table(index=["fam", "param", "name"], columns="city", values=["f1", "quota", "zero_share"]).reset_index()
    lines += ["| แขน | พารามิเตอร์ | f₁* SF | f₁* Rome | เฉลี่ย | โควตา SF / Rome | zero SF / Rome |", "|---|---|---|---|---|---|---|"]
    sel = {}
    for fam, g in T.groupby("fam", sort=False):
        piv = g.pivot_table(index=["param", "name"], columns="city", values="f1").reindex(columns=CITIES)
        piv["avg"] = piv.mean(axis=1, skipna=False)
        for (par, n), r in piv.sort_index().iterrows():
            q = g[g.name == n].set_index("city"); fmt = lambda c, col, f="{:.3f}": (f.format(q.loc[c, col]) if c in q.index and pd.notna(q.loc[c, col]) else "—")
            lines.append(f"| {LABEL.get(fam, fam)} | {'—' if fam in ('grst', 'grfree') else f'{par:g}'} | {fmt('sf', 'f1')} | {fmt('rome', 'f1')} | {r['avg']:.3f} | {fmt('sf', 'quota', '{:.2f}')} / {fmt('rome', 'quota', '{:.2f}')} | {fmt('sf', 'zero_share', '{:.1%}')} / {fmt('rome', 'zero_share', '{:.1%}')} |")
        if fam in FAMS and piv["avg"].notna().any():
            best = piv["avg"].astype(float).idxmax(); sel[fam] = dict(name=best[1], param=best[0], f1_avg=float(piv.loc[best, "avg"]))
    lines += ["", "## ค่าที่เลือก (กฎ: f₁* เฉลี่ยสองเมืองสูงสุด · ล็อกก่อนรัน final)", ""] + [f"- {LABEL[f]}: **{v['param']:g}** (`{v['name']}`, เฉลี่ย {v['f1_avg']:.3f})" for f, v in sel.items()]
    missing = [f for f in FAMS if f not in sel]
    if missing: lines.append(f"- ⚠️ ยังไม่มีผล: {missing}")
    # repro ของ B0 เดิม
    if a.b0_old:
        lines += ["", "## repro B0 เดิม (v1.0 ไม่ใส่ flag ใหม่ ต้องเท่าผล dev เดิมทุก instance)", ""]
        for c in CITIES:
            new, tln = load(root / "b0_k1.5", c); old, _ = load(Path(a.b0_old), c)
            if new is None or old is None: lines.append(f"- {c}: ไม่มีไฟล์"); continue
            ks = new.index.intersection(old.index); dd = (new.loc[ks, "f1_real"] - old.loc[ks, "f1_real"]).abs().max()
            lines.append(f"- {c}: n {len(ks)} · max|Δf₁*| {dd:.6f} {'✅' if dd == 0 else '❌ ไม่เท่า — หยุด หาสาเหตุ'} · offer ใหม่ที่ C = 0 ใน B0 เดิม {zshare(tln):.1%}")
    (root / "GATEDEF_select.json").write_text(json.dumps(dict(version=VERSION, rule="max mean f1* over sf, rome", select=sel), ensure_ascii=False, indent=1), encoding="utf-8")
    (root / "GATEDEF_dev.md").write_text("\n".join(lines) + "\n", encoding="utf-8"); print("\n".join(lines))


def do_final(a):
    root = Path(a.final); sel = json.load(open(a.select, encoding="utf-8"))["select"]
    if "b0dz" not in sel: sys.exit("select ไม่มี b0dz")
    lines = [f"# ชุดป้องกัน gate — final (test, รันครั้งเดียว) · {VERSION}", "", f"ค่าที่ล็อกจาก dev: " + " · ".join(f"{LABEL[f]} = {v['param']:g}" for f, v in sel.items()), "",
             "Δ = แขน − B0 (paired ต่อ instance) · CI bootstrap 95 % · p = Wilcoxon · p_H = Holm (3 การเทียบต่อเมือง)", ""]
    for metric, mname in [("f1_real", "f₁*"), ("f2_real", "f₂*"), ("f3_real", "f₃*")]:
        lines += [f"## {mname}", "", "| แขน | เมือง | ค่าเฉลี่ย | Δ vs B0 [95 % CI] | W/L | p | p_H |", "|---|---|---|---|---|---|---|"]
        for c in CITIES:
            ref, _ = load(root / sel["b0dz"]["name"], c)
            if ref is None: lines.append(f"| — | {c} | ไม่มี B0 | | | | |"); continue
            lines.append(f"| {LABEL['b0dz']} (κ {sel['b0dz']['param']:g}) | {c} | {ref[metric].mean():.3f} | — | | | |")
            res = []
            for f in ["g1dz", "g2dz", "g3dz"]:
                if f not in sel: continue
                d, _ = load(root / sel[f]["name"], c)
                if d is None: continue
                res.append((f, d[metric].mean(), paired(d[metric], ref[metric])))
            adj = holm([r[2]["p"] for r in res]) if res else []
            for (f, m_, s), ph in zip(res, adj):
                lines.append(f"| {LABEL[f]} ({sel[f]['param']:g}) | {c} | {m_:.3f} | {s['delta']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}] | {s['wins']}/{s['losses']} | {s['p']:.3g} | {ph:.3g}{' ✱' if ph < 0.05 else ''} |")
        lines.append("")
    if a.b0_final:
        lines += ["## B0 + drop-zero เทียบ B0 final เดิม (ผลของการแก้ offer ที่ C = 0)", ""]
        for c in CITIES:
            new, tln = load(root / sel["b0dz"]["name"], c); old, tlo = load(Path(a.b0_final), c)
            if new is None or old is None: continue
            s = paired(new.f1_real, old.f1_real)
            lines.append(f"- {c}: B0-dz {new.f1_real.mean():.3f} vs B0 เดิม {old.f1_real.mean():.3f} · Δ {s['delta']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}] · {s['wins']}/{s['losses']} · p {s['p']:.3g} · โควตา {new.quota_used_final.mean():.2f} vs {old.quota_used_final.mean():.2f}")
    # ---- greedy baseline (κ ของ greedy + gate = κ ที่เลือกให้ B0-dz · ประกาศล่วงหน้า ไม่จูนแยก)
    kb = sel["b0dz"]["param"]; gname = f"grdz_k{kb:g}" if (root / f"grdz_k{kb:g}").exists() else f"grdz_k{kb}"
    comps = [("Greedy vs NSGA-II (sticky, ไม่มี gate)", root / "grst_none", Path(a.nsga_sticky) if a.nsga_sticky else None),
             ("Greedy + gate vs Greedy (ผลของ gate บน greedy)", root / gname, root / "grst_none"),
             ("Greedy + gate vs NSGA-II + gate (B0-dz)", root / gname, root / sel["b0dz"]["name"]),
             ("Greedy vs NSGA-II (free, ไม่มี gate)", root / "grfree_none", Path(a.nsga_free) if a.nsga_free else None)]
    lines += ["", f"## Greedy baseline (κ = {kb:g} จาก B0-dz) · Δ = ตัวแรก − ตัวหลัง · Holm 4 การเทียบต่อเมือง/ตัวชี้วัด", ""]
    for metric, mname in [("f1_real", "f₁*"), ("f2_real", "f₂*"), ("f3_real", "f₃*")]:
        lines += [f"### {mname}", "", "| การเทียบ | เมือง | ค่า (ตัวแรก / ตัวหลัง) | Δ [95 % CI] | W/L | p_H |", "|---|---|---|---|---|---|"]
        for c in CITIES:
            res = []
            for lab, fa, fb in comps:
                if fb is None: continue
                da, _ = load(fa, c); db, _ = load(fb, c)
                if da is None or db is None: continue
                res.append((lab, da[metric].mean(), db[metric].mean(), paired(da[metric], db[metric])))
            adj = holm([r[3]["p"] for r in res]) if res else []
            for (lab, ma, mb, s_), ph in zip(res, adj):
                lines.append(f"| {lab} | {c} | {ma:.3f} / {mb:.3f} | {s_['delta']:+.3f} [{s_['lo']:+.3f}, {s_['hi']:+.3f}] | {s_['wins']}/{s_['losses']} | {ph:.3g}{' ✱' if ph < 0.05 else ''} |")
        lines.append("")
    out = root / "GATEDEF_final.md"; out.write_text("\n".join(lines) + "\n", encoding="utf-8"); print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=""); ap.add_argument("--b0-old", default=""); ap.add_argument("--final", default=""); ap.add_argument("--select", default=""); ap.add_argument("--b0-final", default=""); ap.add_argument("--nsga-sticky", default=""); ap.add_argument("--nsga-free", default="")
    a = ap.parse_args()
    if a.final:
        if "instances_devsplit" in a.final: sys.exit("--final ต้องเป็นผลบน instances_v3")
        do_final(a)
    elif a.root: do_dev(a)
    else: sys.exit("ต้องมี --root (dev) หรือ --final")


if __name__ == "__main__":
    main()
