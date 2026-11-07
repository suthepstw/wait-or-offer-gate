#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sE6_nullgate.py (SPEC #6 · v0.1, 2026-10-24) — gate + ตัวทำนายแบบง่าย (physics null, qm_null_dv) เทียบ gate + ตัวทำนายของเรา (qm_model)
=================================================================================================================
คำถาม : เมื่อมี gate แล้ว ตัวทำนายที่เรียนรู้ (Part 1) ยังให้ผลดีกว่าตัวทำนายฟิสิกส์ที่ปรับเทียบแล้วไหม
แขน   : null_none (NSGA-II, sticky, ไม่มี C̃/gate) · nullg_kκ (NSGA-II + C̃_null + gate + drop-zero · ρ, V ของ null จาก dev: e6_rho_null)
dev   : เลือก κ ของ null = f₁* เฉลี่ยสองเมืองสูงสุด (grid เดียวกับ B0-dz) → <root>/NULLGATE_select.json · NULLGATE_dev.md
final : (1) ของเรา + gate (e6gd_final/b0dz_k1.5) vs null + gate  ← คำถามหลัก
        (2) null + gate vs null ไม่มี gate (gate ช่วย null ไหม)
        (3) ของเรา ไม่มี gate (e6final_sticky/nsga2) vs null ไม่มี gate
        paired bootstrap 95 % CI · Wilcoxon · Holm 3 การเทียบต่อเมือง/ตัวชี้วัด → <final>/NULLGATE_final.md
ใช้:  python sE6_nullgate.py --root e6ng_dev --model-gate e6gd_dev/b0dz_k1.5 --model-none e6dev_sticky/nsga2
      python sE6_nullgate.py --final e6ng_final --select e6ng_dev/NULLGATE_select.json --model-gate e6gd_final/b0dz_k1.5 --model-none e6final_sticky/nsga2
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sE6_gatedef import paired, holm, CITIES

VERSION = "E6-nullgate v0.1 (2026-10-24)"


def load(folder: Path, city: str, arm: str):
    r = folder / city / "e5_runs.csv"
    if not r.exists(): return None
    d = pd.read_csv(r); d = d[(d.arm == arm) & (d["mode"] == "adaptive")].set_index("k").sort_index()
    return d if len(d) else None


def do_dev(a):
    root = Path(a.root); rows = []
    for p in sorted(x for x in root.iterdir() if x.is_dir()):
        n = p.name; kap = None if n == "null_none" else float(n.split("_k")[1])
        for c in CITIES:
            d = load(p, c, "qm_null_dv")
            if d is not None: rows.append(dict(name=n, kappa=kap, city=c, f1=d.f1_real.mean(), f2=d.f2_real.mean(), f3=d.f3_real.mean(), quota=d.quota_used_final.mean()))
    T = pd.DataFrame(rows)
    if T.empty: sys.exit(f"ไม่พบผลใน {root}")
    lines = [f"# gate + ตัวทำนายแบบง่าย (physics null) — dev · {VERSION}", "", "sticky · NSGA-II pop 80 × gen 480 · arm qm_null_dv · drop-zero ทุกแขนที่มี gate", "",
             "| แขน | κ | f₁* SF | f₁* Rome | เฉลี่ย | f₂* SF / Rome | โควตา SF / Rome |", "|---|---|---|---|---|---|---|"]
    best = None
    for n, g in T.groupby("name", sort=False):
        q = g.set_index("city"); v = lambda c, col, f="{:.3f}": f.format(q.loc[c, col]) if c in q.index else "—"
        avg = q.f1.mean() if len(q) == 2 else float("nan")
        lines.append(f"| {'null ไม่มี gate' if n == 'null_none' else 'null + C̃ + gate'} | {'—' if n == 'null_none' else f'{g.kappa.iloc[0]:g}'} | {v('sf', 'f1')} | {v('rome', 'f1')} | {avg:.3f} | {v('sf', 'f2')} / {v('rome', 'f2')} | {v('sf', 'quota', '{:.2f}')} / {v('rome', 'quota', '{:.2f}')} |")
        if n != "null_none" and np.isfinite(avg) and (best is None or avg > best[2]): best = (n, float(g.kappa.iloc[0]), avg)
    for lab, path, arm in [("ของเรา + gate (B0-dz κ 1.5)", a.model_gate, "qm_model"), ("ของเรา ไม่มี gate", a.model_none, "qm_model")]:
        if not path: continue
        vals = [load(Path(path), c, arm) for c in CITIES]
        if all(x is not None for x in vals): lines.append(f"| {lab} (อ้างอิง) | — | {vals[0].f1_real.mean():.3f} | {vals[1].f1_real.mean():.3f} | {np.mean([x.f1_real.mean() for x in vals]):.3f} | {vals[0].f2_real.mean():.3f} / {vals[1].f2_real.mean():.3f} | — |")
    if best: lines += ["", f"## κ ที่เลือกให้ null (กฎ: f₁* เฉลี่ยสองเมืองสูงสุด · ล็อกก่อน final): **{best[1]:g}** (`{best[0]}`, เฉลี่ย {best[2]:.3f})"]
    (root / "NULLGATE_select.json").write_text(json.dumps(dict(version=VERSION, name=best[0], kappa=best[1], f1_avg=best[2]) if best else {}, ensure_ascii=False, indent=1), encoding="utf-8")
    (root / "NULLGATE_dev.md").write_text("\n".join(lines) + "\n", encoding="utf-8"); print("\n".join(lines))


def do_final(a):
    root = Path(a.final); sel = json.load(open(a.select, encoding="utf-8"))
    comps = [("ของเรา + gate vs null + gate  (คำถามหลัก)", Path(a.model_gate), "qm_model", root / sel["name"], "qm_null_dv"),
             ("null + gate vs null ไม่มี gate", root / sel["name"], "qm_null_dv", root / "null_none", "qm_null_dv"),
             ("ของเรา ไม่มี gate vs null ไม่มี gate", Path(a.model_none), "qm_model", root / "null_none", "qm_null_dv")]
    lines = [f"# gate + ตัวทำนายแบบง่าย — final (test, รันครั้งเดียว) · {VERSION}", "", f"κ ของ null ที่ล็อกจาก dev = {sel['kappa']:g} · ของเรา κ = 1.5 · Δ = ตัวแรก − ตัวหลัง · Holm 3 การเทียบต่อเมือง/ตัวชี้วัด", ""]
    for metric, mname in [("f1_real", "f₁*"), ("f2_real", "f₂*"), ("f3_real", "f₃*")]:
        lines += [f"## {mname}", "", "| การเทียบ | เมือง | ค่า (ตัวแรก / ตัวหลัง) | Δ [95 % CI] | W/L | p | p_H |", "|---|---|---|---|---|---|---|"]
        for c in CITIES:
            res = []
            for lab, fa, aa, fb, ab in comps:
                da, db = load(fa, c, aa), load(fb, c, ab)
                if da is None or db is None: lines.append(f"| {lab} | {c} | ไม่มีไฟล์ | | | | |"); continue
                res.append((lab, da[metric].mean(), db[metric].mean(), paired(da[metric], db[metric])))
            for (lab, ma, mb, s), ph in zip(res, holm([r[3]["p"] for r in res]) if res else []):
                lines.append(f"| {lab} | {c} | {ma:.3f} / {mb:.3f} | {s['delta']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}] | {s['wins']}/{s['losses']} | {s['p']:.3g} | {ph:.3g}{' ✱' if ph < 0.05 else ''} |")
        lines.append("")
    (root / "NULLGATE_final.md").write_text("\n".join(lines) + "\n", encoding="utf-8"); print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=""); ap.add_argument("--final", default=""); ap.add_argument("--select", default=""); ap.add_argument("--model-gate", default=""); ap.add_argument("--model-none", default="")
    a = ap.parse_args()
    if a.final: do_final(a)
    elif a.root: do_dev(a)
    else: sys.exit("ต้องมี --root (dev) หรือ --final")


if __name__ == "__main__":
    main()
