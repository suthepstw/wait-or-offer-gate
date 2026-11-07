#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sX_compare_runs.py — รวมผลเปรียบเทียบ encoder (Bi-LSTM / Bi-GRU / Bi-RNN / Transformer / ST-Transformer) เป็นตารางเดียว
  อ่าน  models/<city>/<run>/summary.json                 (D5: params, เวลา, minADE/ADE_π/FDE_π/NLL/hit@R บน test)
        models/<city>/<run>/D7_dev/pred_metrics.csv      (D7 v7 split dev: ADE/FDE รายขอบฟ้า, AUC/AUPRC/ECE ของ qm_ub_cal, null gauss_dv_cal)
        models/<city>/<run>/D7_dev/G1.json               (gate G1 + CI)
  เขียน reports/encoder_compare_<city>.md  และพิมพ์ตารางบนจอ
  ใช้:  python sX_compare_runs.py --city sf --runs base_M3,abl_gru,abl_rnn,abl_tfm,abl_sttfm,abl_sttfm_L
        python sX_compare_runs.py --city sf,rome                      (ค่าเริ่มต้น = ทุก run ที่มี D7_dev)
หมายเหตุ: ประเมินบน split **dev** เท่านั้น (final สงวนไว้สำหรับรายงานหลัก) · mean ± std ข้าม seed · CI = cluster bootstrap จาก D7
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
LABEL = {"base_M3": "Bi-LSTM M=3 skip (หลัก)", "abl_uni": "LSTM ทางเดียว", "abl_gru": "Bi-GRU", "abl_rnn": "Bi-RNN (vanilla)",
         "abl_M1": "M = 1 (single mode)", "abl_M5": "M = 5", "abl_noskip": "ไม่มี damped-vel skip", "abl_occ": "+ occ feature", "abl_novel": "ไม่มีฟีเจอร์ความเร็ว (v,dx,dy) + no-skip",
         "abl_tfm": "Transformer (temporal)", "abl_sttfm": "Transformer + spatial enc.", "abl_sttfm_L": "Transformer-L (2.7 M, + spatial enc.)"}
R_MAIN = 200.0

def fm(mean, std=None, nd=3):
    if mean is None or (isinstance(mean, float) and np.isnan(mean)): return "—"
    return f"{mean:.{nd}f}" + (f" ± {std:.{nd}f}" if std is not None and np.isfinite(std) else "")

def load_run(city: str, run: str, models: Path, split: str) -> dict | None:
    rd = models / city / run
    sj = rd / "summary.json"
    if not sj.exists(): return None
    S = json.load(open(sj, encoding="utf-8")); m, s = S["mean"], S["std"]
    row = {"run": run, "encoder": LABEL.get(run, run), "n_seeds": len(S.get("seeds", [])),
           "params_M": m.get("params", np.nan) / 1e6, "s_per_epoch": m.get("s_per_epoch", np.nan), "best_epoch": m.get("best_epoch", np.nan),
           "infer_ms_1k": m.get("infer_ms_per_1k_windows", np.nan),
           "minADE": (m.get("test_minADE"), s.get("test_minADE")), "ADE_pi": (m.get("test_ADE_pi"), s.get("test_ADE_pi")),
           "FDE_pi": (m.get("test_FDE_pi"), s.get("test_FDE_pi")), "NLL": (m.get("test_nll"), s.get("test_nll")),
           "hitR_pi": (m.get("test_hitR_pi"), s.get("test_hitR_pi"))}
    # run_config → รายละเอียดสถาปัตย์
    rc = rd / "run_config.json"
    if rc.exists():
        c = json.load(open(rc, encoding="utf-8")); row["arch"] = f"{c.get('rnn')}{' bi' if c.get('bi') and c.get('rnn') in ('lstm','gru','rnn') else ''}"
        if c.get("rnn") in ("tfm", "sttfm"): row["arch"] += f" d{c.get('tfm_d', 128)}×{c.get('tfm_blocks', 2)} h{c.get('tfm_heads', 4)}"
        row["arch"] += f" M{c.get('M', 3)}" + ("" if c.get("skip", True) else " noskip")
    # D7 dev
    d7 = rd / f"D7_{split}"
    if (d7 / "pred_metrics.csv").exists():
        df = pd.read_csv(d7 / "pred_metrics.csv")
        seeds = df[df.seed != "all"]
        def g(var, met, R=None, h=None):
            q = seeds[(seeds.variant == var) & (seeds.metric == met)]
            if R is not None: q = q[np.isclose(q.R.astype(float), R)]
            if h is not None: q = q[q.h == h]
            v = q.groupby("seed")["value"].mean()
            return (float(v.mean()), float(v.std(ddof=0))) if len(v) else (np.nan, np.nan)
        def g_all(var, met, R=None):
            q = df[(df.seed == "all") & (df.variant == var) & (df.metric == met)]
            if R is not None: q = q[np.isclose(q.R.astype(float), R)]
            return float(q.value.iloc[0]) if len(q) else np.nan
        row.update({"d7_ADE_pi": g("model_pi", "ADE"), "d7_FDE_pi": g("model_pi", "FDE"), "d7_minADE": g("model_min", "ADE"),
                    "d7_pers_ADE": g("persistence", "ADE"), "d7_NLL_cal": g("model_cal", "NLL"), "d7_ECE_pi": g("model_cal", "ECE_pi"),
                    "AUC": g("qm_ub_cal", "AUC", R_MAIN), "AUPRC": g("qm_ub_cal", "AUPRC", R_MAIN), "ECE_qm": g("qm_ub_cal", "ECE_qm", R_MAIN),
                    "AUC_hard": g("qm_ub_cal", "AUC_hard", R_MAIN), "AUC_null": g("gauss_dv_cal", "AUC", R_MAIN),
                    "AUPRC_pers": g("persistence", "AUPRC", R_MAIN),
                    "AUC_ci": (g_all("qm_ub_cal", "AUC_ci_lo", R_MAIN), g_all("qm_ub_cal", "AUC_ci_hi", R_MAIN)),
                    "latency_ms_1k": g("model_pi", "latency_ms_per_1k")[0]})
        row["AUC_gain"] = row["AUC"][0] - row["AUC_null"][0]
        row["lift"] = row["AUPRC"][0] / row["AUPRC_pers"][0] if row["AUPRC_pers"][0] > 0 else np.nan
        if (d7 / "G1.json").exists():
            G = json.load(open(d7 / "G1.json", encoding="utf-8")); row["G1"] = "PASS" if G.get("PASS") else "FAIL"
            row["G1_parts"] = "".join("✔" if G.get(k) else "✘" for k in ("a_pass", "b_pass", "c_pass", "d_pass"))
    return row

def build_tables(rows: list[dict], city: str, split: str) -> str:
    ref = next((r for r in rows if r["run"] == "base_M3"), None)
    L = [f"# เปรียบเทียบ run · {city} · D7 split = **{split}**", "",
         "ทุก run: ข้อมูล D4 · loss ε-WTA · D6 (λ_h, T, isotonic) · D7 v7 เดียวกัน → ต่างจาก base_M3 ที่ส่วนประกอบเดียวตามคอลัมน์ variant  ",
         "mean ± std ข้าม seed · AUC/AUPRC/ECE ของ **qm_ub_cal** ที่ R = 200 m, pt = 60 s · CI = cluster bootstrap (cab×วัน) จาก D7  ",
         "อ่านค่า: ADE/FDE ต่ำ = ดี · NLL ต่ำ = ดี · AUC/AUPRC สูง = ดี · ECE ต่ำ = ดี · gain = AUC − AUC(null gauss_dv_cal)", ""]
    # ตาราง 1: ขนาด/ต้นทุน
    L += ["## T1 · สถาปัตย์และต้นทุน", "| run | variant | สถาปัตย์ | seeds | params (M) | s/epoch | best epoch | infer ms/1k windows |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['run']} | {r['encoder']} | {r.get('arch','—')} | {r['n_seeds']} | {r['params_M']:.2f} | {r['s_per_epoch']:.0f} | {r['best_epoch']:.0f} | {r['infer_ms_1k']:.1f} |")
    # ตาราง 2: ความแม่นของวิถี (D5 test)
    L += ["", "## T2 · ความแม่นของวิถี (D5 · test ทั้งก้อน · เมตร)", "| run | minADE | ADE_π | FDE_π | NLL | hit@R(π) |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['run']} | {fm(*r['minADE'], nd=0)} | {fm(*r['ADE_pi'], nd=0)} | {fm(*r['FDE_pi'], nd=0)} | {fm(*r['NLL'], nd=3)} | {fm(*r['hitR_pi'], nd=3)} |")
    # ตาราง 3: D7 dev — สิ่งที่ allocation ใช้จริง
    L += ["", f"## T3 · Reachability / QM (D7 · split {split} · R = 200 m) — สิ่งที่สาย E ใช้จริง",
          "| run | ADE_π (D7) | NLL_cal | AUC qm_ub_cal | 95% CI | AUC null | gain | AUPRC | lift vs persistence | ECE_qm | AUC hard | G1 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "AUC" not in r: L.append(f"| {r['run']} | — | — | — | — | — | — | — | — | — | — | (ยังไม่มี D7_{split}) |"); continue
        ci = r["AUC_ci"]; ci_txt = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if np.isfinite(ci[0]) else "—"
        L.append(f"| {r['run']} | {fm(*r['d7_ADE_pi'], nd=0)} | {fm(*r['d7_NLL_cal'], nd=3)} | **{fm(*r['AUC'])}** | {ci_txt} | {fm(*r['AUC_null'])} | {r['AUC_gain']:+.3f} | {fm(*r['AUPRC'])} | {r['lift']:.1f}× | {fm(*r['ECE_qm'], nd=4)} | {fm(*r['AUC_hard'])} | {r.get('G1','—')} {r.get('G1_parts','')} |")
    # ตาราง 4: ต่างจาก Bi-LSTM
    if ref is not None and "AUC" in ref:
        L += ["", "## T4 · ผลต่างจาก Bi-LSTM (base_M3) — บวก = ดีกว่า LSTM", "| run | ΔADE_π (m, ลด = บวก) | ΔNLL (ลด = บวก) | ΔAUC | ΔAUPRC | ΔECE_qm (ลด = บวก) | params ×LSTM | เวลา/epoch ×LSTM |", "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["run"] == "base_M3" or "AUC" not in r: continue
            L.append(f"| {r['run']} | {ref['d7_ADE_pi'][0]-r['d7_ADE_pi'][0]:+.0f} | {ref['d7_NLL_cal'][0]-r['d7_NLL_cal'][0]:+.3f} | {r['AUC'][0]-ref['AUC'][0]:+.3f} | {r['AUPRC'][0]-ref['AUPRC'][0]:+.3f} | {ref['ECE_qm'][0]-r['ECE_qm'][0]:+.4f} | {r['params_M']/ref['params_M']:.2f} | {r['s_per_epoch']/ref['s_per_epoch']:.2f} |")
        L += ["", "เกณฑ์อ่าน: |ΔAUC| < ครึ่งหนึ่งของความกว้าง CI ของ LSTM ≈ ไม่ต่างอย่างมีนัย (CI ซ้อนกัน) · ถ้า encoder อื่นชนะทั้ง ADE และ AUC พร้อมกันจึงคุ้มเปลี่ยน — ถ้าชนะแค่ ADE แต่ AUC เท่ากัน ให้คง Bi-LSTM (สิ่งที่ allocation ใช้คือ AUC/ECE ของ C_ij ไม่ใช่ ADE)"]
    return "\n".join(L) + "\n"

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", default="sf"); ap.add_argument("--runs", default="", help="คั่นด้วย , · ว่าง = ทุก run ที่มี D7_<split>")
    ap.add_argument("--models", default="models"); ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default="", help="ไฟล์ .md ปลายทาง (ค่าเริ่มต้น reports/encoder_compare_<city>.md) — ใช้ได้เมื่อระบุเมืองเดียว")
    a = ap.parse_args(); models = Path(a.models) if Path(a.models).is_absolute() else HERE / a.models
    (HERE / "reports").mkdir(exist_ok=True)
    for city in [c.strip() for c in a.city.split(",") if c.strip()]:
        if a.runs: runs = [r.strip() for r in a.runs.split(",") if r.strip()]
        else: runs = sorted(p.parents[1].name for p in (models / city).glob(f"*/D7_{a.split}/pred_metrics.csv"))   # <run>/D7_<split>/G1.json
        if "base_M3" in runs: runs = ["base_M3"] + [r for r in runs if r != "base_M3"]
        rows = [r for r in (load_run(city, run, models, a.split) for run in runs) if r is not None]
        if not rows: print(f"{city}: ไม่พบ run"); continue
        txt = build_tables(rows, city, a.split)
        out = (HERE / a.out if a.out and not Path(a.out).is_absolute() else Path(a.out)) if a.out else HERE / "reports" / f"encoder_compare_{city}.md"
        out.parent.mkdir(parents=True, exist_ok=True); out.write_text(txt, encoding="utf-8")
        print(txt); print(f"→ {out}")

if __name__ == "__main__":
    main()
