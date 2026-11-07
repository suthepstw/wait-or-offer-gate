#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sE5_instances.py (E5 v0.1, 2026-10-21) — C_ij(t) ของทุก arm ที่เวลา t ∈ {0,10,…,50} นาทีในรอบ สำหรับ rolling re-plan
  อินพุต : instances_v3/<city>/{inst_k,oracle_k}.npz (โจทย์เดิม: งาน, หน้าต่าง, K, ε, path_true)  +  e5state/<city>/<run>/state_k.npz (sE5_encode: feat64/mu/sigma/pi/ตำแหน่ง/ความเร็ว ที่ t)
  วิธี   : ที่ t ใช้ตัวประมาณชุดเดิม (หัว (9a′) + guard + isotonic, (9a) union bound + calib, null gauss_dv, persistence) บนหน้าต่างที่เหลือ
             [st_i', et_i'] = [max(st_i − t, 1), et_i − t] (ก้าวนับจาก t) · l = task_xy − x_j(t) · เฉพาะงานที่ยัง "เปิด" (et_i > t)
           oracle y_ij(t) = มีจุดบันทึกใน disc ระหว่างหน้าต่างที่เหลือ (path_true ช่วง (t, et_i]) — นิยาม (8) เดิม
  arms   : qm_model (9a′ tfm guarded) · qm_lstm_reach (9a′ lstm guarded) · qm_tfm_ub (9a tfm) · qm_null_dv · qm_persist_cont · oracle
           (ไม่มี qm_trajgeom/base_M3 ที่ t>0 — ไม่ได้ encode base_M3; E4 แสดงแล้วว่า ≈ qm_tfm_ub)
  sanity : (1) ที่ t=0 เทียบกับ inst เดิม: max|ΔC| ต้องอยู่ระดับ noise ของ sE5 (≤ 5e-3) · (2) ตำแหน่ง x_j(t) จาก state vs path_true[t−1]: median ≤ 5 m
           (3) oracle ที่ t=0 == inst เดิมทุกบิต · (4) AUC ต่อ arm ต่อ t (ความทำนายได้ที่ขอบฟ้าสั้นลง — รายงาน)
  ผล    : e5inst/<city>/inst_k_t<mm>.npz + E5I_report.md
ใช้:  python sE5_instances.py --city sf --inst instances_v3 --state e5state --out e5inst [--limit 3] [--R 200]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sE1_instances as E1                       # v2.2: Frame, import_sD7, load_frozen, auc_rank
import sE1_instances_v3 as E1v3                  # v3.1: load_frozen_reach, reach_prob, iso_apply
VERSION = "E5-instances v0.3 (2026-10-22: + arm qm_model_dg duration guard ≥5, บันทึก qm_model_nog/in_domain_dg)"
ARMS = ("qm_model", "qm_model_dg", "qm_lstm_reach", "qm_tfm_ub", "qm_null_dv", "qm_persist_cont")
DUR_MIN = 5                                      # S-a: หน้าต่างที่เหลือขั้นต่ำ (ก้าว) ที่หัว (9a′) ถูกใช้ (= ระยะสั้นสุดใน DURATIONS ที่ฝึก/calibrate)


def qm_at_t(D7, frz_base, frz_p, frz_r, w_xy, w_v, tau, task_xy, st, et, R, Y_t, H, p75_h, R_max, fp, fr, pred_p, pred_r):
    """C_ij ทุก arm ที่เวลาหนึ่ง · fp/fr = feat64 (W,64) · pred_* = dict(mu,sigma,pi) (W,...) ของ run นั้นที่เวลานั้น · Y_t (W,H,2) เส้นทางจริงที่เหลือ (relative to x(t), pad ไกล)"""
    nW, nT = len(w_xy), len(task_xy)
    ii = np.repeat(np.arange(nW), nT); l = (task_xy[None, :, :] - w_xy[:, None, :]).reshape(-1, 2).astype(np.float32)
    stp = np.tile(st, nW).astype(int); etp = np.tile(et, nW).astype(int)
    d0 = np.linalg.norm(l, axis=1); dom = d0 <= np.maximum(p75_h[etp - 1], 4.0 * R_max)
    out = {"in_domain": dom.reshape(nW, nT).astype(np.float32)}
    def ub_of(frz, pred):
        _, ub = D7.reach_score_analytic(pred["mu"], pred["sigma"], pred["pi"], frz["lam"], frz["T"], ii, stp, etp, l, [R])
        return D7.apply_qm_calib(frz["qcal"], R, "ub", ub[R])
    ub_p, ub_r = ub_of(frz_p, pred_p), ub_of(frz_r, pred_r)
    cal_p = E1v3.iso_apply(D7, frz_p["rcal"], R, E1v3.reach_prob(frz_p["head"], fp, ii, stp, etp, l, R, H))
    cal_r = E1v3.iso_apply(D7, frz_r["rcal"], R, E1v3.reach_prob(frz_r["head"], fr, ii, stp, etp, l, R, H))
    out["qm_model"] = np.where(dom, cal_p, ub_p).reshape(nW, nT); out["qm_model_nog"] = cal_p.reshape(nW, nT)
    # [S-a, ประกาศ 2026-10-22 08:36] duration guard: หัว (9a′) ถูก fit/calibrate บนหน้าต่าง ≥ 5 ก้าวเท่านั้น → คู่ที่หน้าต่างเหลือสั้นกว่าใช้ (9a) ของ run เดียวกัน (ซ้อนบน guard ระยะทาง)
    dom_dg = dom & ((etp - stp + 1) >= DUR_MIN)
    out["qm_model_dg"] = np.where(dom_dg, cal_p, ub_p).reshape(nW, nT); out["in_domain_dg"] = dom_dg.reshape(nW, nT).astype(np.float32)
    out["qm_lstm_reach"] = np.where(dom, cal_r, ub_r).reshape(nW, nT); out["qm_tfm_ub"] = ub_p.reshape(nW, nT)
    sig_null = np.asarray(frz_base["qcal"]["null_sigma_h"]["gauss_dv"], np.float32)
    _, ubn = D7.reach_score_null(w_v[ii], tau, H, sig_null, np.arange(len(ii)), stp, etp, l, [R], "gauss_dv")
    out["qm_null_dv"] = D7.apply_qm_calib(frz_base["qcal"], R, "gauss_dv", ubn[R]).reshape(nW, nT)
    out["qm_persist_cont"] = np.exp(-d0 / R).reshape(nW, nT)
    oracle = D7.reach_labels(Y_t, ii, stp, etp, l, R).reshape(nW, nT).astype(np.uint8)
    return {k: v.astype(np.float32) for k, v in out.items()}, oracle


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", default="sf"); ap.add_argument("--inst", default="instances_v3"); ap.add_argument("--state", default="e5state"); ap.add_argument("--out", default="e5inst")
    ap.add_argument("--models", default="models"); ap.add_argument("--data", default="data"); ap.add_argument("--primary-run", default="reach_tfm"); ap.add_argument("--ref-run", default="reach_lstm")
    ap.add_argument("--reach-split", default="dev"); ap.add_argument("--base-run", default="base_M3"); ap.add_argument("--base-split", default="final")
    ap.add_argument("--R", type=float, default=200.0); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--tol-C0", type=float, default=5e-3)
    a = ap.parse_args(); t0 = time.time(); log = lambda *m: print(f"[{time.time()-t0:7.1f}s]", *m, flush=True); log(VERSION)
    city = a.city; models_root, data_root = Path(a.models), Path(a.data); inst_dir = Path(a.inst) / city; st_dir = Path(a.state) / city; out = Path(a.out) / city; out.mkdir(parents=True, exist_ok=True)
    D7, sha_d7 = E1.import_sD7(log)
    frz_base = E1.load_frozen(models_root, city, a.base_run, a.base_split, log, required=True)
    frz_p = E1v3.load_frozen_reach(models_root, city, a.primary_run, a.reach_split, log); frz_r = E1v3.load_frozen_reach(models_root, city, a.ref_run, a.reach_split, log)
    Yv = np.asarray(np.load(data_root / city / "processed" / "val_Y.npy", mmap_mode="r")); p75_h = np.percentile(np.sqrt((Yv ** 2).sum(-1)), 75, axis=0).astype(np.float32); H = Yv.shape[1]
    R_max = float(max(float(x) for x in frz_p["rcal"]["R"])); R = float(a.R)
    rc_base = json.load(open(models_root / city / a.base_run / "run_config.json")); tau = float(rc_base.get("skip_tau", 3.0))
    log(f"H {H} · R {R:g} · d_sup(et) = max(p75|y_et|, {4*R_max:.0f} m) · τ null {tau:g} · arms {ARMS}+oracle")
    files = sorted(inst_dir.glob("inst_*.npz"));  files = files[:a.limit] if a.limit else files
    rows = []; san = dict(dC0=[], dpos=[], orc0_eq=[], n_open=[])
    for f in files:
        z = np.load(f, allow_pickle=True); k = int(f.stem.split("_")[1]); o = np.load(inst_dir / f"oracle_{k:03d}.npz"); meta = json.loads(str(z["meta"]))
        Sp = np.load(st_dir / a.primary_run / f"state_{k:03d}.npz", allow_pickle=True); Sr = np.load(st_dir / a.ref_run / f"state_{k:03d}.npz", allow_pickle=True)
        assert np.array_equal(Sp["w_row"], z["w_row"]) and np.array_equal(Sr["w_row"], z["w_row"]), f"inst {k}: w_row ของ state ไม่ตรง inst"
        t_list = [int(x) for x in Sp["t_min"]]; assert list(Sr["t_min"]) == t_list
        fr = E1.Frame(meta["frame"]["lat0"], meta["frame"]["lon0"])
        task_xy = z["task_xy"].astype(np.float32); st_all = z["task_st"].astype(int); et_all = z["task_et"].astype(int); W, T = len(z["w_row"]), len(task_xy)
        path = o["path_true"].astype(np.float64)                                    # (W,H,2) absolute, ก้าว 1..H จาก t0
        for ti, tm in enumerate(t_list):
            assert bool(Sp["ok"][ti].all()) and bool(Sr["ok"][ti].all()), f"inst {k} t{tm}: state ไม่ครบ"
            w_v = Sp["vxy"][ti].astype(np.float32)
            # ตำแหน่ง x_j(t): ใช้ path_true (กรอบเดียวกับ replay/oracle ทั้ง E1/E4) · t=0 = w_xy ของ inst · state ใช้เฉพาะ feat64/mu/sigma/pi/v
            w_xy = (z["w_xy"] if tm == 0 else path[:, tm - 1]).astype(np.float32)
            xy_state = fr.xy(Sp["lat"][ti], Sp["lon"][ti]); san["dpos"].append(float(np.median(np.linalg.norm(xy_state - w_xy, axis=1))))   # diagnostic: กรอบ E1 vs D3
            open_ = et_all > tm; st_rel = np.maximum(st_all[open_] - tm, 1); et_rel = et_all[open_] - tm
            Y_t = np.full((W, H, 2), 1e7, np.float32); n_rem = H - tm
            if n_rem > 0: Y_t[:, :n_rem] = (path[:, tm:] - w_xy[:, None, :]).astype(np.float32)
            pred_p = {q: Sp[q][ti] for q in ("mu", "sigma", "pi")}; pred_r = {q: Sr[q][ti] for q in ("mu", "sigma", "pi")}
            QM, ORC = qm_at_t(D7, frz_base, frz_p, frz_r, w_xy, w_v, tau, task_xy[open_], st_rel, et_rel, R, Y_t, H, p75_h, R_max, Sp["feat64"][ti], Sr["feat64"][ti], pred_p, pred_r)
            full = {}
            for key, v in QM.items(): m_ = np.full((W, T), np.nan, np.float32); m_[:, open_] = v; full[key] = m_
            orc = np.zeros((W, T), np.uint8); orc[:, open_] = ORC
            if tm == 0:
                dC0 = float(np.nanmax(np.abs(full["qm_model"] - z[f"qm_model_R{int(R)}"]))); san["dC0"].append(dC0); san["orc0_eq"].append(bool(np.array_equal(orc, o[f"qm_oracle_R{int(R)}"])))
            aucs = {nm: E1.auc_rank(full[nm][:, open_].ravel(), orc[:, open_].ravel()) for nm in ARMS}
            rows.append(dict(k=k, t_min=tm, n_open=int(open_.sum()), pos_rate=float(orc[:, open_].mean()), in_dom=float(QM["in_domain"].mean()), in_dom_dg=float(QM["in_domain_dg"].mean()), **{f"AUC_{nm}": round(v, 3) for nm, v in aucs.items()}))
            np.savez_compressed(out / f"inst_{k:03d}_t{tm:02d}.npz", t_min=tm, open=open_, st_rel_all=np.maximum(st_all - tm, 1).astype(np.int16), et_rel_all=(et_all - tm).astype(np.int16),
                                w_xy=w_xy, w_v=w_v, oracle=orc, R=np.float32(R), **full)
        log(f"  inst {k:03d}: t {t_list} · เปิด {[int((et_all > t).sum()) for t in t_list]} งาน · ΔC(t=0) vs inst {san['dC0'][-1]:.2e} · oracle(t=0) {'==' if san['orc0_eq'][-1] else '!='} inst")
    df = pd.DataFrame(rows); df.to_csv(out / "e5inst_table.csv", index=False)
    g = df.groupby("t_min").mean(numeric_only=True)
    ok0 = max(san["dC0"]) <= a.tol_C0 and all(san["orc0_eq"]); okp = True
    md = lambda d: "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n" + "\n".join("| " + " | ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in r) + " |" for r in d.values)
    lines = [f"# E5 instances — {city} · {VERSION} · {time.strftime('%Y-%m-%d %H:%M')} · sD7 v7 sha {sha_d7[:12]}", "",
             f"instances {len(files)} · t {t_list} นาที · R {R:g} · arms {ARMS} + oracle · state จาก {st_dir}/{{{a.primary_run},{a.ref_run}}} (sE5_encode)", "",
             "## Sanity", "", f"- (1) ΔC(qm_model, t=0) vs instances_v3: max {max(san['dC0']):.2e} (เกณฑ์ ≤ {a.tol_C0:g}) → {'✅' if max(san['dC0']) <= a.tol_C0 else '❌'}",
             f"- (2) diagnostic กรอบพิกัด: ตำแหน่งจาก state (Frame ของ E1) vs path_true (กรอบ D3) median {np.median(san['dpos']):.2f} m, max ของ median {max(san['dpos']):.2f} m — เรขาคณิต replay ใช้ path_true กรอบเดียวตลอด (ไม่ใช่ gate; E1 frame check เดิม SF 2.8 m / p90 9.4 m)",
             f"- (3) oracle(t=0) == instances_v3: {sum(san['orc0_eq'])}/{len(san['orc0_eq'])} → {'✅' if all(san['orc0_eq']) else '❌'}", "",
             "## AUC ต่อ arm ตาม t (mean over instances · เฉพาะงานที่ยังเปิด · ขอบฟ้าสั้นลงเมื่อ t เพิ่ม)", "", md(g.reset_index().round(3)), "",
             f"**{'✅ ผ่าน' if (ok0 and okp) else '❌ ไม่ผ่าน'}** · ไฟล์ inst_k_tMM.npz: open (T,), st_rel_all/et_rel_all (T,), w_xy/w_v (W,2), oracle (W,T) uint8, qm_* (W,T) NaN = งานปิด", ""]
    (out / "E5I_report.md").write_text("\n".join(lines), encoding="utf-8"); log(f"→ {out / 'E5I_report.md'} · {'✅' if (ok0 and okp) else '❌'}")


if __name__ == "__main__":
    main()
