#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sE1_instances_v3.py (E1 v3.1, 2026-10-20) — E1 ด้วยตัวประมาณ (9a′) + support guard: C_ij จาก reach head ที่ freeze แล้ว
  [v3.1] support guard: (9a′) ใช้เฉพาะคู่ที่ |l| ≤ d_sup(et) = max(p75|y_et| (val), 4·R_max) = domain ของ sampler D7/การฝึก · นอกนั้นใช้ (9a) ของ run เดียวกัน
         (E1 diag: หัวไม่ extrapolate นอก domain — C_ij ค้าง 0.02–0.05 ที่ > 5 km ขณะ oracle ≤ 0.01; ใน domain (9a′) ยังชนะ (9a) +0.02) · เก็บ qm_model_nog (ไม่ guard) + in_domain ไว้ด้วย
  ไฟล์ใหม่ — ไม่แก้ sE1_instances.py (v2.2) · import v2.2 เป็นโมดูลเพื่อใช้ Frame/Area/place_tasks/make_K/make_figs เดิม → โจทย์ (งาน, K_i, B, เวลา, รางวัล)
  **เหมือน v2.2 ทุกบิต** (seed เดียวกัน) — เปลี่ยนเฉพาะแหล่ง QM

QM ที่เก็บใน inst_*.npz (optimizer เห็น) — key เดิม qm_model ยังเป็น "ตัวหลัก" เพื่อให้ E4/optimizer เดิมใช้ได้โดยไม่แก้:
     qm_model        = (9a′) ของ encoder หลัก (reach_tfm) + isotonic (9a-iii), guarded   ← ตัวหลักของ paper
     qm_model_nog    = (9a′) calibrated ไม่ guard (diag) · in_domain = mask (|W|,|T|)
     qm_model_uncal  = σ(g) ดิบของ reach_tfm (ก่อน isotonic)
     qm_lstm_reach   = (9a′) ของ reference encoder (reach_lstm) + isotonic, guarded    ← arm "encoder" สำหรับ E4-lite · qm_lstm_ub = (9a) ของ reach_lstm
     qm_tfm_ub       = (9a) union bound ของ run reach_tfm (encoder เดียวกัน คนละ estimator) ← arm "estimator" บน encoder เดียวกัน
     qm_trajgeom     = (9a) ของ base_M3 (v7 frozen, baseline control)                        ← arm "trajectory-then-geometry"
     qm_null_dv · qm_persist_bin · qm_persist_cont                                         (เหมือน v2.2)
  oracle_*.npz: qm_oracle + path_true (E4 replay เท่านั้น)

อ่าน  models/<city>/base_M3/D7_final/E1_handoff.json              (v7 — null σ, (9a) control, ตรวจ sha)
      models/<city>/reach_tfm/D7reach_dev/E1_handoff_reach.json    (ตรวจ sha → pred_test.npz[feat64], reach_head.npz, reach_calib.json, calib.json, qm_calib*.json)
      models/<city>/reach_lstm/D7reach_dev/E1_handoff_reach.json
      data/<city>/processed/… · config/data_<city>.yaml                  (ผ่าน loader ของ sD7_pred_eval v7 — mask dates.test_final เหมือน D7)
เขียน instances_v3/<city>[/pilot]/inst_{k:03d}.npz · oracle_{k:03d}.npz · E1_report.md · E1_config.json · fig_E1_*.png   (ไม่แตะ models/, data/, instances/ เดิม)

ใช้:  python sE1_instances_v3.py --city sf --pilot 3        # นำร่อง → instances_v3/sf/pilot/
      python sE1_instances_v3.py --city sf,rome             # ชุดเต็ม 30 instance/เมือง
      python sE1_instances_v3.py --selftest                 # ข้อมูลสังเคราะห์ ครบทุกขั้น (ไม่ต้องมีข้อมูลจริง, ไม่ใช้ TF)
      python sE1_instances_v3.py --city sf --pilot 1 --verify-head   # เทียบ logits ของ numpy head กับ sD5_train_v4 (ต้องมี TF) แล้วหยุด
ต้องมี sE1_instances.py (v2.2) และ sD7_pred_eval.py (v7) ในโฟลเดอร์เดียวกัน · ไม่ใช้ GPU/TF (ยกเว้น --verify-head)
"""
from __future__ import annotations
import argparse, importlib.util, json, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
VERSION = "E1 v3.2 (2026-10-22: v3.1 + --dev N = สร้าง instance ชุด dev จาก cycle hold-out (ไม่ซ้อน 30 ตัวทดสอบ) สำหรับออกแบบ/จูน MOPSO-IQ v2)"

def _import(name, fname):
    p = HERE / fname
    if not p.exists(): sys.exit(f"ไม่พบ {p}")
    spec = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
E1 = _import("sE1_instances", "sE1_instances.py")                       # v2.2: Frame, Area, place_tasks, make_K, seed_of, sha256_of, import_sD7, to_str, to_epoch_s, auc_rank, make_figs, _md_table, load_frozen
if "v2.2" not in E1.VERSION: sys.exit(f"sE1_instances.py ไม่ใช่ v2.2 (พบ {E1.VERSION}) — v3 ออกแบบให้ต่อยอด v2.2")
PYCHARM = dict(E1.PYCHARM, OUT="instances_v3", PRIMARY_RUN="reach_tfm", REF_RUN="reach_lstm", REACH_SPLIT="dev", VERIFY_HEAD=False)
DURATIONS, THETAS = E1.DURATIONS, E1.THETAS

# ----------------------------------------------------------------------------- หัว (9a′) แบบ numpy — คัดลอกตรงจาก sD5_train_v4 (query_feats / reach_head_np) เพื่อไม่ต้อง import TF
Q_SCALE_M = 5000.0; R_SCALE_M = 250.0; N_QF = 7
def query_feats(l, st, et, R, H):
    r = np.sqrt((l ** 2).sum(-1, keepdims=True))
    return np.concatenate([l / Q_SCALE_M, r / Q_SCALE_M, (st / H)[..., None], (et / H)[..., None], ((et - st) / H)[..., None], (R / R_SCALE_M)[..., None]], -1).astype(np.float32)
def reach_head_np(head, feat64, Q):
    z = np.concatenate([np.repeat(feat64[:, None, :], Q.shape[1], 1), Q], -1)
    z = np.maximum(z @ head["W1"] + head["b1"], 0); z = np.maximum(z @ head["W2"] + head["b2"], 0)
    return (z @ head["W3"] + head["b3"])[..., 0]
def reach_prob(head, feat, i, st, et, l, R, H):
    """คู่ (i: แถว feat, st, et, l สัมพัทธ์) → σ(g) (n,) ที่รัศมี R"""
    Q = query_feats(l[:, None, :].astype(np.float32), st[:, None].astype(np.float32), et[:, None].astype(np.float32), np.full((len(i), 1), float(R), np.float32), float(H))
    lg = reach_head_np(head, feat[i].astype(np.float32), Q)[:, 0]; return (1.0 / (1.0 + np.exp(-lg))).astype(np.float32)

def verify_head_vs_v4(head, feat, log, n=2000, seed=0):
    """ตรวจว่า numpy head ที่คัดลอกมาให้ logits เท่ากับ sD5_train_v4 (import TF) — ใช้ครั้งเดียวตอนติดตั้ง"""
    V4 = _import("sD5_train_v4", "sD5_train_v4.py"); rng = np.random.RandomState(seed)
    i = rng.randint(0, len(feat), n); l = rng.randn(n, 2).astype(np.float32) * 1500; st = rng.randint(1, 30, n); et = st + rng.randint(5, 30, n); H = 60
    Q = V4.query_feats(l[:, None, :], st[:, None].astype(np.float32), et[:, None].astype(np.float32), np.full((n, 1), 200.0, np.float32), float(H))
    a = V4.reach_head_np(head, feat[i].astype(np.float32), Q)[:, 0]; b = reach_head_np(head, feat[i].astype(np.float32), query_feats(l[:, None, :], st[:, None].astype(np.float32), et[:, None].astype(np.float32), np.full((n, 1), 200.0, np.float32), float(H)))[:, 0]
    err = float(np.abs(a - b).max()); log(f"verify-head: max|Δlogit| = {err:.2e} ({'ตรง ✅' if err < 1e-5 else 'ไม่ตรง ❌'})"); return err < 1e-5

# ----------------------------------------------------------------------------- โหลด reach run ที่ freeze (E1_handoff_reach.json)
def load_frozen_reach(models_root: Path, city: str, run: str, split: str, log):
    hp = models_root / city / run / f"D7reach_{split}" / "E1_handoff_reach.json"
    if not hp.exists(): sys.exit(f"ไม่พบ {hp} — ต้องรัน sD7_reach_eval.py --city {city} --runs {run} --split {split} --handoff ก่อน")
    hand = json.load(open(hp, encoding="utf-8")); sd = models_root / city / run / hand["seed"]; shas = {}
    for f, rel in hand["files"].items():
        p = models_root / rel
        if not p.exists(): sys.exit(f"ไฟล์ใน handoff หาย: {p}")
        shas[f] = E1.sha256_of(p)
        if shas[f] != hand["sha256"].get(f): sys.exit(f"sha256 ไม่ตรง handoff: {p} — ไฟล์ freeze ถูกแก้? หยุด")
    for need in ("pred_test.npz", "reach_head.npz", "reach_calib.json", "calib.json"):
        if need not in hand["files"]: sys.exit(f"handoff ของ {run} ไม่มี {need}")
    P = np.load(sd / "pred_test.npz", allow_pickle=True); pred = {k: P[k] for k in P.files}
    if "feat64" not in pred: sys.exit(f"{sd}/pred_test.npz ไม่มี feat64 — ต้องมาจาก sD5_train_v4")
    for k in ("mu", "sigma", "pi"): pred[k] = pred[k].astype(np.float32)
    pred["feat64"] = pred["feat64"].astype(np.float32)
    head = {k: np.asarray(v, np.float32) for k, v in dict(np.load(sd / "reach_head.npz")).items() if k in ("W1", "b1", "W2", "b2", "W3", "b3")}
    if head["W1"].shape[0] != pred["feat64"].shape[1] + N_QF: sys.exit(f"{run}: มิติหัว {head['W1'].shape} ไม่ตรง feat64+{N_QF}")
    rc = json.load(open(sd / "reach_calib.json")); c = json.load(open(sd / "calib.json")); lam = np.asarray(c["lambda_h"], np.float32); T = float(c["T"])
    qc = None; qsrc = None
    for nm in ("qm_calib.json", "qm_calib_reachrun.json"):
        if (sd / nm).exists(): qc = json.load(open(sd / nm)); qsrc = nm; break
    aucs = hand.get("AUC_qm_reach_cal_per_seed", {})
    log(f"  {run}/{hand['seed']} ({hp.parent.name}) · AUC qm_reach_cal = {aucs.get(hand['seed'], float('nan')):.4f} (median {hand.get('median_AUC', float('nan')):.4f}) · feat64 {pred['feat64'].shape} · (9a) calib: {qsrc or '— ไม่มี → ข้าม qm_<run>_ub'} · sha ตรง {len(shas)} ไฟล์")
    return dict(run=run, seed=hand["seed"], split=hp.parent.name, pred=pred, head=head, rcal=rc, lam=lam, T=T, qcal=qc, qcal_src=qsrc, sha=shas, handoff=str(hp.relative_to(models_root)), M=int(pred["mu"].shape[1]))

def iso_apply(D7, rcal, R, s):
    c = rcal["R"][str(int(R))]; return D7.pav_apply(np.asarray(c["x"], np.float64), np.asarray(c["y"], np.float64), s)

# ----------------------------------------------------------------------------- QM ทุกแหล่ง (v3)
def qm_all_v3(D7, frz_base, frz_p, frz_r, w_rows, w_xy, w_v, tau, task_xy, st, et, R_list, Y_all, H, p75_h, R_max):
    """[v3.1] support guard: (9a′) ใช้เฉพาะคู่ใน domain ของ sampler D7 (|l| ≤ max(p75|y_et|, 4·R_max)) · นอก domain ใช้ (9a) ของ run เดียวกัน
    เหตุผล: หัวที่ฝึกไม่ extrapolate นอก domain (E1 diag 2026-10-20: C_ij ค้าง ~0.02–0.05 ที่ > 5 km ขณะ oracle ≤ 0.01) · D7 ไม่เปลี่ยนเพราะคู่ D7 อยู่ใน domain ทั้งหมด"""
    nW, nT = len(w_rows), len(task_xy)
    ii = np.repeat(np.arange(nW), nT); l = (task_xy[None, :, :] - w_xy[:, None, :]).reshape(-1, 2).astype(np.float32)
    stp = np.tile(st, nW).astype(int); etp = np.tile(et, nW).astype(int); R_list = [float(R) for R in R_list]; out = {}
    d0 = np.linalg.norm(l, axis=1); d_sup = np.maximum(p75_h[etp - 1], 4.0 * R_max); dom = (d0 <= d_sup)
    out["in_domain"] = dom.reshape(nW, nT).astype(np.float32)
    # (9a) ของ run หลัก และ reference (ใช้เป็น fallback นอก domain และเป็น arm "estimator" บน encoder เดียวกัน)
    def ub_of(frz, key):
        if frz["qcal"] is None: sys.exit(f"{frz['run']}: ไม่มี qm_calib*.json — support guard ต้องใช้ (9a) ของ run เดียวกัน (รัน sD7_reach_eval ของ run นี้ก่อน)")
        mu, sg, pi = (frz["pred"][k][w_rows] for k in ("mu", "sigma", "pi"))
        _, ub = D7.reach_score_analytic(mu, sg, pi, frz["lam"], frz["T"], ii, stp, etp, l, R_list)
        return {R: D7.apply_qm_calib(frz["qcal"], R, "ub", ub[R]) for R in R_list}
    ub_p, ub_r = ub_of(frz_p, "tfm"), ub_of(frz_r, "lstm")
    fp, fr = frz_p["pred"]["feat64"][w_rows], frz_r["pred"]["feat64"][w_rows]
    for R in R_list:
        raw = reach_prob(frz_p["head"], fp, ii, stp, etp, l, R, H); cal = iso_apply(D7, frz_p["rcal"], R, raw)
        out[f"qm_model_uncal_R{int(R)}"] = raw.reshape(nW, nT)                      # σ(g) ดิบ ไม่ guard (diag)
        out[f"qm_model_nog_R{int(R)}"] = cal.reshape(nW, nT)                        # (9a′) calibrated ไม่ guard (diag)
        out[f"qm_model_R{int(R)}"] = np.where(dom, cal, ub_p[R]).reshape(nW, nT)    # ตัวหลัก: guarded
        out[f"qm_tfm_ub_R{int(R)}"] = ub_p[R].reshape(nW, nT)
        cal_r = iso_apply(D7, frz_r["rcal"], R, reach_prob(frz_r["head"], fr, ii, stp, etp, l, R, H))
        out[f"qm_lstm_reach_R{int(R)}"] = np.where(dom, cal_r, ub_r[R]).reshape(nW, nT)
        out[f"qm_lstm_ub_R{int(R)}"] = ub_r[R].reshape(nW, nT)
    # (9a) base_M3 (v7 control) + null + persistence — เหมือน v2.2
    mu, sg, pi = (frz_base["pred"][k][w_rows] for k in ("mu", "sigma", "pi"))
    _, ub0 = D7.reach_score_analytic(mu, sg, pi, frz_base["lam"], frz_base["T"], ii, stp, etp, l, R_list)
    for R in R_list: out[f"qm_trajgeom_R{int(R)}"] = D7.apply_qm_calib(frz_base["qcal"], R, "ub", ub0[R]).reshape(nW, nT)
    sig_null = np.asarray(frz_base["qcal"]["null_sigma_h"]["gauss_dv"], np.float32)
    _, ubn = D7.reach_score_null(w_v[ii], tau, H, sig_null, np.arange(len(ii)), stp, etp, l, R_list, "gauss_dv")
    for R in R_list: out[f"qm_null_dv_R{int(R)}"] = D7.apply_qm_calib(frz_base["qcal"], R, "gauss_dv", ubn[R]).reshape(nW, nT)
    d0 = np.linalg.norm(l, axis=1).reshape(nW, nT)
    for R in R_list: out[f"qm_persist_bin_R{int(R)}"] = (d0 <= R).astype(np.float32); out[f"qm_persist_cont_R{int(R)}"] = np.exp(-d0 / R).astype(np.float32)
    oracle = {f"qm_oracle_R{int(R)}": D7.reach_labels(Y_all, w_rows[ii], stp, etp, l, R).reshape(nW, nT).astype(np.uint8) for R in R_list}
    return {k: v.astype(np.float32) for k, v in out.items()}, oracle

ARMS = ("qm_model", "qm_model_nog", "qm_lstm_reach", "qm_tfm_ub", "qm_lstm_ub", "qm_trajgeom", "qm_null_dv", "qm_persist_cont")

# ----------------------------------------------------------------------------- main
def run(a, log):
    t0 = time.time(); city = a.city
    data_root = Path(a.data) if Path(a.data).is_absolute() else HERE / a.data; models_root = Path(a.models) if Path(a.models).is_absolute() else HERE / a.models
    out_root = Path(a.out) if Path(a.out).is_absolute() else HERE / a.out
    if getattr(a, "dev", 0) and Path(a.out).resolve() == Path(PYCHARM["OUT"]).resolve(): sys.exit("--dev ต้องใช้ --out โฟลเดอร์แยก (เช่น instances_dev) เพื่อไม่ปนกับ instances_v3 — โครง <out>/<city>/ เหมือนเดิมให้ sE5_* ใช้ต่อได้")
    out = out_root / city / ("pilot" if a.pilot else ""); out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("inst_*.npz")) and not a.force: sys.exit(f"{out} มี instance อยู่แล้ว — ใช้ --force ถ้าต้องการสร้างทับ")
    D7, sha_d7 = E1.import_sD7(log)
    if not hasattr(D7, "pav_apply"): sys.exit("sD7_pred_eval.py v7 ไม่มี pav_apply")
    R_list = [float(R) for R in a.R]; R_main = float(a.r_main)
    if R_main not in R_list: R_list.append(R_main)

    log("โหลดโมเดลที่ freeze")
    frz_base = E1.load_frozen(models_root, city, a.base_run, a.base_split, log, required=True)            # v7: base_M3 D7_final
    frz_p = load_frozen_reach(models_root, city, a.primary_run, a.reach_split, log)                        # (9a′) หลัก
    frz_r = load_frozen_reach(models_root, city, a.ref_run, a.reach_split, log)                            # (9a′) reference
    if getattr(a, "verify_head", False):
        ok = verify_head_vs_v4(frz_p["head"], frz_p["pred"]["feat64"], log); sys.exit(0 if ok else 1)
    for R in R_list:
        for f in (frz_p, frz_r):
            if str(int(R)) not in f["rcal"]["R"]: sys.exit(f"reach_calib.json ของ {f['run']} ไม่มี R = {R:g} — มี {list(f['rcal']['R'])} (เพิ่ม --r-list ตอน --handoff)")
        if str(int(R)) not in frz_base["qcal"]["R"]: sys.exit(f"qm_calib.json ของ {frz_base['run']} ไม่มี R = {R:g}")

    ns = SimpleNamespace(split=a.base_split, clean_only=False)
    Y_all, v_all, meta, mask, clusters, tau, cfg, note = D7.load_split(data_root, city, models_root / city / a.base_run, "test", ns, log)
    N_all, H = Y_all.shape[:2]
    for f in (frz_base, frz_p, frz_r):
        if f["pred"]["mu"].shape[0] != N_all: sys.exit(f"{f['run']}: pred_test.npz ({f['pred']['mu'].shape[0]} แถว) ไม่ตรง test_Y ({N_all}) — คนละ D4?")
    if a.cycle_min > H: sys.exit(f"--cycle-min {a.cycle_min} > H = {H}")
    log(f"test split: {'; '.join(note) or 'all'} · หน้าต่าง {int(mask.sum()):,}/{N_all:,} · H = {H} · τ = {tau:g}")
    P = frz_base["pred"]; src = P if "lat_T" in P else (meta if meta is not None else {})
    for k in ("cab", "date", "hour", "is_cycle_start", "lat_T", "lon_T"):
        if k not in src: sys.exit(f"ไม่พบคีย์ {k} ใน pred_test.npz/meta")
    get = lambda k: (src[k] if isinstance(src, dict) else src[k].values)
    cab = E1.to_str(get("cab")); date = E1.to_str(get("date")); hour = np.asarray(get("hour")).astype(int); cyc = np.asarray(get("is_cycle_start")).astype(bool)
    lat = np.asarray(get("lat_T"), np.float64); lon = np.asarray(get("lon_T"), np.float64); tT = E1.to_epoch_s(get("t_T")) if "t_T" in src else None
    # ตรวจว่า pred ของ reach run อ้างแถวเดียวกับ base (ป้องกันสลับลำดับ)
    for f in (frz_p, frz_r):
        if "cab" in f["pred"] and "t_T" in f["pred"]:
            same = np.array_equal(E1.to_str(f["pred"]["cab"]), cab) and np.allclose(E1.to_epoch_s(f["pred"]["t_T"]), tT if tT is not None else E1.to_epoch_s(f["pred"]["t_T"]))
            if not same: sys.exit(f"{f['run']}: ลำดับหน้าต่างใน pred_test.npz ไม่ตรง base_M3 — คนละ D4/processed?")
    rows_cyc = np.flatnonzero(mask & cyc)
    if len(rows_cyc) == 0: sys.exit("ไม่มีหน้าต่าง is_cycle_start ใน split นี้")

    bb_cfg = cfg.get("bbox") or cfg.get("BBOX")
    fr = E1.Frame(np.median(lat[rows_cyc]), np.median(lon[rows_cyc])); bbox_xy = fr.bbox_xy(bb_cfg); xy_all = fr.xy(lat, lon)
    frame_err = None
    if tT is not None:
        dt_s = float(cfg.get("dt_s", 60)); df_ = pd.DataFrame({"cab": cab[mask], "date": date[mask], "t": tT[mask], "row": np.flatnonzero(mask)}).sort_values(["cab", "date", "t"]); errs = []
        for (_, _), g in df_.groupby(["cab", "date"]):
            if len(g) < 2: continue
            r = g["row"].values; t = g["t"].values; k = np.round((t[1:] - t[:-1]) / dt_s).astype(int); ok = (k >= 1) & (k <= H)
            if ok.any(): r0, r1, kk = r[:-1][ok], r[1:][ok], k[ok]; errs.append(np.linalg.norm(Y_all[r0, kk - 1] - (xy_all[r1] - xy_all[r0]), axis=1))
            if sum(len(e) for e in errs) > 20000: break
        if errs: e = np.concatenate(errs); frame_err = dict(n=int(len(e)), median_m=float(np.median(e)), p90_m=float(np.percentile(e, 90)))
        log(f"frame check: {frame_err}")

    W_min = int((cfg.get("usable_cycle_rule") or {}).get("min_workers_ready", cfg.get("W", 60)))
    dfc = pd.DataFrame({"row": rows_cyc, "cab": cab[rows_cyc], "date": date[rows_cyc], "hour": hour[rows_cyc]}).drop_duplicates(["date", "hour", "cab"])
    per = dfc.groupby(["date", "hour"])["row"].apply(lambda s: np.asarray(s)).reset_index(); per["n"] = per["row"].apply(len)
    usable = per[per["n"] >= W_min].sort_values(["date", "hour"]).reset_index(drop=True)
    if len(usable) == 0: sys.exit(f"ไม่มี cycle ที่มีคันพร้อม ≥ W_min = {W_min}")
    order = np.random.RandomState(E1.seed_of(a.master_seed, "E1", city)).permutation(len(usable))          # seed เดิม → cycle ชุดเดียวกับ v2.2
    n_inst = min(a.n_inst, len(usable)); chosen = usable.iloc[order[:n_inst]].reset_index(drop=True); holdout = usable.iloc[order[n_inst:]]
    n_make = min(a.pilot, n_inst) if a.pilot else n_inst
    dev_n = int(getattr(a, "dev", 0) or 0); k_off = 0
    if dev_n:                                                                                             # [v3.2] dev = cycle hold-out ถัดจาก 30 ตัวทดสอบในลำดับสุ่มเดิม · seed ต่อ instance เลื่อน +1000 กันซ้ำกับชุดทดสอบ
        if len(holdout) == 0: sys.exit("ไม่มี cycle hold-out สำหรับ dev")
        n_make = min(dev_n, len(holdout)); chosen = holdout.iloc[:n_make].reset_index(drop=True); k_off = 1000
        log(f"[dev] ใช้ cycle hold-out {n_make}/{len(holdout)} (ถัดจากชุดทดสอบ {n_inst} ตัวในลำดับสุ่มเดิม) · seed index เลื่อน +{k_off}")
    log(f"cycle ใช้ได้ (≥ {W_min} คัน): {len(usable)} · เลือก {n_inst} (hold-out {len(holdout)}) · สร้าง {n_make}{' (pilot)' if a.pilot else (' (dev)' if dev_n else '')}")
    Yv = np.load(data_root / city / "processed" / "val_Y.npy", mmap_mode="r"); Yv_ = np.asarray(Yv); expand = float(np.percentile(np.linalg.norm(Yv_[:, a.cycle_min - 1], axis=1), 75))
    p75_h = np.percentile(np.sqrt((Yv_ ** 2).sum(-1)), 75, axis=0).astype(np.float32); R_max = float(max(float(x) for x in frz_p["rcal"]["R"]))    # [v3.1] domain ของ sampler (จาก val ไม่ใช้ test)
    log(f"พื้นที่วางงาน = hull(fleet) + {expand:.0f} m · support guard: d_sup(et) = max(p75|y_et|, {4*R_max:.0f} m) · p75|y_60| = {p75_h[min(59, H-1)]:.0f} m")

    rows_report = []; hist = dict(E=[], K=[], dens=[], qm=[])
    for k in range(n_make):
        c = chosen.iloc[k]; cand = np.asarray(c["row"]); n_cand = len(cand)
        S = lambda tag: np.random.RandomState(E1.seed_of(a.master_seed, tag, city, k + k_off))
        rng_W, rng_T, rng_time, rng_rw, rng_K, rng_e = S("W"), S("T"), S("time"), S("reward"), S("K"), S("battery")
        fleet_xy = xy_all[cand]; nW = min(a.W, n_cand); w_rows = np.sort(cand[rng_W.choice(n_cand, nW, replace=False)])
        w_xy = xy_all[w_rows]; w_v = v_all[w_rows].astype(np.float32)
        area = E1.Area(fleet_xy, expand, bbox_xy)
        task_xy, pinfo = E1.place_tasks(area, a.T, a.placement, rng_T, fleet_xy, a.centre_src, a.fleet_frac, a.centre_jitter, a.dens_alpha)
        durs = np.array([d for d in DURATIONS if d < a.cycle_min]); u = rng_time.rand(a.T, 2)
        D = durs[(u[:, 0] * len(durs)).astype(int)]; st = 1 + (u[:, 1] * (a.cycle_min - D)).astype(int); et = st + D
        u2 = rng_rw.rand(a.T, 2); r = 0.5 + u2[:, 0]; eps = 2.0 + 6.0 * u2[:, 1]; e_j = rng_e.uniform(40.0, 100.0, nW)
        K, dens, k_used = E1.make_K(task_xy, fleet_xy, a.k_rule, rng_K)
        B = a.beta * float((r * (K + a.m_extra)).sum()); outlay_max = float((r * (K + a.m_extra)).sum())
        cell = np.floor(task_xy / a.cell_m).astype(int); n_cells = len({tuple(x) for x in cell})
        QM, ORC = qm_all_v3(D7, frz_base, frz_p, frz_r, w_rows, w_xy, w_v, tau, task_xy, st, et, R_list, Y_all, H, p75_h, R_max)
        path_true = (w_xy[:, None, :] + Y_all[w_rows]).astype(np.float32)
        Q = QM[f"qm_model_R{int(R_main)}"]; O = ORC[f"qm_oracle_R{int(R_main)}"]
        E_i = Q.sum(0); g21 = float((Q.max(1) < 0.05).mean() * 100); g22 = float((E_i < 0.5).mean() * 100); g22b = float((E_i < K).mean() * 100)
        n_theta = {th: float((Q >= th).sum(0).mean()) for th in THETAS}
        aucs = {nm: E1.auc_rank(QM[f"{nm}_R{int(R_main)}"].ravel(), O.ravel()) for nm in ARMS if f"{nm}_R{int(R_main)}" in QM}
        dom = QM["in_domain"].ravel() > 0.5; aucs["in_dom_model_nog"] = E1.auc_rank(QM[f"qm_model_nog_R{int(R_main)}"].ravel()[dom], O.ravel()[dom]); aucs["in_dom_tfm_ub"] = E1.auc_rank(QM[f"qm_tfm_ub_R{int(R_main)}"].ravel()[dom], O.ravel()[dom]); frac_dom = float(dom.mean())
        meta_i = dict(k=k, city=city, date=str(c["date"]), hour=int(c["hour"]), n_candidates=int(n_cand), W=int(nW), T=int(a.T), cycle_min=a.cycle_min, R_list=R_list, R_main=R_main,
                      placement=a.placement, centre_src=a.centre_src, fleet_frac=a.fleet_frac, centre_jitter_m=a.centre_jitter, dens_alpha=a.dens_alpha, n_centres=int(pinfo["n_centres"]), n_centres_fleet=int(pinfo["n_centres_fleet"]),
                      cell_m=a.cell_m, n_cells_with_task=int(n_cells), pt_s=60, Q_max=a.q_max, M_extra=a.m_extra, beta=a.beta, B=B, outlay_max=outlay_max, K_rule=k_used, battery_on=bool(a.battery_on),
                      area_km2=float(area.area_km2(rng_T)), expand_m=expand, frame=dict(lat0=fr.lat0, lon0=fr.lon0), seeds=dict(W=E1.seed_of(a.master_seed, "W", city, k + k_off), T=E1.seed_of(a.master_seed, "T", city, k + k_off)), dev=bool(dev_n), k_seed=k + k_off,
                      estimator="(9a') learned reachability head + isotonic", qm_source=dict(model=f"{frz_p['run']}/{frz_p['seed']}", lstm_reach=f"{frz_r['run']}/{frz_r['seed']}", tfm_ub=(f"{frz_p['run']}/{frz_p['seed']} ({frz_p['qcal_src']})" if frz_p["qcal"] else None), trajgeom=f"{frz_base['run']}/{frz_base['seed']}"), version=VERSION)
        np.savez_compressed(out / f"inst_{k:03d}.npz",
                            w_cab=cab[w_rows], w_row=w_rows.astype(np.int64), w_xy=w_xy.astype(np.float32), w_v=w_v, w_e=e_j.astype(np.float32),
                            w_sw=np.zeros(nW, np.float32), w_ew=np.full(nW, float(a.cycle_min), np.float32), fleet_xy=fleet_xy.astype(np.float32),
                            task_xy=task_xy.astype(np.float32), task_R=np.full(a.T, R_main, np.float32), task_st=st.astype(np.int16), task_et=et.astype(np.int16),
                            task_pt_s=np.full(a.T, 60, np.int16), task_r=r.astype(np.float32), task_eps=eps.astype(np.float32), task_K=K.astype(np.int8), task_cell=cell.astype(np.int32),
                            task_dens=dens.astype(np.float32), B=np.float32(B), meta=json.dumps(meta_i, ensure_ascii=False), **QM)
        np.savez_compressed(out / f"oracle_{k:03d}.npz", path_true=path_true, w_row=w_rows.astype(np.int64), **ORC)
        rows_report.append(dict(k=k, date=meta_i["date"], hour=meta_i["hour"], n_cand=n_cand, W=nW, T=a.T, area_km2=round(meta_i["area_km2"], 1), n_centres=pinfo["n_centres"],
                                K_mean=round(float(K.mean()), 2), B=round(B, 1), qm_mean=round(float(Q.mean()), 4), E_mean=round(float(E_i.mean()), 2), pos_oracle=round(float(O.mean()), 4),
                                **{f"n_i(θ={th})": round(v_, 1) for th, v_ in n_theta.items()}, G21_pct=round(g21, 1), G22_pct=round(g22, 1), G22b_pct=round(g22b, 1), in_dom_pct=round(frac_dom * 100, 1), **{f"AUC_{nm}": round(v_, 3) for nm, v_ in aucs.items()}))
        hist["E"].append(E_i); hist["K"].append(K); hist["dens"].append(dens); hist["qm"].append(Q.ravel())
        log(f"  inst {k:03d} {meta_i['date']} h{meta_i['hour']:02d} · cand {n_cand} · K̄ {K.mean():.2f} · B {B:.0f} · QM̄ {Q.mean():.3f} · G2.1 {g21:.0f}% · G2.2 {g22:.0f}% · G2.2b {g22b:.0f}% · "
            + " ".join(f"{nm.replace('qm_', '')} {v_:.3f}" for nm, v_ in aucs.items()))

    df = pd.DataFrame(rows_report); df.to_csv(out / "E1_instances.csv", index=False)
    g21_all, g22_all, g22b_all = df["G21_pct"].mean(), df["G22_pct"].mean(), df["G22b_pct"].mean()
    verdict = {"G2.1 (% worker ไม่มีงานเอื้อม < 40)": (g21_all < 40, f"{g21_all:.1f} %"), "G2.2 (% task E_i < 0.5 < 20)": (g22_all < 20, f"{g22_all:.1f} %"),
               "G2.2b (% task E_i < K_i ≤ 70)": (g22b_all <= 70, f"{g22b_all:.1f} %")}
    if frame_err: verdict["frame check (median |Y − Δx_T| ≤ 5 m)"] = (frame_err["median_m"] <= 5.0, f"{frame_err['median_m']:.2f} m (p90 {frame_err['p90_m']:.1f}, n={frame_err['n']})")
    auc_cols = [c_ for c_ in df.columns if c_.startswith("AUC_")]
    verdict["sanity A: guarded (9a′) ≥ (9a) same encoder, overall"] = (bool(df["AUC_qm_model"].mean() >= df["AUC_qm_tfm_ub"].mean() - 0.002), f"model {df['AUC_qm_model'].mean():.3f} · tfm_ub {df['AUC_qm_tfm_ub'].mean():.3f} · nog {df['AUC_qm_model_nog'].mean():.3f} · in-domain {df['in_dom_pct'].mean():.0f} % ของคู่")
    verdict["sanity B: in-domain (9a′) > (9a) (ตรงกับ D7)"] = (bool(df["AUC_in_dom_model_nog"].mean() > df["AUC_in_dom_tfm_ub"].mean()), f"in-dom (9a′) {df['AUC_in_dom_model_nog'].mean():.3f} vs (9a) {df['AUC_in_dom_tfm_ub'].mean():.3f}")
    verdict["sanity C: model > null > persistence"] = (bool(df["AUC_qm_model"].mean() > df["AUC_qm_null_dv"].mean() > df["AUC_qm_persist_cont"].mean()), " · ".join(f"{c_[4:]} {df[c_].mean():.3f}" for c_ in auc_cols if "in_dom" not in c_))
    if not a.no_figs: E1.make_figs(out, df, hist, city, R_main, a, log)
    lines = [f"# E1 report — {city}{' (pilot)' if a.pilot else (' (dev: hold-out cycles)' if dev_n else '')} · {VERSION} · {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"**qm_model = (9a′) {frz_p['run']}/{frz_p['seed']}** ({frz_p['split']}, sha ตรง) · qm_lstm_reach = {frz_r['run']}/{frz_r['seed']} · qm_tfm_ub = (9a) ของ {frz_p['run']} ({frz_p['qcal_src'] or 'ไม่มี'}) · qm_trajgeom = (9a) {frz_base['run']}/{frz_base['seed']} ({frz_base['split']}) · sD7 v7 sha `{sha_d7[:12]}` · "
             f"split {'; '.join(note) or 'all'} · cycle ใช้ได้ {len(usable)} → เลือก {n_inst} · สร้าง {n_make}", "",
             f"พารามิเตอร์ (เหมือน v2.2): |W| {a.W} · |T| {a.T} · cycle {a.cycle_min} นาที · R {R_list} (หลัก {R_main:g}) · placement {a.placement} (centre {a.centre_src} {a.fleet_frac:g}, jitter {a.centre_jitter:g} m, α {a.dens_alpha:g}) · cell_m {a.cell_m} · β {a.beta} · M_i {a.m_extra} · Q_max {a.q_max} · K rule {a.k_rule} · master seed {a.master_seed}", "",
             "## Gate G2 (ตรวจใน E1)", "", "| ข้อ | ผล | ค่า |", "|---|---|---|"] + [f"| {k_} | {'✅' if ok else '❌'} | {v_} |" for k_, (ok, v_) in verdict.items()] + ["",
             "## ต่อ instance", "", E1._md_table(df), "",
             "## หมายเหตุ", "- [v3.1] qm_model = (9a′) guarded: ใน domain |l| ≤ max(p75|y_et|, 4R_max) ใช้หัว, นอก domain ใช้ (9a) ของ run เดียวกัน · qm_model_nog = ไม่ guard (diag) · in_domain mask อยู่ใน inst",
             "- E_i, G2.* คำนวณจาก qm_model = (9a′) guarded ของ encoder หลัก · AUC_* = AUC ของแต่ละ arm เทียบ oracle บนคู่ทั้งหมดของ instance (sanity vs D7 final: model ≈ 0.85 SF / 0.82 Rome · trajgeom ≈ 0.81 / 0.79 · null ≈ 0.70 / 0.72)",
             "- โจทย์ (งาน, K_i, B, เวลา, รางวัล, worker ที่สุ่ม) เหมือน v2.2 ทุกบิตเพราะ seed เดียวกัน → เทียบกับ instances/ เดิมได้โดยตรง",
             "- arms สำหรับ E4-lite: qm_model (9a′ tfm) · qm_lstm_reach (9a′ lstm) · qm_tfm_ub (9a tfm) · qm_trajgeom (9a base_M3) · qm_null_dv · qm_persist_* · oracle (replay เท่านั้น)", ""]
    (out / "E1_report.md").write_text("\n".join(lines), encoding="utf-8")
    cfg_out = dict(version=VERSION, city=city, params={k_: (v_ if not isinstance(v_, Path) else str(v_)) for k_, v_ in vars(a).items()}, R_list=R_list, H=H, W_min=W_min, tau=tau, split_note=note, n_usable=int(len(usable)),
                   chosen_cycles=[dict(k=i, date=str(r_["date"]), hour=int(r_["hour"]), n=int(r_["n"])) for i, r_ in chosen.iterrows()], expand_m=expand, frame_check=frame_err,
                   primary=dict(run=frz_p["run"], seed=frz_p["seed"], handoff=frz_p["handoff"], sha256=frz_p["sha"], qcal_src=frz_p["qcal_src"]), reference=dict(run=frz_r["run"], seed=frz_r["seed"], handoff=frz_r["handoff"], sha256=frz_r["sha"]),
                   base=dict(run=frz_base["run"], seed=frz_base["seed"], handoff=frz_base["handoff"], sha256=frz_base["sha"]), sD7_pred_eval_sha256=sha_d7,
                   gate_G2={k_: dict(ok=bool(ok), value=v_) for k_, (ok, v_) in verdict.items()}, thomas=E1.THOMAS, durations=DURATIONS)
    json.dump(cfg_out, open(out / "E1_config.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=str)
    log(f"เสร็จ {n_make} instance → {out}  ({time.time() - t0:.0f}s) · " + " · ".join(f"{k_.split(' ')[0]} {'✅' if ok else '❌'}" for k_, (ok, _) in verdict.items()))

# ----------------------------------------------------------------------------- selftest (สังเคราะห์ — เพิ่ม reach files จากโครงของ v2.2)
def selftest(log):
    import tempfile
    root = Path(tempfile.mkdtemp(prefix="e1v3_selftest_")); city = "_e1test"; rng = np.random.RandomState(0)
    N, H, M, F, Min = 1500, 60, 3, 9, 8; proc = root / "data" / city / "processed"; proc.mkdir(parents=True)
    lat0, lon0 = 37.76, -122.42; kx, ky = 111320 * np.cos(np.radians(lat0)), 110574
    cabs = np.array([f"cab{i:03d}" for i in rng.randint(0, 120, N)]); dates = np.where(rng.rand(N) < 0.5, "2008-06-07", "2008-06-08"); hours = rng.randint(8, 14, N)
    xy0 = rng.randn(N, 2) * np.array([3000, 2500]); v = rng.randn(N, 2) * 150
    Y = (v[:, None, :] * (3 * (1 - np.exp(-np.arange(1, H + 1) / 3)))[None, :, None] + np.cumsum(rng.randn(N, H, 2) * 60, 1)).astype(np.float32)
    X = rng.randn(N, Min, F).astype(np.float32); X[:, -1, 3] = v[:, 0] / 150; X[:, -1, 4] = v[:, 1] / 150
    np.save(proc / "test_X.npy", X); np.save(proc / "test_Y.npy", Y); np.save(proc / "val_X.npy", X[:500]); np.save(proc / "val_Y.npy", Y[:500])
    tT = 1212800000 + hours * 3600 + (dates == "2008-06-08") * 86400
    meta = pd.DataFrame({"cab": cabs, "seg": 0, "t_T": tT, "lat_T": lat0 + xy0[:, 1] / ky, "lon_T": lon0 + xy0[:, 0] / kx, "date": dates, "hour": hours, "is_cycle_start": True, "is_cycle_clean": True})
    meta.to_pickle(proc / "test_meta.pkl"); meta.iloc[:500].to_pickle(proc / "val_meta.pkl")
    json.dump({"names": ["x", "y", "v", "dx", "dy", "dow_sin", "dow_cos", "slot_sin", "slot_cos"], "mean": [0] * 9, "std": [1, 1, 1, 150, 150, 1, 1, 1, 1]}, open(root / "data" / city / "scaler.json", "w"))
    (HERE / "config").mkdir(exist_ok=True); cfgp = HERE / "config" / f"data_{city}.yaml"; cfgp.write_text("dt_s: 60\nW: 60\ndates:\n  test_final: ['2008-06-07', '2008-06-08']\n", encoding="utf-8")
    sig = np.linspace(30, 1200, H).astype(np.float32); iso = {"x": [0.0, 0.5, 1.0], "y": [0.0, 0.45, 0.9], "n": [1, 1, 1], "tail_merged": 0, "y_max": 0.9, "pos_rate_val": 0.05}
    qcal = {"version": 7, "null_sigma_h": {"gauss_pers": sig.tolist(), "gauss_dv": sig.tolist()}, "R": {str(R): {"ub": iso, "lb": iso, "gauss_dv": iso, "gauss_pers": iso} for R in (150, 200, 250)}}
    common = dict(cab=cabs, t_T=tT, lat_T=meta["lat_T"].values, lon_T=meta["lon_T"].values, date=dates, hour=hours, is_cycle_start=np.ones(N, bool))
    for rn, kind, seed in (("base_M3", "v7", "seed1"), ("reach_tfm", "reach", "seed2"), ("reach_lstm", "reach", "seed0")):
        sd = root / "models" / city / rn / seed; sd.mkdir(parents=True); json.dump({"features": ["x", "y", "v", "dx", "dy", "dow_sin", "dow_cos", "slot_sin", "slot_cos"], "skip_tau": 3.0}, open(sd.parent / "run_config.json", "w"))
        mu = (Y[:, None] + rng.randn(N, M, H, 2) * sig[None, None, :, None] * 0.7).astype(np.float32); sg = np.broadcast_to(sig, (N, M, H)).astype(np.float32) * np.float32(0.9); pi = np.full((N, M), 1 / M, np.float32)
        extra = {}
        if kind == "reach":
            feat = np.concatenate([Y[:, 29] / 3000, Y[:, 59] / 5000, rng.randn(N, 60) * 0.1], 1).astype(np.float16)      # feat64 "รู้" จุดอนาคต → หัวสังเคราะห์ทำนายได้
            extra["feat64"] = feat
        np.savez(sd / "pred_test.npz", mu=mu, sigma=sg, pi=pi, **common, **extra); np.savez(sd / "pred_val.npz", mu=mu[:500], sigma=sg[:500], pi=pi[:500])
        json.dump({"lambda_h": [1.1] * H, "T": 1.0}, open(sd / "calib.json", "w")); json.dump(qcal, open(sd / "qm_calib.json", "w"))
        if kind == "v7":
            files = {f: f"{city}/{rn}/{seed}/{f}" for f in ("pred_test.npz", "pred_val.npz", "calib.json", "qm_calib.json")}; hd = sd.parent / "D7_final"; hd.mkdir()
            json.dump({"version": 7, "seed": seed, "qm_variant_for_E1": "qm_ub_cal", "R_main": 200, "AUC_qm_ub_cal@200_per_seed": {seed: 0.83}, "files": files, "sha256": {f: E1.sha256_of(sd / f) for f in files}}, open(hd / "E1_handoff.json", "w"))
        else:
            # หัวสังเคราะห์: logit = 4·(1 − |Δ_pred| / R·2) โดย Δ_pred อ่านจาก feat[0:2]·3000 (ก้าว 30) − l  → เขียนเป็น W1 เชิงเส้น + ReLU ไม่ได้ตรง ๆ จึงใช้หัวสุ่มที่ "พอใช้" + ตรวจแค่ pipeline/shape/CRN
                                                                                        # W1 (64+7)→128, W2 128→64, W3 64→1
            W1 = (rng.randn(64 + N_QF, 128) * 0.3).astype(np.float32); W1[64:67, :] *= 4   # ให้ query เชิงพื้นที่มีน้ำหนัก
            head = dict(W1=W1, b1=np.zeros(128, np.float32), W2=(rng.randn(128, 64) * 0.2).astype(np.float32), b2=np.zeros(64, np.float32), W3=(rng.randn(64, 1) * 0.3).astype(np.float32), b3=np.array([-1.5], np.float32))
            np.savez(sd / "reach_head.npz", **head, q_scale_m=Q_SCALE_M, r_scale_m=R_SCALE_M, n_qf=N_QF, feat_dim=64)
            rc = {"version": 1, "n_pairs": 100, "n_min_tail": 5, "R": {str(R): {"x": [0.0, 0.5, 1.0], "y": [0.0, 0.3, 0.8], "n": [1, 1, 1], "tail_merged": 0, "y_max": 0.8, "pos_rate_val": 0.05, "auc_val_raw": 0.6} for R in (100, 150, 200, 250)}}
            json.dump(rc, open(sd / "reach_calib.json", "w"))
            files = {f: f"{city}/{rn}/{seed}/{f}" for f in ("pred_test.npz", "pred_val.npz", "calib.json", "reach_head.npz", "reach_calib.json", "qm_calib.json")}; hd = sd.parent / "D7reach_dev"; hd.mkdir()
            json.dump({"version": "reach-1", "seed": seed, "R_main": 200, "AUC_qm_reach_cal_per_seed": {seed: 0.86}, "median_AUC": 0.86, "files": files, "sha256": {f: E1.sha256_of(sd / f) for f in files}}, open(hd / "E1_handoff_reach.json", "w"))
    a = SimpleNamespace(**{k.lower(): v for k, v in PYCHARM.items()}); a.city = city; a.data = str(root / "data"); a.models = str(root / "models"); a.out = str(root / "instances_v3")
    a.R = PYCHARM["R_LIST"]; a.pilot = 2; a.W = 40; a.T = 80; a.force = True; a.verify_head = False
    run(a, log)
    z = np.load(root / "instances_v3" / city / "pilot" / "inst_000.npz", allow_pickle=True); o = np.load(root / "instances_v3" / city / "pilot" / "oracle_000.npz")
    for key in ("qm_model", "qm_model_nog", "qm_model_uncal", "qm_lstm_reach", "qm_tfm_ub", "qm_lstm_ub", "qm_trajgeom", "qm_null_dv", "qm_persist_bin", "qm_persist_cont"):
        assert z[f"{key}_R200"].shape == (40, 80), key
        assert np.all((z[f"{key}_R200"] >= 0) & (z[f"{key}_R200"] <= 1)), f"{key} นอก [0,1]"
    assert o["qm_oracle_R200"].shape == (40, 80) and o["path_true"].shape == (40, H, 2)
    assert np.all(z["qm_model_nog_R200"] <= 0.8 + 1e-6), "isotonic y_max ไม่ถูกใช้"
    dom = z["in_domain"] > 0.5; assert dom.shape == (40, 80) and np.allclose(z["qm_model_R200"][dom], z["qm_model_nog_R200"][dom]) and np.allclose(z["qm_model_R200"][~dom], z["qm_tfm_ub_R200"][~dom]), "support guard ผิด"
    assert np.allclose(z["qm_lstm_reach_R200"][~dom], z["qm_lstm_ub_R200"][~dom]), "support guard (lstm) ผิด"
    # CRN กับ v2.2: โจทย์เดียวกัน
    a2 = SimpleNamespace(**{k.lower(): v for k, v in E1.PYCHARM.items()}); a2.city = city; a2.data = a.data; a2.models = a.models; a2.out = str(root / "instances_v22"); a2.R = E1.PYCHARM["R_LIST"]; a2.pilot = 2; a2.W = 40; a2.T = 80; a2.force = True; a2.base_split = "final"; a2.m1_run = "none"
    E1.run(a2, log); z2 = np.load(root / "instances_v22" / city / "pilot" / "inst_000.npz", allow_pickle=True)
    for key in ("task_xy", "w_row", "task_K", "task_st", "task_et", "task_r", "B"): assert np.array_equal(z[key], z2[key]), f"โจทย์ต่างจาก v2.2: {key}"
    assert np.allclose(z["qm_trajgeom_R200"], z2["qm_model_R200"]), "qm_trajgeom ต้อง = qm_model ของ v2.2 (base_M3 (9a))"
    cfgp.unlink(); log(f"selftest ผ่าน ✅ → {root}  (ลบได้)"); print((root / "instances_v3" / city / "pilot" / "E1_report.md").read_text(encoding="utf-8")[:1500])

def main():
    P = PYCHARM; ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", default=P["CITY"]); ap.add_argument("--data", default=P["DATA"]); ap.add_argument("--models", default=P["MODELS"]); ap.add_argument("--out", default=P["OUT"])
    ap.add_argument("--n-inst", type=int, default=P["N_INST"]); ap.add_argument("--pilot", type=int, default=P["PILOT"]); ap.add_argument("--dev", type=int, default=0, help="[v3.2] สร้าง N instance ชุด dev จาก cycle hold-out → ต้องใช้ --out แยก เช่น instances_dev (ไม่แตะ 30 ตัวทดสอบ)")
    ap.add_argument("--W", type=int, default=P["W"]); ap.add_argument("--T", type=int, default=P["T"]); ap.add_argument("--cycle-min", type=int, default=P["CYCLE_MIN"])
    ap.add_argument("--R", default=",".join(map(str, P["R_LIST"]))); ap.add_argument("--r-main", type=float, default=P["R_MAIN"])
    ap.add_argument("--placement", choices=["clustered", "uniform"], default=P["PLACEMENT"]); ap.add_argument("--centre-src", choices=["fleet", "mixed", "uniform"], default=P["CENTRE_SRC"]); ap.add_argument("--fleet-frac", type=float, default=P["FLEET_FRAC"]); ap.add_argument("--centre-jitter", type=float, default=P["CENTRE_JITTER"]); ap.add_argument("--dens-alpha", type=float, default=P["DENS_ALPHA"]); ap.add_argument("--cell-m", type=float, default=P["CELL_M"])
    ap.add_argument("--beta", type=float, default=P["BETA"]); ap.add_argument("--m-extra", type=int, default=P["M_EXTRA"]); ap.add_argument("--q-max", type=int, default=P["Q_MAX"])
    ap.add_argument("--k-rule", choices=["density", "random"], default=P["K_RULE"]); ap.add_argument("--battery-on", action="store_true", default=P["BATTERY_ON"]); ap.add_argument("--master-seed", type=int, default=P["MASTER_SEED"])
    ap.add_argument("--base-run", default=P["BASE_RUN"]); ap.add_argument("--base-split", default=P["BASE_SPLIT"], help="split ของ base_M3 handoff และของ window ที่ใช้ (final)")
    ap.add_argument("--primary-run", default=P["PRIMARY_RUN"]); ap.add_argument("--ref-run", default=P["REF_RUN"]); ap.add_argument("--reach-split", default=P["REACH_SPLIT"], help="โฟลเดอร์ D7reach_<split> ที่มี E1_handoff_reach.json (freeze ทำบน dev)")
    ap.add_argument("--verify-head", action="store_true", default=P["VERIFY_HEAD"]); ap.add_argument("--no-figs", action="store_true", default=P["NO_FIGS"]); ap.add_argument("--force", action="store_true", default=P["FORCE"]); ap.add_argument("--selftest", action="store_true", default=P["SELFTEST"])
    a = ap.parse_args(); a.R = [float(x) for x in str(a.R).split(",") if x.strip()]
    t0 = time.time(); log = lambda *m: print(f"[{time.time()-t0:6.1f}s]", *m, flush=True)
    if a.selftest: selftest(log); return
    for city in [c.strip() for c in a.city.split(",") if c.strip()]:
        a.city = city; run(a, log)

if __name__ == "__main__":
    main()
