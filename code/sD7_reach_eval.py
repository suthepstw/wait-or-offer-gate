#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sD7_reach_eval.py — ประเมิน reachability ของหัว "ถามจุด" (9a′) เทียบ (9a) qm_ub บน **คู่ (i, st, et, l) ชุดเดียวกัน** กับ sD7 v7
  · import sD7_pred_eval.py (v7, ห้ามแก้) ใช้ sample_pairs / reach_labels / pav_fit / ClusterBoot / load_split เดิมทุกตัว → CRN_SEED เดียวกัน = คู่เดียวกับ D7_dev ของทุก run
  · ไม่เขียนไฟล์ใด ๆ ของ v7: ผลลง models/<city>/<run>/D7reach_<split>/ · isotonic ของหัวใหม่ลง seed*/reach_calib.json · ledger models/D7reach_results.csv
  · run ที่ไม่มีหัว reach (base_M3, abl_tfm …) ก็ประเมินได้ → ได้ qm_ub_cal บนคู่เดียวกันเพื่อเทียบ (ค่าต้องเท่ากับ D7_dev ของ v7 ถ้าชุด window เดียวกัน)
score ที่รายงาน (ต่อ R):
  qm_ub, qm_ub_cal    (9a)+(9b) จาก mixture heads ของโมเดลนั้น (λ_h, T จาก calib.json; isotonic จาก qm_calib.json ถ้ามี ไม่มีก็ fit ใหม่บน val คู่เดียวกับ v7 → เก็บแยกชื่อ qm_calib_reachrun.json)
  qm_reach, qm_reach_cal  (9a′) σ(g(feat64, l − x_T, st, et, R)) จาก seed*/reach_head.npz + feat64 ใน pred_*.npz · isotonic fit บน val (CRN_SEED+7, n_val_pairs เท่า v7)
  gauss_dv_cal        null (9a) รอบ damped-velocity — อ้างอิง
CI: cluster bootstrap (cab×วัน) seed 2/3 เท่า v7 · ΔAUC แบบ paired (คู่เดียวกัน) ของ qm_reach_cal − qm_ub_cal ภายใน run และของทุก run เทียบ --ref
CLI: python sD7_reach_eval.py --city sf --runs reach_lstm,base_M3,abl_tfm --split dev --ref base_M3 --out reports/reach_eval_sf.md
     python sD7_reach_eval.py --city sf --runs reach_tfm,reach_lstm --ref reach_lstm --ref-variant qm_reach_cal --out reports/reach_enc_sf.md   # encoder เทียบกันบนหัวใหม่
     python sD7_reach_eval.py --city sf --runs reach_tfm --split dev --handoff      # freeze → D7reach_dev/E1_handoff_reach.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import sD7_pred_eval as V7                                                               # noqa: E402
from sD7_pred_eval import (sample_pairs, reach_labels, reach_score_analytic, reach_score_null, fit_null_sigma, pav_fit, pav_apply, ClusterBoot,   # noqa: E402
                           auc_rank, auprc, brier, reliability, load_split, load_npz, softmax_T, CRN_SEED, NULLS, sha256_of)
from sD5_train_v4 import query_feats, reach_head_np                                      # noqa: E402

VARS = ("qm_ub", "qm_ub_cal", "qm_reach", "qm_reach_cal", "qm_reach_cal_g", "gauss_dv_cal")   # _g = (9a′) ฉบับสมบูรณ์: หัวใน domain d_sup(et) = max(p75|y_et| (val), 4R_max), (9a) ub_cal นอก domain

def reach_scores(head, feat, i, st, et, l, r_list, H):
    out = {}
    for R in r_list:
        Q = query_feats(l[:, None, :], st[:, None].astype(np.float32), et[:, None].astype(np.float32), np.full((len(i), 1), R, np.float32), float(H))
        lg = reach_head_np(head, feat[i].astype(np.float32), Q)[:, 0]; out[R] = (1.0 / (1.0 + np.exp(-lg))).astype(np.float32)
    return out

def get_or_fit_qm_calib(sd, data_root, city, rd, a, lam, T, tau, log, Pv, Yv, vv, i, st, et, l):
    """qm_calib.json ของ v7 ถ้ามี (ไม่แตะ) · ไม่มี → fit เหมือน v7 แต่เก็บเป็น qm_calib_reachrun.json"""
    p7 = sd / "qm_calib.json"
    if p7.exists():
        c = json.load(open(p7)); return c, {k: np.asarray(v, np.float32) for k, v in c["null_sigma_h"].items()}, "qm_calib.json (v7)"
    scores = {}; scores["lb"], scores["ub"] = reach_score_analytic(Pv["mu"], Pv["sigma"], Pv["pi"], lam, T, i, st, et, l, a.r_list, 1)
    null_sigma = {kind: fit_null_sigma(Yv, vv, tau, kind) for kind in NULLS}
    for kind in NULLS: scores[kind] = reach_score_null(vv, tau, Yv.shape[1], null_sigma[kind], i, st, et, l, a.r_list, kind, 1)[1]
    calib = {"version": "reachrun(v7-like)", "n_pairs": a.n_val_pairs, "n_min_tail": a.n_min_tail, "null_sigma_h": {k: [float(x) for x in s] for k, s in null_sigma.items()}, "R": {}}
    for R in a.r_list:
        lab = reach_labels(Yv, i, st, et, l, R, 1).astype(float); calib["R"][str(int(R))] = {}
        for name, sc_ in scores.items():
            bx, by, bn, nm = pav_fit(sc_[R], lab, a.n_min_tail)
            calib["R"][str(int(R))][name] = {"x": [float(x) for x in bx], "y": [float(y) for y in by], "n": [int(x) for x in bn], "tail_merged": int(nm), "y_max": float(by[-1]), "pos_rate_val": float(lab.mean())}
    json.dump(calib, open(sd / "qm_calib_reachrun.json", "w")); return calib, null_sigma, "qm_calib_reachrun.json (fit ใหม่)"

def apply_cal(c, R, name, s): cc = c["R"][str(int(R))][name]; return pav_apply(np.array(cc["x"]), np.array(cc["y"]), s)

def evaluate_run(city, run_name, data_root, models_root, a, log, shared):
    rd = models_root / city / run_name; out = rd / f"D7reach_{a.split}"; out.mkdir(parents=True, exist_ok=True)
    seeds = sorted(p for p in rd.glob("seed*") if (p / "pred_test.npz").exists())
    if not seeds: log(f"ข้าม {rd}: ไม่พบ pred_test.npz"); return None
    Y_all, v_all, meta, mask, clusters_all, tau, cfg, note = load_split(data_root, city, rd, "test", a, log)
    Y, v, clusters = Y_all[mask], v_all[mask], clusters_all[mask]; N, H = Y.shape[:2]
    rng_pairs = np.random.RandomState(CRN_SEED); i, st, et, l, near = sample_pairs(Y, a.n_pairs, max(a.r_list), rng_pairs, a.n_hard); main = ~near
    pc = clusters[i]; cb_pairs = ClusterBoot(pc[main], a.n_boot, seed=2); cb_hard = ClusterBoot(pc[near], a.n_boot, seed=3) if near.any() else None
    subsets = {"hard": near, "st<=15": main & (st <= 15), "st16-30": main & (st > 15) & (st <= 30), "st31-45": main & (st > 30) & (st <= 45), "st>45": main & (st > 45)}
    labels = {R: reach_labels(Y, i, st, et, l, R, 1) for R in a.r_list}
    sig_pairs = f"{N}:{int(mask.sum())}:{float(Y[:1000].sum()):.3f}"          # ลายเซ็นชุด window → เทียบข้าม run ได้เฉพาะเมื่อเท่ากัน
    has_head = all((sd / "reach_head.npz").exists() for sd in seeds)
    log(f"=== reach-eval {city}/{run_name} · split={a.split} ({'; '.join(note) or 'all'}) · {len(seeds)} seeds · หน้าต่าง {N:,} · คู่หลัก {int(main.sum()):,} + hard {int(near.sum()):,} · หัว reach: {'มี' if has_head else 'ไม่มี (เทียบ qm_ub เท่านั้น)'} ===")
    # val pairs เดียวกับ v7 (ใช้ fit isotonic)
    Yv, vv, _, _, _, _, _, _ = load_split(data_root, city, rd, "val", a, log)
    rngv = np.random.RandomState(CRN_SEED + 7); iv, stv, etv, lv, _ = sample_pairs(Yv, a.n_val_pairs, max(a.r_list), rngv, 0)
    labv = {R: reach_labels(Yv, iv, stv, etv, lv, R, 1).astype(float) for R in a.r_list}
    # [guard] domain ของ (9a′) ตามนิยาม E1 v3.1: p75 ของ |y_h| จาก val (ไม่ใช้ test) · d_sup(et) = max(p75[et−1], 4·R_max)
    p75v = np.percentile(np.sqrt((np.asarray(Yv) ** 2).sum(-1)), 75, axis=0).astype(np.float32); d_sup = np.maximum(p75v[et - 1], 4.0 * max(a.r_list)); dom = np.linalg.norm(l, axis=1) <= d_sup
    log(f"  domain guard: นอก domain {int((~dom).sum()):,}/{len(dom):,} คู่ ({(~dom).mean()*100:.2f} %) · หลัก {(~dom[main]).mean()*100:.2f} % · hard {(~dom[near]).mean()*100 if near.any() else 0:.2f} %")
    rows = []; S_all = {R: {} for R in a.r_list}; seed_auc = {}
    add = lambda seed, var, met, val, R=None, sub=None: rows.append({"city": city, "run": run_name, "split": a.split, "seed": seed, "variant": var, "metric": met, "R": R, "subset": sub, "value": float(val)})
    for sd in seeds:
        P = load_npz(sd / "pred_test.npz"); Pv = load_npz(sd / "pred_val.npz")
        if len(P["mu"]) != len(mask) or len(Pv["mu"]) != len(Yv):
            log(f"  ข้าม {run_name}: จำนวนหน้าต่างใน pred ({len(P['mu'])}/{len(Pv['mu'])}) ≠ data root ({len(mask)}/{len(Yv)}) — run นี้ใช้ processed อื่น ต้องประเมินแยกด้วย --data"); return None
        mu, sig, pi = P["mu"][mask], P["sigma"][mask], P["pi"][mask]
        if (sd / "calib.json").exists(): c = json.load(open(sd / "calib.json")); lam = np.asarray(c["lambda_h"], np.float32); T = float(c["T"])
        else: lam = np.ones(H, np.float32); T = 1.0; log(f"  {sd.name}: ไม่มี calib.json → λ=1, T=1 (ควรรัน sD6 ก่อน)")
        qc, null_sigma, src = get_or_fit_qm_calib(sd, data_root, city, rd, a, lam, T, tau, log, Pv, Yv, vv, iv, stv, etv, lv)
        _, ub = reach_score_analytic(mu, sig, pi, lam, T, i, st, et, l, a.r_list, 1)
        sc = {"qm_ub": ub, "qm_ub_cal": {R: apply_cal(qc, R, "ub", ub[R]) for R in a.r_list}}
        gdv = reach_score_null(v, tau, H, null_sigma["gauss_dv"], i, st, et, l, a.r_list, "gauss_dv", 1)[1]
        sc["gauss_dv_cal"] = {R: apply_cal(qc, R, "gauss_dv", gdv[R]) for R in a.r_list}
        if has_head:
            head = dict(np.load(sd / "reach_head.npz")); fk = "feat64"
            if fk not in P or fk not in Pv: sys.exit(f"{sd}: pred_*.npz ไม่มี key feat64 — ต้องมาจาก sD5_train_v4")
            raw = reach_scores(head, P[fk][mask], i, st, et, l, a.r_list, H); rawv = reach_scores(head, Pv[fk], iv, stv, etv, lv, a.r_list, H)
            rc = {"version": 1, "n_pairs": a.n_val_pairs, "n_min_tail": a.n_min_tail, "R": {}}
            for R in a.r_list:
                bx, by, bn, nm = pav_fit(rawv[R], labv[R], a.n_min_tail)
                rc["R"][str(int(R))] = {"x": [float(x) for x in bx], "y": [float(y) for y in by], "n": [int(x) for x in bn], "tail_merged": int(nm), "y_max": float(by[-1]), "pos_rate_val": float(labv[R].mean()),
                                        "auc_val_raw": float(auc_rank(rawv[R], labv[R] > 0.5))}
            json.dump(rc, open(sd / "reach_calib.json", "w"))
            sc["qm_reach"] = raw; sc["qm_reach_cal"] = {R: pav_apply(np.array(rc["R"][str(int(R))]["x"]), np.array(rc["R"][str(int(R))]["y"]), raw[R]) for R in a.r_list}
            sc["qm_reach_cal_g"] = {R: np.where(dom, sc["qm_reach_cal"][R], sc["qm_ub_cal"][R]).astype(np.float32) for R in a.r_list}
        for R in a.r_list:
            lab = labels[R]; add(sd.name, "label", "pos_rate", lab[main].mean(), R); add(sd.name, "label", "pos_rate_hard", lab[near].mean(), R)
            for var, s_ in sc.items():
                sm, lm = s_[R][main], lab[main]
                add(sd.name, var, "AUC", auc_rank(sm, lm), R); add(sd.name, var, "AUPRC", auprc(sm, lm), R)
                b, bss = brier(sm, lm.astype(float)); _, ece = reliability(sm, lm.astype(float)); add(sd.name, var, "Brier", b, R); add(sd.name, var, "BSS", bss, R); add(sd.name, var, "ECE", ece, R)
                for nm, msk in subsets.items():
                    if msk.sum(): add(sd.name, var, "AUC", auc_rank(s_[R][msk], lab[msk]), R, nm)
                if near.any(): _, ece_h = reliability(s_[R][near], lab[near].astype(float)); add(sd.name, var, "ECE", ece_h, R, "hard")
                S_all[R].setdefault(var, []).append(s_[R])
        Rm = a.r_main; lm = labels[Rm][main]
        msg = f"  {sd.name}: AUC@{Rm:g} qm_ub_cal {auc_rank(sc['qm_ub_cal'][Rm][main], lm):.4f} · null gauss_dv_cal {auc_rank(sc['gauss_dv_cal'][Rm][main], lm):.4f}"
        if has_head: msg += f" · **qm_reach_cal {auc_rank(sc['qm_reach_cal'][Rm][main], lm):.4f}** (raw {auc_rank(sc['qm_reach'][Rm][main], lm):.4f}, hard {auc_rank(sc['qm_reach_cal'][Rm][near], labels[Rm][near]):.3f} vs ub {auc_rank(sc['qm_ub_cal'][Rm][near], labels[Rm][near]):.3f})"
        log(msg + f" · isotonic ub จาก {src}")
        if has_head: seed_auc[sd.name] = float(auc_rank(sc["qm_reach_cal"][Rm][main], lm))
    # ---- CI (คะแนนเฉลี่ยข้าม seed) + paired Δ ภายใน run
    Rm = a.r_main; lm = labels[Rm][main]; lh = labels[Rm][near]; ci = {}; mean_scores = {}
    for var in S_all[Rm]:
        s_all = np.mean(S_all[Rm][var], 0); mean_scores[var] = s_all; sm = s_all[main]
        ci[var] = {"AUC": cb_pairs.ci(lambda idx: auc_rank(sm[idx], lm[idx])), "AUPRC": cb_pairs.ci(lambda idx: auprc(sm[idx], lm[idx]))}
        if cb_hard is not None: sh = s_all[near]; ci[var]["AUC_hard"] = cb_hard.ci(lambda idx: auc_rank(sh[idx], lh[idx]))
        for met, (lo, hi) in ci[var].items(): add("all", var, f"{met}_ci_lo", lo, Rm); add("all", var, f"{met}_ci_hi", hi, Rm)
        add("all", var, "AUC_meanscore", auc_rank(sm, lm), Rm)
    d_in = None
    if "qm_reach_cal" in mean_scores:
        sa, sb = mean_scores["qm_reach_cal"][main], mean_scores["qm_ub_cal"][main]
        d_in = (auc_rank(sa, lm) - auc_rank(sb, lm), *cb_pairs.ci(lambda idx: auc_rank(sa[idx], lm[idx]) - auc_rank(sb[idx], lm[idx])))
        add("all", "delta_reach_minus_ub", "dAUC", d_in[0], Rm); add("all", "delta_reach_minus_ub", "dAUC_ci_lo", d_in[1], Rm); add("all", "delta_reach_minus_ub", "dAUC_ci_hi", d_in[2], Rm)
    df = pd.DataFrame(rows); df.to_csv(out / "reach_metrics.csv", index=False)
    ledger = models_root / "D7reach_results.csv"; df.to_csv(ledger, mode="a", header=not ledger.exists(), index=False)
    # ---- report ต่อ run
    g = lambda var, met, R=None, sub=None: df[(df.seed != "all") & (df.variant == var) & (df.metric == met) & (df.R == R) & (df.subset.isna() if sub is None else df.subset == sub)].value
    fm = lambda q, f="{:.3f}": (f.format(q.mean()) + f" ± {q.std(ddof=0):.3f}") if len(q) else "—"
    md = lambda d: "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n" + "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in d.values)
    vars_here = [v_ for v_ in VARS if v_ in S_all[Rm]]
    T1 = pd.DataFrame([{"variant": v_, **{f"AUC@{R:g}": fm(g(v_, "AUC", R)) for R in a.r_list}} for v_ in vars_here])
    T2 = pd.DataFrame([{"variant": v_, "AUPRC": fm(g(v_, "AUPRC", Rm)), "Brier": fm(g(v_, "Brier", Rm), "{:.4f}"), "BSS": fm(g(v_, "BSS", Rm)), "ECE": fm(g(v_, "ECE", Rm), "{:.4f}"),
                        "AUC cluster-CI": f"[{ci[v_]['AUC'][0]:.3f}, {ci[v_]['AUC'][1]:.3f}]", **{nm: fm(g(v_, "AUC", Rm, nm)) for nm in subsets}, "ECE hard": fm(g(v_, "ECE", Rm, "hard"), "{:.4f}")} for v_ in vars_here])
    ci_txt = lambda c: f"[{c[1]:.4f}, {c[2]:.4f}]"
    text = [f"# reach-eval — {city} / {run_name} · split **{a.split}** ({'; '.join(note) or 'all'}) · {len(seeds)} seeds · หน้าต่าง {N:,} · คู่หลัก {int(main.sum()):,} (pos rate @R{Rm:g} {labels[Rm][main].mean():.4f}) + hard {int(near.sum()):,} (pos rate {labels[Rm][near].mean():.3f}) · CRN_SEED {CRN_SEED} = คู่เดียวกับ D7 v7 · window-signature `{sig_pairs}`",
            "", "qm_ub(_cal) = (9a)+(9b) จาก mixture heads · qm_reach(_cal) = (9a′) หัวถามจุด σ(g(feat64, l−x_T, st, et, R)) + isotonic บน val · gauss_dv_cal = null", "",
            "## AUC ต่อ R (ชุดหลัก, mean ± std ข้าม seed)", md(T1), "", f"## @R={Rm:g}: AUPRC / calibration / CI / subset", md(T2), ""]
    if d_in is not None: text.append(f"**ΔAUC paired (qm_reach_cal − qm_ub_cal) บนคู่เดียวกัน @R={Rm:g}: {d_in[0]:+.4f} cluster-CI {ci_txt(d_in)}** → {'มีนัย (CI ไม่คร่อม 0)' if d_in[1] > 0 or d_in[2] < 0 else 'ไม่มีนัย (CI คร่อม 0)'}")
    (out / "reach_report.md").write_text("\n".join(text), encoding="utf-8")
    if getattr(a, "handoff", False) and has_head and seed_auc:                        # freeze: seed ที่ AUC(qm_reach_cal) ใกล้ median + sha256 (แบบเดียวกับ v7 E1_handoff.json)
        med = float(np.median(list(seed_auc.values()))); pick = min(seed_auc, key=lambda k: (abs(seed_auc[k] - med), k)); sdp = rd / pick
        files = {f: str((sdp / f).relative_to(models_root)) for f in ("pred_test.npz", "pred_val.npz", "calib.json", "reach_head.npz", "reach_calib.json", "qm_calib.json") if (sdp / f).exists()}
        hand = {"version": "reach-1", "estimator": "(9a') direct reachability head + isotonic (9a-iii)", "city": city, "run": run_name, "split": a.split, "R_main": Rm, "seed": pick,
                "AUC_qm_reach_cal_per_seed": seed_auc, "median_AUC": med, "files": files, "sha256": {f: sha256_of(sdp / f) for f in files},
                "how_to_compute_C_ij": "z_j = pred_*.npz['feat64'] (N×64 float16→float32) · q_ij = query_feats(l_i − x_j(τ0), st, et, R, H) · logit = reach_head_np(reach_head.npz, z_j, q_ij) · C̃ = sigmoid(logit) · C_ij = isotonic(reach_calib.json['R'][str(R)]) · อ่าน sD5_train_v4.query_feats / reach_head_np · ตรวจ sha256 ก่อนใช้",
                "note": "สาย E ใช้ seed นี้ตัวเดียว · base_M3 E1_handoff.json (v7) ยังใช้สำหรับ arm trajectory-then-geometry"}
        json.dump(hand, open(out / "E1_handoff_reach.json", "w"), indent=2, ensure_ascii=False)
        log(f"  freeze → {out / 'E1_handoff_reach.json'}: seed {pick} (AUC {seed_auc[pick]:.4f}, median {med:.4f}; " + ", ".join(f"{k} {v:.4f}" for k, v in seed_auc.items()) + ")")
    log(f"  → {out / 'reach_report.md'}" + (f" · ΔAUC(reach−ub) {d_in[0]:+.4f} {ci_txt(d_in)}" if d_in else ""))
    shared[run_name] = {"sig": sig_pairs, "mean_scores": mean_scores, "lab": lm, "lab_hard": lh, "main": main, "near": near, "cb": cb_pairs, "cb_hard": cb_hard, "ci": ci, "df": df, "n_seeds": len(seeds), "has_head": has_head, "delta_in": d_in}
    return shared[run_name]

def cross_run_report(city, shared, a, out_path: Path, log):
    """ตารางรวม + paired ΔAUC ของแต่ละ run เทียบ --ref บนคู่เดียวกัน (ต้องมี window-signature เท่ากัน)"""
    Rm = a.r_main; ref = a.ref if a.ref in shared else next(iter(shared)); S0 = shared[ref]
    rv = a.ref_variant if a.ref_variant in S0["mean_scores"] else "qm_ub_cal"; col = f"ΔAUC vs {ref}/{rv} (paired)"
    rows = []
    for run, S in shared.items():
        same = S["sig"] == S0["sig"]; g = lambda var, met: S["df"][(S["df"].seed != "all") & (S["df"].variant == var) & (S["df"].metric == met) & (S["df"].R == Rm) & S["df"].subset.isna()].value
        for var in ("qm_ub_cal", "qm_reach_cal", "qm_reach_cal_g"):
            if var not in S["mean_scores"]: continue
            s = S["mean_scores"][var][S["main"]]; auc = auc_rank(s, S["lab"])
            r = {"run": run, "variant": var, "seeds": S["n_seeds"], f"AUC@{Rm:g}": f"{g(var,'AUC').mean():.4f} ± {g(var,'AUC').std(ddof=0):.4f}", "cluster-CI": f"[{S['ci'][var]['AUC'][0]:.4f}, {S['ci'][var]['AUC'][1]:.4f}]",
                 "AUPRC": f"{g(var,'AUPRC').mean():.3f}", "ECE": f"{g(var,'ECE').mean():.4f}", "AUC hard": f"{auc_rank(S['mean_scores'][var][S['near']], S['lab_hard']):.3f}" if S["near"].any() else "—"}
            if same and not (run == ref and var == rv):
                s0 = S0["mean_scores"][rv][S0["main"]]; d = auc - auc_rank(s0, S0["lab"]); lo, hi = S["cb"].ci(lambda idx: auc_rank(s[idx], S["lab"][idx]) - auc_rank(s0[idx], S0["lab"][idx]))
                r[col] = f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]" + (" ✔" if lo > 0 else (" ✘" if hi < 0 else ""))
            else: r[col] = "—" if same else "ชุด window ต่างกัน — เทียบไม่ได้"
            rows.append(r)
    df = pd.DataFrame(rows); md = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n" + "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in df.values)
    text = [f"# reach-eval รวม — {city} · split {a.split} · R = {Rm:g} m · ref = {ref} · คู่ชุดเดียวกัน (CRN_SEED {CRN_SEED}) → ΔAUC แบบ paired, CI = cluster bootstrap {a.n_boot} รอบ (✔ = CI > 0 ทั้งช่วง)", "", md, "",
            "อ่านผล: (9a′) ชนะ (9a) อย่างมีนัยเมื่อ ΔAUC paired ของ qm_reach_cal มี CI ไม่คร่อม 0 · ECE ต้อง ≤ ของ qm_ub_cal + 0.001 · ดู AUC hard ประกอบ (งานใกล้เส้นทาง = สิ่งที่ optimizer ต้องแยกจริง)"]
    out_path.parent.mkdir(parents=True, exist_ok=True); out_path.write_text("\n".join(text), encoding="utf-8"); log(f"ตารางรวม → {out_path}")

def main():
    P = V7.PYCHARM
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", default=P["CITY"]); ap.add_argument("--runs", required=True); ap.add_argument("--data", default=P["DATA"]); ap.add_argument("--models", default=P["MODELS"])
    ap.add_argument("--split", choices=["dev", "final", "all"], default="dev"); ap.add_argument("--clean-only", action="store_true", default=False)
    ap.add_argument("--r-list", default=",".join(map(str, P["R_LIST"]))); ap.add_argument("--r-main", type=float, default=P["R_MAIN"])
    ap.add_argument("--n-pairs", type=int, default=P["N_PAIRS"]); ap.add_argument("--n-hard", type=int, default=P["N_HARD"]); ap.add_argument("--n-val-pairs", type=int, default=P["N_VAL_PAIRS"])
    ap.add_argument("--n-min-tail", type=int, default=P["N_MIN_TAIL"]); ap.add_argument("--n-boot", type=int, default=P["N_BOOT"])
    ap.add_argument("--handoff", action="store_true", help="เขียน D7reach_<split>/E1_handoff_reach.json (seed ใกล้ median ของ AUC qm_reach_cal + sha256)"); ap.add_argument("--ref", default="base_M3"); ap.add_argument("--ref-variant", default="qm_ub_cal", help="score ของ run อ้างอิงที่ใช้คิด ΔAUC paired (qm_ub_cal | qm_reach_cal)"); ap.add_argument("--out", default="")
    a = ap.parse_args(); a.r_list = [float(x) for x in a.r_list.split(",")]; a.sub_steps = 1
    data_root = Path(a.data) if Path(a.data).is_absolute() else HERE / a.data; models_root = Path(a.models) if Path(a.models).is_absolute() else HERE / a.models
    t0 = time.time(); log = lambda *m: print(f"[{time.time()-t0:6.1f}s]", *m, flush=True)
    shared = {}
    for r in [x.strip() for x in a.runs.split(",") if x.strip()]: evaluate_run(a.city, r, data_root, models_root, a, log, shared)
    if shared: cross_run_report(a.city, shared, a, Path(a.out) if a.out else HERE / "reports" / f"reach_eval_{a.city}.md", log)

if __name__ == "__main__":
    main()
