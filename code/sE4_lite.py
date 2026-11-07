#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sE4_lite.py (E4-lite v1.0, 2026-10-20) — Value-of-Prediction: NSGA-II บน instance ของ E1 v3 × หลาย arm ของ C_ij → replay ด้วยเส้นทางจริง
  Def 6 **v5** (Short_Formulation v5): f₁ = Poisson-binomial Φ_i (10)(11) · f₂ = Jain(RW_j) บน W⁺ (13)(14) · f₃ = Jain(s_i) บน T⁺ (15a)(15b)(15)
  ข้อจำกัด (16): ก Σ_j x_ij ≤ K_i · ข Σ_i x_ij ≤ Q_max · ค Σ_i x_ij ε_i ≤ e_j (ถ้า --battery-on) · ไม่มีงบ (v5)
  L_i(1) = 1 (ค่าหลัก; --L-rule K = L_i = K_i เป็น sensitivity) · θ_T = θ_W = 0.05 · Q_max = 3

arms (C_ij ที่ optimizer เห็น) — ทุก arm แก้โจทย์เดียวกัน (งาน/คน/K/ε เดียวกัน) ต่างกันแค่ C:
  qm_model (9a′ tfm, guarded) · qm_lstm_reach (9a′ lstm) · qm_tfm_ub (9a tfm) · qm_trajgeom (9a base_M3) · qm_null_dv · qm_persist_cont · oracle (y จริง — ขอบบน)
replay: ทุกจุดบน front ให้คะแนนด้วย y จริง (เงื่อนไข ฎ): f₁* = mean_i min(N*_i, L_i)/L_i · f₂* = Jain(RW* บน W⁺ ของ arm) · f₃* = Jain(s*_i) โดย Φ^max อ้างอิงร่วม = ของ oracle (เทียบข้าม arm ได้)
เลือก X* จาก front: จุดใกล้ ideal (1,1,1) ที่สุด (L2 บนค่าที่ optimizer เห็น) — กฎเดียว ประกาศก่อนดูผล · รายงาน max-f₁ และ HV ของ front จริงด้วย
VoP: Δf₁* ของแต่ละ arm เทียบ qm_model ต่อ instance (paired) → mean, bootstrap CI (instances), Wilcoxon
Gate: G2.4 corr(f₁,f₃) บน front < 0.9 · G2.6 oracle > model > persistence (f₁*)

ใช้:  pip install pymoo   (ครั้งเดียว)
      python sE4_lite.py --city sf --inst instances_v3 --pilot 3                # 3 instance × 7 arm
      python sE4_lite.py --city sf,rome --inst instances_v3 --out e4lite       # ชุดเต็ม
      python sE4_lite.py --selftest
ผล: <out>/<city>/e4_runs.csv (ทุก arm × instance × จุดที่เลือก) · e4_fronts.npz · E4_report.md · fig_e4_*.png
"""
from __future__ import annotations
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
VERSION = "E4-lite v1.1 (2026-10-22: decode vectorised ×26 — ผลเท่า v1.0 ทุกบิต · decode_ref = เดิม)"
ARMS = ["qm_model", "qm_lstm_reach", "qm_tfm_ub", "qm_trajgeom", "qm_null_dv", "qm_persist_cont", "oracle"]
LABEL = {"qm_model": "(9a′) Transformer [main]", "qm_lstm_reach": "(9a′) Bi-LSTM", "qm_tfm_ub": "(9a) Transformer", "qm_trajgeom": "(9a) base_M3", "qm_null_dv": "physics null", "qm_persist_cont": "persistence", "oracle": "oracle (upper bound)"}

# ----------------------------------------------------------------------------- Def 6 v5 objectives (vectorised over population)
def jain(v, mask=None):
    """Jain index ต่อแถว · v (P, n) · mask (n,) เลือกสมาชิก · J(0) = 0"""
    if mask is not None: v = v[:, mask]
    n = v.shape[1]
    if n == 0: return np.zeros(v.shape[0])
    s = v.sum(1); q = (v ** 2).sum(1); out = np.zeros(v.shape[0])
    np.divide(s ** 2, n * q, out=out, where=q > 0); return out

def phi_pb(C_sel, L):
    """Φ_i = E[min(N_i, L_i)]/L_i · C_sel (P, T, S) ความน่าจะเป็นของช่องที่มอบ (0 = ไม่มอบ) · L (T,) · DP Poisson-binomial ถึง k ≤ L_max"""
    P, T, S = C_sel.shape; Lmax = int(L.max())
    d = np.zeros((P, T, Lmax + 1)); d[:, :, 0] = 1.0
    for u in range(S):
        c = C_sel[:, :, u][..., None]
        new = np.concatenate([d[:, :, :1] * (1 - c[:, :, 0:1]), d[:, :, 1:] * (1 - c) + d[:, :, :-1] * c], 2)
        new[:, :, -1] = d[:, :, -1] + d[:, :, -2] * c[:, :, 0]                     # ช่องสุดท้าย = Pr[N ≥ Lmax] สะสม (มวลไม่หาย)
        d = new
    # Pr[N ≥ k] = 1 − Σ_{k'<k} d_k'
    cdf = np.cumsum(d, 2)                                              # Pr[N ≤ k]
    ge = 1.0 - np.concatenate([np.zeros((P, T, 1)), cdf[:, :, :-1]], 2)   # Pr[N ≥ k] สำหรับ k = 0..Lmax
    ks = np.arange(Lmax + 1)[None, None, :]
    phi = (ge * ((ks >= 1) & (ks <= L[None, :, None]))).sum(2) / L[None, :]
    full = np.take_along_axis(ge, L[None, :, None].repeat(P, 0), 2)[:, :, 0]   # Pr[N ≥ L_i]
    return phi, full

class Problem:
    """โจทย์ของ 1 instance × 1 arm · encoding: ต่องาน i มี K_i ช่อง ค่า ∈ {0..n_i} (0 = ว่าง, k = candidate ที่ k) · candidates = top-N worker ตาม C ของ arm"""
    def __init__(self, C, K, L, r, eps, e, RWpast, Qmax, battery_on, theta_T, theta_W, n_cand, phimax_ref=None, oracle_y=None):
        self.C = C.astype(np.float64); self.W, self.T = C.shape; self.K = K.astype(int); self.L = L.astype(int); self.r = r; self.eps = eps; self.e = e; self.RWpast = RWpast
        self.Qmax = Qmax; self.battery_on = battery_on; self.n_cand = n_cand
        order = np.argsort(-self.C, 0)[:n_cand]                       # (n_cand, T) worker index ของ candidate
        self.cand = order.T                                             # (T, n_cand)
        self.cand_C = np.take_along_axis(self.C.T, self.cand, 1)        # (T, n_cand)
        self.Smax = int(self.K.max()); self.slot_task = np.repeat(np.arange(self.T), self.K); self.n_var = len(self.slot_task)
        self.slot_pos = np.concatenate([np.arange(k) for k in self.K])
        # W⁺, T⁺, Φ^max (ต้นรอบ จาก C ของ arm นี้ · เงื่อนไข ฌ/ญ)
        self.Wplus = self.C.max(1) >= theta_W
        phimax_own = np.zeros(self.T)
        for i in range(self.T):
            top = self.cand_C[i, :self.K[i]]; phimax_own[i] = 1 - np.prod(1 - top) if self.L[i] == 1 else phi_pb(top[None, None, :], self.L[i:i + 1])[0][0, 0]
        self.phimax_own = phimax_own; self.Tplus = phimax_own >= theta_T
        self.phimax_ref = phimax_own if phimax_ref is None else phimax_ref   # สำหรับ f₃* ใช้อ้างอิงร่วม
        self.oracle_y = oracle_y
    # ---- decode genome → x_sel (P, T, Smax) worker index (−1 = ว่าง) หลัง repair
    def _assign(self, G):
        """ขั้น 1 + repair 1 แบบ vectorised (ผลเท่า decode_ref ทุกบิต)"""
        P = G.shape[0]; sel = G.astype(int) - 1
        cand_v = self.cand[self.slot_task]                                           # (n_var, n_cand)
        w = cand_v[np.arange(self.n_var)[None, :], np.clip(sel, 0, self.n_cand - 1)]; w = np.where(sel >= 0, w, -1)
        x = np.full((P, self.T, self.Smax), -1, int); x[:, self.slot_task, self.slot_pos] = w
        for s in range(1, self.Smax):
            for s2 in range(s):
                dup = (x[:, :, s] == x[:, :, s2]) & (x[:, :, s] >= 0); x[:, :, s][dup] = -1
        return x
    def _repair_quota(self, x, Q_w, pending=None):
        """repair 2 แบบ vectorised: ต่อ (particle, worker) เรียง C น้อย→มาก (เสมอ → ลำดับ index) เก็บ offer ใหม่ได้ไม่เกิน Q_w[j] ตัวที่ C สูงสุด · pending (T,W) เก็บฟรี · เฉพาะ battery off"""
        p_i, t_i, s_i = np.nonzero(x >= 0)
        if not len(p_i): return x
        wj = x[p_i, t_i, s_i]; cj = self.C[wj, t_i]; flat = t_i * self.Smax + s_i
        o = np.lexsort((flat, cj, wj, p_i)); p_o, t_o, s_o, w_o = p_i[o], t_i[o], s_i[o], wj[o]
        key = p_o * self.W + w_o; new = np.r_[True, key[1:] != key[:-1]]
        g_start = np.flatnonzero(new); g_len = np.diff(np.r_[g_start, len(key)]); g_end = np.repeat(g_start + g_len, g_len)   # index หลังตัวสุดท้ายของกลุ่ม
        is_new = np.ones(len(key), bool) if pending is None else ~pending[t_o, w_o]
        cum = np.cumsum(is_new); cum_end = cum[g_end - 1]; after = cum_end - cum                      # จำนวน offer ใหม่ที่ C สูงกว่า (อยู่หลัง) ในกลุ่ม
        drop = is_new & (after >= Q_w[w_o])
        x[p_o[drop], t_o[drop], s_o[drop]] = -1
        return x
    def decode(self, G):
        if self.battery_on: return self.decode_ref(G)
        return self._repair_quota(self._assign(G), np.full(self.W, self.Qmax))
    def decode_ref(self, G):
        """ฉบับอ้างอิง (loop เดิม v1.0) — ใช้เมื่อ battery on และสำหรับทดสอบความเท่ากัน"""
        P = G.shape[0]; x = np.full((P, self.T, self.Smax), -1, int)
        g = G.astype(int)
        for v in range(self.n_var):
            i, s = self.slot_task[v], self.slot_pos[v]; sel = g[:, v] - 1
            x[:, i, s] = np.where(sel >= 0, self.cand[i, np.clip(sel, 0, self.n_cand - 1)], -1)
        # repair 1: worker ซ้ำในงานเดียวกัน → เก็บช่องแรก
        for s in range(1, self.Smax):
            for s2 in range(s):
                dup = (x[:, :, s] == x[:, :, s2]) & (x[:, :, s] >= 0); x[:, :, s][dup] = -1
        # repair 2: Q_max และพลังงาน ต่อ worker — ตัดงานที่ C ต่ำสุดออกจนผ่าน
        for p in range(P):
            xs = x[p]; flat = xs.ravel(); idx = np.flatnonzero(flat >= 0)
            if not len(idx): continue
            wj = flat[idx]; ti = idx // self.Smax; cj = self.C[wj, ti]
            order = np.lexsort((cj, wj))                             # จัดกลุ่มตาม worker, C น้อย→มาก
            wj_o, ti_o, idx_o = wj[order], ti[order], idx[order]
            for j in np.unique(wj_o):
                m = np.flatnonzero(wj_o == j); n_j = len(m)
                # เก็บเฉพาะ Q_max ตัวที่ C สูงสุด (ท้ายลิสต์) และตัดต่อถ้าพลังงานเกิน
                keep = m[max(0, n_j - self.Qmax):]
                if self.battery_on:
                    while len(keep) and self.eps[ti_o[keep]].sum() > self.e[j]: keep = keep[1:]
                drop = np.setdiff1d(m, keep); flat[idx_o[drop]] = -1
            x[p] = flat.reshape(self.T, self.Smax)
        return x
    # ---- objectives from x_sel with a given probability matrix (C = plan, y = replay)
    def objectives(self, x, Cmat, phimax_for_f3, use_RWpast=True):
        P = x.shape[0]; assigned = x >= 0
        Ci = np.where(assigned, Cmat[np.clip(x, 0, self.W - 1), np.arange(self.T)[None, :, None]], 0.0)   # (P,T,S)
        phi, full = phi_pb(Ci, self.L)
        f1 = phi.mean(1)
        RW = np.tile(self.RWpast if use_RWpast else np.zeros(self.W), (P, 1))
        rw_add = Ci * self.r[None, :, None]
        for s in range(self.Smax):
            np.add.at(RW, (np.repeat(np.arange(P), self.T), np.where(assigned[:, :, s], x[:, :, s], 0).ravel()), np.where(assigned[:, :, s], rw_add[:, :, s], 0).ravel())
        f2 = jain(RW, self.Wplus)
        s_i = np.where(phimax_for_f3[None, :] > 0, phi / np.maximum(phimax_for_f3[None, :], 1e-12), 0.0); s_i = np.clip(s_i, 0, 1)
        f3 = jain(s_i, self.Tplus)
        n_offer = assigned.sum((1, 2)); outlay = (Ci * self.r[None, :, None]).sum((1, 2))
        return dict(f1=f1, f2=f2, f3=f3, full=full.mean(1), n_offer=n_offer, outlay=outlay)

def run_nsga2(prob, pop, gen, seed):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem as PmProblem
    from pymoo.operators.sampling.rnd import IntegerRandomSampling
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.repair.rounding import RoundingRepair
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    class PP(PmProblem):
        def __init__(s): super().__init__(n_var=prob.n_var, n_obj=3, n_ieq_constr=0, xl=0, xu=prob.n_cand, vtype=int)
        def _evaluate(s, G, out, *a, **k):
            x = prob.decode(G); o = prob.objectives(x, prob.C, prob.phimax_own)
            out["F"] = -np.stack([o["f1"], o["f2"], o["f3"]], 1)
    from pymoo.core.sampling import Sampling
    class GreedySeeded(Sampling):
        """สุ่ม + ใส่คำตอบ greedy (ทุกช่อง = candidate อันดับ 1,2,3 ตามลำดับช่อง) 1 ตัว และ greedy ที่เติมเฉพาะช่องแรก 1 ตัว — ทุก arm ได้ seed แบบเดียวกัน"""
        def _do(s, problem, n_samples, **kw):
            X = IntegerRandomSampling()._do(problem, n_samples, **kw).astype(float)
            X[0] = prob.slot_pos + 1; X[1] = np.where(prob.slot_pos == 0, 1, 0); return X
    alg = NSGA2(pop_size=pop, sampling=GreedySeeded(), crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()), mutation=PM(prob=1.0 / max(1, prob.n_var) * 3, eta=20, vtype=float, repair=RoundingRepair()), eliminate_duplicates=True)
    res = minimize(PP(), alg, get_termination("n_gen", gen), seed=seed, verbose=False)
    G = np.atleast_2d(res.X); F = -np.atleast_2d(res.F)
    return G, F

def hv3(F, ref=(0, 0, 0)):
    """hypervolume ของจุด max ใน [0,1]^3 เทียบ ref (0,0,0) — ใช้ pymoo"""
    try:
        from pymoo.indicators.hv import HV
        return float(HV(ref_point=np.array([1.0, 1.0, 1.0]))(1.0 - np.atleast_2d(F)))
    except Exception: return float("nan")

# ----------------------------------------------------------------------------- per instance
def solve_instance(inst_path, R, arms, pop, gen, seed, Qmax, battery_on, theta_T, theta_W, n_cand, L_rule, log):
    z = np.load(inst_path, allow_pickle=True); o = np.load(str(inst_path).replace("inst_", "oracle_"))
    meta = json.loads(str(z["meta"])); k = meta["k"]
    K = z["task_K"].astype(int); L = np.ones_like(K) if L_rule == "one" else K.copy(); r = z["task_r"].astype(float); eps = z["task_eps"].astype(float); e = z["w_e"].astype(float)
    Y = o[f"qm_oracle_R{R}"].astype(float)                           # (W,T) y จริง
    W, T = Y.shape; RWpast = np.zeros(W)
    # Φ^max อ้างอิงร่วมสำหรับ f₃* = จาก oracle
    ref_prob = Problem(Y, K, L, r, eps, e, RWpast, Qmax, battery_on, theta_T, theta_W, n_cand)
    phimax_ref = ref_prob.phimax_own; Tplus_ref = ref_prob.Tplus
    rows = []; fronts = {}
    for arm in arms:
        Cm = Y if arm == "oracle" else z[f"{arm}_R{R}"].astype(float)
        if arm not in ("oracle",) and f"{arm}_R{R}" not in z: log(f"    ข้าม arm {arm} (ไม่มีใน inst)"); continue
        prob = Problem(Cm, K, L, r, eps, e, RWpast, Qmax, battery_on, theta_T, theta_W, n_cand, phimax_ref=phimax_ref)
        t0 = time.time(); G, F = run_nsga2(prob, pop, gen, seed + k)
        x = prob.decode(G); plan = prob.objectives(x, prob.C, prob.phimax_own)
        # replay: y จริง · f₃* ใช้ Φ^max_ref และ T⁺ ของ oracle (เทียบข้าม arm) · f₂* ใช้ W⁺ ของ arm (ตามนิยาม)
        prob.Tplus = Tplus_ref; real = prob.objectives(x, Y, phimax_ref)
        Fp = np.stack([plan["f1"], plan["f2"], plan["f3"]], 1); Fr = np.stack([real["f1"], real["f2"], real["f3"]], 1)
        i_sel = int(np.argmin(np.linalg.norm(1.0 - Fp, axis=1))); i_f1 = int(np.argmax(Fp[:, 0]))
        corr13 = float(np.corrcoef(Fp[:, 0], Fp[:, 2])[0, 1]) if len(Fp) > 2 and Fp[:, 0].std() > 0 and Fp[:, 2].std() > 0 else float("nan")
        rows.append(dict(k=k, arm=arm, n_front=len(Fp), sec=round(time.time() - t0, 1), corr_f1f3=round(corr13, 3), hv_plan=hv3(Fp), hv_real=hv3(Fr),
                         f1_plan=plan["f1"][i_sel], f2_plan=plan["f2"][i_sel], f3_plan=plan["f3"][i_sel],
                         f1_real=real["f1"][i_sel], f2_real=real["f2"][i_sel], f3_real=real["f3"][i_sel], full_real=real["full"][i_sel], n_offer=int(plan["n_offer"][i_sel]), outlay_real=real["outlay"][i_sel],
                         f1_real_maxf1=real["f1"][i_f1], f1_plan_maxf1=plan["f1"][i_f1], f1_real_frontmean=real["f1"].mean(), n_Wplus=int(prob.Wplus.sum()), n_Tplus_own=int((prob.phimax_own >= theta_T).sum())))
        fronts[f"{k}_{arm}_plan"] = Fp; fronts[f"{k}_{arm}_real"] = Fr; fronts[f"{k}_{arm}_x"] = x[i_sel]
        log(f"    inst {k:03d} {arm:16s} front {len(Fp):3d} · plan f1 {plan['f1'][i_sel]:.3f} → real f1* {real['f1'][i_sel]:.3f} · f2* {real['f2'][i_sel]:.3f} · f3* {real['f3'][i_sel]:.3f} · corr(f1,f3) {corr13:+.2f} · {time.time()-t0:.0f}s")
    return rows, fronts

def paired_summary(df, ref="qm_model", col="f1_real", n_boot=2000, seed=0):
    rng = np.random.RandomState(seed); out = []
    piv = df.pivot(index="k", columns="arm", values=col)
    for arm in piv.columns:
        if arm == ref or ref not in piv: continue
        d = (piv[arm] - piv[ref]).dropna().to_numpy()
        if not len(d): continue
        bs = np.array([rng.choice(d, len(d)).mean() for _ in range(n_boot)])
        try:
            from scipy.stats import wilcoxon; p = float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0
        except Exception: p = float("nan")
        out.append(dict(arm=arm, n=len(d), mean=piv[arm].mean(), ref_mean=piv[ref].mean(), delta=d.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), wilcoxon_p=p, wins=int((d > 0).sum()), losses=int((d < 0).sum())))
    return pd.DataFrame(out)

def main_city(a, city, log):
    inst_dir = Path(a.inst) / city / ("pilot" if a.pilot else ""); files = sorted(glob.glob(str(inst_dir / "inst_*.npz")))
    if a.pilot: files = files[:a.pilot]
    if not files: sys.exit(f"ไม่พบ inst_*.npz ใน {inst_dir}")
    out = Path(a.out) / city / ("pilot" if a.pilot else ""); out.mkdir(parents=True, exist_ok=True)
    arms = [x for x in a.arms.split(",") if x]; rows = []; fronts = {}
    log(f"{city}: {len(files)} instance × {len(arms)} arm · NSGA-II pop {a.pop} gen {a.gen} · R {a.R} · L rule {a.L_rule} · Q_max {a.q_max} · θ {a.theta_T}/{a.theta_W} · cand {a.n_cand}")
    for f in files:
        rr, ff = solve_instance(Path(f), int(a.R), arms, a.pop, a.gen, a.seed, a.q_max, a.battery_on, a.theta_T, a.theta_W, a.n_cand, a.L_rule, log); rows += rr; fronts.update(ff)
    df = pd.DataFrame(rows); df.to_csv(out / "e4_runs.csv", index=False); np.savez_compressed(out / "e4_fronts.npz", **fronts)
    # ---- summary
    agg = df.groupby("arm").agg(n=("k", "count"), f1_plan=("f1_plan", "mean"), f1_real=("f1_real", "mean"), f2_real=("f2_real", "mean"), f3_real=("f3_real", "mean"), full_real=("full_real", "mean"), hv_real=("hv_real", "mean"), corr_f1f3=("corr_f1f3", "mean"), n_offer=("n_offer", "mean"), f1_real_maxf1=("f1_real_maxf1", "mean")).reindex([x for x in ARMS if x in df.arm.unique()])
    vop = paired_summary(df, "qm_model", "f1_real"); vop3 = paired_summary(df, "qm_model", "f3_real"); voph = paired_summary(df, "qm_model", "hv_real")
    g24 = float(df[df.arm == "qm_model"].corr_f1f3.mean()); o_ = agg.f1_real
    g26 = bool(o_.get("oracle", 1) > o_.get("qm_model", 0) > o_.get("qm_persist_cont", -1))
    md = lambda d: "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n" + "\n".join("| " + " | ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r) + " |" for r in d.values)
    agg_r = agg.reset_index(); agg_r.insert(1, "label", agg_r.arm.map(LABEL))
    lines = [f"# E4-lite — {city}{' (pilot)' if a.pilot else ''} · {VERSION} · {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"instances {len(files)} ({inst_dir}) · arms {len(arms)} · NSGA-II pop {a.pop} × gen {a.gen} (seed {a.seed}+k) · R {a.R} m · L_i = {'1' if a.L_rule == 'one' else 'K_i'} · Q_max {a.q_max} · battery {'on' if a.battery_on else 'off'} · θ_T/θ_W {a.theta_T}/{a.theta_W} · candidates/task {a.n_cand}",
             "X* = จุดบน front ใกล้ ideal (1,1,1) ที่สุด (L2 บนค่าที่ optimizer เห็น) · *_real = replay ด้วย y จริง (เงื่อนไข ฎ) · f₃* ใช้ Φ^max และ T⁺ ของ oracle เป็นอ้างอิงร่วม · HV ref (0,0,0)", "",
             "## สรุปต่อ arm (mean over instances)", "", md(agg_r.round(4)), "",
             "## Value of Prediction — Δ vs qm_model (paired ต่อ instance · bootstrap 95 % CI · Wilcoxon)", "", "### f₁* (realised fulfilment)", "", md(vop.round(4)) if len(vop) else "—", "",
             "### f₃* (realised service equity)", "", md(vop3.round(4)) if len(vop3) else "—", "", "### HV ของ front จริง", "", md(voph.round(4)) if len(voph) else "—", "",
             "## Gates", "", f"- G2.4 corr(f₁,f₃) บน front ของ qm_model: {g24:+.3f} → {'✅ < 0.9' if g24 < 0.9 else '❌ ≥ 0.9 (fallback f₃ ตาม Remark 2)'}",
             f"- G2.6 oracle > model > persistence (f₁*): {o_.get('oracle', float('nan')):.3f} > {o_.get('qm_model', float('nan')):.3f} > {o_.get('qm_persist_cont', float('nan')):.3f} → {'✅' if g26 else '❌'}", "",
             "## อ่าน", "- Δ ลบ = arm นั้นแย่กว่า (9a′) Transformer · ตัวเลขหลักของ paper: Δf₁*(qm_tfm_ub) = ผลของ estimator บน encoder เดียวกัน · Δf₁*(qm_lstm_reach) = ผลของ encoder บน estimator เดียวกัน · Δ(qm_trajgeom) = เทียบ control เดิม",
             "- f1_plan vs f1_real ต่างกันมาก = C_ij ปรับเทียบไม่ตรงกับคู่ที่ถูกเลือก (optimizer เลือกคู่ที่ C สูงเกินจริง = winner's curse) — ดู oracle gap", ""]
    (out / "E4_report.md").write_text("\n".join(lines), encoding="utf-8"); log(f"→ {out / 'E4_report.md'}")
    if not a.no_figs: make_figs(out, df, city, log)
    return df

def make_figs(out, df, city, log):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e: log(f"ไม่มี matplotlib ({e})"); return
    arms = [x for x in ARMS if x in df.arm.unique()]; piv = df.pivot(index="k", columns="arm", values="f1_real")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].boxplot([piv[x].dropna() for x in arms], tick_labels=[LABEL[x].replace(" [main]", "") for x in arms], showmeans=True); ax[0].set_ylabel("realised f₁*"); ax[0].tick_params(axis="x", rotation=25, labelsize=8); ax[0].set_title(f"{city} · realised fulfilment by C_ij source")
    if "qm_model" in piv:
        for x in arms:
            if x == "qm_model": continue
            d = (piv[x] - piv["qm_model"]).dropna(); ax[1].scatter(np.full(len(d), arms.index(x)), d, s=12, alpha=.6); ax[1].scatter([arms.index(x)], [d.mean()], s=60, marker="_", color="k")
        ax[1].axhline(0, color="#888", lw=1, ls=":"); ax[1].set_xticks(range(len(arms))); ax[1].set_xticklabels([LABEL[x].replace(" [main]", "") for x in arms], rotation=25, fontsize=8); ax[1].set_ylabel("Δf₁* vs (9a′) Transformer (paired)"); ax[1].set_title("Value of Prediction")
    fig.tight_layout(); fig.savefig(out / "fig_e4_vop.png", dpi=140); plt.close(fig)

# ----------------------------------------------------------------------------- selftest
def selftest(log):
    import tempfile
    root = Path(tempfile.mkdtemp(prefix="e4_selftest_")); rng = np.random.RandomState(0); W, T = 40, 80
    for k in range(2):
        Y = (rng.rand(W, T) < 0.04).astype(np.uint8)                      # oracle
        base = np.clip(Y * rng.uniform(0.2, 0.8, (W, T)) + (1 - Y) * rng.uniform(0, 0.08, (W, T)), 0, 1)
        z = dict(task_K=rng.choice([1, 2, 3], T, p=[.6, .3, .1]).astype(np.int8), task_r=rng.uniform(.5, 1.5, T).astype(np.float32), task_eps=rng.uniform(2, 8, T).astype(np.float32), w_e=rng.uniform(40, 100, W).astype(np.float32),
                 meta=json.dumps({"k": k}), qm_model_R200=base, qm_lstm_reach_R200=np.clip(base + rng.normal(0, .02, (W, T)), 0, 1), qm_tfm_ub_R200=np.clip(base * 0.8 + rng.normal(0, .03, (W, T)), 0, 1),
                 qm_trajgeom_R200=np.clip(base * 0.75 + rng.normal(0, .04, (W, T)), 0, 1), qm_null_dv_R200=rng.uniform(0, .1, (W, T)), qm_persist_cont_R200=rng.uniform(0, .1, (W, T)))
        d = root / "inst" / "_t"; d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / f"inst_{k:03d}.npz", **z); np.savez_compressed(d / f"oracle_{k:03d}.npz", qm_oracle_R200=Y)
    a = argparse.Namespace(city="_t", inst=str(root / "inst"), out=str(root / "e4"), pilot=0, arms=",".join(ARMS), pop=40, gen=30, seed=0, R=200, q_max=3, battery_on=False, theta_T=.05, theta_W=.05, n_cand=6, L_rule="one", no_figs=False)
    df = main_city(a, "_t", log)
    assert set(df.arm) == set(ARMS) and len(df) == 14
    assert (df.f1_real.between(0, 1) & df.f2_real.between(0, 1) & df.f3_real.between(0, 1)).all()
    o = df[df.arm == "oracle"].f1_real.mean(); m = df[df.arm == "qm_model"].f1_real.mean(); n = df[df.arm == "qm_null_dv"].f1_real.mean()
    assert o >= m >= n, f"oracle {o:.3f} ≥ model {m:.3f} ≥ null {n:.3f} ไม่เป็นจริง"
    # ตรวจ DP Poisson-binomial กับ L = 2
    phi, full = phi_pb(np.array([[[0.5, 0.5, 0.0]]]), np.array([2])); assert abs(phi[0, 0] - (0.75 + 0.25) / 2) < 1e-9 and abs(full[0, 0] - 0.25) < 1e-9, (phi, full)   # E[min(N,2)]/2 = (Pr[N≥1]+Pr[N≥2])/2
    phi1, _ = phi_pb(np.array([[[0.3, 0.2, 0.0]]]), np.array([1])); assert abs(phi1[0, 0] - (1 - 0.7 * 0.8)) < 1e-9
    log(f"selftest ผ่าน ✅ → {root}"); print((root / "e4" / "_t" / "E4_report.md").read_text(encoding="utf-8")[:2500])

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", default="sf"); ap.add_argument("--inst", default="instances_v3"); ap.add_argument("--out", default="e4lite"); ap.add_argument("--pilot", type=int, default=0, help="ใช้ <inst>/<city>/pilot/ และจำกัด N instance")
    ap.add_argument("--arms", default=",".join(ARMS)); ap.add_argument("--pop", type=int, default=80); ap.add_argument("--gen", type=int, default=120); ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--R", type=int, default=200); ap.add_argument("--q-max", type=int, default=3); ap.add_argument("--battery-on", action="store_true"); ap.add_argument("--theta-T", type=float, default=0.05); ap.add_argument("--theta-W", type=float, default=0.05)
    ap.add_argument("--n-cand", type=int, default=8, help="candidate worker ต่องาน (top-N ตาม C ของ arm)"); ap.add_argument("--L-rule", choices=["one", "K"], default="one")
    ap.add_argument("--no-figs", action="store_true"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(); t0 = time.time(); log = lambda *m: print(f"[{time.time()-t0:7.1f}s]", *m, flush=True)
    try: import pymoo  # noqa
    except ImportError: sys.exit("ต้องติดตั้ง pymoo: pip install pymoo")
    if a.selftest: selftest(log); return
    for city in [c.strip() for c in a.city.split(",") if c.strip()]: main_city(a, city, log)

if __name__ == "__main__": main()
