#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sE5_replan.py (E5 v0.1, 2026-10-21) — rolling re-plan ภายในรอบ 60 นาที เทียบ static (E4) ด้วย replay เดียวกัน
================================================================================================= กติกา (= scope §10 ฉบับยืนยัน — ประกาศก่อนรัน)
  หน่วย       : instance ของ E1 v3.1 (งาน/คน/K/ε/หน้าต่าง/path_true เดิมทุกบิต) · จุดตัดสินใจ t ∈ T = {0, Δ, 2Δ, …} (ค่าหลัก Δ = 10 → 6 จุด)
  C_ij(t)     : จาก sE5_instances (ตัวประมาณชุดเดิม บนประวัติที่มีถึง t และหน้าต่างที่เหลือ) · arm เดียวกันทุกจุดเวลา
  งานที่คิด    : ที่ t วางแผนเฉพาะงาน "เปิด" = ยังไม่สำเร็จ ∧ et_i > t · งานที่สำเร็จแล้วนับถาวร (L_i = 1: สำเร็จครั้งแรกจบ)
  offer       : แผน X(t) คือ allocation เต็มของงานเปิด · offer ที่ยังไม่เกิดผลถอนได้ (--withdraw free, ค่าหลัก) → โควตาถูกใช้เฉพาะเมื่อเกิดผล
                 sensitivity --withdraw sticky: offer ที่เคยให้แล้วนับโควตา Q_max/พลังงานถาวรแม้ถอน
  โควตา       : Q_rem_j = Q_max − #งานที่ j ทำสำเร็จแล้ว (sticky: − #offer ที่เคยให้) · e_rem_j = e_j − Σ ε ของงานสำเร็จ (ถ้า battery on) · K_i คงเดิม (งานเปิดยังไม่มีใครสำเร็จ)
  f₂ ที่ t     : RW_past_j = Σ r_i ของงานที่ j ทำสำเร็จแล้ว (จ่ายเมื่อส่งมอบ A10) · W⁺ ของ arm จาก C(t=0) คงที่ทั้งรอบ
  f₃ ที่ t     : Φ^max_own(t) จาก C(t) ของงานเปิด · T⁺(t) ตามนั้น · f₃* ตอนจบ: Jain(done_i / Φ^max_ref(oracle,t=0)) บน T⁺_ref (เหมือน E4)
  optimizer   : NSGA-II เดิมของ E4 (pop 80, SBX/PM เดิม, greedy seed 2 ตัว) + **warm-start**: แผน t−Δ (เฉพาะงานที่ยังเปิด) เป็นสมาชิกตัวที่ 3
                 งบ: --gen ต่อจุดตัดสินใจ (ค่าหลัก 480 = เท่า static ต่อครั้ง; sensitivity --budget total → gen = 480/|T|)
  E6 (v0.5)   : --algo mopso|mopso_q|mopso_iq เรียก sE6_algos.run_algo ด้วย (prob, pop, gen, seed, warm) เดียวกัน → เทียบอัลกอริทึมบนลูป/replay/instance เดียวกัน · ค่าหลัก nsga2 ไม่เปลี่ยนพฤติกรรม v0.4
  X*(t)       : จุดบน front ใกล้ ideal (1,1,1) ที่สุด (L2 บนค่าที่ optimizer เห็น ณ t) — กฎเดียวกับ E4 · รายงาน max-f₁ ด้วย
  replay      : ระหว่าง (t, t+Δ] คู่ (i,j) ใน X*(t) สำเร็จถ้า ∃ ก้าว h ∈ (t, min(t+Δ, et_i)] ∩ [st_i, et_i] ที่ ‖path_j(h) − l_i‖ ≤ R (นิยาม (8), P=1)
                 static = loop เดียวกันที่ T = {0} (Δ = 60) → ต้องให้ f₁* ≈ E4 (sanity S1)
  ตัวชี้วัด   : f₁* = #done/|T_tasks| · f₂* = Jain(RW_final) บน W⁺ · f₃* ดังบน · Δ(adaptive − static) paired ต่อ instance ต่อ arm, bootstrap CI, Wilcoxon · timeline done(t)
  gates       : G5.1 adaptive ≥ static สำหรับ qm_model (arm ที่มีสารสนเทศ) — baseline ไร้สารสนเทศอาจแย่ลงจากการถอน offer (รายงาน ไม่ใช่ gate) · G5.2 |oracle adaptive − oracle static| ≤ 0.02 (oracle รู้อนาคตแล้ว re-plan ไม่ควรเพิ่ม)
  ห้าม        : ปรับ Δ/กฎเลือกจุด/θ/n_cand หลังเห็นผล · ทุกชุดที่รันอยู่ใน appendix
=================================================================================================
ใช้:  python sE5_replan.py --city sf --limit 10 --arms qm_model,qm_persist_cont,oracle --gen 120 --out e5replan_pilot     # pilot
      python sE5_replan.py --city sf --gen 480 --out e5replan                                                              # ชุดเต็ม
ผล:   <out>/<city>/e5_runs.csv · e5_timeline.csv · E5_report.md · (v0.6) e5_trace.csv (HV ทุก 20 gen) · e5_fronts.npz (front plan/real ทุกจุดตัดสินใจ)
"""
from __future__ import annotations
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sE4_lite as E4                                            # Problem, jain, phi_pb, paired_summary, hv3 (ไม่แก้)
VERSION = "E5-replan v1.1 (2026-10-24: v1.0 + --algo greedy = baseline เรียงคู่ตาม C มาก→น้อย เคารพ K_i/โควตา/pending · ไม่มีการค้นหา) · v1.0 (2026-10-24: v0.9 + ชุดป้องกัน gate (SPEC #4): --delay t_d · --gate-fixed τ · --gate-raw (gate บน C ดิบ, V จาก rho.json) · --drop-zero (ไม่ commit offer ใหม่ที่ C = 0) · log n_zero_new · ไม่ใส่ flag = v0.9 ทุกบิต) · v0.9 (2026-10-24: v0.8 + --qgate qgate.json (B1): ต่อคนขับ เดินคู่เรียง C̃ มาก→น้อย เสนอเมื่อ Q_offer ≥ Q_wait → m_j ช่องใหม่ · ตัดคู่ที่ C̃ ต่ำกว่าคู่ที่ m_j · โควตาใหม่ ≤ m_j · --qgate-cv fold = k mod 5 · sticky เท่านั้น · ไม่ใส่ flag = เดิมทุกบิต)"
LAST_TRACE = []                                                   # (gen, evals, hv) ของการแก้ครั้งล่าสุด (nsga2 ผ่าน callback · mopso ผ่าน run_mopso.last)
ARMS = ["qm_model", "qm_lstm_reach", "qm_tfm_ub", "qm_null_dv", "qm_persist_cont", "oracle"]
LABEL = dict(E4.LABEL, oracle="oracle (y จริง)", qm_model_dg="(9a′) Transformer + duration guard ≥5 [S-a]", qm_model_nog="(9a′) Transformer no guard (diag)")


class ProblemT(E4.Problem):
    """Problem ของ E4 + โควตาคงเหลือต่อ worker สำหรับ offer ใหม่ (Q_rem) · pending[i,j] = offer ที่ค้างอยู่จากรอบก่อน เก็บไว้ได้ฟรี (ไม่กินโควตาเพิ่ม)
    free   : Q_rem = Q_max − #สำเร็จ, pending = ว่าง (offer ถอนได้ คืนโควตา)
    sticky : Q_rem = Q_max − #คู่ที่เคยเสนอ, pending = offer ปัจจุบันบนงานเปิด → เก็บได้ฟรี / ถอนแล้วโควตาไม่คืน / offer ใหม่กินโควตา  [fix 2026-10-22 09:00]"""
    def __init__(self, C, K, L, r, eps, Q_rem, e_rem, RWpast, battery_on, theta_T, theta_W, n_cand, phimax_ref=None, pending=None):
        Q_rem = np.asarray(Q_rem).astype(int); C = np.asarray(C, np.float64)
        self.pending = np.zeros(C.T.shape, bool) if pending is None else np.asarray(pending, bool)          # (T, W)
        usable = (Q_rem[None, :] > 0) | self.pending.astype(bool)                                             # (T, W): เลือกได้ถ้ามีโควตาใหม่ หรือเป็น offer ค้าง
        C2 = np.where(usable.T, C, 0.0)                                                                      # worker ที่ใช้ไม่ได้กับงานนั้น: C = 0 → ไม่เป็น candidate
        super().__init__(C2, K, L, r, eps, e_rem, RWpast, int(np.max(Q_rem)) if len(Q_rem) else 0, battery_on, theta_T, theta_W, n_cand, phimax_ref)
        self.Q_rem = Q_rem
    def decode(self, G):
        if self.battery_on: return self.decode_ref(G)
        return self._repair_quota(self._assign(G), self.Q_rem, self.pending)          # v0.6.1 vectorised (เท่า decode_ref ทุกบิต — ทดสอบแล้ว)
    def decode_ref(self, G):
        P = G.shape[0]; x = np.full((P, self.T, self.Smax), -1, int); g = G.astype(int)
        for v in range(self.n_var):
            i, s = self.slot_task[v], self.slot_pos[v]; sel = g[:, v] - 1
            x[:, i, s] = np.where(sel >= 0, self.cand[i, np.clip(sel, 0, self.n_cand - 1)], -1)
        for s in range(1, self.Smax):
            for s2 in range(s):
                dup = (x[:, :, s] == x[:, :, s2]) & (x[:, :, s] >= 0); x[:, :, s][dup] = -1
        for p in range(P):
            xs = x[p]; flat = xs.ravel(); idx = np.flatnonzero(flat >= 0)
            if not len(idx): continue
            wj = flat[idx]; ti = idx // self.Smax; cj = self.C[wj, ti]; order = np.lexsort((cj, wj)); wj_o, ti_o, idx_o = wj[order], ti[order], idx[order]
            for j in np.unique(wj_o):
                m = np.flatnonzero(wj_o == j); q = int(self.Q_rem[j])
                is_pend = self.pending[ti_o[m], j]; m_pend = m[is_pend]; m_new = m[~is_pend]                # offer ค้าง: เก็บฟรี · offer ใหม่: ไม่เกิน Q_rem (เรียง C น้อย→มาก จึงเก็บท้ายลิสต์)
                keep = np.concatenate([m_pend, m_new[max(0, len(m_new) - q):] if q > 0 else m_new[:0]])
                if self.battery_on:
                    while len(keep) and self.eps[ti_o[keep]].sum() > self.e[j]: keep = keep[1:]
                drop = np.setdiff1d(m, keep); flat[idx_o[drop]] = -1
            x[p] = flat.reshape(self.T, self.Smax)
        return x


def run_nsga2_warm(prob, pop, gen, seed, warm_G=None):
    """NSGA-II เดิม (sE4_lite.run_nsga2) + genome warm-start เพิ่มเป็นสมาชิกตัวที่ 3"""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem as PmProblem
    from pymoo.operators.sampling.rnd import IntegerRandomSampling
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.repair.rounding import RoundingRepair
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    from pymoo.core.sampling import Sampling
    class PP(PmProblem):
        def __init__(s): super().__init__(n_var=prob.n_var, n_obj=3, n_ieq_constr=0, xl=0, xu=prob.n_cand, vtype=int)
        def _evaluate(s, G, out, *a, **k):
            x = prob.decode(G); o = prob.objectives(x, prob.C, prob.phimax_own); out["F"] = -np.stack([o["f1"], o["f2"], o["f3"]], 1)
    class Seeded(Sampling):
        def _do(s, problem, n_samples, **kw):
            X = IntegerRandomSampling()._do(problem, n_samples, **kw).astype(float)
            X[0] = prob.slot_pos + 1; X[1] = np.where(prob.slot_pos == 0, 1, 0)
            if warm_G is not None and n_samples > 2: X[2] = warm_G
            return X
    alg = NSGA2(pop_size=pop, sampling=Seeded(), crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()), mutation=PM(prob=1.0 / max(1, prob.n_var) * 3, eta=20, vtype=float, repair=RoundingRepair()), eliminate_duplicates=True)
    from pymoo.core.callback import Callback
    class Trace(Callback):
        """HV ของ non-dominated set ทุก TRACE_EVERY gen (อ่านอย่างเดียว ไม่แตะ RNG → ผลเท่าเดิม)"""
        def notify(s, algorithm):
            g = int(algorithm.n_gen)
            if g == 1 or g % TRACE_EVERY == 0 or g == gen: LAST_TRACE.append((g, g * pop, E4.hv3(-algorithm.opt.get("F"))))
    LAST_TRACE.clear()
    res = minimize(PP(), alg, get_termination("n_gen", gen), seed=seed, verbose=False, callback=Trace())
    return np.atleast_2d(res.X), -np.atleast_2d(res.F)


TRACE_EVERY = 20


def greedy_plan(prob):
    """[v1.1] baseline greedy: เก็บ offer ค้าง (pending) ไว้ก่อน (ฟรี) → เรียงคู่ (งาน, candidate) ที่ C > 0 จากมาก→น้อย → มอบถ้างานยังไม่ครบ K_i และคนขับยังมีโควตาใหม่ · คืน x (T, Smax)"""
    T, S = prob.T, prob.Smax; x = np.full((T, S), -1, int); nk = np.zeros(T, int); qleft = np.asarray(prob.Q_rem, int).copy(); used = set()
    pend = prob.pending
    for i, j in zip(*np.nonzero(pend)):
        if nk[i] < prob.K[i]: x[i, nk[i]] = j; nk[i] += 1; used.add((int(i), int(j)))
    ii = np.repeat(np.arange(T), prob.cand.shape[1]); jj = prob.cand.ravel(); cc = prob.C[jj, ii]
    keep = cc > 0; ii, jj, cc = ii[keep], jj[keep], cc[keep]
    for o in np.lexsort((jj, ii, -cc)):                                                         # C มาก→น้อย · เสมอ → index งาน, คนขับ
        i, j = int(ii[o]), int(jj[o])
        if (i, j) in used or nk[i] >= prob.K[i] or qleft[j] <= 0: continue
        x[i, nk[i]] = j; nk[i] += 1; qleft[j] -= 1; used.add((i, j))
    return x


def plan_to_genome(x_prev_full, open_idx, prob):
    """แผนเดิม (T_all, Smax) worker index → genome ของ Problem ใหม่ (งานเปิด, candidate ใหม่): slot = อันดับ candidate+1 ถ้ายังอยู่ใน top-n_cand ไม่งั้น 0"""
    G = np.zeros(prob.n_var)
    for v in range(prob.n_var):
        i_new, s = prob.slot_task[v], prob.slot_pos[v]; i_old = open_idx[i_new]
        if s < x_prev_full.shape[1] and x_prev_full[i_old, s] >= 0:
            j = x_prev_full[i_old, s]; pos = np.flatnonzero(prob.cand[i_new] == j)
            if len(pos): G[v] = pos[0] + 1
    return G


def hits_table(path, task_xy, R):
    """hit[j,i,h] = ‖path_j(h) − l_i‖ ≤ R · h = 1..H (index h−1)"""
    d = np.sqrt(((path[:, None, :, :] - task_xy[None, :, None, :]) ** 2).sum(-1))   # (W,T,H)
    return d <= R


def solve_instance(city, k, inst_dir, e5_dir, arms, t_list, a, log):
    z = np.load(inst_dir / f"inst_{k:03d}.npz", allow_pickle=True); o = np.load(inst_dir / f"oracle_{k:03d}.npz")
    K = z["task_K"].astype(int); L = np.ones_like(K); r = z["task_r"].astype(float); eps = z["task_eps"].astype(float); e0 = z["w_e"].astype(float)
    st = z["task_st"].astype(int); et = z["task_et"].astype(int); task_xy = z["task_xy"].astype(np.float64); path = o["path_true"].astype(np.float64)
    W, T = path.shape[0], len(task_xy); H = path.shape[1]; R = float(a.R)
    hit = hits_table(path, task_xy, R)                                                     # (W,T,H)
    win = np.zeros((T, H), bool)
    for i in range(T): win[i, st[i] - 1:et[i]] = True
    Z = {tm: np.load(e5_dir / f"inst_{k:03d}_t{tm:02d}.npz") for tm in t_list}
    Y0 = Z[0]["oracle"].astype(float); ref = E4.Problem(Y0, K, L, r, eps, e0, np.zeros(W), a.q_max, a.battery_on, a.theta_T, a.theta_W, a.n_cand)
    phimax_ref, Tplus_ref = ref.phimax_own, ref.Tplus
    rows, tl, trrows, fronts = [], [], [], {}
    for arm in arms:
        t0_ = time.time(); sec_solves = []; hv_plan0 = hv_real0 = f1_plan0 = f1_exp0 = float("nan"); nfront0 = 0
        C0 = Z[0]["oracle"].astype(float) if arm == "oracle" else np.nan_to_num(Z[0][arm].astype(float))
        Wplus = C0.max(1) >= a.theta_W
        done = np.zeros(T, bool); done_by = np.full(T, -1, int); n_succ_j = np.zeros(W, int); n_off_j = np.zeros(W, int); e_used = np.zeros(W); offered = set()   # sticky: คู่ (i,j) ที่เคยเสนอ นับครั้งเดียว
        x_full = np.full((T, int(K.max())), -1, int); n_changed = []; corr13 = []
        steps = list(t_list) + [H]
        for si, tm in enumerate(t_list):
            t_next = steps[si + 1]
            open_ = (~done) & (et > tm); open_idx = np.flatnonzero(open_)
            if len(open_idx) == 0: x_full[:] = -1; tl.append(dict(k=k, arm=arm, mode=a.mode_tag, t=tm, n_open=0, done=int(done.sum()))); continue
            C = Z[tm]["oracle"].astype(float) if arm == "oracle" else np.nan_to_num(Z[tm][arm].astype(float))
            C = C[:, open_idx]
            if getattr(a, "rho", None) is not None and arm != "oracle" and not getattr(a, "gate_raw", False):   # [v0.7 A1] (v1.0: --gate-raw ไม่ปรับ) C̃ = ρ(h_i)·C · h_i = หน้าต่างที่เหลือของงาน i ที่ t
                import sE6_algos as E6; C = np.minimum(1.0, C * E6.rho_factor(a.rho, et[open_idx] - tm)[None, :])
            q_used = n_off_j if a.withdraw == "sticky" else n_succ_j
            Q_rem = np.maximum(a.q_max - q_used, 0); e_rem = e0 - e_used
            RWpast = np.zeros(W)
            for i in np.flatnonzero(done): RWpast[done_by[i]] += r[i]
            pend = None
            if a.withdraw == "sticky" and tm > 0:
                pend = np.zeros((len(open_idx), W), bool)
                for ii_, i in enumerate(open_idx):
                    for j in x_full[i]:
                        if j >= 0: pend[ii_, j] = True
            n_gated = 0
            if getattr(a, "gate_kappa", None) is not None and arm != "oracle":                        # [v0.8 B0] gate เสนอ/รอ: คู่ใหม่ต้องมี C̃ ≥ κ·V_{Q_rem_j}(t) · pending เก็บได้เสมอ
                Vt = np.asarray(a.rho["V"][str(tm)], float); kq = np.clip(Q_rem, 1, len(Vt)) - 1; thr = a.gate_kappa * Vt[kq]       # (W,)
                blocked = C < thr[:, None]
                if pend is not None: blocked &= ~pend.T
                n_gated = int((blocked & (C > 0)).sum()); C = np.where(blocked, 0.0, C)
            thr_alt = None                                                                           # [v1.0 SPEC #4] gate ทางเลือก (ใช้ทีละแบบ)
            if getattr(a, "gate_fixed", None) is not None: thr_alt = a.gate_fixed                    #   G2: C̃ ≥ τ คงที่ (ไม่ขึ้นกับ t, q)
            elif getattr(a, "delay", None) is not None: thr_alt = np.inf if tm < a.delay else None   #   G1: ไม่เสนอคู่ใหม่ก่อน t_d · หลังจากนั้นไม่มี gate
            if thr_alt is not None and arm != "oracle":
                blocked = C < thr_alt
                if pend is not None: blocked &= ~pend.T
                n_gated = int((blocked & (C > 0)).sum()); C = np.where(blocked, 0.0, C)
            Q_use = Q_rem
            if getattr(a, "qg", None) is not None and arm != "oracle":                                 # [v0.9 B1] Q-gate ต่อคนขับ
                tab = a.qg["tables"]["full" if not a.qgate_cv else str(k % int(a.qg["folds"]))]; Qo = np.asarray(tab["Qo"]); Qw = np.asarray(tab["Qw"])
                ti = a.qg["t_list"].index(tm); ce = np.array(a.qg["c_edges"]); he = np.array(a.qg["h_edges"]); hrem = et[open_idx] - tm
                hb = np.searchsorted(he, hrem, side="left"); Q_use = Q_rem.copy(); blocked = np.zeros_like(C, bool)
                for j in range(W):
                    q = int(Q_rem[j])
                    if q <= 0: continue
                    newc = C[j].copy()
                    if pend is not None: newc[pend[:, j]] = 0.0                                          # pending ไม่นับเป็นคู่ใหม่
                    o_ = np.argsort(-newc); o_ = o_[newc[o_] > 0]; m = 0
                    for p in range(min(q, len(o_))):
                        cb = int(np.searchsorted(ce, newc[o_[p]], side="right")); qq = q - m
                        if Qo[ti, qq, cb, hb[o_[p]]] >= Qw[ti, qq, cb, hb[o_[p]]] - 1e-12: m += 1
                        else: break
                    Q_use[j] = m
                    thr = newc[o_[m - 1]] if m > 0 else np.inf
                    blk = newc < thr
                    if pend is not None: blk &= ~pend[:, j]
                    blocked[j] = blk
                n_gated = int((blocked & (C > 0)).sum()); C = np.where(blocked, 0.0, C)
            prob = ProblemT(C, K[open_idx], L[open_idx], r[open_idx], eps[open_idx], Q_use, e_rem, RWpast, a.battery_on, a.theta_T, a.theta_W, a.n_cand, pending=pend)
            is_greedy = getattr(a, "algo", "nsga2") == "greedy"
            warm = plan_to_genome(x_full, open_idx, prob) if (tm > 0 and a.warm and not is_greedy) else None
            t_solve = time.time()
            if is_greedy: G = None; trace = []
            elif getattr(a, "algo", "nsga2") == "nsga2": G, F = run_nsga2_warm(prob, a.pop, a.gen, a.seed + k, warm); trace = list(LAST_TRACE)
            else:
                import sE6_algos as E6; G, F = E6.run_algo(a.algo, prob, a.pop, a.gen, a.seed + k, warm, informed_frac=getattr(a, "inf_frac_val", None))     # E6: optimizer อื่น งบ/seed/warm เดียวกัน
                tr = E6.run_mopso.last["hv_trace"]; trace = [(int(g) + 2, (int(g) + 2) * a.pop, float(h)) for g, h in tr if (int(g) + 2) % TRACE_EVERY == 0 or int(g) == 0 or int(g) == len(tr) - 1]
            sec_solve = time.time() - t_solve
            x = greedy_plan(prob)[None] if is_greedy else prob.decode(G); plan = prob.objectives(x, prob.C, prob.phimax_own)
            Fp = np.stack([plan["f1"], plan["f2"], plan["f3"]], 1); i_sel = int(np.argmin(np.linalg.norm(1.0 - Fp, axis=1)))
            if len(Fp) > 2 and Fp[:, 0].std() > 0 and Fp[:, 2].std() > 0: corr13.append(float(np.corrcoef(Fp[:, 0], Fp[:, 2])[0, 1]))
            # ---- E6 logging: front ที่ "realised-expectation" ใต้ oracle ของหน้าต่างที่เหลือ (แบบ E4) · HV ทั้งสอง · โควตา/offer · trace (ไม่กระทบผล)
            Yt = Z[tm]["oracle"].astype(float)[:, open_idx]; Tp_save = prob.Tplus; prob.Tplus = Tplus_ref[open_idx]
            real_e = prob.objectives(x, Yt, phimax_ref[open_idx]); prob.Tplus = Tp_save
            Fr = np.stack([real_e["f1"], real_e["f2"], real_e["f3"]], 1)
            hv_p, hv_r = E4.hv3(Fp), E4.hv3(Fr)
            if a.save_fronts: fronts[f"{k}_{arm}_{a.mode_tag}_t{tm:02d}_plan"] = Fp.astype(np.float32); fronts[f"{k}_{arm}_{a.mode_tag}_t{tm:02d}_real"] = Fr.astype(np.float32); fronts[f"{k}_{arm}_{a.mode_tag}_t{tm:02d}_isel"] = np.array([i_sel])
            for g_, ev_, h_ in trace: trrows.append(dict(k=k, arm=arm, mode=a.mode_tag, algo=getattr(a, "algo", "nsga2"), t=tm, gen=g_, evals=ev_, hv=h_))
            if tm == 0: hv_plan0, hv_real0, f1_plan0, f1_exp0, nfront0 = hv_p, hv_r, float(Fp[i_sel, 0]), float(Fr[i_sel, 0]), int(len(Fp))
            sec_solves.append(sec_solve)
            xs = x[i_sel].copy(); tt_, ss_ = np.nonzero(xs >= 0); jj_ = xs[tt_, ss_]                  # [v1.0] offer ใหม่ที่ C (หลัง gate/โควตา) = 0: ไม่มีผลต่อ f1–f3 แต่กินโควตาใต้ sticky
            zmask = prob.C[jj_, tt_] <= 0
            if pend is not None: zmask &= ~pend[tt_, jj_]
            n_zero_new = int(zmask.sum()); n_new_all = int(len(jj_) - (0 if pend is None else pend[tt_, jj_].sum()))
            if getattr(a, "drop_zero", False): xs[tt_[zmask], ss_[zmask]] = -1
            x_new = np.full((T, int(K.max())), -1, int); x_new[open_idx, :x.shape[2]] = xs
            if tm > 0: n_changed.append(int(((x_new != x_full) & ((x_new >= 0) | (x_full >= 0)))[open_idx].sum()))
            x_full = x_new
            if a.withdraw == "sticky":                                                       # [fix 2026-10-22 08:50] เดิมนับ offer เดิมซ้ำทุกจุดตัดสินใจ → โควตาหมดใน 2 รอบ (ผล sticky ชุดแรกใช้ไม่ได้)
                for i in open_idx:
                    for j in x_full[i]:
                        if j >= 0 and (int(i), int(j)) not in offered: offered.add((int(i), int(j))); n_off_j[j] += 1
            # ---- replay (t, t_next]: คู่ใน X*(t) สำเร็จถ้ามี hit ในหน้าต่างงาน ∩ (t, t_next]
            h_lo, h_hi = tm, min(t_next, H)                                                  # index h−1 ∈ [tm, t_next)
            for i in open_idx:
                js = [j for j in x_full[i] if j >= 0]
                if not js: continue
                for j in js:
                    if (hit[j, i, h_lo:h_hi] & win[i, h_lo:h_hi]).any():
                        done[i] = True; done_by[i] = j; n_succ_j[j] += 1; e_used[j] += eps[i]; break
            q_used_now = n_off_j if a.withdraw == "sticky" else n_succ_j
            tl.append(dict(k=k, arm=arm, mode=a.mode_tag, algo=getattr(a, "algo", "nsga2"), t=tm, n_open=int(len(open_idx)), done=int(done.sum()), f1_plan=float(plan["f1"][i_sel]), f1_exp=float(Fr[i_sel, 0]), n_front=int(len(Fp)),
                           hv_plan=hv_p, hv_real=hv_r, offers_out=int((x_full[open_idx] >= 0).sum()), quota_used=float(np.minimum(q_used_now, a.q_max).sum() / max(1, W * a.q_max)), sec_solve=round(sec_solve, 1), n_gated=n_gated, n_zero_new=n_zero_new, n_new=n_new_all))
        # ---- ตัวชี้วัดตอนจบ
        RW = np.zeros(W)
        for i in np.flatnonzero(done): RW[done_by[i]] += r[i]
        f1 = float(done.mean()); f2 = float(E4.jain(RW[None, :], Wplus)[0]) if Wplus.any() else float("nan")
        s_i = np.where(phimax_ref > 0, done / np.maximum(phimax_ref, 1e-12), 0.0); s_i = np.clip(s_i, 0, 1); f3 = float(E4.jain(s_i[None, :], Tplus_ref)[0]) if Tplus_ref.any() else float("nan")
        rows.append(dict(k=k, arm=arm, mode=a.mode_tag, algo=getattr(a, "algo", "nsga2"), shrink=bool(getattr(a, "rho", None) is not None and not getattr(a, "gate_raw", False)), gate_raw=bool(getattr(a, "gate_raw", False)), gate_fixed=getattr(a, "gate_fixed", None), delay=getattr(a, "delay", None), drop_zero=bool(getattr(a, "drop_zero", False)), inf_frac=getattr(a, "inf_frac_val", None), gate_kappa=getattr(a, "gate_kappa", None), qgate=("cv" if a.qgate_cv else "full") if getattr(a, "qg", None) is not None else None, n_steps=len(t_list), f1_real=f1, f2_real=f2, f3_real=f3, n_done=int(done.sum()), outlay=float(RW.sum()),
                         n_changed_mean=float(np.mean(n_changed)) if n_changed else 0.0, corr_f1f3=float(np.mean(corr13)) if corr13 else float("nan"), sec=round(time.time() - t0_, 1),
                         sec_per_solve=round(float(np.mean(sec_solves)), 1) if sec_solves else float("nan"), hv_plan_t0=hv_plan0, hv_real_t0=hv_real0, f1_plan_t0=f1_plan0, f1_exp_t0=f1_exp0, n_front_t0=nfront0,
                         quota_used_final=float(np.minimum(n_off_j if a.withdraw == "sticky" else n_succ_j, a.q_max).sum() / max(1, W * a.q_max))))
        log(f"    inst {k:03d} {a.mode_tag:8s} {arm:16s} {getattr(a, 'algo', 'nsga2'):8s} done {int(done.sum()):3d}/{T} f1* {f1:.3f} · f2* {f2:.3f} · f3* {f3:.3f} · เปลี่ยน/ครั้ง {np.mean(n_changed) if n_changed else 0:.1f} · {time.time()-t0_:.0f}s")
    return rows, tl, trrows, fronts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", default="sf"); ap.add_argument("--inst", default="instances_v3"); ap.add_argument("--e5", default="e5inst"); ap.add_argument("--out", default="e5replan")
    ap.add_argument("--arms", default=",".join(ARMS)); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--t", default="0,10,20,30,40,50")
    ap.add_argument("--pop", type=int, default=80); ap.add_argument("--gen", type=int, default=480); ap.add_argument("--budget", choices=["per-step", "total"], default="per-step")
    ap.add_argument("--seed", type=int, default=12345); ap.add_argument("--R", type=float, default=200.0); ap.add_argument("--q-max", type=int, default=3); ap.add_argument("--battery-on", action="store_true")
    ap.add_argument("--theta-T", type=float, default=0.05); ap.add_argument("--theta-W", type=float, default=0.05); ap.add_argument("--n-cand", type=int, default=8)
    ap.add_argument("--algo", choices=["nsga2", "greedy", "mopso", "mopso_q", "mopso_iq", "mopso_rk", "mopso_q_rk", "mopso_iq_rk"], default="nsga2", help="optimizer (E6; nsga2 = เดิม) · อื่น ๆ ใช้ sE6_algos.py")
    ap.add_argument("--shrink", default="", help="[v0.7] rho.json ของ sE6_rho (ต่อเมือง) → C̃ = ρ(h)·C ทุก arm ยกเว้น oracle (MOPSO-IQ v2 ส่วน A1 · ablation: ใช้กับ nsga2 ได้)")
    ap.add_argument("--inf-frac", default="", help="[v0.7] สัดส่วน informed init ของ mopso_iq*: ตัวเลข หรือ 'auto' = inf_frac จาก rho.json (A2) · ว่าง = 0.30 เดิม")
    ap.add_argument("--gate", type=float, default=None, help="[v0.8 B0] κ ของ gate เสนอ/รอ (ต้องมี --shrink เพื่ออ่าน V(t)) · ว่าง = ไม่มี gate")
    ap.add_argument("--qgate", default="", help="[v0.9 B1] qgate.json ของ sE6_qgate (ต้องมี --shrink, sticky เท่านั้น)"); ap.add_argument("--qgate-cv", action="store_true", help="[v0.9] ใช้ตาราง fold = k mod folds (ประเมิน dev แบบ cross-fit)")
    ap.add_argument("--delay", type=int, default=None, help="[v1.0 G1] ไม่เสนอคู่ใหม่ที่ t < t_d (นาที) · หลังจากนั้นไม่มี gate")
    ap.add_argument("--gate-fixed", type=float, default=None, help="[v1.0 G2] เสนอคู่ใหม่เมื่อ C̃ ≥ τ (คงที่)")
    ap.add_argument("--gate-raw", action="store_true", help="[v1.0 G3] ใช้กับ --gate: เทียบ C ดิบ (ไม่ใช้ ρ) กับ κ·V(t) — ยังต้องมี --shrink เพื่ออ่าน V")
    ap.add_argument("--drop-zero", action="store_true", help="[v1.0] ไม่ commit offer ใหม่ที่ C (หลัง gate) = 0")
    ap.add_argument("--e4", default="e4lite", help="โฟลเดอร์ผล E4 สำหรับ sanity S1 (static ของ E5 vs E4)"); ap.add_argument("--withdraw", choices=["free", "sticky"], default="free"); ap.add_argument("--no-warm", dest="warm", action="store_false"); ap.add_argument("--static-only", action="store_true"); ap.add_argument("--adaptive-only", action="store_true"); ap.add_argument("--no-fronts", dest="save_fronts", action="store_false", help="ไม่บันทึก e5_fronts.npz")
    a = ap.parse_args(); t0 = time.time(); log = lambda *m: print(f"[{time.time()-t0:7.1f}s]", *m, flush=True); log(VERSION)
    a.rho = None; a.inf_frac_val = None; a.gate_kappa = None; a.qg = None
    if a.shrink:
        import sE6_algos as E6; a.rho = E6.load_rho(a.shrink); log(f"[v2-A1] recalibrate C̃ = min(1, ρ(h)·C) จาก {a.shrink} · ρ {np.round(a.rho.get('rho_used', a.rho['rho_xstar_clipped']), 3).tolist()} bins {a.rho['bins']} mode {a.rho.get('rho_mode', 'clip')} (เมือง {a.rho.get('city')})")
    if a.inf_frac:
        a.inf_frac_val = float(a.rho["inf_frac"]) if a.inf_frac == "auto" else float(a.inf_frac); log(f"[v2-A2] informed init fraction = {a.inf_frac_val:.3f}")
        if a.inf_frac == "auto" and a.rho is None: sys.exit("--inf-frac auto ต้องมี --shrink rho.json")
    if a.qgate:
        if a.rho is None or a.withdraw != "sticky": sys.exit("--qgate ต้องมี --shrink และ --withdraw sticky")
        if a.gate is not None: sys.exit("--qgate กับ --gate ใช้พร้อมกันไม่ได้")
        import json as _js; a.qg = _js.load(open(a.qgate, encoding="utf-8")); log(f"[v2-B1] Q-gate {a.qgate} ({a.qg['version']}, เมือง {a.qg['city']}) · {'cross-fit fold = k mod ' + str(a.qg['folds']) if a.qgate_cv else 'ตาราง full'}")
    n_gate = sum(x is not None and x is not False for x in (a.gate, a.gate_fixed, a.delay)) + bool(a.qgate)
    if n_gate > 1: sys.exit("--gate / --gate-fixed / --delay / --qgate ใช้ได้ทีละแบบ")
    if a.gate_raw and a.gate is None: sys.exit("--gate-raw ต้องใช้คู่กับ --gate")
    if a.gate_fixed is not None: log(f"[v1.0 G2] gate คงที่ τ = {a.gate_fixed:g} บน {'C̃' if a.rho is not None else 'C'}")
    if a.delay is not None: log(f"[v1.0 G1] ไม่เสนอคู่ใหม่ก่อน t = {a.delay} นาที")
    if a.gate_raw: log("[v1.0 G3] gate บน C ดิบ (ไม่ปรับ ρ) · V(t) จาก rho.json")
    if a.drop_zero: log("[v1.0] --drop-zero: ไม่ commit offer ใหม่ที่ C = 0")
    if a.gate is not None:
        if a.rho is None: sys.exit("--gate ต้องมี --shrink rho.json (อ่าน V(t))")
        a.gate_kappa = float(a.gate); log(f"[v2-B0] gate เสนอ/รอ κ = {a.gate_kappa:g} · V1(t) {[round(a.rho['V'][str(t_)][0], 3) for t_ in sorted(int(x) for x in a.rho['V'])]}")
    city = a.city; inst_dir = Path(a.inst) / city; e5_dir = Path(a.e5) / city; out = Path(a.out) / city; out.mkdir(parents=True, exist_ok=True)
    arms = [x for x in a.arms.split(",") if x]; t_adapt = [int(x) for x in a.t.split(",")]
    files = sorted(inst_dir.glob("inst_*.npz")); files = files[:a.limit] if a.limit else files
    ks = [int(f.stem.split("_")[1]) for f in files]
    gen_static = a.gen; gen_step = a.gen if a.budget == "per-step" else max(10, a.gen // len(t_adapt))
    log(f"{city}: {len(ks)} instance × {len(arms)} arm · t {t_adapt} · optimizer {a.algo}{' +shrink' if a.rho is not None else ''}{(' inf_frac ' + str(a.inf_frac_val)) if a.inf_frac_val is not None else ''}{(' gate κ=' + str(a.gate_kappa) + (' (C ดิบ)' if a.gate_raw else '')) if a.gate_kappa is not None else ''}{(' τ=' + str(a.gate_fixed)) if a.gate_fixed is not None else ''}{(' delay ' + str(a.delay)) if a.delay is not None else ''}{' drop0' if a.drop_zero else ''}{(' qgate ' + ('cv' if a.qgate_cv else 'full')) if a.qg is not None else ''} pop {a.pop} × gen {gen_step}/จุด ({a.budget}) · warm {a.warm} · withdraw {a.withdraw} · Q_max {a.q_max} · θ {a.theta_T}/{a.theta_W} · cand {a.n_cand} · R {a.R:g}")
    rows, tl, trrows, fronts = [], [], [], {}
    modes = [("static", [0]), ("adaptive", t_adapt)]
    if a.static_only: modes = modes[:1]
    if a.adaptive_only: modes = modes[1:]
    for k in ks:
        for tag, tlist in modes:
            a.mode_tag = tag; a.gen = gen_static if tag == "static" else gen_step
            rr, tt, tr, ff = solve_instance(city, k, inst_dir, e5_dir, arms, tlist, a, log); rows += rr; tl += tt; trrows += tr; fronts.update(ff)
    df = pd.DataFrame(rows); df.to_csv(out / "e5_runs.csv", index=False); pd.DataFrame(tl).to_csv(out / "e5_timeline.csv", index=False); pd.DataFrame(trrows).to_csv(out / "e5_trace.csv", index=False)
    if a.save_fronts and fronts: np.savez_compressed(out / "e5_fronts.npz", **fronts)
    # ---- summary
    piv = df.pivot_table(index=["k", "arm"], columns="mode", values="f1_real").reset_index()
    lines = [f"# E5 rolling re-plan — {city} · {VERSION} · {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"instances {len(ks)} · arms {arms} · Δ = {t_adapt[1] - t_adapt[0] if len(t_adapt) > 1 else 60} นาที ({len(t_adapt)} จุดตัดสินใจ) · optimizer **{a.algo}** pop {a.pop} × gen {gen_step}/จุด ({a.budget}) · warm-start {a.warm} · withdraw {a.withdraw} · R {a.R:g} · L_i = 1 · Q_max {a.q_max} · θ {a.theta_T}/{a.theta_W} · cand {a.n_cand}",
             "X* = ใกล้ ideal (1,1,1) ที่แต่ละจุด · replay ด้วย path_true · f₃* ใช้ Φ^max/T⁺ ของ oracle ที่ t=0", ""]
    md = lambda d: "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n" + "\n".join("| " + " | ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r) + " |" for r in d.values)
    if "static" in piv and "adaptive" in piv:
        piv["delta"] = piv["adaptive"] - piv["static"]; summ = []
        for arm in arms:
            d = piv[piv.arm == arm]["delta"].dropna().to_numpy()
            if not len(d): continue
            rng = np.random.RandomState(0); bs = np.array([rng.choice(d, len(d)).mean() for _ in range(2000)])
            try:
                from scipy.stats import wilcoxon; p = float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0
            except Exception: p = float("nan")
            summ.append(dict(arm=arm, label=LABEL.get(arm, arm), n=len(d), f1_static=piv[piv.arm == arm]["static"].mean(), f1_adaptive=piv[piv.arm == arm]["adaptive"].mean(), delta=d.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), wilcoxon_p=p, wins=int((d > 0).sum()), losses=int((d < 0).sum())))
        S = pd.DataFrame(summ)
        g51 = bool((S[S.arm == "qm_model"].f1_adaptive >= S[S.arm == "qm_model"].f1_static - 1e-9).all()) if "qm_model" in set(S.arm) else True
        g52 = (abs(S[S.arm == "oracle"].delta.iloc[0]) <= 0.02) if "oracle" in set(S.arm) else True
        lines += ["## f₁* static vs adaptive (mean over instances · Δ paired · bootstrap 95 % CI · Wilcoxon)", "", md(S.round(4)), "",
                  "## Gates", "", f"- G5.1 adaptive ≥ static สำหรับ qm_model (ตัวประมาณที่มีสารสนเทศ — ถ้าได้ข้อมูลเพิ่มแล้วแย่ลง = ตรวจ replay/โควตา): {'✅' if g51 else '❌'} · baseline ไร้สารสนเทศ (persistence/null) อาจแย่ลงได้จากการถอน offer — รายงานตามจริง ไม่ใช่ gate",
                  f"- G5.2 |Δ oracle| ≤ 0.02: {'✅' if g52 else '❌'} " + (f"(Δ = {S[S.arm == 'oracle'].delta.iloc[0]:+.4f})" if 'oracle' in set(S.arm) else "(ไม่มี arm oracle)"), ""]
        # VoP ภายใต้ adaptive: Δ vs qm_model
        if "qm_model" in arms:
            vop = E4.paired_summary(df[df["mode"] == "adaptive"], "qm_model", "f1_real")
            lines += ["## Value of Prediction ภายใต้ adaptive — Δf₁* vs qm_model (paired)", "", md(vop.round(4)) if len(vop) else "—", ""]
    T_ = pd.DataFrame(tl); Ta = T_[(T_["mode"] == "adaptive") & T_["arm"].isin(arms)]
    if len(Ta):
        tt = Ta.groupby(["arm", "t"]).agg(done=("done", "mean"), n_open=("n_open", "mean")).reset_index()
        lines += ["## timeline (adaptive) — งานสำเร็จสะสม (mean) หลัง replay ช่วง (t, t+Δ] · static = ค่าสุดท้ายในตาราง f₁* × |T|", "", md(tt.pivot(index="t", columns="arm", values="done").reset_index().round(2)), ""]
    # ---- S1: static ของสคริปต์นี้ vs E4 (replay/optimizer เดียวกัน → ต้องใกล้กันภายใน ΔC ของ sE5)
    e4p = Path(a.e4) / city / "e4_runs.csv" if a.e4 else None
    if e4p and e4p.exists() and "static" in set(df["mode"]):
        e4 = pd.read_csv(e4p)[["k", "arm", "f1_real"]].rename(columns={"f1_real": "f1_E4"}); s1 = df[df["mode"] == "static"][["k", "arm", "f1_real"]].merge(e4, on=["k", "arm"])
        if len(s1):
            d_ = (s1.f1_real - s1.f1_E4); lines += ["## S1 — static ของ E5 vs E4 (instance/arm เดียวกัน)", "", f"- n {len(s1)} · mean Δ {d_.mean():+.4f} · max|Δ| {d_.abs().max():.4f} · เท่ากันเป๊ะ {(d_ == 0).sum()}/{len(s1)} (ต่างได้จาก ΔC ≤ 5e-3 ของ sE5 และ gen ที่ต่างกัน: E4 ใช้ 480)", ""]
    (out / "E5_report.md").write_text("\n".join(lines), encoding="utf-8"); log(f"→ {out / 'E5_report.md'}")


if __name__ == "__main__":
    main()
