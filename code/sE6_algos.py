#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sE6_algos.py (E6 v0.2, 2026-10-22) — optimizer ทางเลือกบน Problem ของ E4/E5 (Def 6 v5) · เสียบแทน run_nsga2_warm ด้วย interface เดียวกัน
=================================================================================================================== กติกา (ประกาศก่อนรัน)
  interface   : run_algo(name, prob, pop, gen, seed, warm_G=None) → (X_int (A,n), F (A,3) ค่าสูง = ดี) — เหมือน sE5_replan.run_nsga2_warm ทุกประการ
  genome      : ต่อเนื่อง x ∈ [0, n_cand]^n · ประเมินที่ round(x) (= RoundingRepair ของ NSGA-II) · decode/repair/objectives = ของ E4/E5 ไม่แก้
  งบประมาณ   : pop × gen evaluations เท่ากันทุกอัลกอริทึม (ค่าหลัก 80 × 480 = 38,400) · seed เริ่มเหมือนกัน: greedy 2 ตัว + warm-start (ถ้ามี)
  ผลลัพธ์     : archive ไม่ถูกครอบงำ (cap = pop) ↔ NSGA-II คืน non-dominated ของประชากรสุดท้าย · X* เลือกใน E5 เหมือนเดิม (ใกล้ ideal)

  nsga2       : sE5_replan.run_nsga2_warm (pymoo NSGA-II, SBX 0.9/η15, PM η20 p = 3/n) — ไม่แก้
  mopso       : Coello, Pulido & Lechuga (2004) IEEE TEVC 8(3) — external archive + leader จาก archive · ใช้ crowding distance แทน adaptive grid
                (Raquel & Naval 2005 GECCO) · pbest ด้วย dominance (เสมอ → สุ่ม 50/50) · w 0.9→0.4 เชิงเส้น (Shi & Eberhart 1998) · c₁ = c₂ = 1.49445 (Eberhart & Shi 2000)
                · v_max = 0.25·พิสัย · ขอบ: หนีบค่า + กลับทิศความเร็ว · turbulence: PM η20 p = 3/n กับ particle ที่สุ่มได้ด้วย p_t = (1 − it/gen)^(5/mr), mr = 0.5 (Coello 2004 §III-C)
  mopso_q     : mopso + Q-learning ควบคุมพารามิเตอร์ระหว่างรัน (adaptive parameter control แนว Samma et al. 2016, Appl. Soft Comput.; Liu et al. 2019 CEC)
                state = (HV ของ archive ดีขึ้นใน 10 รอบล่าสุด? ) × (ความหลากหลายฝูง < มัธยฐานประวัติ?) = 4 · action = 4 ชุด (w, c₁, c₂, ตัวคูณ turbulence)
                reward (v0.2) = สัดส่วน particle รอบนี้ที่ archive เดิมครอบงำไม่ได้ (หนาแน่น; เดิม ΔHV เป็น 0 ใน ~90 % ของรอบ → ไม่เรียน) · argmax สุ่มเมื่อเสมอ (เดิมเอียง action 0)
                ε-greedy ε 0.3→0.05 · α 0.1 · γ 0.9 · Q เริ่ม 0 · เรียนภายในการแก้ 1 ครั้ง (ไม่ข้าม instance → ไม่มี leakage)
  mopso_iq    : mopso_q + (i) informed init (QM-init): 30 % ของฝูงสุ่มยีนตาม C_ij (ความน่าจะเป็นของ candidate · ไม่เสนอ ∝ 1 − max C)
                + (ii) decomposition: λ_p ~ Dirichlet(1,1,1) ต่อ particle · pbest/leader ด้วย Tchebycheff บน F ที่ normalise ด้วย ideal/nadir ของ archive (แนว MOPSO/D, Peng & Zhang 2008)
                + (iii) local search เมื่อ HV นิ่ง ≥ 10 รอบ: reinit 20 % ที่แย่สุด (ตาม g_λ) ด้วย informed init · PM ×3 กับ 20 % ถัดไป (ไม่ใช้ evaluation เพิ่ม)
  encoding    : rank (ค่าหลักเดิม) = อันดับผู้สมัคร ∈ [0, n_cand] ต่อช่อง ปัดเป็นจำนวนเต็ม · **key** (*_rk, v0.2) = random-key/propensity u_ik ∈ [0,1] ต่อคู่ (งาน, ผู้สมัคร)
                decode: เรียง u มาก→น้อยต่องาน ช่อง s = ผู้สมัครอันดับ s ถ้า u ≥ 0.5 ไม่งั้นว่าง → genome เดิม → decode/repair เดิม (Bean 1994; ใกล้ binary-threshold ของโค้ดเก่า)
  v0.2 fix    : leader ใช้ตำแหน่งต่อเนื่องของสมาชิก archive (เดิมใช้ genome ที่ปัดแล้ว) — ผลต่อ synthetic ขนาดจริง: mopso_iq −13 % → −2…+4 % เทียบ NSGA-II
  การตรวจ     : ทุกการเปลี่ยนแปลงออกแบบ/ตรวจบนโจทย์สังเคราะห์เท่านั้น (selftest + ขนาดจริง W60 T200) ไม่ใช้ instance ทดสอบ 30 ตัว · ประกาศก่อนรัน E6 เต็ม
  ไม่มี       : ค่าคงที่จูนมือผูกกับสเกล reward (โค้ดเก่า MOPSO_*_fast_v4 ใช้ไม่ได้: genome binary/slot · objectives เก่า · ~60 ค่าคงที่) — เขียนใหม่ทั้งหมด
===================================================================================================================
ใช้:  python sE6_algos.py --selftest
      python sE6_algos.py --inst instances_v3 --e5 e5inst_v3 --city sf --k 0 --arm qm_model --algos nsga2,mopso,mopso_q,mopso_iq --pop 80 --gen 480 --seeds 2
ผล:   ตารางเทียบ HV_plan · max f₁_plan · f₁ ของ X* · f₁_real(X*) (replay แบบ E4 ด้วย oracle t=0) · เวลา
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sE4_lite as E4

VERSION = "E6-algos v0.3 (2026-10-23: v0.2 + informed_frac ปรับได้ (A2) + rho helpers สำหรับ C̃ (A1))"
ALGOS = ["nsga2", "mopso", "mopso_q", "mopso_iq", "mopso_rk", "mopso_q_rk", "mopso_iq_rk"]
LABEL = {"nsga2": "NSGA-II (pymoo)", "mopso": "MOPSO (Coello 2004, crowding; rank enc.)", "mopso_q": "MOPSO-Q (Q-learning parameter control; rank)", "mopso_iq": "MOPSO-IQ (informed init + decomposition + LS; rank)",
         "mopso_rk": "MOPSO (random-key enc.)", "mopso_q_rk": "MOPSO-Q (random-key)", "mopso_iq_rk": "MOPSO-IQ (random-key)"}
C_PSO = 1.49445


# ----------------------------------------------------------------------------- ส่วนร่วม
def evaluate(prob, G_int):
    """genome จำนวนเต็ม (P, n) → F (P, 3) ค่าสูง = ดี (ผ่าน decode/repair/objectives ของ Problem เดิม)"""
    x = prob.decode(G_int); o = prob.objectives(x, prob.C, prob.phimax_own)
    return np.stack([o["f1"], o["f2"], o["f3"]], 1)


def to_int(X, n_cand):
    return np.clip(np.rint(X), 0, n_cand).astype(int)


def nondominated_mask(F):
    """True = ไม่มีใครครอบงำ (max ทุกมิติ)"""
    P = len(F); nd = np.ones(P, bool)
    for p in range(P):
        dom = (F >= F[p]).all(1) & (F > F[p]).any(1)
        if dom.any(): nd[p] = False
    return nd


def crowding(F):
    P, M = F.shape
    if P <= 2: return np.full(P, np.inf)
    d = np.zeros(P)
    for m in range(M):
        o = np.argsort(F[:, m], kind="stable"); fm = F[o, m]; span = fm[-1] - fm[0]
        d[o[0]] = d[o[-1]] = np.inf
        if span > 0: d[o[1:-1]] += (fm[2:] - fm[:-2]) / span
    return d


class Archive:
    """external archive ของคำตอบไม่ถูกครอบงำ · cap ด้วย crowding distance (ตัดตัวที่แน่นสุดทีละตัว) · เก็บ genome (X), F และตำแหน่งใน encoding (pos) สำหรับ leader"""
    def __init__(self, cap):
        self.cap = cap; self.X = None; self.F = np.zeros((0, 3)); self.pos = None
    def add(self, X, F, pos=None):
        pos = np.asarray(X, float) if pos is None else pos
        Xa = X if self.X is None else np.vstack([self.X, X]); Fa = F if self.X is None else np.vstack([self.F, F]); Pa = pos if self.X is None else np.vstack([self.pos, pos])
        _, ui = np.unique(Xa, axis=0, return_index=True); ui = np.sort(ui); Xa, Fa, Pa = Xa[ui], Fa[ui], Pa[ui]
        nd = nondominated_mask(Fa); Xa, Fa, Pa = Xa[nd], Fa[nd], Pa[nd]
        while len(Fa) > self.cap:
            cd = crowding(Fa); i = int(np.argmin(cd)); keep = np.ones(len(Fa), bool); keep[i] = False; Xa, Fa, Pa = Xa[keep], Fa[keep], Pa[keep]
        self.X, self.F, self.pos = Xa, Fa, Pa
    def hv(self):
        return E4.hv3(self.F) if len(self.F) else 0.0


def pm_mutation(X, lo, hi, eta, p_gene, rng):
    """polynomial mutation (Deb & Agrawal 1999) — สูตรเดียวกับ pymoo PM · ต่อยีนด้วยความน่าจะเป็น p_gene"""
    span = hi - lo; mask = rng.random(X.shape) < p_gene; u = rng.random(X.shape); mp = 1.0 / (eta + 1.0)
    d1 = (X - lo) / span; d2 = (hi - X) / span
    v1 = 2.0 * u + (1.0 - 2.0 * u) * (1.0 - d1) ** (eta + 1.0); dq1 = np.abs(v1) ** mp - 1.0
    v2 = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - d2) ** (eta + 1.0); dq2 = 1.0 - np.abs(v2) ** mp
    dq = np.where(u <= 0.5, dq1, dq2); Xn = np.clip(X + dq * span, lo, hi)
    return np.where(mask, Xn, X)


def informed_genome(prob, rng, n):
    """QM-init: ยีน v (งาน i) สุ่มจาก {0..n_cand} ด้วยน้ำหนัก k≥1: C_i,cand_k · k=0 (ไม่เสนอ): 1 − max_k C — งานที่ไม่มีผู้สมัครดีจะว่างบ่อย"""
    Wc = prob.cand_C[prob.slot_task]                                   # (n_var, n_cand) เรียงมาก→น้อย
    w0 = np.clip(1.0 - Wc.max(1, keepdims=True), 0.02, 1.0)
    Wt = np.concatenate([w0, np.maximum(Wc, 1e-6)], 1); Wt = Wt / Wt.sum(1, keepdims=True)
    cdf = np.cumsum(Wt, 1); u = rng.random((n, prob.n_var, 1))
    return (u > cdf[None, :, :]).sum(2).astype(float)                  # ค่า 0..n_cand


def init_swarm(prob, pop, rng, warm_G, informed_frac):
    """seed เหมือน NSGA-II: [0] greedy เต็ม · [1] greedy ช่องแรก · [2] warm-start (ถ้ามี) · ที่เหลือสุ่มสม่ำเสมอ (± informed สัดส่วนที่ขอ)"""
    X = rng.integers(0, prob.n_cand + 1, (pop, prob.n_var)).astype(float)
    X[0] = prob.slot_pos + 1; X[1] = np.where(prob.slot_pos == 0, 1, 0); k0 = 2
    if warm_G is not None and pop > 2: X[2] = np.asarray(warm_G, float); k0 = 3
    n_inf = min(pop - k0, int(round(informed_frac * pop)))
    if n_inf > 0: X[k0:k0 + n_inf] = informed_genome(prob, rng, n_inf)
    return X


def dominates(Fa, Fb):
    return (Fa >= Fb).all(1) & (Fa > Fb).any(1)


class QController:
    """Q-learning เลือกชุดพารามิเตอร์ PSO ต่อรอบ · state 4 (HV ดีขึ้น? × ความหลากหลายต่ำ?) · action 4
    v0.2: reward หนาแน่น = สัดส่วน particle ที่ไม่ถูก archive เดิมครอบงำ (ดัน front) · argmax สุ่มเมื่อเสมอ (เดิมเอียงไป action 0 เสมอเมื่อ Q = 0)"""
    ACTIONS = [(0.9, 2.0, 1.0, 1.0),        # explore: inertia สูง cognitive มาก
               (0.7, C_PSO, C_PSO, 1.0),    # balanced
               (0.4, 1.0, 2.0, 1.0),        # exploit: ตาม leader
               (0.6, C_PSO, C_PSO, 3.0)]    # turbulence ×3
    def __init__(self, rng, alpha=0.1, gamma=0.9, eps0=0.3, eps_min=0.05):
        self.rng = rng; self.alpha = alpha; self.gamma = gamma; self.eps0 = eps0; self.eps_min = eps_min
        self.Q = np.zeros((4, len(self.ACTIONS))); self.visits = np.zeros((4, len(self.ACTIONS)), int)
    @staticmethod
    def state(hv_hist, div_hist):
        improve = (len(hv_hist) < 11) or (hv_hist[-1] > hv_hist[-11] + 1e-6)
        div_low = len(div_hist) >= 2 and div_hist[-1] < float(np.median(div_hist))
        return int(improve) * 2 + int(div_low)
    def act(self, s, it, gen):
        eps = max(self.eps_min, self.eps0 * (1.0 - it / max(1, gen)))
        if self.rng.random() < eps: a = int(self.rng.integers(len(self.ACTIONS)))
        else:
            q = self.Q[s]; best = np.flatnonzero(q >= q.max() - 1e-12); a = int(self.rng.choice(best))
        self.visits[s, a] += 1; return a
    def update(self, s, a, r, s2):
        self.Q[s, a] += self.alpha * (r + self.gamma * self.Q[s2].max() - self.Q[s, a])


# ----------------------------------------------------------------------------- encodings
class RankEnc:
    """genome = อันดับผู้สมัคร ∈ [0, n_cand] ต่อช่อง (ปัดเป็นจำนวนเต็ม = RoundingRepair ของ NSGA-II)"""
    name = "rank"
    def __init__(self, prob): self.prob = prob; self.n = prob.n_var; self.lo, self.hi = 0.0, float(prob.n_cand)
    def to_genome(self, X): return to_int(X, self.prob.n_cand)
    def from_genome(self, G): return np.asarray(G, float)
    def informed(self, rng, n): return informed_genome(self.prob, rng, n)
    def random(self, rng, n): return rng.integers(0, self.prob.n_cand + 1, (n, self.n)).astype(float)
    def greedy(self):
        p = self.prob; return np.stack([p.slot_pos + 1.0, np.where(p.slot_pos == 0, 1.0, 0.0)])


class KeyEnc:
    """random-key / propensity: ตำแหน่ง u_ik ∈ [0,1] ต่อคู่ (งาน i, ผู้สมัครอันดับ k) = ความโน้มเอียงที่จะเสนอ
    decode: ต่องาน เรียง u มาก→น้อย ช่อง s ของงาน i = ผู้สมัครอันดับ s ของการเรียง ถ้า u ≥ 0.5 ไม่งั้นว่าง → genome จำนวนเต็มเดิม → decode/repair เดิม
    ต่อเนื่องจริง: PSO เคลื่อน u ทีละน้อย การตัดสินใจเปลี่ยนเมื่อข้าม 0.5 หรือสลับอันดับ (แนว random-key ของ Bean 1994 · ใกล้ binary-threshold ของโค้ดเก่า)"""
    name = "key"
    def __init__(self, prob): self.prob = prob; self.T = prob.T; self.nc = prob.n_cand; self.n = self.T * self.nc; self.lo, self.hi = 0.0, 1.0
    def to_genome(self, X):
        p = self.prob; U = X.reshape(len(X), self.T, self.nc); order = np.argsort(-U, axis=2, kind="stable"); Us = np.take_along_axis(U, order, 2)
        us = Us[:, p.slot_task, p.slot_pos]; oc = order[:, p.slot_task, p.slot_pos]
        return np.where(us >= 0.5, oc + 1, 0).astype(int)
    def from_genome(self, G):
        """genome จำนวนเต็ม → keys: ที่เลือก 0.9 ที่เหลือ 0.1 (ใช้กับ warm-start / informed)"""
        p = self.prob; G = np.atleast_2d(np.asarray(G, int)); U = np.full((len(G), self.T, self.nc), 0.1)
        for v in range(p.n_var):
            i = p.slot_task[v]; k = G[:, v] - 1; m = k >= 0
            U[np.flatnonzero(m), i, k[m]] = 0.9 - 0.05 * p.slot_pos[v]          # ช่องแรกสูงกว่า → รักษาลำดับช่อง
        return U.reshape(len(G), self.n)
    def informed(self, rng, n): return np.clip(self.from_genome(informed_genome(self.prob, rng, n)) + rng.uniform(-0.1, 0.1, (n, self.n)), 0, 1)
    def random(self, rng, n): return rng.random((n, self.n))
    def greedy(self):
        p = self.prob; U = np.zeros((2, self.T, self.nc))
        for i in range(self.T): U[0, i, :p.K[i]] = 1.0
        U[1, :, 0] = 1.0
        return U.reshape(2, self.n)


def init_swarm_enc(enc, pop, rng, warm_G, informed_frac):
    X = enc.random(rng, pop); X[:2] = enc.greedy(); k0 = 2
    if warm_G is not None and pop > 2: X[2] = enc.from_genome(np.asarray(warm_G, int))[0]; k0 = 3
    n_inf = min(pop - k0, int(round(informed_frac * pop)))
    if n_inf > 0: X[k0:k0 + n_inf] = enc.informed(rng, n_inf)
    return X


# ----------------------------------------------------------------------------- MOPSO family
def run_mopso(prob, pop, gen, seed, warm_G=None, variant="mopso", log=None, encoding="rank", informed_frac=None):
    """variant ∈ {mopso, mopso_q, mopso_iq} × encoding ∈ {rank, key} · คืน (X_int archive, F archive)"""
    rng = np.random.default_rng(seed); enc = KeyEnc(prob) if encoding == "key" else RankEnc(prob)
    n = enc.n; lo, hi = enc.lo, enc.hi; vmax = 0.25 * (hi - lo)
    use_q = variant in ("mopso_q", "mopso_iq"); use_iq = variant == "mopso_iq"
    inf_frac = (0.30 if informed_frac is None else float(informed_frac)) if use_iq else 0.0
    X = init_swarm_enc(enc, pop, rng, warm_G, inf_frac); V = np.zeros_like(X)
    G = enc.to_genome(X); F = evaluate(prob, G)
    Pb, PbF = X.copy(), F.copy(); arch = Archive(pop); arch.add(G, F, X)
    lam = rng.dirichlet(np.ones(3), pop) if use_iq else None
    qc = QController(rng) if use_q else None
    hv_hist = [arch.hv()]; div_hist = []; stall = 0; s_prev = a_prev = None; p_gene = 3.0 / max(1, n); trace = []
    n_iter = max(1, gen - 1)                                            # งบ = pop × gen evaluations พอดี (init 1 + อีก gen−1 รอบ) = pymoo n_gen

    def tcheb(Fv, lam_v):
        ideal = arch.F.max(0); nadir = arch.F.min(0); span = np.maximum(ideal - nadir, 1e-3)
        Fn = np.clip((Fv - nadir) / span, 0.0, 1.0)
        return (lam_v * (1.0 - Fn)).max(-1)

    for it in range(n_iter):
        if use_q:
            div_hist.append(float(np.mean(X.std(0)) / (hi - lo)))
            s = qc.state(hv_hist, div_hist); a = qc.act(s, it, n_iter); w, c1, c2, tmul = qc.ACTIONS[a]
        else:
            w = 0.9 - 0.5 * it / max(1, n_iter - 1); c1 = c2 = C_PSO; tmul = 1.0
        # ---- leader (ตำแหน่งของสมาชิก archive ในปริภูมิ encoding)
        A = arch.pos; nA = len(A)
        if use_iq:
            gl = tcheb(arch.F[None, :, :], lam[:, None, :]); Lb = A[np.argmin(gl, 1)]
        else:
            cd = crowding(arch.F); i1 = rng.integers(nA, size=pop); i2 = rng.integers(nA, size=pop)
            Lb = A[np.where(cd[i1] >= cd[i2], i1, i2)]
        # ---- PSO update
        r1, r2 = rng.random(X.shape), rng.random(X.shape)
        V = w * V + c1 * r1 * (Pb - X) + c2 * r2 * (Lb - X); V = np.clip(V, -vmax, vmax)
        Xn = X + V; out = (Xn < lo) | (Xn > hi); V[out] *= -1.0; X = np.clip(Xn, lo, hi)
        p_t = min(1.0, tmul * (1.0 - it / max(1, n_iter)) ** 10)
        mpart = rng.random(pop) < p_t
        if mpart.any(): X[mpart] = pm_mutation(X[mpart], lo, hi, 20.0, p_gene, rng)
        # ---- evaluate
        G = enc.to_genome(X); F = evaluate(prob, G)
        push = 1.0 - np.mean([dominates(arch.F, F[p:p + 1]).any() for p in range(pop)])   # สัดส่วน particle ที่ archive เดิมครอบงำไม่ได้ (reward ของ Q)
        arch.add(G, F, X)
        # ---- pbest
        if use_iq: better = tcheb(F, lam) < tcheb(PbF, lam)
        else:
            dn, do = dominates(F, PbF), dominates(PbF, F); better = dn | (~dn & ~do & (rng.random(pop) < 0.5))
        Pb[better] = X[better]; PbF[better] = F[better]
        hv_now = arch.hv(); hv_hist.append(hv_now)
        if use_q:
            r = push
            if s_prev is not None: qc.update(s_prev, a_prev, r_prev, s)
            s_prev, a_prev, r_prev = s, a, r
        stall = stall + 1 if hv_now <= hv_hist[-2] + 1e-9 else 0
        if use_iq and stall >= 10:
            g = tcheb(PbF, lam); order = np.argsort(-g); n_re = max(1, pop // 5); n_mu = max(1, pop // 5)
            X[order[:n_re]] = enc.informed(rng, n_re); V[order[:n_re]] = 0.0
            X[order[n_re:n_re + n_mu]] = pm_mutation(X[order[n_re:n_re + n_mu]], lo, hi, 20.0, 3 * p_gene, rng)
            stall = 0
        if log and (it % 50 == 0 or it == n_iter - 1): log(f"      {variant}/{enc.name} it {it:4d} archive {len(arch.F):3d} HV {hv_now:.4f} max f1 {arch.F[:, 0].max():.3f}" + (f" · Q a{a} s{s}" if use_q else ""))
        trace.append((it, hv_now))
    if use_q and s_prev is not None: qc.update(s_prev, a_prev, r_prev, qc.state(hv_hist, div_hist))
    run_mopso.last = dict(hv_trace=np.array(trace), Q=qc.Q if use_q else None, visits=qc.visits if use_q else None, encoding=enc.name)
    return arch.X.astype(int), arch.F


def load_rho(path):
    """อ่าน rho.json ของ sE6_rho (ต่อเมือง) → dict"""
    import json
    return json.load(open(path, encoding="utf-8"))


def rho_factor(rho, h_rem):
    """ตัวคูณหดตัว ρ(h) ต่องาน (h_rem = นาทีที่เหลือของหน้าต่างงาน) จาก rho_xstar_clipped ตาม bins"""
    b = np.searchsorted(np.array(rho["bins"]), np.asarray(h_rem), side="left")
    return np.asarray(rho.get("rho_used", rho["rho_xstar_clipped"]), float)[b]


def run_algo(name, prob, pop, gen, seed, warm_G=None, log=None, informed_frac=None):
    if name == "nsga2":
        import sE5_replan as E5
        return E5.run_nsga2_warm(prob, pop, gen, seed, warm_G)
    base, enc = (name[:-3], "key") if name.endswith("_rk") else (name, "rank")
    if base in ("mopso", "mopso_q", "mopso_iq"): return run_mopso(prob, pop, gen, seed, warm_G, variant=base, log=log, encoding=enc, informed_frac=informed_frac)
    raise ValueError(f"unknown algo {name} (เลือกจาก {ALGOS})")


# ----------------------------------------------------------------------------- selftest / one-instance compare
def _synthetic_problem(seed=0, W=40, T=80, n_cand=6):
    rng = np.random.RandomState(seed); Y = (rng.rand(W, T) < 0.04).astype(float)
    C = np.clip(Y * rng.uniform(0.2, 0.8, (W, T)) + (1 - Y) * rng.uniform(0, 0.08, (W, T)), 0, 1)
    K = rng.choice([1, 2, 3], T, p=[.6, .3, .1]); L = np.ones(T, int); r = rng.uniform(.5, 1.5, T); eps = rng.uniform(2, 8, T); e = rng.uniform(40, 100, W)
    ref = E4.Problem(Y, K, L, r, eps, e, np.zeros(W), 3, False, .05, .05, n_cand)
    prob = E4.Problem(C, K, L, r, eps, e, np.zeros(W), 3, False, .05, .05, n_cand, phimax_ref=ref.phimax_own)
    return prob, Y, ref


def compare_table(prob, Y, ref, algos, pop, gen, seeds, log):
    rows = []
    for name in algos:
        for sd in range(seeds):
            t0 = time.time(); X, F = run_algo(name, prob, pop, gen, 1000 + sd, None, log=None); sec = time.time() - t0
            assert len(F) >= 1 and np.isfinite(F).all() and (F >= -1e-9).all() and (F <= 1 + 1e-9).all(), (name, F.min(), F.max())
            assert nondominated_mask(F).all(), f"{name}: คืนจุดที่ถูกครอบงำ"
            x = prob.decode(X); plan = prob.objectives(x, prob.C, prob.phimax_own); Fp = np.stack([plan["f1"], plan["f2"], plan["f3"]], 1)
            assert np.allclose(Fp, F, atol=1e-9), f"{name}: F ที่คืนไม่ตรงกับการประเมินซ้ำ"
            i_sel = int(np.argmin(np.linalg.norm(1.0 - Fp, axis=1)))
            Tplus_save = prob.Tplus; prob.Tplus = ref.Tplus; real = prob.objectives(x, Y, ref.phimax_own); prob.Tplus = Tplus_save
            rows.append(dict(algo=name, seed=sd, n_front=len(F), hv_plan=E4.hv3(Fp), f1_plan_max=float(Fp[:, 0].max()), f1_plan_sel=float(Fp[i_sel, 0]), f2_sel=float(Fp[i_sel, 1]), f3_sel=float(Fp[i_sel, 2]), f1_real_sel=float(real["f1"][i_sel]), n_offer=int(plan["n_offer"][i_sel]), sec=round(sec, 1)))
            log(f"    {name:9s} seed {sd} · front {len(F):3d} · HV {rows[-1]['hv_plan']:.4f} · max f1 {rows[-1]['f1_plan_max']:.3f} · X*: f1 {rows[-1]['f1_plan_sel']:.3f} f2 {rows[-1]['f2_sel']:.3f} f3 {rows[-1]['f3_sel']:.3f} → f1_real {rows[-1]['f1_real_sel']:.3f} · offers {rows[-1]['n_offer']} · {sec:.0f}s")
    import pandas as pd
    df = pd.DataFrame(rows); agg = df.groupby("algo", sort=False).agg(n=("seed", "count"), hv_plan=("hv_plan", "mean"), f1_plan_max=("f1_plan_max", "mean"), f1_plan_sel=("f1_plan_sel", "mean"), f1_real_sel=("f1_real_sel", "mean"), n_front=("n_front", "mean"), sec=("sec", "mean"))
    return df, agg


def selftest(log):
    prob, Y, ref = _synthetic_problem()
    # หน่วยย่อย
    F = np.array([[.5, .5, .5], [.6, .5, .5], [.5, .6, .4], [.6, .5, .5]]); nd = nondominated_mask(F); assert nd.tolist() == [False, True, True, True], nd
    cd = crowding(np.array([[0, 1.], [.5, .5], [1, 0.]])); assert np.isinf(cd[0]) and np.isinf(cd[2]) and abs(cd[1] - 2.0) < 1e-9, cd
    rng = np.random.default_rng(0); Xm = pm_mutation(np.full((5, 10), 3.0), 0.0, 6.0, 20.0, 1.0, rng); assert (Xm >= 0).all() and (Xm <= 6).all() and not np.allclose(Xm, 3.0)
    gi = informed_genome(prob, rng, 50); assert gi.shape == (50, prob.n_var) and gi.min() >= 0 and gi.max() <= prob.n_cand
    a = Archive(5); a.add(np.arange(12).reshape(12, 1), np.c_[np.linspace(0, 1, 12), np.linspace(1, 0, 12), np.full(12, .5)]); assert len(a.F) == 5 and nondominated_mask(a.F).all()
    s = QController.state([0.1] * 5, [0.2, 0.3]); assert s in range(4)
    ke = KeyEnc(prob); Gk = ke.to_genome(ke.greedy()); assert np.array_equal(Gk[0], prob.slot_pos + 1) and np.array_equal(Gk[1], np.where(prob.slot_pos == 0, 1, 0)), "KeyEnc greedy"
    Gr = rng.integers(0, prob.n_cand + 1, (20, prob.n_var)); assert np.array_equal(np.sort(prob.decode(ke.to_genome(ke.from_genome(Gr))), 2), np.sort(prob.decode(Gr), 2)), "KeyEnc from→to ไม่ให้การจัดสรรเดิม (เทียบเป็นเซตต่องาน)"
    Gi = ke.to_genome(ke.informed(rng, 30)); assert Gi.min() >= 0 and Gi.max() <= prob.n_cand
    # เทียบ 4 อัลกอริทึม งบเท่ากัน
    df, agg = compare_table(prob, Y, ref, ALGOS, pop=40, gen=40, seeds=2, log=log)
    log("\n" + agg.round(4).to_string())
    assert set(df.algo) == set(ALGOS)
    log("selftest ผ่าน ✅")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--inst", default="instances_v3"); ap.add_argument("--e5", default="e5inst_v3"); ap.add_argument("--city", default="sf"); ap.add_argument("--k", type=int, default=0); ap.add_argument("--arm", default="qm_model")
    ap.add_argument("--algos", default=",".join(ALGOS)); ap.add_argument("--pop", type=int, default=80); ap.add_argument("--gen", type=int, default=480); ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--R", type=float, default=200.0); ap.add_argument("--q-max", type=int, default=3); ap.add_argument("--theta-T", type=float, default=0.05); ap.add_argument("--theta-W", type=float, default=0.05); ap.add_argument("--n-cand", type=int, default=8)
    ap.add_argument("--out", default="e6_compare"); ap.add_argument("--trace", action="store_true", help="พิมพ์ HV ทุก 50 รอบ")
    a = ap.parse_args(); t0 = time.time(); log = lambda *m: print(f"[{time.time()-t0:7.1f}s]", *m, flush=True); log(VERSION)
    if a.selftest: selftest(log); return
    import sE5_replan as E5
    inst_dir = Path(a.inst) / a.city; e5_dir = Path(a.e5) / a.city
    z = np.load(inst_dir / f"inst_{a.k:03d}.npz", allow_pickle=True); Z0 = np.load(e5_dir / f"inst_{a.k:03d}_t00.npz")
    K = z["task_K"].astype(int); L = np.ones_like(K); r = z["task_r"].astype(float); eps = z["task_eps"].astype(float); e0 = z["w_e"].astype(float)
    Y0 = Z0["oracle"].astype(float); W = Y0.shape[0]
    ref = E4.Problem(Y0, K, L, r, eps, e0, np.zeros(W), a.q_max, False, a.theta_T, a.theta_W, a.n_cand)
    C = Y0 if a.arm == "oracle" else np.nan_to_num(Z0[a.arm].astype(float))
    prob = E5.ProblemT(C, K, L, r, eps, np.full(W, a.q_max), e0, np.zeros(W), False, a.theta_T, a.theta_W, a.n_cand, phimax_ref=ref.phimax_own)
    algos = [x for x in a.algos.split(",") if x]
    log(f"{a.city} inst {a.k:03d} arm {a.arm} · W {W} T {len(K)} n_var {prob.n_var} · pop {a.pop} × gen {a.gen} = {a.pop*a.gen} evals/solve · seeds {a.seeds} · algos {algos}")
    df, agg = compare_table(prob, Y0, ref, algos, a.pop, a.gen, a.seeds, log)
    out = Path(a.out) / a.city; out.mkdir(parents=True, exist_ok=True); df.to_csv(out / f"compare_k{a.k:03d}_{a.arm}.csv", index=False)
    log("\n" + agg.round(4).to_string()); log(f"→ {out / f'compare_k{a.k:03d}_{a.arm}.csv'}")


if __name__ == "__main__":
    main()
