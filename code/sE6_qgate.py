#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sE6_qgate.py (E6 v2 ส่วน B1 · v0.1, 2026-10-24) — เรียนนโยบาย "เสนอตอนนี้หรือรอ" ต่อคนขับจาก replay ของชุด **dev** (sticky เท่านั้น)
================================================================================================================ นิยาม (ประกาศก่อนรัน — SPEC แก้ไข #3)
  หน่วยตัดสินใจ : คนขับ j ที่จุดตัดสินใจ t มีโควตาเหลือ q ∈ {1,2,3} · คู่ผู้สมัคร = งานเปิดเรียงตาม C̃_ij(t) มาก→น้อย (C̃ = min(1, ρ(h)·C) จาก rho.json)
  ลำดับภายในขั้น : พิจารณาคู่อันดับ p = 0,1,2 ทีละคู่ — "เสนอ" → ใช้ 1 ช่อง แล้วไปพิจารณาคู่ p+1 ที่ t เดิมด้วย q−1 · "รอ" → หยุดเสนอในขั้นนี้ ไป t+Δ ด้วย q เดิม
  state         : s = (t ∈ {0..50}, q, bin C̃ ของคู่ [<.02, .02–.05, .05–.10, .10–.20, ≥.20], bin หน้าต่างที่เหลือของงาน h [≤10, 11–20, 21–40, >40])  = 6×3×5×4 = 360
  reward        : เสนอ → 1 ถ้าคู่นั้นสำเร็จใน replay (∃ hit ใน [t, H) ∩ หน้าต่างงาน) · 0 ไม่สำเร็จ · รอ → 0
                  (ต้นทุนของการเสนอแล้วพลาด = ช่องโควตาที่หายไป ถูกนับโดย V(q−1) อยู่แล้ว → ไม่มี c_waste แยก · SPEC เดิมเขียน −0.3 = นับซ้ำ → แก้ไข #3)
  การแก้        : backward induction แบบตารางบน MDP เชิงประจักษ์ (ขอบฟ้าจำกัด 6 ขั้น, γ = 1 — วัตถุประสงค์คือจำนวนงาน ไม่มีเหตุให้ลดค่าเวลา)
                  Q_offer(t,q,s) = mean[ r + V(t, q−1, s_{p+1}) ] · Q_wait(t,q,s) = mean[ V(t+Δ, q, s_top(t+Δ)) ] · V = max(Q_offer, Q_wait) · V(·, 0, ·) = 0 · V(t_last+Δ) = 0
  ช่องข้อมูลน้อย : n < 20 → ถอยไปใช้ค่าเฉลี่ยระดับ (t,q,c) → (t,q)
  สมมติฐาน      : ไม่จำลองการแย่งงานระหว่างคนขับ (งานอาจสำเร็จโดยคนอื่นก่อน) — บันทึกเป็นข้อจำกัด
  cross-fitting : 5 fold ตาม k mod 5 → fold f ฝึกบน instance ที่ k mod 5 ≠ f (ใช้ประเมิน dev อย่างเป็นธรรม) · "full" = ฝึกทั้งหมด (ใช้กับ final)
================================================================================================================
ใช้:  python sE6_qgate.py --city sf --inst instances_devsplit --e5 e5inst_devsplit --rho e6_rho/sf/rho.json --out e6_qgate
ผล:   <out>/<city>/qgate.json · QGATE_report.md
ใช้งาน: sE5_replan.py ... --shrink e6_rho/<city>/rho.json --qgate e6_qgate/<city>/qgate.json [--qgate-cv]   (sticky เท่านั้น)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
VERSION = "E6-qgate v0.1 (2026-10-24)"
C_EDGES = [0.02, 0.05, 0.10, 0.20]
H_EDGES = [10, 20, 40]
QMAX = 3
N_MIN = 20


def cbin(c): return np.searchsorted(np.array(C_EDGES), np.asarray(c), side="right")      # 0..4
def hbin(h): return np.searchsorted(np.array(H_EDGES), np.asarray(h), side="left")       # 0..3


def rho_factor(rho, h_rem):
    b = np.searchsorted(np.array(rho["bins"]), np.asarray(h_rem), side="left")
    return np.asarray(rho.get("rho_used", rho["rho_xstar_clipped"]), float)[b]


def collect(inst_dir, e5_dir, ks, t_list, rho, arm, R, log):
    """ตัวอย่างระดับคู่ในสายลำดับต่อ (k, t, j): (k, ti, p, c, h, succ, next_c, next_h, has_next) และ top ของ (k, ti, j) สำหรับ t ถัดไป"""
    pair_rows, top_rows = [], []
    for k in ks:
        z = np.load(inst_dir / f"inst_{k:03d}.npz", allow_pickle=True); o = np.load(inst_dir / f"oracle_{k:03d}.npz")
        st = z["task_st"].astype(int); et = z["task_et"].astype(int); txy = z["task_xy"].astype(np.float64); path = o["path_true"].astype(np.float64)
        W, H = path.shape[:2]; T = len(txy)
        hit = np.sqrt(((path[:, None, :, :] - txy[None, :, None, :]) ** 2).sum(-1)) <= R                 # (W,T,H)
        win = np.zeros((T, H), bool)
        for i in range(T): win[i, st[i] - 1:et[i]] = True
        for ti, tm in enumerate(t_list):
            Zt = np.load(e5_dir / f"inst_{k:03d}_t{tm:02d}.npz"); op = Zt["open"].astype(bool) & (et > tm); idx = np.flatnonzero(op)
            if not len(idx): continue
            C = np.nan_to_num(Zt[arm].astype(float))[:, idx]; hrem = et[idx] - tm
            C = np.minimum(1.0, C * rho_factor(rho, hrem)[None, :])
            succ = (hit[:, idx, tm:] & win[None, idx, tm:]).any(2)                                         # (W, n_open): เสนอที่ t แล้วสำเร็จ (offer คงอยู่)
            order = np.argsort(-C, axis=1)[:, :QMAX + 1]
            for j in range(W):
                o_ = order[j]; cs = C[j, o_]; ok = cs > 0
                o_ = o_[ok]; cs = cs[ok]
                if not len(o_): continue
                top_rows.append((k, ti, j, int(cbin(cs[0])), int(hbin(hrem[o_[0]]))))
                for p in range(min(QMAX, len(o_))):
                    has_n = p + 1 < len(o_)
                    pair_rows.append((k, ti, j, p, int(cbin(cs[p])), int(hbin(hrem[o_[p]])), int(succ[j, o_[p]]),
                                      int(cbin(cs[p + 1])) if has_n else -1, int(hbin(hrem[o_[p + 1]])) if has_n else -1, int(has_n)))
        log(f"  inst {k:03d}: pairs {len(pair_rows):,}")
    P = pd.DataFrame(pair_rows, columns=["k", "ti", "j", "p", "c", "h", "succ", "nc", "nh", "has_next"])
    TP = pd.DataFrame(top_rows, columns=["k", "ti", "j", "c", "h"])
    return P, TP


def solve(P, TP, nT):
    """backward induction · คืน Qo, Qw (nT, QMAX+1, 5, 4) พร้อม count และฟังก์ชัน V แบบ backoff"""
    nC, nH = len(C_EDGES) + 1, len(H_EDGES) + 1
    Qo = np.zeros((nT, QMAX + 1, nC, nH)); Qw = np.zeros_like(Qo); N = np.zeros((nT, nC, nH), int)
    V = np.zeros((nT + 1, QMAX + 1, nC, nH))                                                               # V[nT] = 0, V[:,0] = 0
    # ตัวอย่าง wait: (k, ti, j) → top ของ (k, ti+1, j)
    TPn = TP.copy(); TPn["ti"] = TPn["ti"] - 1
    Pw = P[P.p == 0].merge(TPn.rename(columns={"c": "wc", "h": "wh"}), on=["k", "ti", "j"], how="left")      # wait เกิดที่คู่ใดก็ได้ แต่ next ขึ้นกับ (k,ti,j) เท่านั้น → ใช้ map ต่อ (k,ti,j)
    wmap = {(r.k, r.ti, r.j): (r.wc, r.wh) for r in Pw.itertuples()}
    for ti in range(nT - 1, -1, -1):
        Pt = P[P.ti == ti]
        if not len(Pt): continue
        cc, hh = Pt.c.to_numpy(), Pt.h.to_numpy()
        np.add.at(N[ti], (cc, hh), 1)
        wn = np.array([wmap.get((a, ti, b), (np.nan, np.nan)) for a, b in zip(Pt.k.to_numpy(), Pt.j.to_numpy())], float)
        for q in range(1, QMAX + 1):
            # wait target: V(ti+1, q, top at ti+1) (0 ถ้าไม่มี)
            wt = np.zeros(len(Pt)); okw = ~np.isnan(wn[:, 0])
            wt[okw] = V[ti + 1, q, wn[okw, 0].astype(int), wn[okw, 1].astype(int)]
            # offer target: succ + V(ti, q−1, next pair)
            ot = Pt.succ.to_numpy().astype(float).copy()
            if q - 1 >= 1:
                hn = Pt.has_next.to_numpy().astype(bool); ot[hn] += V[ti, q - 1, Pt.nc.to_numpy()[hn], Pt.nh.to_numpy()[hn]]
            Qo[ti, q] = cell_mean(cc, hh, ot, nC, nH); Qw[ti, q] = cell_mean(cc, hh, wt, nC, nH)
            V[ti, q] = np.maximum(Qo[ti, q], Qw[ti, q])
    return Qo, Qw, N


def cell_mean(cc, hh, y, nC, nH):
    """ค่าเฉลี่ยต่อ (c,h) พร้อม backoff → (c) → ทั้งหมด เมื่อ n < N_MIN"""
    S = np.zeros((nC, nH)); n = np.zeros((nC, nH)); np.add.at(S, (cc, hh), y); np.add.at(n, (cc, hh), 1)
    Sc = S.sum(1); nc = n.sum(1); tot = S.sum() / max(n.sum(), 1)
    out = np.where(n >= N_MIN, S / np.maximum(n, 1), np.nan)
    for c_ in range(nC):
        fill = Sc[c_] / nc[c_] if nc[c_] >= N_MIN else tot
        out[c_] = np.where(np.isnan(out[c_]), fill, out[c_])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", default="sf"); ap.add_argument("--inst", default="instances_devsplit"); ap.add_argument("--e5", default="e5inst_devsplit"); ap.add_argument("--rho", required=True); ap.add_argument("--out", default="e6_qgate")
    ap.add_argument("--arm", default="qm_model"); ap.add_argument("--t", default="0,10,20,30,40,50"); ap.add_argument("--R", type=float, default=200.0); ap.add_argument("--folds", type=int, default=5); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--allow-test", action="store_true")
    a = ap.parse_args(); t0 = time.time(); log = lambda *m: print(f"[{time.time()-t0:7.1f}s]", *m, flush=True); log(VERSION)
    if "instances_v3" in a.inst and not a.allow_test: sys.exit("ห้ามฝึก qgate บน instances_v3 (ชุดทดสอบ)")
    rho = json.load(open(a.rho, encoding="utf-8")); t_list = [int(x) for x in a.t.split(",")]
    inst_dir = Path(a.inst) / a.city; e5_dir = Path(a.e5) / a.city; out = Path(a.out) / a.city; out.mkdir(parents=True, exist_ok=True)
    files = sorted(inst_dir.glob("inst_*.npz")); files = files[:a.limit] if a.limit else files; ks = [int(f.stem.split("_")[1]) for f in files]
    log(f"{a.city}: {len(ks)} dev instance · arm {a.arm} · ρ {np.round(rho.get('rho_used', rho['rho_xstar_clipped']), 3).tolist()} · folds {a.folds}")
    P, TP = collect(inst_dir, e5_dir, ks, t_list, rho, a.arm, a.R, log)
    res = dict(version=VERSION, city=a.city, arm=a.arm, rho=a.rho, t_list=t_list, c_edges=C_EDGES, h_edges=H_EDGES, qmax=QMAX, n_min=N_MIN, folds=a.folds, tables={})
    for name, kk in [("full", ks)] + [(str(f), [k for k in ks if k % a.folds != f]) for f in range(a.folds)]:
        Qo, Qw, N = solve(P[P.k.isin(kk)], TP[TP.k.isin(kk)], len(t_list))
        res["tables"][name] = dict(Qo=Qo.tolist(), Qw=Qw.tolist(), N=N.tolist(), n_inst=len(kk))
    json.dump(res, open(out / "qgate.json", "w", encoding="utf-8"))
    # ---- รายงาน: นโยบายจาก full (offer = 1 ถ้า Qo ≥ Qw)
    Qo = np.array(res["tables"]["full"]["Qo"]); Qw = np.array(res["tables"]["full"]["Qw"]); N = np.array(res["tables"]["full"]["N"])
    lines = [f"# Q-gate (B1) — {a.city} · {VERSION} · {time.strftime('%Y-%m-%d %H:%M')}", "", f"dev {len(ks)} instance · ตัวอย่างคู่ {len(P):,} · อัตราสำเร็จเมื่อเสนอ (เฉลี่ย) {P.succ.mean():.4f}", "",
             "## นโยบาย (full): เสนอเมื่อ Q_offer ≥ Q_wait · แถว = bin C̃ (<.02 … ≥.20) · คอลัมน์ = bin หน้าต่างที่เหลือ (≤10, 11–20, 21–40, >40) · ✔ เสนอ · · รอ", ""]
    for ti, tm in enumerate(t_list):
        for q in range(1, QMAX + 1):
            pol = Qo[ti, q] >= Qw[ti, q] - 1e-12
            rows_ = [" ".join("✔" if pol[c_, h_] else "·" for h_ in range(len(H_EDGES) + 1)) for c_ in range(len(C_EDGES) + 1)]
            lines.append(f"- t={tm:02d} q={q}: " + " | ".join(rows_) + f"   (n/cell min {N[ti].min()}, median {int(np.median(N[ti]))})")
    lines += ["", "## ค่าเฉลี่ย V(t, q) ถ่วงด้วยจำนวนตัวอย่าง (มูลค่าคาดหวังของคนขับที่มีโควตา q)", ""]
    for ti, tm in enumerate(t_list):
        w_ = N[ti] / max(N[ti].sum(), 1); lines.append(f"- t={tm:02d}: " + " · ".join(f"q={q}: {float((np.maximum(Qo[ti, q], Qw[ti, q]) * w_).sum()):.3f}" for q in range(1, QMAX + 1)))
    lines += ["", "## ใช้ต่อ", f"- sE5_replan --withdraw sticky --shrink {a.rho} --qgate {out / 'qgate.json'} --qgate-cv   (dev: cross-fit fold = k mod {a.folds})", f"- final: ไม่ใส่ --qgate-cv (ใช้ตาราง full)", ""]
    (out / "QGATE_report.md").write_text("\n".join(lines), encoding="utf-8"); log(f"→ {out / 'QGATE_report.md'}")


if __name__ == "__main__":
    main()
