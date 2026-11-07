#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sE1_instances.py (v2.2) — E1: สร้าง instance ของ MTA-OW จาก cycle ของ final test  (ตาม E1_DESIGN_v2.md)
  1 instance = 1 cycle (date, hour) ของ final test ที่มีคันพร้อม ≥ W_min · worker = ตัวอย่างสุ่ม |W| จาก W(ĉ) · task |T| วางด้วย Thomas process
  QM_ij (Def 5 (9a) + isotonic) คำนวณที่นี่เลยด้วยฟังก์ชันของ sD7_pred_eval.py **v7** (import — ไม่เขียนสูตรซ้ำ) 6 แหล่ง:
     qm_model · qm_model_uncal · qm_M1 · qm_null_dv · qm_persist_bin · qm_persist_cont      (ไฟล์ inst_*.npz — optimizer เห็น)
     qm_oracle + path_true                                                                  (ไฟล์ oracle_*.npz — E4 replay เท่านั้น)
  โจทย์ (ตำแหน่งงาน, K_i, B, เวลา, รางวัล) ไม่ขึ้นกับ QM แหล่งใด (หลักการข้อ 5) · ทุกอย่างที่จะ sensitivity เป็นพารามิเตอร์ · CRN:
     เปลี่ยน --R / --W / --cell-m / --battery-on → งานชุดเดิม · --T น้อยลง → prefix ของชุดเดิม

อ่าน  models/<city>/base_M3/D7_final/E1_handoff.json  (seed + sha256 → ตรวจก่อนอ่าน pred_test.npz, calib.json, qm_calib.json)
      models/<city>/abl_M1/D7_dev/E1_handoff.json     (qm_M1; ถ้าไม่มี → ข้าม key นี้พร้อมเตือน)
      data/<city>/processed/{test,val}_{X,Y}.npy, test_meta, scaler.json · config/data_<city>.yaml (dates.test_final, bbox, W_min)
เขียน instances/<city>[/pilot]/inst_{k:03d}.npz · oracle_{k:03d}.npz · E1_report.md · E1_config.json · fig_E1_*.png
      ไม่แตะ models/ และ data/ เลย

ใช้:  python sE1_instances.py --city sf --pilot 3            # นำร่อง 3 instance → instances/sf/pilot/  → ดู G2.1/2.2/2.2b ใน E1_report.md
      python sE1_instances.py --city sf                      # ชุดเต็ม 30 instance → instances/sf/
      python sE1_instances.py --city sf --R 150              # sensitivity (งานชุดเดิม)  · --placement uniform · --W 120 · --T 100 · --cycle-min 30 · --cell-m 250
      python sE1_instances.py --city sf --centre-src uniform # [v2.1] ศูนย์ cluster สม่ำเสมอแบบ v2 เดิม (sensitivity) · ค่าหลัก mixed 0.85 (ศูนย์จากตำแหน่ง fleet 85 % ถ่วงน้ำหนักตามความหนาแน่น α=1) · --dens-alpha 0 = v2.1
      python sE1_instances.py --selftest                     # สร้างข้อมูลสังเคราะห์เล็ก ๆ แล้วรันครบทุกขั้น (ไม่ต้องมีข้อมูลจริง)
ต้องมี sD7_pred_eval.py (v7) ในโฟลเดอร์เดียวกัน · numpy / pandas / scipy (+ matplotlib สำหรับรูป) · ไม่ใช้ GPU
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
VERSION = "E1 v2.2 (2026-10-08)"

PYCHARM = dict(
    CITY="sf", DATA="data", MODELS="models", OUT="instances",
    N_INST=30, PILOT=0,                 # PILOT > 0 → สร้างแค่ N ตัวแรก (prefix ของชุดเต็ม) ลง <out>/<city>/pilot/
    W=60, T=200, CYCLE_MIN=60,          # |W| · |T| · ความยาวรอบ (นาที; ≤ H ของสาย D)
    R_LIST=[150, 200, 250], R_MAIN=200, # R_i = R เท่ากันทั้ง instance (Def 1) · เก็บ QM ทุก R
    PLACEMENT="clustered",              # clustered (Thomas + log-normal weights) | uniform
    CENTRE_SRC="mixed", FLEET_FRAC=0.85, CENTRE_JITTER=500.0, DENS_ALPHA=1.0,   # [v2.1/2.2] ศูนย์ cluster: fleet | mixed | uniform · ถ่วงน้ำหนักคัน ∝ dens^α (α=0 → ทุกคันเท่ากัน = v2.1) — v2 เดิม = uniform
    CELL_M=500,                         # grid ของ coverage (= grid_acc_500m ของ D7) — cell_id เก็บไว้ไม่ว่า f₃ จะใช้นิยามใด
    BETA=0.65, M_EXTRA=2, Q_MAX=3,      # B = β Σ r_i (K_i + M_i) · M_i · Q_max
    K_RULE="density",                   # density (K จากความหนาแน่น fleet ที่สังเกตได้) | random (choice{1,2,3} p=.5,.3,.2)
    BATTERY_ON=False,                   # ฟิลด์ e_j, ε_i มีเสมอ · flag บอก E4 ว่าใช้ constraint (ค) หรือไม่
    MASTER_SEED=12345,                  # = CRN_SEED ของ D7
    BASE_RUN="base_M3", BASE_SPLIT="final", M1_RUN="abl_M1", M1_SPLIT="dev",
    NO_FIGS=False, FORCE=False, SELFTEST=False,
)
DURATIONS = [5, 10, 15, 20, 30]         # หน้าต่างงาน (นาที) = ที่ D7 calibrate
THETAS = (0.02, 0.05, 0.10)             # รายงาน n_i(θ) เท่านั้น ไม่ใช่ส่วนของโจทย์
DENS_RADIUS_M = 2000.0                  # รัศมีนับความหนาแน่น fleet สำหรับ K_i
CENTRE_DENS_RADIUS_M = 1000.0           # [v2.2] รัศมีนับความหนาแน่นรอบคันสำหรับถ่วงน้ำหนักเลือกศูนย์ cluster
THOMAS = dict(n_centres_mean=10, w_sigma=0.5, sig_c=(300.0, 800.0))

# ----------------------------------------------------------------------------- utils
def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def seed_of(master: int, *tags) -> int:
    """seed ย่อยที่กำหนดโดย (master, tag, city, k) → เปลี่ยนพารามิเตอร์หนึ่งตัวแล้ว stream อื่นไม่เลื่อน (CRN)"""
    s = "|".join([str(master)] + [str(t) for t in tags]).encode()
    return int(hashlib.sha256(s).hexdigest()[:8], 16)

def import_sD7(log):
    p = HERE / "sD7_pred_eval.py"
    if not p.exists(): sys.exit(f"ไม่พบ {p} — ต้องมี sD7_pred_eval.py (v7) ในโฟลเดอร์เดียวกัน")
    head = p.read_text(encoding="utf-8", errors="ignore").splitlines()[:6]
    if not any("(v7)" in ln for ln in head): sys.exit("sD7_pred_eval.py ที่พบไม่ใช่ v7 (หัวไฟล์ไม่มี '(v7)') — E1 ต้อง import v7 เท่านั้น (v8 มี pt_steps/noise ที่ไม่เกี่ยว)")
    spec = importlib.util.spec_from_file_location("sD7_pred_eval", p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    for fn in ("load_split", "load_npz", "reach_score_analytic", "reach_score_null", "reach_labels", "apply_qm_calib"):
        if not hasattr(m, fn): sys.exit(f"sD7_pred_eval.py ไม่มีฟังก์ชัน {fn}")
    log(f"import sD7_pred_eval.py v7  sha256 {sha256_of(p)[:12]}…"); return m, sha256_of(p)

def to_str(a): return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in np.asarray(a).ravel()])

def to_epoch_s(a):
    """t_T → วินาที (float) ถ้าแปลงได้ · None ถ้าไม่ใช่เวลา"""
    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.number): return a.astype(np.float64)
    try: return pd.to_datetime(to_str(a)).values.astype("datetime64[s]").astype(np.int64).astype(np.float64)
    except Exception: return None

# ----------------------------------------------------------------------------- พิกัด
class Frame:
    """ระนาบ equirectangular รอบศูนย์เมือง (เมตร) — ใช้วางงาน/นับความหนาแน่น · เวกเตอร์สัมพัทธ์ l_i − x_T,j ส่งเข้าสูตร D7 ซึ่งทำงานบนเมตรสัมพัทธ์อยู่แล้ว
    (สเกลต่างจากระนาบของ D2 ได้ไม่เกินระดับ 1e-3 ภายในเมืองเดียว — ตรวจด้วย frame check ใน report)"""
    def __init__(self, lat0, lon0):
        self.lat0, self.lon0 = float(lat0), float(lon0); self.kx = 111320.0 * np.cos(np.radians(self.lat0)); self.ky = 110574.0
    def xy(self, lat, lon): return np.stack([(np.asarray(lon, np.float64) - self.lon0) * self.kx, (np.asarray(lat, np.float64) - self.lat0) * self.ky], -1)
    def bbox_xy(self, bb):
        """bbox จาก config → [[xmin,ymin],[xmax,ymax]] · รองรับ dict(lat_min,lat_max,lon_min,lon_max) หรือ [lon_min, lat_min, lon_max, lat_max]"""
        if bb is None: return None
        try:
            if isinstance(bb, dict):
                la0, la1 = float(bb.get("lat_min", bb.get("min_lat"))), float(bb.get("lat_max", bb.get("max_lat")))
                lo0, lo1 = float(bb.get("lon_min", bb.get("min_lon"))), float(bb.get("lon_max", bb.get("max_lon")))
            else:
                lo0, la0, lo1, la1 = [float(x) for x in bb]
            p = self.xy([la0, la1], [lo0, lo1]); return np.array([p.min(0), p.max(0)])
        except Exception: return None

class Area:
    """พื้นที่วางงาน = convex hull ของ fleet_xy ขยายออก `expand` m ตัดด้วย bbox เมือง (ถ้ามี) — ไม่ใช้เส้นทางอนาคต"""
    def __init__(self, pts, expand, bbox=None):
        from scipy.spatial import ConvexHull
        pts = np.asarray(pts, np.float64); self.expand = float(expand)
        if len(pts) >= 3 and np.linalg.matrix_rank(pts - pts.mean(0)) == 2: self.poly = pts[ConvexHull(pts).vertices]
        else: self.poly = None; self.ctr = pts.mean(0)
        lo, hi = pts.min(0) - expand, pts.max(0) + expand
        if bbox is not None: lo, hi = np.maximum(lo, bbox[0]), np.minimum(hi, bbox[1])
        self.lo, self.hi, self.bbox = lo, hi, bbox
    def _inside_poly(self, p):
        P = self.poly; n = len(P); inside = np.zeros(len(p), bool)
        for a in range(n):                                                # crossing number
            x1, y1 = P[a]; x2, y2 = P[(a + 1) % n]
            c = ((y1 > p[:, 1]) != (y2 > p[:, 1])) & (p[:, 0] < (x2 - x1) * (p[:, 1] - y1) / (y2 - y1 + 1e-12) + x1)
            inside ^= c
        return inside
    def _dist_poly(self, p):
        P = self.poly; n = len(P); d = np.full(len(p), np.inf)
        for a in range(n):
            A, B = P[a], P[(a + 1) % n]; AB = B - A; t = np.clip(((p - A) @ AB) / (AB @ AB + 1e-12), 0, 1)
            d = np.minimum(d, np.linalg.norm(p - (A + t[:, None] * AB), axis=1))
        return d
    def contains(self, p):
        p = np.atleast_2d(np.asarray(p, np.float64))
        ok = np.all((p >= self.lo) & (p <= self.hi), 1)
        if self.poly is None: return ok & (np.linalg.norm(p - self.ctr, axis=1) <= self.expand)
        return ok & (self._inside_poly(p) | (self._dist_poly(p) <= self.expand))
    def sample(self, rng, n=1):
        out = []
        while len(out) < n:
            cand = self.lo + rng.rand(max(8, 4 * (n - len(out))), 2) * (self.hi - self.lo); out.extend(cand[self.contains(cand)].tolist())
        return np.asarray(out[:n])
    def area_km2(self, rng, n=4000):
        box = float(np.prod(self.hi - self.lo)); p = self.lo + rng.rand(n, 2) * (self.hi - self.lo); return box * self.contains(p).mean() / 1e6

# ----------------------------------------------------------------------------- โหลดโมเดลที่ freeze
def load_frozen(models_root: Path, city: str, run: str, split: str, log, required=True):
    """อ่าน E1_handoff.json → ตรวจ sha256 → คืน dict(seed, pred, lam, T, qcal, sha, handoff) · required=False → None ถ้าไม่มี"""
    hp = models_root / city / run / f"D7_{split}" / "E1_handoff.json"
    if not hp.exists():
        alt = sorted((models_root / city / run).glob("D7_*/E1_handoff.json"))
        if alt and not required: hp = alt[-1]; log(f"  {run}: ไม่พบ D7_{split}/E1_handoff.json → ใช้ {hp.relative_to(models_root)}")
        elif required: sys.exit(f"ไม่พบ {hp} — ต้องรัน D7 v7 --split {split} ของ {run} ก่อน (เขียน E1_handoff.json)")
        else: log(f"  {run}: ไม่พบ E1_handoff.json → ข้าม"); return None
    hand = json.load(open(hp, encoding="utf-8")); sd = models_root / city / run / hand["seed"]
    shas = {}
    for f, rel in hand["files"].items():
        p = models_root / rel
        if not p.exists(): sys.exit(f"ไฟล์ใน handoff หาย: {p}")
        shas[f] = sha256_of(p)
        if shas[f] != hand["sha256"].get(f): sys.exit(f"sha256 ไม่ตรง handoff: {p} — ไฟล์ freeze ถูกแก้? หยุด (หลักการข้อ 2)")
    P = np.load(sd / "pred_test.npz", allow_pickle=True); pred = {k: P[k] for k in P.files}
    for k in ("mu", "sigma", "pi"): pred[k] = pred[k].astype(np.float32)
    c = json.load(open(sd / "calib.json")); lam = np.asarray(c["lambda_h"], np.float32); T = float(c["T"])
    qc_name = [f for f in hand["files"] if f.startswith("qm_calib")][0]; qcal = json.load(open(sd / qc_name))
    log(f"  {run}/{hand['seed']} ({hp.parent.name}) · AUC {hand.get('qm_variant_for_E1','qm_ub_cal')} = {hand[[k for k in hand if k.startswith('AUC_')][0]][hand['seed']]:.3f} · mu {pred['mu'].shape} · sha ตรง {len(shas)} ไฟล์")
    return dict(run=run, seed=hand["seed"], split=hp.parent.name, pred=pred, lam=lam, T=T, qcal=qcal, sha=shas, handoff=str(hp.relative_to(models_root)), M=int(pred["mu"].shape[1]))

# ----------------------------------------------------------------------------- สร้างงาน
def place_tasks(area: Area, n: int, placement: str, rng, fleet_xy=None, centre_src="mixed", fleet_frac=0.85, jitter_m=500.0, dens_alpha=1.0):
    """คืน (xy (n,2), info) · วางทีละงาน → T น้อยลง = prefix ของชุดเดิม (CRN)
    [v2.1] ศูนย์ cluster (parent ของ Thomas process) มาจาก `centre_src`:
      fleet   = ตำแหน่งปัจจุบันของรถ 1 คันใน fleet (สุ่มถ่วงน้ำหนัก ∝ ความหนาแน่นรอบคัน^α [v2.2]) + N(0, jitter²)  → "platform ตั้งงานในย่านที่มีกิจกรรม" (สังเกตได้ ณ ต้น cycle ไม่ใช้อนาคต)
      uniform = สม่ำเสมอในพื้นที่ hull (แบบ v2 เดิม — pilot SF: 87 % ของงานไม่มีใครถึง เพราะศูนย์ตกในที่ว่าง)
      mixed   = แต่ละศูนย์เป็น fleet ด้วยความน่าจะเป็น fleet_frac ไม่งั้น uniform (ค่าหลัก: มีงาน "ยาก" ปนไว้ให้ optimizer ต้องเลี่ยง)"""
    if placement == "uniform":
        return np.vstack([area.sample(rng, 1) for _ in range(n)]), dict(n_centres=0, n_centres_fleet=0)
    n_c = max(1, int(rng.poisson(THOMAS["n_centres_mean"])))
    centres = np.zeros((n_c, 2)); is_fleet = np.zeros(n_c, bool)
    if fleet_xy is not None and len(fleet_xy):                                   # [v2.2] น้ำหนักคัน ∝ (จำนวนคันใน 1 km รอบมัน)^α — ศูนย์ไปอยู่ย่านที่รถหนาแน่น ไม่ใช่คันโดดเดี่ยวชานเมือง
        from scipy.spatial import cKDTree
        dloc = np.asarray(cKDTree(fleet_xy).query_ball_point(fleet_xy, CENTRE_DENS_RADIUS_M, return_length=True), float)
        fleet_w = dloc ** dens_alpha if dens_alpha > 0 else np.ones(len(fleet_xy)); fleet_w = fleet_w / fleet_w.sum()
    for c in range(n_c):
        src_fleet = (centre_src == "fleet") or (centre_src == "mixed" and rng.rand() < fleet_frac)
        if src_fleet and fleet_xy is not None and len(fleet_xy):
            for _ in range(100):
                p = fleet_xy[rng.choice(len(fleet_xy), p=fleet_w)] + rng.normal(0.0, jitter_m, 2)   # [v2.2] เลือกคันตามความหนาแน่นรอบตัว
                if area.contains(p)[0]: break
            else: p = area.sample(rng, 1)[0]; src_fleet = False
        else: p = area.sample(rng, 1)[0]
        centres[c] = p; is_fleet[c] = src_fleet
    w = rng.lognormal(0.0, THOMAS["w_sigma"], n_c); w /= w.sum(); sig_c = rng.uniform(*THOMAS["sig_c"], size=n_c)
    xy = np.zeros((n, 2)); which = np.zeros(n, int)
    for t in range(n):
        c = int(rng.choice(n_c, p=w)); which[t] = c
        for _ in range(1000):
            p = centres[c] + rng.normal(0.0, sig_c[c], 2)
            if area.contains(p)[0]: break
        else: p = area.sample(rng, 1)[0]
        xy[t] = p
    return xy, dict(n_centres=n_c, n_centres_fleet=int(is_fleet.sum()), centres=centres, is_fleet=is_fleet, weights=w, sig_c=sig_c, which=which)

def make_K(task_xy, fleet_xy, rule, rng):
    """K_i จากความหนาแน่น fleet ที่สังเกตได้ (ไม่ต้องทำนาย) · fallback สุ่มอิสระ"""
    from scipy.spatial import cKDTree
    dens = np.asarray(cKDTree(fleet_xy).query_ball_point(task_xy, DENS_RADIUS_M, return_length=True), float)
    ref = float(np.median(dens))
    if rule == "density" and ref > 0 and dens.std() > 0:
        K = np.clip(np.round(dens / ref * rng.uniform(0.8, 1.2, len(dens))), 1, 3).astype(int); used = "density"
    else:
        K = rng.choice([1, 2, 3], size=len(dens), p=[0.5, 0.3, 0.2]); used = "random" + ("" if rule == "random" else " (fallback: dens แทบไม่แปรผัน)")
    return K, dens, used

# ----------------------------------------------------------------------------- QM ทุกแหล่ง
def qm_all(D7, frz, frz_m1, w_rows, w_xy, w_v, tau, task_xy, st, et, R_list, Y_cyc, H):
    """คู่ทั้ง |W|×|T| · คืน dict key → (|W|,|T|) ต่อ R  และ oracle/path"""
    nW, nT = len(w_rows), len(task_xy)
    ii = np.repeat(np.arange(nW), nT); l = (task_xy[None, :, :] - w_xy[:, None, :]).reshape(-1, 2).astype(np.float32)   # l สัมพัทธ์กับ x_T ของ worker
    stp = np.tile(st, nW).astype(int); etp = np.tile(et, nW).astype(int)
    out = {}; R_list = [float(R) for R in R_list]
    def sub(frz_): return (frz_["pred"]["mu"][w_rows], frz_["pred"]["sigma"][w_rows], frz_["pred"]["pi"][w_rows])
    mu, sg, pi = sub(frz)
    _, ub = D7.reach_score_analytic(mu, sg, pi, frz["lam"], frz["T"], ii, stp, etp, l, R_list)
    for R in R_list:
        out[f"qm_model_uncal_R{int(R)}"] = ub[R].reshape(nW, nT); out[f"qm_model_R{int(R)}"] = D7.apply_qm_calib(frz["qcal"], R, "ub", ub[R]).reshape(nW, nT)
    if frz_m1 is not None:
        mu1, sg1, pi1 = sub(frz_m1); _, ub1 = D7.reach_score_analytic(mu1, sg1, pi1, frz_m1["lam"], frz_m1["T"], ii, stp, etp, l, R_list)
        for R in R_list: out[f"qm_M1_R{int(R)}"] = D7.apply_qm_calib(frz_m1["qcal"], R, "ub", ub1[R]).reshape(nW, nT)
    sig_null = np.asarray(frz["qcal"]["null_sigma_h"]["gauss_dv"], np.float32)
    _, ubn = D7.reach_score_null(w_v[ii], tau, H, sig_null, np.arange(len(ii)), stp, etp, l, R_list, "gauss_dv")
    for R in R_list: out[f"qm_null_dv_R{int(R)}"] = D7.apply_qm_calib(frz["qcal"], R, "gauss_dv", ubn[R]).reshape(nW, nT)
    d0 = np.linalg.norm(l, axis=1).reshape(nW, nT)
    for R in R_list: out[f"qm_persist_bin_R{int(R)}"] = (d0 <= R).astype(np.float32); out[f"qm_persist_cont_R{int(R)}"] = np.exp(-d0 / R).astype(np.float32)
    oracle = {f"qm_oracle_R{int(R)}": D7.reach_labels(Y_cyc, w_rows[ii], stp, etp, l, R).reshape(nW, nT).astype(np.uint8) for R in R_list}
    return {k: v.astype(np.float32) for k, v in out.items()}, oracle

def auc_rank(score, label):
    from scipy.stats import rankdata
    label = label.astype(bool); n1 = label.sum(); n0 = len(label) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    r = rankdata(score); return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

# ----------------------------------------------------------------------------- main
def run(a, log):
    t0 = time.time(); city = a.city
    data_root = Path(a.data) if Path(a.data).is_absolute() else HERE / a.data; models_root = Path(a.models) if Path(a.models).is_absolute() else HERE / a.models
    out_root = Path(a.out) if Path(a.out).is_absolute() else HERE / a.out
    out = out_root / city / ("pilot" if a.pilot else ""); out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("inst_*.npz")) and not a.force: sys.exit(f"{out} มี instance อยู่แล้ว — ใช้ --force ถ้าต้องการสร้างทับ (ไฟล์ของ E1 เท่านั้น ไม่กระทบ models/)")
    D7, sha_d7 = import_sD7(log)
    R_list = [float(R) for R in a.R]; R_main = float(a.r_main)
    if R_main not in R_list: R_list.append(R_main)

    # ---- โมเดลที่ freeze
    log("โหลดโมเดลที่ freeze")
    frz = load_frozen(models_root, city, a.base_run, a.base_split, log, required=True)
    frz_m1 = load_frozen(models_root, city, a.m1_run, a.m1_split, log, required=False)
    if frz_m1 is not None and frz_m1["M"] != 1: log(f"  เตือน: {a.m1_run} มี M = {frz_m1['M']} ไม่ใช่ 1")
    for R in R_list:
        if str(int(R)) not in frz["qcal"]["R"]: sys.exit(f"qm_calib.json ของ {frz['run']} ไม่มี R = {R:g} — เลือก R จาก {list(frz['qcal']['R'])}")

    # ---- ข้อมูล test (final) ผ่าน loader ของ D7 (mask dates.test_final เหมือน D7)
    ns = SimpleNamespace(split=a.base_split, clean_only=False)
    Y_all, v_all, meta, mask, clusters, tau, cfg, note = D7.load_split(data_root, city, models_root / city / a.base_run, "test", ns, log)
    N_all, H = Y_all.shape[:2]
    if frz["pred"]["mu"].shape[0] != N_all: sys.exit(f"pred_test.npz ({frz['pred']['mu'].shape[0]} แถว) ไม่ตรง test_Y ({N_all}) — คนละ D4?")
    if frz_m1 is not None and frz_m1["pred"]["mu"].shape[0] != N_all: log("  เตือน: abl_M1 pred_test แถวไม่ตรง → ข้าม qm_M1"); frz_m1 = None
    if a.cycle_min > H: sys.exit(f"--cycle-min {a.cycle_min} > H = {H}")
    log(f"test split: {'; '.join(note) or 'all'} · หน้าต่าง {int(mask.sum()):,}/{N_all:,} · H = {H} · τ = {tau:g}")
    P = frz["pred"]; src = P if "lat_T" in P else (meta if meta is not None else {})
    for k in ("cab", "date", "hour", "is_cycle_start", "lat_T", "lon_T"):
        if k not in src: sys.exit(f"ไม่พบคีย์ {k} ใน pred_test.npz/meta")
    get = lambda k: (src[k] if isinstance(src, dict) else src[k].values)
    cab = to_str(get("cab")); date = to_str(get("date")); hour = np.asarray(get("hour")).astype(int); cyc = np.asarray(get("is_cycle_start")).astype(bool)
    lat = np.asarray(get("lat_T"), np.float64); lon = np.asarray(get("lon_T"), np.float64); tT = to_epoch_s(get("t_T")) if "t_T" in src else None
    rows_cyc = np.flatnonzero(mask & cyc)
    if len(rows_cyc) == 0: sys.exit("ไม่มีหน้าต่าง is_cycle_start ใน split นี้")

    # ---- ระนาบพิกัด
    bb_cfg = cfg.get("bbox") or cfg.get("BBOX")
    fr = Frame(np.median(lat[rows_cyc]), np.median(lon[rows_cyc])); bbox_xy = fr.bbox_xy(bb_cfg)
    xy_all = fr.xy(lat, lon)
    # frame check: คู่หน้าต่างคันเดียวกัน วันเดียวกัน ห่าง k ก้าว → Y[row1][k−1] ควร ≈ x_T2 − x_T1
    frame_err = None
    if tT is not None:
        dt_s = float(cfg.get("dt_s", 60)); df_ = pd.DataFrame({"cab": cab[mask], "date": date[mask], "t": tT[mask], "row": np.flatnonzero(mask)}).sort_values(["cab", "date", "t"])
        errs = []
        for (_, _), g in df_.groupby(["cab", "date"]):
            if len(g) < 2: continue
            r = g["row"].values; t = g["t"].values
            k = np.round((t[1:] - t[:-1]) / dt_s).astype(int); ok = (k >= 1) & (k <= H)
            if ok.any():
                r0, r1, kk = r[:-1][ok], r[1:][ok], k[ok]
                errs.append(np.linalg.norm(Y_all[r0, kk - 1] - (xy_all[r1] - xy_all[r0]), axis=1))
            if sum(len(e) for e in errs) > 20000: break
        if errs: e = np.concatenate(errs); frame_err = dict(n=int(len(e)), median_m=float(np.median(e)), p90_m=float(np.percentile(e, 90)))
        log(f"frame check (ระนาบ E1 vs เมตรของ D2): {frame_err}")
    else: log("frame check: ไม่มี t_T → ข้าม")

    # ---- cycle ที่ใช้ได้
    W_min = int((cfg.get("usable_cycle_rule") or {}).get("min_workers_ready", cfg.get("W", 60)))
    dfc = pd.DataFrame({"row": rows_cyc, "cab": cab[rows_cyc], "date": date[rows_cyc], "hour": hour[rows_cyc]}).drop_duplicates(["date", "hour", "cab"])
    per = dfc.groupby(["date", "hour"])["row"].apply(lambda s: np.asarray(s)).reset_index(); per["n"] = per["row"].apply(len)
    usable = per[per["n"] >= W_min].sort_values(["date", "hour"]).reset_index(drop=True)
    if len(usable) == 0: sys.exit(f"ไม่มี cycle ที่มีคันพร้อม ≥ W_min = {W_min}")
    order = np.random.RandomState(seed_of(a.master_seed, "E1", city)).permutation(len(usable))
    n_inst = min(a.n_inst, len(usable)); chosen = usable.iloc[order[:n_inst]].reset_index(drop=True); holdout = usable.iloc[order[n_inst:]]
    n_make = min(a.pilot, n_inst) if a.pilot else n_inst
    log(f"cycle ใช้ได้ (≥ {W_min} คัน): {len(usable)} · median คัน {int(usable['n'].median())} · เลือก {n_inst} (hold-out {len(holdout)}) · สร้าง {n_make}{' (pilot)' if a.pilot else ''}")

    # ---- ค่าประชากรจาก val: p75(|y_cycle_min|) สำหรับขยายพื้นที่
    Yv = np.load(data_root / city / "processed" / "val_Y.npy", mmap_mode="r"); expand = float(np.percentile(np.linalg.norm(np.asarray(Yv[:, a.cycle_min - 1]), axis=1), 75))
    log(f"พื้นที่วางงาน = hull(fleet) + {expand:.0f} m (p75 |y_{a.cycle_min}| ของ val){' ∩ bbox เมือง' if bbox_xy is not None else ''}")

    # ---- สร้างทีละ instance
    Y_cyc_view = Y_all                                                   # reach_labels รับ index แถวดิบ
    rows_report = []; hist = dict(E=[], K=[], dens=[], qm=[])
    for k in range(n_make):
        c = chosen.iloc[k]; cand = np.asarray(c["row"]); n_cand = len(cand)
        rng_W = np.random.RandomState(seed_of(a.master_seed, "W", city, k)); rng_T = np.random.RandomState(seed_of(a.master_seed, "T", city, k))
        rng_time = np.random.RandomState(seed_of(a.master_seed, "time", city, k)); rng_rw = np.random.RandomState(seed_of(a.master_seed, "reward", city, k))
        rng_K = np.random.RandomState(seed_of(a.master_seed, "K", city, k)); rng_e = np.random.RandomState(seed_of(a.master_seed, "battery", city, k))
        fleet_xy = xy_all[cand]
        nW = min(a.W, n_cand); w_rows = np.sort(cand[rng_W.choice(n_cand, nW, replace=False)])
        if nW < a.W: log(f"  inst {k:03d}: คันพร้อม {n_cand} < |W| = {a.W} → ใช้ {nW}")
        w_xy = xy_all[w_rows]; w_v = v_all[w_rows].astype(np.float32)
        area = Area(fleet_xy, expand, bbox_xy)
        task_xy, pinfo = place_tasks(area, a.T, a.placement, rng_T, fleet_xy, a.centre_src, a.fleet_frac, a.centre_jitter, a.dens_alpha)
        durs = np.array([d for d in DURATIONS if d < a.cycle_min]); u = rng_time.rand(a.T, 2)                 # ดึงเป็นแถวต่องาน → |T| น้อยลง = prefix (CRN)
        D = durs[(u[:, 0] * len(durs)).astype(int)]; st = 1 + (u[:, 1] * (a.cycle_min - D)).astype(int); et = st + D   # st ∈ [1, cycle_min − D] · et ≤ cycle_min (เหมือน sample_pairs ของ D7)
        u2 = rng_rw.rand(a.T, 2); r = 0.5 + u2[:, 0]; eps = 2.0 + 6.0 * u2[:, 1]; e_j = rng_e.uniform(40.0, 100.0, nW)
        K, dens, k_used = make_K(task_xy, fleet_xy, a.k_rule, rng_K)
        B = a.beta * float((r * (K + a.m_extra)).sum()); outlay_max = float((r * (K + a.m_extra)).sum())
        cell = np.floor(task_xy / a.cell_m).astype(int); n_cells = len({tuple(x) for x in cell})
        QM, ORC = qm_all(D7, frz, frz_m1, w_rows, w_xy, w_v, tau, task_xy, st, et, R_list, Y_cyc_view, H)
        path_true = (w_xy[:, None, :] + Y_all[w_rows]).astype(np.float32)

        # ---- สถิติ G2 (ที่ R หลัก) — ไม่ใช่ส่วนของโจทย์
        Q = QM[f"qm_model_R{int(R_main)}"]; O = ORC[f"qm_oracle_R{int(R_main)}"]
        E_i = Q.sum(0); g21 = float((Q.max(1) < 0.05).mean() * 100); g22 = float((E_i < 0.5).mean() * 100); g22b = float((E_i < K).mean() * 100)
        n_theta = {th: float((Q >= th).sum(0).mean()) for th in THETAS}
        aucs = {nm: auc_rank(QM[f"{nm}_R{int(R_main)}"].ravel(), O.ravel()) for nm in ("qm_model", "qm_null_dv", "qm_persist_cont") + (("qm_M1",) if frz_m1 is not None else ())}
        meta_i = dict(k=k, city=city, date=str(c["date"]), hour=int(c["hour"]), n_candidates=int(n_cand), W=int(nW), T=int(a.T), cycle_min=a.cycle_min, R_list=R_list, R_main=R_main,
                      placement=a.placement, centre_src=a.centre_src, fleet_frac=a.fleet_frac, centre_jitter_m=a.centre_jitter, dens_alpha=a.dens_alpha, n_centres=int(pinfo["n_centres"]), n_centres_fleet=int(pinfo["n_centres_fleet"]), cell_m=a.cell_m, n_cells_with_task=int(n_cells), pt_s=60, Q_max=a.q_max, M_extra=a.m_extra, beta=a.beta,
                      B=B, outlay_max=outlay_max, K_rule=k_used, battery_on=bool(a.battery_on), area_km2=float(area.area_km2(rng_T)), expand_m=expand,
                      frame=dict(lat0=fr.lat0, lon0=fr.lon0), seeds=dict(W=seed_of(a.master_seed, "W", city, k), T=seed_of(a.master_seed, "T", city, k)),
                      qm_source=dict(base=f"{frz['run']}/{frz['seed']}", M1=(f"{frz_m1['run']}/{frz_m1['seed']}" if frz_m1 else None)), version=VERSION)
        np.savez_compressed(out / f"inst_{k:03d}.npz",
                            w_cab=cab[w_rows], w_row=w_rows.astype(np.int64), w_xy=w_xy.astype(np.float32), w_v=w_v, w_e=e_j.astype(np.float32),
                            w_sw=np.zeros(nW, np.float32), w_ew=np.full(nW, float(a.cycle_min), np.float32), fleet_xy=fleet_xy.astype(np.float32),
                            task_xy=task_xy.astype(np.float32), task_R=np.full(a.T, R_main, np.float32), task_st=st.astype(np.int16), task_et=et.astype(np.int16),
                            task_pt_s=np.full(a.T, 60, np.int16), task_r=r.astype(np.float32), task_eps=eps.astype(np.float32), task_K=K.astype(np.int8), task_cell=cell.astype(np.int32),
                            task_dens=dens.astype(np.float32), B=np.float32(B), meta=json.dumps(meta_i, ensure_ascii=False), **QM)
        np.savez_compressed(out / f"oracle_{k:03d}.npz", path_true=path_true, w_row=w_rows.astype(np.int64), **ORC)
        rows_report.append(dict(k=k, date=meta_i["date"], hour=meta_i["hour"], n_cand=n_cand, W=nW, T=a.T, area_km2=round(meta_i["area_km2"], 1), n_centres=pinfo["n_centres"], n_fleet_c=pinfo["n_centres_fleet"],
                                K_mean=round(float(K.mean()), 2), B=round(B, 1), outlay_max=round(outlay_max, 1), qm_mean=round(float(Q.mean()), 4), E_mean=round(float(E_i.mean()), 2),
                                pos_oracle=round(float(O.mean()), 4), **{f"n_i(θ={th})": round(v_, 1) for th, v_ in n_theta.items()},
                                G21_pct=round(g21, 1), G22_pct=round(g22, 1), G22b_pct=round(g22b, 1), **{f"AUC_{nm}": round(v_, 3) for nm, v_ in aucs.items()}))
        hist["E"].append(E_i); hist["K"].append(K); hist["dens"].append(dens); hist["qm"].append(Q.ravel())
        log(f"  inst {k:03d} {meta_i['date']} h{meta_i['hour']:02d} · cand {n_cand} · area {meta_i['area_km2']:.0f} km² · K̄ {K.mean():.2f} ({k_used}) · B {B:.0f}/{outlay_max:.0f} · "
            f"QM̄ {Q.mean():.3f} · G2.1 {g21:.0f}% · G2.2 {g22:.0f}% · G2.2b {g22b:.0f}% · AUC model {aucs['qm_model']:.3f} null {aucs['qm_null_dv']:.3f}")

    # ---- report + config
    df = pd.DataFrame(rows_report); df.to_csv(out / "E1_instances.csv", index=False)
    g21_all, g22_all, g22b_all = df["G21_pct"].mean(), df["G22_pct"].mean(), df["G22b_pct"].mean()
    verdict = {"G2.1 (% worker ไม่มีงานเอื้อม < 40)": (g21_all < 40, f"{g21_all:.1f} %"), "G2.2 (% task E_i < 0.5 < 20)": (g22_all < 20, f"{g22_all:.1f} %"),
               "G2.2b (% task E_i < K_i — รายงาน; > 70 ควรลด |T|/เพิ่ม |W|)": (g22b_all <= 70, f"{g22b_all:.1f} %")}
    if frame_err: verdict["frame check (median |Y − Δx_T| ≤ 5 m)"] = (frame_err["median_m"] <= 5.0, f"{frame_err['median_m']:.2f} m (p90 {frame_err['p90_m']:.1f}, n={frame_err['n']})")
    if not a.no_figs: make_figs(out, df, hist, city, R_main, a, log)
    lines = [f"# E1 report — {city}{' (pilot)' if a.pilot else ''} · {VERSION} · {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"QM จาก **{frz['run']}/{frz['seed']}** ({frz['split']}, sha ตรง handoff) · qm_M1 จาก {frz_m1['run'] + '/' + frz_m1['seed'] if frz_m1 else '— (ไม่มี)'} · sD7 v7 sha `{sha_d7[:12]}` · "
             f"split {'; '.join(note) or 'all'} · cycle ใช้ได้ {len(usable)} (≥ {W_min} คัน) → เลือก {n_inst} · สร้าง {n_make}", "",
             f"พารามิเตอร์: |W| {a.W} · |T| {a.T} · cycle {a.cycle_min} นาที · R {R_list} (หลัก {R_main:g}) · placement {a.placement} (centre {a.centre_src}{f' {a.fleet_frac:g}' if a.centre_src == 'mixed' else ''}, jitter {a.centre_jitter:g} m, α {a.dens_alpha:g}) · cell_m {a.cell_m} · β {a.beta} · M_i {a.m_extra} · Q_max {a.q_max} · K rule {a.k_rule} · battery {'on' if a.battery_on else 'off'} · master seed {a.master_seed}", "",
             "## Gate G2 (ตรวจใน E1)", "", "| ข้อ | ผล | ค่า |", "|---|---|---|"] + [f"| {k_} | {'✅' if ok else '❌'} | {v_} |" for k_, (ok, v_) in verdict.items()] + ["",
             "G2.3–G2.6 ตรวจหลัง E4-lite นำร่อง (utilization ของ B · corr(f₁,f₃) · clustered vs uniform · oracle > model > persistence)", "",
             "## ต่อ instance", "", _md_table(df), "",
             "## หมายเหตุ", f"- E_i = Σ_j QM_model[j,i] (จำนวนคนที่คาดว่าจะถึง) · n_i(θ) = จำนวน worker ที่ QM ≥ θ เฉลี่ยต่องาน · pos_oracle = สัดส่วนคู่ (j,i) ที่เส้นทางจริงถึงจริง",
             f"- AUC_* = AUC ของ QM แหล่งนั้นเทียบ oracle บนคู่ทั้งหมดของ instance (sanity: ควรใกล้ค่า D7 ≈ 0.80 สำหรับ model, ≈ 0.72 สำหรับ null)",
             f"- f₃ ยังเปิด 2 นิยาม (budget saving ตาม Short_Formulation v3 (15) / coverage ตาม E1 v2) — instance เก็บทั้ง r_i, K_i (สำหรับ (15)) และ cell_id (สำหรับ coverage) ตัดสินที่ G2.4 ด้วยผล pilot",
             f"- ไฟล์ oracle_*.npz (path_true, qm_oracle) ให้ E4 replay เท่านั้น — optimizer ห้ามอ่าน", ""]
    (out / "E1_report.md").write_text("\n".join(lines), encoding="utf-8")
    cfg_out = dict(version=VERSION, city=city, params={k_: (v_ if not isinstance(v_, Path) else str(v_)) for k_, v_ in vars(a).items()}, R_list=R_list, H=H, W_min=W_min, tau=tau, split_note=note,
                   n_usable=int(len(usable)), chosen_cycles=[dict(k=i, date=str(r_["date"]), hour=int(r_["hour"]), n=int(r_["n"])) for i, r_ in chosen.iterrows()],
                   holdout_cycles=[dict(date=str(r_["date"]), hour=int(r_["hour"]), n=int(r_["n"])) for _, r_ in holdout.iterrows()], expand_m=expand, frame=dict(lat0=fr.lat0, lon0=fr.lon0, bbox_xy=(bbox_xy.tolist() if bbox_xy is not None else None)),
                   frame_check=frame_err, base=dict(run=frz["run"], seed=frz["seed"], handoff=frz["handoff"], sha256=frz["sha"]),
                   M1=(dict(run=frz_m1["run"], seed=frz_m1["seed"], handoff=frz_m1["handoff"], sha256=frz_m1["sha"]) if frz_m1 else None),
                   sD7_pred_eval_sha256=sha_d7, gate_G2={k_: dict(ok=bool(ok), value=v_) for k_, (ok, v_) in verdict.items()}, thomas=THOMAS, durations=DURATIONS, dens_radius_m=DENS_RADIUS_M)
    json.dump(cfg_out, open(out / "E1_config.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=str)
    log(f"เสร็จ {n_make} instance → {out}  ({time.time() - t0:.0f}s) · " + " · ".join(f"{k_.split(' ')[0]} {'✅' if ok else '❌'}" for k_, (ok, _) in verdict.items()))

def _md_table(df):
    try: return df.to_markdown(index=False)
    except Exception: return "```\n" + df.to_string(index=False) + "\n```"          # ไม่มี tabulate

def make_figs(out, df, hist, city, R_main, a, log):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e: log(f"  ไม่มี matplotlib ({e}) → ข้ามรูป"); return
    E = np.concatenate(hist["E"]); K = np.concatenate(hist["K"]); dens = np.concatenate(hist["dens"]); qm = np.concatenate(hist["qm"])
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
    ax[0].hist(E, bins=40, color="#4C78A8"); ax[0].axvline(0.5, color="r", ls="--", lw=1); ax[0].set_title("E_i = sum_j QM_model (expected arrivals)"); ax[0].set_xlabel("E_i")
    ax[1].hist(K, bins=[0.5, 1.5, 2.5, 3.5], color="#F58518", rwidth=0.8); ax[1].set_title("K_i"); ax[1].set_xticks([1, 2, 3])
    ax[2].hist(dens, bins=30, color="#54A24B"); ax[2].set_title(f"dens_i (fleet within {DENS_RADIUS_M/1000:g} km)")
    ax[3].hist(np.clip(qm, 1e-4, 1), bins=np.logspace(-4, 0, 40), color="#B279A2"); ax[3].set_xscale("log"); ax[3].set_title(f"QM_model (R={R_main:g})"); ax[3].set_yscale("log")
    fig.suptitle(f"E1 {city} - {len(df)} instances | W={a.W} T={a.T} {a.placement}"); fig.tight_layout(); fig.savefig(out / "fig_E1_hist.png", dpi=130); plt.close(fig)
    z = np.load(out / "inst_000.npz", allow_pickle=True); o = np.load(out / "oracle_000.npz")
    fig, ax = plt.subplots(figsize=(6.5, 6.5)); Q = z[f"qm_model_R{int(R_main)}"]
    ax.scatter(*z["fleet_xy"].T / 1000, s=6, c="#bbb", label=f"fleet in cycle ({len(z['fleet_xy'])})")
    for p in o["path_true"][:: max(1, len(o["path_true"]) // 20)]: ax.plot(p[:, 0] / 1000, p[:, 1] / 1000, color="#4C78A8", lw=0.6, alpha=0.6)
    ax.scatter(*z["w_xy"].T / 1000, s=14, c="#4C78A8", label=f"workers W={len(z['w_xy'])} (+ some true paths)")
    sc = ax.scatter(*z["task_xy"].T / 1000, s=10 + 14 * z["task_K"], c=Q.sum(0), cmap="viridis", vmin=0, label=f"tasks T={len(z['task_xy'])} (size=K_i, color=E_i)")
    plt.colorbar(sc, ax=ax, label="E_i"); ax.set_aspect("equal"); ax.set_xlabel("km"); ax.set_ylabel("km"); ax.legend(fontsize=7, loc="upper left")
    m = json.loads(str(z["meta"])); ax.set_title(f"inst_000 {city} {m['date']} h{m['hour']:02d} | {a.placement} ({m['n_centres']} centres) | R={R_main:g}", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "fig_E1_map_000.png", dpi=130); plt.close(fig)

# ----------------------------------------------------------------------------- selftest (ข้อมูลสังเคราะห์ — ไม่แตะของจริง)
def selftest(log):
    """สร้าง data/_e1test + models/_e1test ขนาดเล็กในรูปแบบเดียวกับ D4/D5/D6/D7 แล้วรัน E1 ครบทุกขั้น"""
    import tempfile
    root = Path(tempfile.mkdtemp(prefix="e1_selftest_")); city = "_e1test"; rng = np.random.RandomState(0)
    N, H, M, F, Min = 1500, 60, 3, 9, 8; proc = root / "data" / city / "processed"; proc.mkdir(parents=True)
    lat0, lon0 = 37.76, -122.42; kx, ky = 111320 * np.cos(np.radians(lat0)), 110574
    cabs = np.array([f"cab{i:03d}" for i in rng.randint(0, 120, N)]); dates = np.where(rng.rand(N) < 0.5, "2008-06-07", "2008-06-08"); hours = rng.randint(8, 14, N)
    xy0 = rng.randn(N, 2) * np.array([3000, 2500]); v = rng.randn(N, 2) * 150
    steps = v[:, None, :] * (3 * (1 - np.exp(-np.arange(1, H + 1) / 3)))[None, :, None] + np.cumsum(rng.randn(N, H, 2) * 60, 1)
    Y = steps.astype(np.float32); X = rng.randn(N, Min, F).astype(np.float32); X[:, -1, 3] = v[:, 0] / 150; X[:, -1, 4] = v[:, 1] / 150
    np.save(proc / "test_X.npy", X); np.save(proc / "test_Y.npy", Y); np.save(proc / "val_X.npy", X[:500]); np.save(proc / "val_Y.npy", Y[:500])
    meta = pd.DataFrame({"cab": cabs, "seg": 0, "t_T": 1212800000 + hours * 3600 + (dates == "2008-06-08") * 86400, "lat_T": lat0 + xy0[:, 1] / ky, "lon_T": lon0 + xy0[:, 0] / kx,
                         "date": dates, "hour": hours, "is_cycle_start": True, "is_cycle_clean": True})
    meta.to_pickle(proc / "test_meta.pkl"); meta.iloc[:500].to_pickle(proc / "val_meta.pkl")
    json.dump({"names": ["x", "y", "v", "dx", "dy", "dow_sin", "dow_cos", "slot_sin", "slot_cos"], "mean": [0] * 9, "std": [1, 1, 1, 150, 150, 1, 1, 1, 1]}, open(root / "data" / city / "scaler.json", "w"))
    (HERE / "config").mkdir(exist_ok=True)
    cfgp = HERE / "config" / f"data_{city}.yaml"; cfgp.write_text("dt_s: 60\nW: 60\ndates:\n  test_final: ['2008-06-07', '2008-06-08']\n", encoding="utf-8")
    sig = np.linspace(30, 1200, H).astype(np.float32)
    for rn, Mr, split, seed in (("base_M3", 3, "final", "seed1"), ("abl_M1", 1, "dev", "seed0")):
        sd = root / "models" / city / rn / seed; sd.mkdir(parents=True); json.dump({"features": ["x", "y", "v", "dx", "dy", "dow_sin", "dow_cos", "slot_sin", "slot_cos"], "skip_tau": 3.0}, open(sd.parent / "run_config.json", "w"))
        mu = (Y[:, None] + rng.randn(N, Mr, H, 2) * sig[None, None, :, None] * 0.7).astype(np.float32); sg = np.broadcast_to(sig, (N, Mr, H)).astype(np.float32) * np.float32(0.9)
        pi = np.full((N, Mr), 1.0 / Mr, np.float32)
        np.savez(sd / "pred_test.npz", mu=mu, sigma=sg, pi=pi, cab=cabs, t_T=meta["t_T"].values, lat_T=meta["lat_T"].values, lon_T=meta["lon_T"].values, date=dates, hour=hours, is_cycle_start=np.ones(N, bool))
        np.savez(sd / "pred_val.npz", mu=mu[:500], sigma=sg[:500], pi=pi[:500]); json.dump({"lambda_h": [1.1] * H, "T": 1.0}, open(sd / "calib.json", "w"))
        iso = {"x": [0.0, 0.5, 1.0], "y": [0.0, 0.45, 0.9], "n": [1, 1, 1], "tail_merged": 0, "y_max": 0.9, "pos_rate_val": 0.05}
        json.dump({"version": 7, "null_sigma_h": {"gauss_pers": sig.tolist(), "gauss_dv": sig.tolist()}, "R": {str(R): {"ub": iso, "lb": iso, "gauss_dv": iso, "gauss_pers": iso} for R in (150, 200, 250)}}, open(sd / "qm_calib.json", "w"))
        files = {f: f"{city}/{rn}/{seed}/{f}" for f in ("pred_test.npz", "pred_val.npz", "calib.json", "qm_calib.json")}
        hd = sd.parent / f"D7_{split}"; hd.mkdir()
        json.dump({"version": 7, "seed": seed, "qm_variant_for_E1": "qm_ub_cal", "R_main": 200, "AUC_qm_ub_cal@200_per_seed": {seed: 0.83}, "files": files, "sha256": {f: sha256_of(sd / f) for f in files}}, open(hd / "E1_handoff.json", "w"))
    a = SimpleNamespace(**{k.lower(): v for k, v in PYCHARM.items()}); a.city = city; a.data = str(root / "data"); a.models = str(root / "models"); a.out = str(root / "instances")
    a.R = PYCHARM["R_LIST"]; a.pilot = 2; a.W = 40; a.T = 80; a.force = True
    run(a, log)
    z = np.load(root / "instances" / city / "pilot" / "inst_000.npz", allow_pickle=True); o = np.load(root / "instances" / city / "pilot" / "oracle_000.npz")
    assert z["qm_model_R200"].shape == (40, 80) and o["qm_oracle_R200"].shape == (40, 80) and o["path_true"].shape == (40, H, 2)
    assert np.all((z["task_et"] - z["task_st"]) >= 1) and z["task_et"].max() <= 60 and z["task_K"].min() >= 1 and z["task_K"].max() <= 3
    assert np.allclose(z["qm_model_uncal_R200"] >= z["qm_model_uncal_R150"] - 1e-6, True), "QM ควรไม่ลดเมื่อ R โต"
    # CRN: R เปลี่ยน → งานเดิม
    a.R = [150]; a.r_main = 150; a.out = str(root / "instances_R150"); run(a, log)
    z2 = np.load(root / "instances_R150" / city / "pilot" / "inst_000.npz", allow_pickle=True)
    assert np.array_equal(z["task_xy"], z2["task_xy"]) and np.array_equal(z["w_row"], z2["w_row"]) and np.array_equal(z["task_K"], z2["task_K"]), "CRN เสีย: เปลี่ยน R แล้วงานเปลี่ยน"
    # prefix: T น้อยลง → prefix ของเดิม
    a.R = PYCHARM["R_LIST"]; a.r_main = 200; a.T = 30; a.out = str(root / "instances_T30"); run(a, log)
    z3 = np.load(root / "instances_T30" / city / "pilot" / "inst_000.npz", allow_pickle=True)
    assert np.array_equal(z["task_xy"][:30], z3["task_xy"]) and np.array_equal(z["task_st"][:30], z3["task_st"]), "prefix เสีย: |T| น้อยลงแล้วงานไม่ใช่ prefix"
    cfgp.unlink(); log(f"selftest ผ่าน ✅ → {root}  (ลบได้)"); print((root / "instances" / city / "pilot" / "E1_report.md").read_text(encoding="utf-8")[:1800])

def main():
    P = PYCHARM; ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", default=P["CITY"]); ap.add_argument("--data", default=P["DATA"]); ap.add_argument("--models", default=P["MODELS"]); ap.add_argument("--out", default=P["OUT"])
    ap.add_argument("--n-inst", type=int, default=P["N_INST"]); ap.add_argument("--pilot", type=int, default=P["PILOT"], help="สร้างแค่ N ตัวแรก → <out>/<city>/pilot/")
    ap.add_argument("--W", type=int, default=P["W"]); ap.add_argument("--T", type=int, default=P["T"]); ap.add_argument("--cycle-min", type=int, default=P["CYCLE_MIN"])
    ap.add_argument("--R", default=",".join(map(str, P["R_LIST"])), help="รายการ R ที่เก็บ QM เช่น 150,200,250"); ap.add_argument("--r-main", type=float, default=P["R_MAIN"])
    ap.add_argument("--placement", choices=["clustered", "uniform"], default=P["PLACEMENT"]); ap.add_argument("--centre-src", choices=["fleet", "mixed", "uniform"], default=P["CENTRE_SRC"]); ap.add_argument("--fleet-frac", type=float, default=P["FLEET_FRAC"]); ap.add_argument("--centre-jitter", type=float, default=P["CENTRE_JITTER"]); ap.add_argument("--dens-alpha", type=float, default=P["DENS_ALPHA"], help="น้ำหนักเลือกคันเป็นศูนย์ ∝ dens^α (0 = เท่ากันทุกคัน)"); ap.add_argument("--cell-m", type=float, default=P["CELL_M"])
    ap.add_argument("--beta", type=float, default=P["BETA"]); ap.add_argument("--m-extra", type=int, default=P["M_EXTRA"]); ap.add_argument("--q-max", type=int, default=P["Q_MAX"])
    ap.add_argument("--k-rule", choices=["density", "random"], default=P["K_RULE"]); ap.add_argument("--battery-on", action="store_true", default=P["BATTERY_ON"])
    ap.add_argument("--master-seed", type=int, default=P["MASTER_SEED"])
    ap.add_argument("--base-run", default=P["BASE_RUN"]); ap.add_argument("--base-split", default=P["BASE_SPLIT"]); ap.add_argument("--m1-run", default=P["M1_RUN"]); ap.add_argument("--m1-split", default=P["M1_SPLIT"])
    ap.add_argument("--no-figs", action="store_true", default=P["NO_FIGS"]); ap.add_argument("--force", action="store_true", default=P["FORCE"]); ap.add_argument("--selftest", action="store_true", default=P["SELFTEST"])
    a = ap.parse_args(); a.R = [float(x) for x in str(a.R).split(",") if x.strip()]
    t0 = time.time(); log = lambda *m: print(f"[{time.time()-t0:6.1f}s]", *m, flush=True)
    if a.selftest: selftest(log); return
    for city in [c.strip() for c in a.city.split(",") if c.strip()]:
        a.city = city; run(a, log)

if __name__ == "__main__":
    main()
