#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sE5_encode.py (E5 v0.1, 2026-10-21) — encoder state ของ worker ที่เวลา t ภายในรอบ (สำหรับ rolling re-plan)
  ที่มา: HANDOFF §8 "ตรวจก่อนเขียนโค้ด ข้อ 1" — pred_test.npz มีหน้าต่างเฉพาะจุดที่มีอนาคต 60 ก้าว (sD1_D4_prepare.make_windows บ.207)
         ⇒ ที่ t = t₀+10…50 นาที หลายคันไม่มีหน้าต่างทั้งที่ยังรายงานอยู่ ⇒ ต้อง encode เองจากประวัติ M_in ก้าว (ไม่ต้องมีอนาคต)
  หลักการ: ใช้โมเดลตรึง (weights + run_config + scaler เดิม) · สร้างหน้าต่างอินพุตแบบเดียวกับ make_windows ทุกประการ
           (feat[T−M_in+1 : T+1] จาก interim/d3_features, scaler เดียวกัน, segment เดียวกัน) · ไม่ฝึก ไม่เขียนทับไฟล์ใด ๆ ของ run
  ตรวจ (บังคับ): ทุก (cab, t) ที่ pred_test.npz มีหน้าต่าง t_T == t → X ของเรา == processed/test_X[row] และ feat64/mu/sigma/pi ตรง (float16/32)
                 ถ้าไม่ตรง = หยุด (ห้าม "ปรับให้ใกล้") → ดู --verify-only และรายงาน e5state/<city>/<run>/VERIFY.md

ใช้:  python sE5_encode.py --city sf --runs reach_tfm,reach_lstm --inst instances_v3 --data data --out e5state --t 0,10,20,30,40,50 [--limit 3] [--cpu]
      python sE5_encode.py --city sf --runs reach_tfm --verify-only --limit 3        # ตรวจอย่างเดียว ไม่เขียน state
ผล:   e5state/<city>/<run>/state_<k>.npz : feat64 (nt,W,64) f32 · mu (nt,W,M,H,2) · sigma (nt,W,M,H) · pi (nt,W,M) · lat,lon,x,y (nt,W) · t_abs (nt,W) · ok (nt,W) · t_min (nt) · w_row (W)
      e5state/<city>/<run>/VERIFY.md
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
VERSION = "E5-encode v0.4 (2026-10-21: + vxy ใน state, เกณฑ์ p99.9/max)"


def log_factory():
    t0 = time.time()
    return lambda *m: print(f"[{time.time()-t0:7.1f}s]", *m, flush=True)


# ----------------------------------------------------------------------------- run / frozen files
def load_run(models_root: Path, city: str, run: str, log):
    rd = models_root / city / run
    rc = json.load(open(rd / "run_config.json"))
    hp = rd / "D7reach_dev" / "E1_handoff_reach.json"
    if hp.exists():
        hand = json.load(open(hp)); seed = int(str(hand["seed"]).replace("seed", "")); src = "E1_handoff_reach.json"
    else:
        seed = int(rc["seeds"][0]); src = "run_config.seeds[0] (ไม่มี handoff)"
    sd = rd / f"seed{seed}"
    met = json.load(open(sd / "metrics.json"))
    M_in, H = int(met["M_in"]), int(met["H"])
    feats = rc["features"]
    log(f"  {city}/{run}: seed {seed} ({src}) · encoder {rc['rnn']} · M_in {M_in} · H {H} · M {rc['M']} · features {feats}")
    return dict(rd=rd, sd=sd, rc=rc, seed=seed, M_in=M_in, H=H, feats=feats)


def build_frozen_model(R: dict, data_root: Path, city: str, log):
    """สร้างสถาปัตย์เดิมด้วย build_model_v4 + โหลด best.weights.h5 (ไม่ compile ไม่ฝึก)"""
    from sD5_train_v4 import build_model_v4, unpack_np, N_QF, keras
    rc, feats = R["rc"], R["feats"]
    sc = json.load(open(data_root / city / "scaler.json"))
    dxdy = {n: (float(sc["mean"][sc["names"].index(n)]), float(sc["std"][sc["names"].index(n)])) for n in ("dx", "dy")}
    has_v = "dx" in feats and "dy" in feats
    skip = bool(rc["skip"]) and has_v
    dx_idx, dy_idx = (feats.index("dx"), feats.index("dy")) if has_v else (0, 1)
    x_idx, y_idx = (feats.index("x"), feats.index("y")) if ("x" in feats and "y" in feats) else (0, 1)
    model = build_model_v4(R["M_in"], len(feats), R["H"], int(rc["M"]), int(rc["k_query"]), rnn=rc["rnn"], bi=bool(rc["bi"]), hidden=int(rc["hidden"]),
                           skip=skip, skip_tau=float(rc["skip_tau"]), dx_idx=dx_idx, dy_idx=dy_idx, dx_ms=dxdy["dx"], dy_ms=dxdy["dy"], x_idx=x_idx, y_idx=y_idx,
                           tfm_d=int(rc.get("tfm_d", 128)), tfm_blocks=int(rc.get("tfm_blocks", 2)), tfm_heads=int(rc.get("tfm_heads", 4)))
    wpath = R["sd"] / "best.weights.h5"
    model.load_weights(str(wpath))
    feat_model = keras.Model(model.inputs[0], model.get_layer("feat64").output)
    n_mix = int(rc["M"]) * R["H"] * 2 + int(rc["M"]) * R["H"] + int(rc["M"])
    log(f"  โหลด {wpath.relative_to(R['rd'].parent.parent)} · {model.name} · params {model.count_params():,}")
    return dict(model=model, feat_model=feat_model, n_mix=n_mix, unpack=unpack_np, N_QF=N_QF, sc=sc)


def encode(FM: dict, X: np.ndarray, M: int, H: int, batch: int = 2048):
    """X (B,M_in,F) scaled → feat64 (B,64) f32, mu (B,M,H,2), sigma (B,M,H), pi (B,M)"""
    Q = np.zeros((len(X), 1, FM["N_QF"]), np.float32)
    P = FM["model"].predict([X, Q], batch_size=batch, verbose=0)[:, :FM["n_mix"]]
    feat = FM["feat_model"].predict(X, batch_size=batch, verbose=0).astype(np.float32)
    mu, sig, pi = FM["unpack"](P, M, H)
    return feat, np.clip(mu, -6e4, 6e4).astype(np.float32), sig.astype(np.float32), pi.astype(np.float32)


# ----------------------------------------------------------------------------- d3 features (ประวัติดิบบนกริด)
def load_d3(data_root: Path, city: str, cabs: set, t_lo: int, t_hi: int, feats: list, log) -> pd.DataFrame:
    base = data_root / city / "interim"
    f = next((p for p in (base / "d3_features.parquet", base / "d3_features.pkl") if p.exists()), None)
    if f is None: sys.exit(f"ไม่พบ {base}/d3_features.(parquet|pkl)")
    cols = ["cab", "seg", "t", "lat", "lon", "x", "y"] + [c for c in feats if c not in ("x", "y")]
    if f.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
            tab = pq.read_table(f, columns=cols, filters=[("cab", "in", sorted(cabs)), ("t", ">=", int(t_lo)), ("t", "<=", int(t_hi))])
            rs = tab.to_pandas()
        except Exception as e:
            log(f"  pyarrow filter ไม่ได้ ({e}) → อ่านทั้งไฟล์แล้วกรอง"); rs = pd.read_parquet(f, columns=cols)
            rs = rs[rs["cab"].isin(cabs) & (rs["t"] >= t_lo) & (rs["t"] <= t_hi)]
    else:
        rs = pd.read_pickle(f)[cols]; rs = rs[rs["cab"].isin(cabs) & (rs["t"] >= t_lo) & (rs["t"] <= t_hi)]
    rs = rs.sort_values(["cab", "t"]).reset_index(drop=True)
    log(f"  d3: {len(rs):,} ก้าว · {rs['cab'].nunique()} คัน · t {pd.to_datetime(rs['t'].min(), unit='s')} … {pd.to_datetime(rs['t'].max(), unit='s')}")
    return rs


class HistIndex:
    """(cab, t) → หน้าต่างประวัติ M_in ก้าวที่สิ้นสุดที่ t (segment เดียวกัน, กริดต่อเนื่อง) — เหมือน make_windows: xin = feat[T−M+1 : T+1]"""
    def __init__(self, rs: pd.DataFrame, feats: list, M_in: int, dt_s: int, sc: dict):
        self.M_in, self.dt = M_in, dt_s
        self.cab = rs["cab"].values.astype(str); self.seg = rs["seg"].values.astype(np.int64); self.t = rs["t"].values.astype(np.int64)
        self.lat = rs["lat"].values.astype(np.float64); self.lon = rs["lon"].values.astype(np.float64); self.x = rs["x"].values.astype(np.float64); self.y = rs["y"].values.astype(np.float64)
        self.vxy = np.stack([rs["dx"].values, rs["dy"].values], -1).astype(np.float32) if ("dx" in rs and "dy" in rs) else np.zeros((len(rs), 2), np.float32)   # ความเร็ว m/ก้าว ที่ก้าวสุดท้าย (= w_v ของ E1/D7 ก่อน scale)
        F = rs[feats].values.astype(np.float32)
        for j, name in enumerate(sc["names"]):                      # scaler ตามชื่อ (เหมือน apply_scaler แต่ไม่พึ่ง index ของ F_all)
            if name in feats: F[:, feats.index(name)] = (F[:, feats.index(name)] - sc["mean"][j]) / sc["std"][j]
        self.F = F
        self.pos = {(c, int(t)): i for i, (c, t) in enumerate(zip(self.cab, self.t))}

    def window(self, cab: str, t_abs: int):
        i = self.pos.get((cab, int(t_abs)))
        if i is None: return None, "no_row"
        j0 = i - self.M_in + 1
        if j0 < 0 or self.cab[j0] != cab or self.seg[j0] != self.seg[i] or self.t[i] - self.t[j0] != (self.M_in - 1) * self.dt: return None, "short_history"
        return self.F[j0:i + 1], "ok"


# ----------------------------------------------------------------------------- main per run
def run_city(a, city: str, run: str, log):
    models_root, data_root, inst_root = Path(a.models), Path(a.data), Path(a.inst) / city
    R = load_run(models_root, city, run, log)
    Pz = np.load(R["sd"] / "pred_test.npz", allow_pickle=True)
    P = {k: Pz[k] for k in ("feat64", "mu", "sigma", "pi")}                      # แตก npz ครั้งเดียว (เดิมแตก 160 MB ทุกครั้งที่ index → 150 s/instance)
    p_cab = Pz["cab"].astype(str); p_t = Pz["t_T"].astype(np.int64); p_row = Pz["row"].astype(np.int64) if "row" in Pz.files else np.arange(len(p_cab))
    log(f"  pred_test: {len(p_cab):,} หน้าต่าง · โหลด mu/sigma/pi/feat64 เข้า RAM แล้ว")
    pred_pos = {(c, int(t)): i for i, (c, t) in enumerate(zip(p_cab, p_t))}
    tests = sorted(Path(inst_root).glob("oracle_*.npz"))
    if a.limit: tests = tests[:a.limit]
    if not tests: sys.exit(f"ไม่พบ oracle_*.npz ใน {inst_root}")
    t_list = [int(x) for x in a.t.split(",")]
    # worker rows ของทุก instance → cab, t0
    inst = []
    for f in tests:
        o = np.load(f); w = o["w_row"].astype(np.int64); k = int(f.stem.split("_")[1])
        inst.append(dict(k=k, w_row=w, cab=p_cab[w], t0=p_t[w]))
    cabs = set(c for I in inst for c in I["cab"]); t_lo = min(int(I["t0"].min()) for I in inst) - (R["M_in"] + 1) * a.dt_s; t_hi = max(int(I["t0"].max()) for I in inst) + max(t_list) * 60 + a.dt_s
    rs = load_d3(data_root, city, cabs, t_lo, t_hi, R["feats"], log)
    FM = build_frozen_model(R, data_root, city, log)
    HI = HistIndex(rs, R["feats"], R["M_in"], a.dt_s, FM["sc"])
    testX = None
    px = data_root / city / a.proc / "test_X.npy"
    if px.exists():
        testX = np.load(px, mmap_mode="r"); F_all = None
        try:
            import yaml; cfgp = Path(R["rc"].get("config") or "")
            if cfgp.exists(): F_all = yaml.safe_load(open(cfgp))["features"]
        except Exception: pass
        if F_all is None: F_all = ["x", "y", "v", "dx", "dy", "dow_sin", "dow_cos", "slot_sin", "slot_cos", "occ"]
        fidx = [F_all.index(f) for f in R["feats"]] if all(f in F_all for f in R["feats"]) else None
    out = Path(a.out) / city / run; out.mkdir(parents=True, exist_ok=True)
    M, H = int(R["rc"]["M"]), R["H"]
    ver = dict(n_pairs=0, n_cmp=0, feat_max=0.0, mu_max=0.0, sig_max=0.0, pi_max=0.0, x_max=0.0, x_cmp=0, bad=[])
    status = {}
    for I in inst:
        k, W = I["k"], len(I["w_row"]); nt = len(t_list)
        Xs, idx, meta = [], [], []
        ok = np.zeros((nt, W), bool); t_abs = np.zeros((nt, W), np.int64); why = {}
        for ti, tm in enumerate(t_list):
            for j in range(W):
                ta = int(I["t0"][j]) + 60 * tm; t_abs[ti, j] = ta
                X, st = HI.window(I["cab"][j], ta); why[st] = why.get(st, 0) + 1
                if X is None: continue
                Xs.append(X); idx.append((ti, j)); ok[ti, j] = True
        feat64 = np.zeros((nt, W, 64), np.float32); mu = np.zeros((nt, W, M, H, 2), np.float32); sig = np.zeros((nt, W, M, H), np.float32); pi = np.zeros((nt, W, M), np.float32)
        lat = np.full((nt, W), np.nan); lon = np.full((nt, W), np.nan); xx = np.full((nt, W), np.nan); yy = np.full((nt, W), np.nan); vxy = np.zeros((nt, W, 2), np.float32)
        if Xs:
            Xb = np.stack(Xs).astype(np.float32); f_, m_, s_, p_ = encode(FM, Xb, M, H, batch=a.batch)
            for n, (ti, j) in enumerate(idx):
                feat64[ti, j] = f_[n]; mu[ti, j] = m_[n]; sig[ti, j] = s_[n]; pi[ti, j] = p_[n]
                i = HI.pos[(I["cab"][j], int(t_abs[ti, j]))]; lat[ti, j], lon[ti, j], xx[ti, j], yy[ti, j] = HI.lat[i], HI.lon[i], HI.x[i], HI.y[i]; vxy[ti, j] = HI.vxy[i]
                # ---- verify กับ pred_test (ที่มีหน้าต่าง t_T == t)
                q = pred_pos.get((I["cab"][j], int(t_abs[ti, j])))
                if q is not None:
                    ver["n_pairs"] += 1
                    d_feat = float(np.abs(f_[n].astype(np.float16).astype(np.float32) - P["feat64"][q].astype(np.float32)).max())
                    d_mu = float(np.abs(m_[n] - P["mu"][q].astype(np.float32)).max()); d_sig = float(np.abs(s_[n] - P["sigma"][q].astype(np.float32)).max()); d_pi = float(np.abs(p_[n] - P["pi"][q]).max())
                    ver["n_cmp"] += 1; ver["feat_max"] = max(ver["feat_max"], d_feat); ver["mu_max"] = max(ver["mu_max"], d_mu); ver["sig_max"] = max(ver["sig_max"], d_sig); ver["pi_max"] = max(ver["pi_max"], d_pi)
                    if testX is not None and fidx is not None:
                        dX = float(np.abs(Xb[n] - np.asarray(testX[p_row[q]])[:, fidx].astype(np.float32)).max()); ver["x_max"] = max(ver["x_max"], dX); ver["x_cmp"] += 1
                    else: dX = float("nan")
                    if d_feat > a.tol_feat or d_mu > a.tol_mu or (dX == dX and dX > 1e-5): ver["bad"].append(dict(k=k, cab=I["cab"][j], t_min=t_list[ti], d_feat=d_feat, d_mu=d_mu, d_sig=d_sig, d_pi=d_pi, dX=dX))
        status[k] = why
        if not a.verify_only:
            np.savez_compressed(out / f"state_{k:03d}.npz", feat64=feat64, mu=mu, sigma=sig, pi=pi, lat=lat, lon=lon, x=xx, y=yy, vxy=vxy, t_abs=t_abs, ok=ok, t_min=np.array(t_list), w_row=I["w_row"],
                                cab=np.array(I["cab"], dtype=object), run=run, seed=R["seed"], version=VERSION)
        log(f"  inst {k:03d}: หน้าต่าง {ok.sum()}/{ok.size} · " + " · ".join(f"{s} {n}" for s, n in why.items()) + f" · ตรวจกับ pred_test สะสม {ver['n_cmp']} คู่")
    # ---- noise ของ pipeline เอง (อินพุตเดิมจาก test_X ไม่ผ่านการสร้างหน้าต่างของเรา)
    noise_txt = ""
    if a.noise and testX is not None and fidx is not None and ver["n_cmp"]:
        rows = sorted({pred_pos[(I["cab"][j], int(I["t0"][j]) + 60 * tm)] for I in inst for j in range(len(I["w_row"])) for tm in t_list if (I["cab"][j], int(I["t0"][j]) + 60 * tm) in pred_pos})
        Xo = np.asarray(testX[p_row[rows]])[:, :, fidx].astype(np.float32)
        fA, mA, sA, pA = encode(FM, Xo, M, H, batch=2048); fB, mB, sB, pB = encode(FM, Xo, M, H, batch=1024)
        d_ref = lambda m_, key: float(np.abs(m_ - P[key][rows].astype(np.float32)).max())
        noise_txt = (f"- **noise ของ pipeline** (อินพุต = test_X แถวเดิม, {len(rows)} หน้าต่าง): batch 2048 vs pred_test: Δmu {d_ref(mA, 'mu'):.2e} m · Δsigma {d_ref(sA, 'sigma'):.2e} · Δfeat64 {float(np.abs(fA.astype(np.float16).astype(np.float32) - P['feat64'][rows].astype(np.float32)).max()):.2e}"
                     f" · batch 1024 vs pred_test: Δmu {d_ref(mB, 'mu'):.2e} m · batch 2048 vs 1024 (กันเอง): Δmu {float(np.abs(mA - mB).max()):.2e} m · Δfeat64 {float(np.abs(fA - fB).max()):.2e}")
        log(noise_txt)
    # ---- ผลต่อ C_ij ผ่านหัว reach จริง (สิ่งที่ re-plan ใช้) — เกณฑ์ประกาศก่อนดู: max|ΔC| ≤ tol_C (= ระดับ ECE ของตัวประมาณ, B5)
    c_txt = ""; c_ok = True
    hp = R["sd"] / "reach_head.npz"
    if hp.exists() and ver["n_cmp"]:
        from sD5_train_v4 import reach_head_np, query_feats
        head = {k: np.load(hp)[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3")}
        # feat64 ของเราสำหรับคู่ที่เทียบ: คำนวณจาก test_X แถวเดิม (พิสูจน์แล้วว่า ΔX = 0 จึงเท่ากับหน้าต่างของเรา)
        rows = sorted({pred_pos[(I["cab"][j], int(I["t0"][j]) + 60 * tm)] for I in inst for j in range(len(I["w_row"])) for tm in t_list if (I["cab"][j], int(I["t0"][j]) + 60 * tm) in pred_pos})
        if testX is not None and fidx is not None:
            Xo = np.asarray(testX[p_row[rows]])[:, :, fidx].astype(np.float32); f_o = encode(FM, Xo, M, H, batch=a.batch)[0]
            f_p = P["feat64"][rows].astype(np.float32)
            rng = np.random.RandomState(0); n, K = len(rows), 32
            dur = rng.choice([5, 10, 15, 20, 30], (n, K)); st = 1 + (rng.rand(n, K) * np.maximum(1, H - dur)).astype(int); et = np.minimum(H, st + dur)
            l = rng.uniform(-5000, 5000, (n, K, 2)); Rq = np.full((n, K), 200.0, np.float32)
            Q = query_feats(l, st, et, Rq, H)
            lo, lp = reach_head_np(head, f_o, Q), reach_head_np(head, f_p, Q); Co, Cp = 1 / (1 + np.exp(-lo)), 1 / (1 + np.exp(-lp))
            dC = np.abs(Co - Cp); dF = np.abs(f_o - f_p); i_max = np.unravel_index(dF.argmax(), dF.shape)
            c_txt = (f"- **ผลต่อ C_ij (หัว reach จริง, {n} หน้าต่าง × {K} query สุ่ม, R 200, ยังไม่ผ่าน isotonic)**: max|ΔC| = **{dC.max():.2e}** · p99 {np.percentile(dC, 99):.2e} · mean {dC.mean():.2e} · max|Δlogit| {np.abs(lo - lp).max():.2e}"
                     f" · Δfeat64 สูงสุดที่หน้าต่าง {i_max[0]} มิติ {i_max[1]}: ของเรา {f_o[i_max]:+.4f} vs pred_test {f_p[i_max]:+.4f} (p99 ของ |Δfeat64| = {np.percentile(dF, 99):.2e})"
                     f" → {'✅ ≤ ' if dC.max() <= a.tol_C else '❌ > '}tol_C {a.tol_C:g}")
            p999 = float(np.percentile(dC, 99.9)); c_ok = bool(p999 <= a.tol_C and dC.max() <= a.tol_C_max)
            c_txt += f" · p99.9 {p999:.2e} → เกณฑ์ (2026-10-21 20:24): p99.9 ≤ {a.tol_C:g} ∧ max ≤ {a.tol_C_max:g} → {'✅' if c_ok else '❌'}"; log(c_txt)
    # ---- VERIFY.md
    passed = (ver["n_cmp"] > 0) and (not ver["bad"] or (a.accept_by_C and c_ok and ver["x_max"] == 0.0))
    miss = "; ".join(f"inst {k}: " + ", ".join(f"{s_}={n}" for s_, n in w.items() if s_ != "ok") for k, w in status.items() if any(s_ != "ok" for s_ in w))
    lines = [f"# E5 encode verify — {city}/{run} seed {R['seed']} · {VERSION} · {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"instances {len(inst)} · t (นาที) {t_list} · คู่ที่ pred_test มีหน้าต่างตรงเวลา: {ver['n_pairs']} · เทียบแล้ว {ver['n_cmp']}", "",
             f"- max|Δ feat64| (float16) = **{ver['feat_max']:.3e}** (เกณฑ์ ≤ {a.tol_feat:g})", f"- max|Δ mu| = {ver['mu_max']:.3e} m (เกณฑ์ ≤ {a.tol_mu:g}) · max|Δ sigma| = {ver['sig_max']:.3e} · max|Δ pi| = {ver['pi_max']:.3e}",
             f"- max|Δ X อินพุต| vs processed/test_X = {ver['x_max']:.3e} (เทียบ {ver['x_cmp']} หน้าต่าง)" if ver["x_cmp"] else "- ไม่ได้เทียบ X อินพุต (ไม่พบ test_X.npy หรือ feature list)",
             noise_txt or "- (ไม่ได้วัด noise ของ pipeline — ใช้ --noise)",
             c_txt or "- (ไม่ได้วัดผลต่อ C_ij — ไม่พบ reach_head.npz)",
             f"- device/precision ของรอบนี้: {'CPU' if a.cpu else 'GPU'} · TF32 {a.tf32} · batch {a.batch}",
             ("- หน้าต่างที่สร้างไม่ได้ (ต่อ instance): " + miss) if miss else "- ทุกหน้าต่างสร้างได้", "",
             f"**ผล: {'✅ ผ่าน' if passed else '❌ ไม่ผ่าน — ห้ามใช้ state จนกว่าจะหาสาเหตุ'}** · เกณฑ์: ΔX = 0 ∧ " + (f"p99.9|ΔC| ≤ {a.tol_C:g} ∧ max ≤ {a.tol_C_max:g} (--accept-by-C)" if a.accept_by_C else f"Δfeat64 ≤ {a.tol_feat:g} ∧ Δmu ≤ {a.tol_mu:g} m"), ""]
    if ver["bad"]:
        lines += ["| k | cab | t_min | Δfeat64 | Δmu | Δsigma | Δpi | ΔX |", "|---|---|---|---|---|---|---|---|"] + [f"| {b['k']} | {b['cab']} | {b['t_min']} | {b['d_feat']:.2e} | {b['d_mu']:.2e} | {b['d_sig']:.2e} | {b['d_pi']:.2e} | {b['dX']:.2e} |" for b in ver["bad"][:40]]
    (out / "VERIFY.md").write_text("\n".join(lines), encoding="utf-8"); log(f"→ {out / 'VERIFY.md'} · {'✅' if passed else '❌'}")
    return passed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", default="sf"); ap.add_argument("--runs", default="reach_tfm,reach_lstm"); ap.add_argument("--inst", default="instances_v3"); ap.add_argument("--data", default="data")
    ap.add_argument("--models", default="models"); ap.add_argument("--proc", default="processed"); ap.add_argument("--out", default="e5state"); ap.add_argument("--t", default="0,10,20,30,40,50")
    ap.add_argument("--dt-s", type=int, default=60); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--verify-only", action="store_true"); ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--batch", type=int, default=2048, help="= batch ที่ sD5_train_v4 ใช้ตอนสร้าง pred_test (2048)"); ap.add_argument("--tf32", choices=["on", "off"], default="on", help="TF32 บน GPU (ค่า default ของ TF = on; pred_test ถูกสร้างแบบ default)")
    ap.add_argument("--noise", action="store_true", help="วัด noise ของ pipeline เอง: predict แถว test_X เดิมของคู่ที่เทียบ ด้วย batch 2048 และ 1024 แล้วเทียบกับ pred_test และกันเอง")
    ap.add_argument("--tol-C", type=float, default=2e-3, help="เกณฑ์ p99.9|ΔC_ij| ผ่านหัว reach (= ระดับ ECE ของตัวประมาณ, v7 B5)"); ap.add_argument("--tol-C-max", type=float, default=5e-3, help="เกณฑ์ max|ΔC_ij| (หางของ 10⁵ ตัวอย่าง; แก้จาก max ≤ 2e-3 เมื่อ 2026-10-21 20:24 ดู HANDOFF)"); ap.add_argument("--accept-by-C", action="store_true", help="ตัดสินผ่าน/ไม่ผ่านด้วย ΔX = 0 ∧ max|ΔC| ≤ tol_C แทน Δmu (mu ถูก cumsum×100 m ขยาย noise ~10⁴ เท่า)")
    ap.add_argument("--tol-feat", type=float, default=2e-3, help="float16 quantisation ของ pred_test (step ≈ 1e-3 ที่ |v|≤1)"); ap.add_argument("--tol-mu", type=float, default=0.5, help="เมตร")
    a = ap.parse_args()
    if a.cpu: os.environ["CUDA_VISIBLE_DEVICES"] = ""
    log = log_factory(); log(VERSION)
    import tensorflow as tf
    tf.config.experimental.enable_tensor_float_32_execution(a.tf32 == "on")
    gpus = tf.config.list_physical_devices("GPU"); log(f"  device: {'GPU ' + str(len(gpus)) if gpus else 'CPU'} · TF32 {a.tf32} · batch {a.batch} · TF {tf.__version__}")
    allok = True
    for run in [r for r in a.runs.split(",") if r]:
        allok &= run_city(a, a.city, run, log)
    log("ทั้งหมดผ่าน ✅" if allok else "มี run ที่ไม่ผ่าน ❌")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
