#!/usr/bin/env python3
"""
sD1_D4_prepare_v2.py — ฟีเจอร์เพิ่มสำหรับ Bi-LSTM+ (M1 occ_dt · M2 stop-event) + D4 ใหม่ลง data/<city>/processed_<tag>/
ไม่แก้ sD1_D4_prepare.py · ไม่แตะ data/<city>/processed/ และ scaler.json เดิม

หลักการ: ใช้ interim/d3_features ของ base (D1–D3 ชุดเดิม) → เพิ่มคอลัมน์ฟีเจอร์ → make_windows เดิม (M_in/H/stride เดิม)
          → หน้าต่างแถวต่อแถวตรงกับ processed/ เดิม (ตรวจ cab/t_T ให้เหมือนกัน) → D6/D7 v7 อ่าน Y/meta จาก processed/ เดิมได้ตามปกติ
          ถ้ายังไม่มี interim/d3_features จะรัน D1–D3 ของ sD1_D4_prepare เดิมให้ก่อน (ต้องหาโฟลเดอร์ raw เจอ)

ฟีเจอร์ใหม่ (ทั้งหมด causal — ใช้เฉพาะก้าวก่อนหน้าถึงก้าวปัจจุบัน; z-score จาก train)
  occ_dt        = min(clip, นาทีนับจากครั้งล่าสุดที่ occ เปลี่ยนค่า)                 [SF; config: occ_dt_clip_min]
  t_since_stop  = min(clip, นาทีนับจากสิ้นสุด stop ล่าสุดใน segment)  (ยังไม่เคยมี stop → clip)
  last_stop_dur = min(clip, ความยาว stop ล่าสุด นาที)                 (ยังไม่เคยมี → 0)
  n_stops_30    = จำนวน stop ที่สิ้นสุดใน 30 นาทีที่ผ่านมา
  moving_streak = min(clip, นาทีที่ v ≥ v_stop ติดต่อกันจนถึงก้าวนี้)
  stop = ช่วง v < stop_v_kmh ติดต่อกัน ≥ stop_min_s (บนกริด Δτ = ceil(stop_min_s/Δτ) ก้าว); จอดยาว (พักกะ) ถูกตัดไปแล้วใน D2

CLI:  python sD1_D4_prepare_v2.py --city sf   --config config/data_sf_occ.yaml
      python sD1_D4_prepare_v2.py --city sf   --config config/data_sf_stop.yaml
      python sD1_D4_prepare_v2.py --city rome --config config/data_rome_stop.yaml
      --demo  = ทดสอบกับ d3_features สังเคราะห์ (ไม่ต้องมีข้อมูลจริง)
ผลลัพธ์:  data/<city>/<processed_dir>/{train,val,test}_X.npy · _Y.npy · _meta · test_cycle_paths.npy · scaler.json
          data/<city>/reports/D4_split_<tag>.csv · D3_features_<tag>.csv · D3_stop_proxy_<tag>.csv (SF) · D1_D4_report_<tag>.md · manifest_<tag>.json
"""
from __future__ import annotations
import argparse, json, os, sys, time, hashlib, math
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sD1_D4_prepare as P1  # noqa: E402  (ฟังก์ชันเดิม: load_config, assign_split, make_windows, apply_scaler, save_table, d1–d3)

NO_SCALE = {"occ", "dow_sin", "dow_cos", "slot_sin", "slot_cos"}
STOP_FEATS = ["t_since_stop", "last_stop_dur", "n_stops_30", "moving_streak"]

# ----------------------------------------------------------------------------- config (base: + overrides)
def load_config_v2(path: str, city: str) -> dict:
    """config ของ variant: ถ้ามี `base: data_<city>.yaml` จะอ่าน base ก่อนแล้วทับด้วยค่าในไฟล์นี้ · ค่า default เหมือน sD1_D4_prepare.load_config"""
    import yaml
    p = Path(path)
    cfg = yaml.safe_load(open(p, encoding="utf-8"))
    if "base" in cfg:
        base = yaml.safe_load(open(p.parent / cfg["base"], encoding="utf-8"))
        if "profile" in base:
            prof = base[base["profile"]]; base = {k: v for k, v in base.items() if k not in ("hf", "full")}; base.update(prof)
        over = {k: v for k, v in cfg.items() if k != "base"}
        base.update(over); base["_base"] = cfg["base"]; cfg = base
    cfg.setdefault("features", ["x", "y", "v", "dx", "dy", "dow_sin", "dow_cos", "slot_sin", "slot_cos", "occ"])
    cfg.setdefault("speed_filter_iters", 5); cfg.setdefault("parked_radius_m", 30); cfg.setdefault("parked_min_minutes", 30)
    cfg.setdefault("min_days_per_cab", 3); cfg.setdefault("min_points_per_cab", 1000)
    cfg.setdefault("train_stride", 5); cfg.setdefault("val_stride", 5); cfg.setdefault("test_stride", 5)
    cfg.setdefault("processed_dir", "processed"); cfg.setdefault("variant", Path(cfg["processed_dir"]).name.replace("processed_", ""))
    cfg["city"] = city
    return cfg

def merged_config_path(path: str) -> str:
    """ถ้า config มี base: → เขียนไฟล์รวม config/merged_<name>.yaml (ไม่มี base:) แล้วคืน path นั้น
    ใช้บันทึกลง run_config.json ให้ sD6/sD7 v7 (ซึ่งอ่าน yaml ตรง ๆ ไม่รู้จัก base:) เห็น dates/dt_s/clean_gap_mult ครบ · ไม่มี base: → คืน path เดิม"""
    import yaml
    p = Path(path); raw = yaml.safe_load(open(p, encoding="utf-8"))
    if "base" not in raw: return str(path)
    city = raw.get("city") or yaml.safe_load(open(p.parent / raw["base"], encoding="utf-8")).get("city", "")
    cfg = {k: v for k, v in load_config_v2(str(p), city).items() if k != "_base"}
    out = p.parent / f"merged_{p.name}"
    out.write_text("# สร้างอัตโนมัติจาก " + p.name + " + " + raw["base"] + " — อย่าแก้ไฟล์นี้ แก้ต้นฉบับ\n" + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return str(out)

# ----------------------------------------------------------------------------- M1 occ_dt
def add_occ_dt(rs: pd.DataFrame, clip_min: float = 60.0) -> pd.DataFrame:
    """นาทีนับจากครั้งล่าสุดที่ occ เปลี่ยนค่า (ต่อคัน เรียงตามเวลา ข้าม segment ได้เพราะเป็นเวลาจริง) · ก่อนเห็นการเปลี่ยนครั้งแรก = clip"""
    cab = rs["cab"].values; t = rs["t"].values.astype(np.int64); occ = rs["occ"].values
    new_cab = np.r_[True, cab[1:] != cab[:-1]]
    chg = np.r_[False, occ[1:] != occ[:-1]] & ~new_cab
    t_chg = np.where(chg, t, -1).astype(np.int64)
    # forward-fill เวลาเปลี่ยนล่าสุดภายในคัน
    last = pd.Series(np.where(chg, t, np.nan)).groupby(np.cumsum(new_cab)).ffill().values
    dt_min = np.where(np.isnan(last), clip_min, (t - np.nan_to_num(last, nan=0)) / 60.0)
    rs["occ_dt"] = np.minimum(dt_min, clip_min).astype(np.float32)
    return rs

# ----------------------------------------------------------------------------- M2 stop-event
def stop_events_segment(slow: np.ndarray, k_min: int):
    """คืน (starts, ends) ของ run ที่ slow ติดกัน ≥ k_min ก้าว (index ภายใน segment, end รวม)"""
    if not slow.any(): return np.zeros(0, int), np.zeros(0, int)
    d = np.diff(np.r_[0, slow.astype(np.int8), 0])
    s = np.flatnonzero(d == 1); e = np.flatnonzero(d == -1) - 1
    ok = (e - s + 1) >= k_min
    return s[ok], e[ok]

def add_stop_features(rs: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    dt_s = int(cfg["dt_s"]); dt_min = dt_s / 60.0
    v_stop = float(cfg.get("stop_v_kmh", 2.0)) / 3.6                       # m/s
    k_min = max(1, int(math.ceil(float(cfg.get("stop_min_s", 90)) / dt_s)))
    c_since, c_dur = float(cfg.get("stop_since_clip_min", 60)), float(cfg.get("stop_dur_clip_min", 30))
    w_cnt = int(round(float(cfg.get("stop_count_window_min", 30)) / dt_min)); c_streak = float(cfg.get("stop_streak_clip_min", 60))
    n = len(rs); seg = rs["seg"].values; v = rs["v"].values
    t_since = np.full(n, c_since, np.float32); last_dur = np.zeros(n, np.float32); n30 = np.zeros(n, np.float32); streak = np.zeros(n, np.float32)
    starts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]]); ends = np.r_[starts[1:], n]
    n_stop, stop_len = 0, []
    for s0, s1 in zip(starts, ends):
        slow = v[s0:s1] < v_stop; m = s1 - s0; i = np.arange(m)
        # moving streak: run-length ของ ~slow จนถึงก้าว i
        mv = (~slow).astype(np.int64)
        reset = np.maximum.accumulate(np.where(mv == 0, i + 1, 0))   # ตำแหน่งหลัง slow ล่าสุด
        run = i + 1 - reset                                                              # จำนวนก้าว moving ติดกันถึง i
        streak[s0:s1] = np.minimum(run * dt_min, c_streak)
        ss, ee = stop_events_segment(slow, k_min)
        if len(ee) == 0: continue
        n_stop += len(ee); stop_len.extend(((ee - ss + 1) * dt_min).tolist())
        durs = (ee - ss + 1) * dt_min
        j = np.searchsorted(ee, i, side="left") - 1          # stop ล่าสุดที่สิ้นสุด "ก่อน" ก้าว i (ee < i)
        has = j >= 0
        t_since[s0:s1] = np.where(has, np.minimum((i - ee[np.clip(j, 0, None)]) * dt_min, c_since), c_since)
        last_dur[s0:s1] = np.where(has, np.minimum(durs[np.clip(j, 0, None)], c_dur), 0.0)
        n30[s0:s1] = np.searchsorted(ee, i, side="left") - np.searchsorted(ee, i - w_cnt, side="left")   # ends ใน [i−w, i−1]
    for name, arr in zip(STOP_FEATS, (t_since, last_dur, n30, streak)): rs[name] = arr
    stat = {"stop_v_kmh": v_stop * 3.6, "stop_min_steps": k_min, "n_stops": int(n_stop), "stops_per_hour": float(n_stop / (n * dt_s / 3600)),
            "stop_len_min_p50": float(np.median(stop_len)) if stop_len else float("nan"), "stop_len_min_p90": float(np.percentile(stop_len, 90)) if stop_len else float("nan"),
            "pct_steps_slow": float((v < v_stop).mean() * 100)}
    return rs, stat

def stop_proxy_check(rs: pd.DataFrame, cfg: dict) -> dict:
    """SF: stop-event ตรงกับ occ-transition แค่ไหน (±tol นาที) → recall/precision ลง limitation ของ paper"""
    dt_s = int(cfg["dt_s"]); tol = int(round(float(cfg.get("stop_proxy_tol_min", 2)) * 60 / dt_s))
    v_stop = float(cfg.get("stop_v_kmh", 2.0)) / 3.6; k_min = max(1, int(math.ceil(float(cfg.get("stop_min_s", 90)) / dt_s)))
    seg = rs["seg"].values; v = rs["v"].values; occ = rs["occ"].values; n = len(rs)
    starts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]]); ends = np.r_[starts[1:], n]
    n_tr = n_tr_hit = n_st = n_st_hit = 0
    for s0, s1 in zip(starts, ends):
        o = occ[s0:s1]; tr = np.flatnonzero(np.r_[False, o[1:] != o[:-1]])
        ss, ee = stop_events_segment(v[s0:s1] < v_stop, k_min)
        n_tr += len(tr); n_st += len(ee)
        if len(tr) and len(ee):
            # transition ชน stop ถ้าอยู่ใน [s−tol, e+tol] ของ stop ใด ๆ
            j = np.searchsorted(ss, tr, side="right") - 1
            hit_prev = (j >= 0) & (tr <= ee[np.clip(j, 0, None)] + tol)
            jn = np.clip(j + 1, 0, len(ss) - 1); hit_next = tr >= ss[jn] - tol
            n_tr_hit += int((hit_prev | hit_next).sum())
            # stop ชน transition ถ้ามี transition ใน [s−tol, e+tol]
            lo = np.searchsorted(tr, ss - tol, side="left"); hi = np.searchsorted(tr, ee + tol, side="right")
            n_st_hit += int((hi > lo).sum())
    return {"tol_min": tol * dt_s / 60, "occ_transitions": n_tr, "stop_events": n_st,
            "recall_transition_has_stop": n_tr_hit / n_tr if n_tr else float("nan"),
            "precision_stop_has_transition": n_st_hit / n_st if n_st else float("nan")}


# ----------------------------------------------------------------------------- M5 fleet transition prior
def build_prior(rs: pd.DataFrame, cfg: dict, log) -> dict:
    """prior จาก train เท่านั้น: ต่อ (cell ตอนนี้, slot) → histogram ของ "การกระจัด" ที่ +H_p นาที (bin ขนาด dcell_m, ±k bin) → shrink → PCA dim
       คืน table สำหรับ lookup ตอนสร้างฟีเจอร์ต่อหน้าต่าง (ไม่รั่ว: ใช้เฉพาะก้าว split==train)"""
    pc = cfg["prior"]; cell = float(pc.get("cell_m", 500)); dcell = float(pc.get("dcell_m", 1000)); k = int(pc.get("k", 5))
    hp = int(round(float(pc.get("horizon_min", 60)) * 60 / int(cfg["dt_s"]))); dim = int(pc.get("dim", 32)); m_shr = float(pc.get("shrink", 20))
    nb = 2 * k + 1
    tr = np.flatnonzero(rs["split"].values == "train")
    seg = rs["seg"].values; x = rs["x"].values; y = rs["y"].values
    i = tr[tr + hp < len(rs)]; i = i[seg[i + hp] == seg[i]]                     # ก้าวที่ +hp ยังอยู่ segment เดิม
    dx = x[i + hp] - x[i]; dy = y[i + hp] - y[i]
    bx = np.clip(np.floor(dx / dcell).astype(int) + k, 0, nb - 1); by = np.clip(np.floor(dy / dcell).astype(int) + k, 0, nb - 1)
    dbin = bx * nb + by
    cx = np.floor(x[i] / cell).astype(np.int64); cy = np.floor(y[i] / cell).astype(np.int64)
    wk = (pd.to_datetime(rs["date"].values[i]).dayofweek >= 5).astype(np.int64); slot = rs["hour"].values[i].astype(np.int64) * 2 + wk
    key = pd.DataFrame({"cx": cx, "cy": cy, "slot": slot, "dbin": dbin})
    cnt = key.groupby(["cx", "cy", "slot", "dbin"]).size().unstack("dbin", fill_value=0).reindex(columns=range(nb * nb), fill_value=0)
    n_row = cnt.sum(1).values[:, None]
    slot_cnt = key.groupby(["slot", "dbin"]).size().unstack("dbin", fill_value=0).reindex(columns=range(nb * nb), fill_value=0)
    slot_P = (slot_cnt.values / np.maximum(slot_cnt.sum(1).values[:, None], 1)).astype(np.float64)   # (48, nb²) ไม่ครบทุก slot ได้
    slot_index = {s_: j for j, s_ in enumerate(slot_cnt.index)}
    glob_P = key.groupby("dbin").size().reindex(range(nb * nb), fill_value=0).values.astype(np.float64); glob_P /= glob_P.sum()
    P_slot_rows = np.stack([slot_P[slot_index[s_]] if s_ in slot_index else glob_P for s_ in cnt.index.get_level_values("slot")])
    P = (cnt.values + m_shr * P_slot_rows) / (n_row + m_shr)                                           # shrink ไปหา slot-global
    mean = P.mean(0); U, S, Vt = np.linalg.svd(P - mean, full_matrices=False)
    comps = Vt[:dim]; expl = float((S[:dim] ** 2).sum() / (S ** 2).sum())
    Z = (P - mean) @ comps.T
    log(f"  prior: train steps {len(i):,} · (cell,slot) {len(cnt):,} rows · bins {nb}×{nb} ({dcell:.0f} m) · PCA {dim} อธิบาย {expl*100:.1f}% · shrink m={m_shr:g}")
    return {"cell_m": cell, "dcell_m": dcell, "k": k, "dim": dim, "horizon_min": pc.get("horizon_min", 60),
            "keys": np.stack([cnt.index.get_level_values("cx"), cnt.index.get_level_values("cy"), cnt.index.get_level_values("slot")], 1).astype(np.int64),
            "Z": Z.astype(np.float32), "mean": mean.astype(np.float32), "comps": comps.astype(np.float32),
            "slot_keys": np.array(list(slot_index.keys()), dtype=np.int64), "slot_Z": ((slot_P - mean) @ comps.T).astype(np.float32),
            "glob_Z": ((glob_P - mean) @ comps.T).astype(np.float32), "explained": expl, "n_train_steps": int(len(i))}

def prior_features(meta: pd.DataFrame, tab: dict, cfg: dict) -> np.ndarray:
    """ฟีเจอร์ต่อหน้าต่าง = Z[cell(x_T,y_T), slot(t_T)] · ไม่เคยเห็น cell → slot-global · ไม่เคยเห็น slot → global"""
    if len(meta) == 0: return np.zeros((0, tab["dim"]), np.float32)
    lat0, lon0 = cfg["origin"]; d = pd.DataFrame({"lat": meta["lat_T"].values, "lon": meta["lon_T"].values}); P1.project(d, lat0, lon0)
    cx = np.floor(d["x"].values / tab["cell_m"]).astype(np.int64); cy = np.floor(d["y"].values / tab["cell_m"]).astype(np.int64)
    wk = (pd.to_datetime(meta["date"].astype(str).values).dayofweek >= 5).astype(np.int64); slot = meta["hour"].values.astype(np.int64) * 2 + wk
    idx = {tuple(r): j for j, r in enumerate(map(tuple, tab["keys"]))}; sidx = {int(s_): j for j, s_ in enumerate(tab["slot_keys"])}
    out = np.empty((len(meta), tab["dim"]), np.float32); n_hit = 0
    for r in range(len(meta)):
        j = idx.get((int(cx[r]), int(cy[r]), int(slot[r])))
        if j is not None: out[r] = tab["Z"][j]; n_hit += 1
        else:
            js = sidx.get(int(slot[r])); out[r] = tab["slot_Z"][js] if js is not None else tab["glob_Z"]
    out_stat = n_hit / max(1, len(meta))
    prior_features.last_hit_rate = out_stat
    return out
prior_features.last_hit_rate = float("nan")

# ----------------------------------------------------------------------------- scaler (z-score ทุกฟีเจอร์ต่อเนื่อง)
def fit_scaler_v2(X_train: np.ndarray, F: list) -> dict:
    cols = [i for i, f in enumerate(F) if f not in NO_SCALE]
    flat = X_train.reshape(-1, X_train.shape[-1])
    return {"cols": cols, "names": [F[i] for i in cols],
            "mean": [float(flat[:, i].mean()) for i in cols], "std": [float(flat[:, i].std() + 1e-9) for i in cols]}

# ----------------------------------------------------------------------------- demo data
def make_demo_d3(n_cabs=6, n_seg=16, seg_len=300, dt_s=60, seed=0, dates=("2008-05-17", "2008-06-08")) -> pd.DataFrame:
    rng = np.random.RandomState(seed); rows = []; seg_id = 0
    t0 = int(pd.Timestamp(dates[0], tz="UTC").timestamp()); t1 = int(pd.Timestamp(dates[1], tz="UTC").timestamp())
    for c in range(n_cabs):
        t = t0 + 3600 * 12
        for _ in range(n_seg):
            n = seg_len; v = np.where(rng.rand(n) < 0.15, rng.uniform(0, 0.4, n), rng.uniform(2, 15, n))
            # ทำให้ช้าติดกันเป็นช่วง
            for k in range(0, n, 50):
                if rng.rand() < 0.5: v[k:k + rng.randint(1, 5)] = 0.1
            ang = np.cumsum(rng.normal(0, 0.2, n)); dx = (v * dt_s * np.cos(ang)).astype(np.float32); dy = (v * dt_s * np.sin(ang)).astype(np.float32)
            x = np.cumsum(dx) + rng.uniform(-5000, 5000); y = np.cumsum(dy) + rng.uniform(-5000, 5000)
            occ = np.zeros(n, np.int8); p = 0
            while p < n: L = rng.randint(5, 40); occ[p:p + L] = rng.randint(0, 2); p += L
            tt = t + np.arange(n) * dt_s
            loc = pd.to_datetime(tt, unit="s", utc=True)
            rows.append(pd.DataFrame({"cab": f"cab{c}", "seg": seg_id, "t": tt, "lat": 37.78 + y / 111000, "lon": -122.41 + x / 88000, "occ": occ.astype(np.float32),
                                      "gap_s": 60.0, "parked": False, "x": x, "y": y, "dx": dx, "dy": dy, "v": v.astype(np.float32),
                                      "dow_sin": np.sin(2 * np.pi * loc.dayofweek / 7), "dow_cos": np.cos(2 * np.pi * loc.dayofweek / 7),
                                      "slot_sin": np.sin(2 * np.pi * (loc.hour * 60 + loc.minute) / 1440), "slot_cos": np.cos(2 * np.pi * (loc.hour * 60 + loc.minute) / 1440),
                                      "date": loc.strftime("%Y-%m-%d"), "hour": loc.hour.astype(np.int8), "minute": loc.minute.astype(np.int8)}))
            seg_id += 1; t = tt[-1] + rng.randint(600, 7200) + (t1 - t0) // n_seg - seg_len * dt_s
    return pd.concat(rows, ignore_index=True)

# ----------------------------------------------------------------------------- main
def run(city: str, cfg_path: str, raw, n_cabs, out_root: str, demo: bool = False):
    t_start = time.time(); log = lambda *m: print(f"[{time.time()-t_start:6.1f}s]", *m, flush=True)
    cfg = load_config_v2(cfg_path, city)
    tag = cfg["variant"]; proc_name = cfg["processed_dir"]; F = list(cfg["features"])
    if proc_name == "processed": sys.exit("processed_dir ต้องไม่ใช่ 'processed' (ห้ามเขียนทับของเดิม) — ตั้งใน config เช่น processed_f11")
    out = Path(out_root) / city
    for sub in (proc_name, "reports", "interim"): (out / sub).mkdir(parents=True, exist_ok=True)
    log(f"config {cfg_path} (base {cfg.get('_base', '-')}) · variant={tag} · processed_dir={proc_name} · ฟีเจอร์ {len(F)}: {F}")

    # ---- d3_features ของ base (หรือสร้าง)
    d3_file = next((f for f in (out / "interim" / "d3_features.parquet", out / "interim" / "d3_features.pkl") if f.exists()), None)
    if demo:
        rs = make_demo_d3(dates=(str(cfg["dates"]["train"][0]), str(cfg["dates"]["test"][1]))); log(f"DEMO: d3_features สังเคราะห์ {len(rs):,} ก้าว")
    elif d3_file:
        log(f"อ่าน {d3_file} (D1–D3 ของ base)")
        rs = pd.read_parquet(d3_file) if d3_file.suffix == ".parquet" else pd.read_pickle(d3_file)
    else:
        log("ไม่พบ interim/d3_features → รัน D1–D3 ของ sD1_D4_prepare เดิม")
        if city == "sf": df = P1.load_sf(raw, n_cabs)
        elif city == "rome": df = P1.load_rome(P1._rome_file(raw), n_cabs)
        else: df = P1.load_tdrive(raw, n_cabs)
        df, t_clean = P1.d1_clean(df, cfg, log); t_clean.to_csv(out / "reports" / "D1_cleaning.csv", index=False)
        rs, t_seg = P1.d2_segment_resample(df, cfg, log); del df; t_seg.to_csv(out / "reports" / "D2_segments.csv", index=False)
        rs = P1.d3_features(rs, cfg, log); P1.save_table(rs, out / "interim" / "d3_features.parquet")
    rs = rs.sort_values(["cab", "t"], kind="stable").reset_index(drop=True)   # ลำดับเดียวกับที่ D1–D3 สร้าง (ปกติเรียงอยู่แล้ว)

    # ---- ฟีเจอร์ใหม่
    feat_stats = {}
    if "occ_dt" in F:
        if not cfg.get("occ_feature", True): sys.exit("occ_dt ต้องการ occ จริง — เมืองนี้ occ_feature: false")
        rs = add_occ_dt(rs, float(cfg.get("occ_dt_clip_min", 60))); log(f"  occ_dt: mean {rs['occ_dt'].mean():.1f} นาที · % clip {(rs['occ_dt'] >= float(cfg.get('occ_dt_clip_min', 60))).mean()*100:.1f}")
    if any(f in F for f in STOP_FEATS):
        rs, st = add_stop_features(rs, cfg); feat_stats.update({f"stop_{k}": v for k, v in st.items()})
        log(f"  stop-event: v<{st['stop_v_kmh']:.1f} km/h ≥{st['stop_min_steps']} ก้าว → {st['n_stops']:,} stops ({st['stops_per_hour']:.2f}/ชม.) · ยาว p50 {st['stop_len_min_p50']:.0f} นาที · ก้าวช้า {st['pct_steps_slow']:.1f}%")
        if cfg.get("occ_feature", True) and "occ" in rs and rs["occ"].nunique() > 1:
            px = stop_proxy_check(rs, cfg); feat_stats.update({f"proxy_{k}": v for k, v in px.items()})
            pd.DataFrame([px]).to_csv(out / "reports" / f"D3_stop_proxy_{tag}.csv", index=False)
            log(f"  proxy (±{px['tol_min']:.0f} นาที): occ-transition {px['occ_transitions']:,} · stop {px['stop_events']:,} · recall {px['recall_transition_has_stop']:.3f} · precision {px['precision_stop_has_transition']:.3f}")
    missing = [f for f in F if f not in rs.columns]
    if missing: sys.exit(f"ไม่มีคอลัมน์ฟีเจอร์ {missing} (รองรับ: occ_dt, {STOP_FEATS})")
    desc = rs[[f for f in F if f not in NO_SCALE]].describe().T[["mean", "std", "min", "50%", "max"]]
    desc.to_csv(out / "reports" / f"D3_features_{tag}.csv")

    # ---- D4 (ฟังก์ชันเดิม) → processed_<tag>/
    rs["split"] = P1.assign_split(rs, cfg)
    wins, paths = P1.make_windows(rs, cfg, log)
    sc = fit_scaler_v2(wins["train"][0], F); json.dump(sc, open(out / proc_name / "scaler.json", "w"), indent=2)
    rows = []; align = {}
    for sp, (X, Y, meta) in wins.items():
        np.save(out / proc_name / f"{sp}_X.npy", P1.apply_scaler(X, sc)); np.save(out / proc_name / f"{sp}_Y.npy", Y)
        P1.save_table(meta, out / proc_name / f"{sp}_meta.parquet")
        a, b = cfg["dates"][sp]
        rows.append({"split": sp, "date_from": str(a), "date_to": str(b), "cabs": int(meta["cab"].nunique()) if len(meta) else 0,
                     "segments": int(meta["seg"].nunique()) if len(meta) else 0, "windows": len(meta), "cycle_windows": int(meta["is_cycle_start"].sum()) if len(meta) else 0})
        # ตรวจว่าแถวตรงกับ processed/ เดิม (D6/D7 v7 อ่าน Y/meta จากที่นั่น)
        base_meta = next((f for f in (out / "processed" / f"{sp}_meta.parquet", out / "processed" / f"{sp}_meta.pkl") if f.exists()), None)
        if base_meta is not None:
            bm = pd.read_parquet(base_meta) if base_meta.suffix == ".parquet" else pd.read_pickle(base_meta)
            same = len(bm) == len(meta) and (bm["cab"].astype(str).values == meta["cab"].astype(str).values).all() and (bm["t_T"].values == meta["t_T"].values).all()
            align[sp] = bool(same)
    np.save(out / proc_name / "test_cycle_paths.npy", paths)
    if cfg.get("prior"):
        tab = build_prior(rs, cfg, log)
        np.savez_compressed(out / proc_name / "prior_table.npz", **{k_: v_ for k_, v_ in tab.items() if isinstance(v_, np.ndarray)},
                            meta=json.dumps({k_: v_ for k_, v_ in tab.items() if not isinstance(v_, np.ndarray)}))
        ztr = prior_features(wins["train"][2], tab, cfg); mu_, sd_ = ztr.mean(0), ztr.std(0) + 1e-6
        for sp, (X, Y, meta) in wins.items():
            z = prior_features(meta, tab, cfg); np.save(out / proc_name / f"{sp}_prior.npy", ((z - mu_) / sd_).astype(np.float32))
            log(f"  prior {sp:<5} ({len(meta):,} × {tab['dim']}) · cell hit {prior_features.last_hit_rate*100:.1f}%"); feat_stats[f"prior_cell_hit_{sp}"] = prior_features.last_hit_rate
        feat_stats.update({"prior_explained": tab["explained"], "prior_rows": int(len(tab["keys"])), "prior_train_steps": tab["n_train_steps"]})
    t_split = pd.DataFrame(rows); t_split.to_csv(out / "reports" / f"D4_split_{tag}.csv", index=False)
    ok = all(align.values()) if align else False
    if align and ok:
        log("  แถวหน้าต่างตรงกับ processed/ เดิมทุก split ✓ (D6/D7 v7 ใช้ data/ เดิมได้)")
    else:
        # หน้าต่างต่างจาก base (เช่น M_in 30) → ทำ data root สำรอง data_<tag>/<city>/{processed → processed_<tag>, scaler.json} ให้ D6/D7 v7 ใช้ --data data_<tag>
        mroot = Path(out_root).parent / f"{Path(out_root).name}_{tag}" / city; mroot.mkdir(parents=True, exist_ok=True)
        for name, target in (("processed", out / proc_name), ("scaler.json", out / proc_name / "scaler.json")):
            lnk = mroot / name
            if lnk.is_symlink() or lnk.exists(): lnk.unlink()
            try: lnk.symlink_to(os.path.relpath(target.resolve(), lnk.parent), target_is_directory=(name == "processed"))
            except OSError:
                import shutil; shutil.copytree(target, lnk) if name == "processed" else shutil.copy(target, lnk)
        msg = f"ไม่ตรงกับ processed/ เดิม {align}" if align else "ไม่มี processed/ เดิมให้เทียบ"
        log(f"  หน้าต่าง{msg} → สร้าง {mroot.parent.name}/{city}/ (symlink) · D6/D7 v7 ใช้ --data {mroot.parent.name}")
        manifest_extra = {"data_root_for_D6_D7": mroot.parent.name}
    manifest_extra = locals().get("manifest_extra", {})

    md = [f"# D1–D4 v2 — {city} · variant `{tag}`", "", f"config: `{cfg_path}` · processed_dir: `{proc_name}` · ฟีเจอร์ {len(F)}: {F}", "",
          "## ฟีเจอร์ต่อเนื่อง (ก่อน z-score, ทุก split)", P1.md_table(desc.reset_index().rename(columns={"index": "feature"}).round(3)), "",
          "## สถิติฟีเจอร์ใหม่", P1.md_table(pd.DataFrame([feat_stats]).T.reset_index().rename(columns={"index": "key", 0: "value"}).round(4)) if feat_stats else "—", "",
          "## D4 split", P1.md_table(t_split), "",
          f"alignment กับ processed/ เดิม: {align if align else 'n/a'} · scaler cols: {sc['names']}"]
    (out / "reports" / f"D1_D4_report_{tag}.md").write_text("\n".join(md), encoding="utf-8")
    manifest = {"city": city, "variant": tag, "config": cfg_path, "config_sha1": hashlib.sha1(open(cfg_path, "rb").read()).hexdigest(),
                "processed_dir": proc_name, "features": F, "d3_source": str(d3_file) if d3_file else ("demo" if demo else "rebuilt"),
                "align_with_processed": align, "feat_stats": feat_stats, "created": time.strftime("%Y-%m-%d %H:%M:%S"), **manifest_extra}
    json.dump(manifest, open(out / f"manifest_{tag}.json", "w"), indent=2, ensure_ascii=False)
    log(f"เสร็จ → {out}/{proc_name}/ · {out}/reports/D1_D4_report_{tag}.md")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", required=True, choices=["sf", "rome", "beijing"])
    ap.add_argument("--config", required=True, help="config/data_<city>_<variant>.yaml (มี base: และ processed_dir:)")
    ap.add_argument("--raw", default=None, help="ใช้เมื่อยังไม่มี interim/d3_features"); ap.add_argument("--n-cabs", type=int, default=None)
    ap.add_argument("--out", default=str(HERE / "data")); ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    raw = a.raw
    if not a.demo and raw is None:
        folder = {"sf": "DataTaxiSanFran", "rome": "DataTaxiRome", "beijing": "DataTaxiBejin"}[a.city]
        raw = P1._find_folder(folder)
    run(a.city, a.config, raw, a.n_cabs, a.out, demo=a.demo)

if __name__ == "__main__":
    main()
