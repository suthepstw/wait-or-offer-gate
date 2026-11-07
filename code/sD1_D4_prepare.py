#!/usr/bin/env python3
"""
sD1_D4_prepare.py — ขั้น D1–D4: clean → segment/resample → ฟีเจอร์หน่วยเมตร → split ตามเวลา + หน้าต่าง X/Y
อ่านค่าทั้งหมดจาก config/data_<city>.yaml (ที่เคาะจาก D0) และใช้ loader ชุดเดียวกับ sD0_audit.py (ต้องอยู่โฟลเดอร์เดียวกัน)

PyCharm: แก้บล็อก PYCHARM แล้วกด Run    ·    CLI: python sD1_D4_prepare.py --city sf [--config config/data_sf.yaml] [--n-cabs 20]

ผลลัพธ์ใน data/<city>/
  interim/d1_clean.parquet · interim/d2_resampled.parquet · interim/d3_features.parquet · scaler.json
  processed/{train,val,test}_X.npy (N×M_in×F) · _Y.npy (N×H×2 เมตรจาก x_T) · _meta.parquet
  processed/test_cycle_paths.npy (เส้นทางจริง lat/lon ของหน้าต่างต้น cycle สำหรับ replay)
  reports/D1_cleaning.csv · D2_segments.csv · D4_split.csv · D1_D4_report.md · manifest.json
ต้องการ: numpy pandas pyyaml   (+ pyarrow ถ้าต้องการ parquet; ไม่มีจะเก็บเป็น .pkl)
"""
from __future__ import annotations
import argparse, json, os, sys, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sD0_audit import load_sf, load_rome, load_tdrive, project, per_cab_steps, flag_parked, _find_folder, _rome_file, md_table, R_EARTH  # noqa: E402

# ============================================================================= PyCharm
PYCHARM = dict(
    CITY="beijing",                 # "sf" | "rome" | "beijing"
    CONFIG=r"",                # เว้นว่าง = config/data_<city>.yaml ข้างสคริปต์
    RAW=r"",                   # เว้นว่าง = หาโฟลเดอร์ DataTaxi* อัตโนมัติ (เหมือน sD0)
    N_CABS=None,               # ทดสอบ: 20 · รันจริง: None
    OUT="data",                # โฟลเดอร์ผลลัพธ์ (สร้าง data/<city>/ ข้างสคริปต์)
    RESUME=False,               # True = ถ้ามี interim/d3_features แล้ว ข้าม D1–D3 รันแค่ D4 (เร็ว) · False = ทำใหม่ทั้งหมด
)

# ----------------------------------------------------------------------------- config
def load_config(path: str, city: str) -> dict:
    try:
        import yaml
    except ImportError:
        sys.exit("ต้องติดตั้ง pyyaml: pip install pyyaml")
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    if "profile" in cfg:                          # beijing: hf / full
        prof = cfg[cfg["profile"]]
        base = {k: v for k, v in cfg.items() if k not in ("hf", "full")}
        base.update(prof); cfg = base
    cfg.setdefault("features", ["x", "y", "v", "dx", "dy", "dow_sin", "dow_cos", "slot_sin", "slot_cos", "occ"])
    cfg.setdefault("speed_filter_iters", 5); cfg.setdefault("parked_radius_m", 30); cfg.setdefault("parked_min_minutes", 30)
    cfg.setdefault("min_days_per_cab", 3); cfg.setdefault("min_points_per_cab", 1000)
    cfg.setdefault("train_stride", 5); cfg.setdefault("val_stride", 5); cfg.setdefault("test_stride", 5)
    cfg["city"] = city
    return cfg

def local_tz(cfg): return cfg["tz"]

def save_table(df: pd.DataFrame, path: Path) -> Path:
    """parquet ถ้ามี pyarrow/fastparquet ไม่งั้นเก็บเป็น .pkl (pandas pickle)"""
    try:
        df.to_parquet(path, index=False); return path
    except Exception:
        alt = path.with_suffix(".pkl"); df.to_pickle(alt); return alt

# ----------------------------------------------------------------------------- D1 clean
def d1_clean(df: pd.DataFrame, cfg: dict, log) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    def rec(step, d): rows.append({"step": step, "points": len(d), "cabs": d["cab"].nunique()}); log(f"  {step:<28} {len(d):>12,} จุด  {d['cab'].nunique():>5} คัน")
    rec("0 raw", df)
    # 1.1 sort + dedupe + NaN/0
    df = df.dropna(subset=["lat", "lon", "t"])
    df = df[(df["lat"] != 0) & (df["lon"] != 0)]
    df = df.drop_duplicates(["cab", "t"]).sort_values(["cab", "t"]).reset_index(drop=True)
    rec("1.1 sort/dedupe", df)
    # (beijing hf) per-cab sampling-rate filter
    cf = cfg.get("cab_filter")
    if cf and cf.get("median_interval_le_s"):
        med = df.groupby("cab")["t"].diff().groupby(df["cab"]).median()
        df = df[df["cab"].isin(med.index[med <= cf["median_interval_le_s"]])].reset_index(drop=True)
        rec(f"1.1b cab median dt ≤ {cf['median_interval_le_s']}s", df)
    # 1.2 bbox
    b = cfg["bbox"]
    df = df[df["lat"].between(*b["lat"]) & df["lon"].between(*b["lon"])].reset_index(drop=True)
    rec("1.2 bbox", df)
    # 1.3 iterative speed filter
    lat0, lon0 = cfg["origin"]
    project(df, lat0, lon0)
    for it in range(int(cfg["speed_filter_iters"])):
        df = per_cab_steps(df)
        bad = (df["v"] > cfg["v_max_kmh"]) & (df["dt"] >= cfg.get("speed_min_dt_s", 1))
        if bad.sum() == 0: break
        df = df[~bad].reset_index(drop=True)
        rec(f"1.3 speed>{cfg['v_max_kmh']} รอบ {it+1}", df)
    df = per_cab_steps(df)
    # 1.4 parked flag (ไม่ลบ)
    flag_parked(df, cfg["parked_radius_m"], cfg["parked_min_minutes"])
    rows.append({"step": "1.4 parked flagged (kept)", "points": int(df["parked"].sum()), "cabs": int(df.loc[df["parked"], "cab"].nunique())})
    # 1.5 cab filter
    g = df.groupby("cab")
    days = (g["t"].max() - g["t"].min()) / 86400
    npts = g.size(); med_dt = g["dt"].median()
    ok = (days >= cfg["min_days_per_cab"]) & (npts >= cfg["min_points_per_cab"]) & (med_dt <= 3 * cfg["dt_s"])
    df = df[df["cab"].isin(ok.index[ok])].reset_index(drop=True)
    rec("1.5 cab filter", df)
    return df, pd.DataFrame(rows)

# ----------------------------------------------------------------------------- D2 segment + resample
def d2_segment_resample(df: pd.DataFrame, cfg: dict, log) -> tuple[pd.DataFrame, pd.DataFrame]:
    dt_s, gap = int(cfg["dt_s"]), float(cfg["gap_max_s"])
    min_len = (int(cfg["M_in"]) + int(cfg["H"])) * dt_s
    # ตัด segment: คันเปลี่ยน / gap / ช่วง parked ที่ "ยาว" (พักกะ) — ช่วงจอดสั้นกว่านั้นเก็บไว้ในเส้นทาง (รถยังอยู่ตรงนั้นจริง)
    brk_min = float(cfg.get("parked_break_minutes", 180))
    parked = df["parked"].values
    run_id = np.cumsum(np.r_[True, (parked[1:] != parked[:-1]) | (df["cab"].values[1:] != df["cab"].values[:-1])]) - 1
    t = df["t"].values
    run_start = np.flatnonzero(np.r_[True, run_id[1:] != run_id[:-1]]); run_end = np.r_[run_start[1:], len(df)] - 1
    run_dur = (t[run_end] - t[run_start]) / 60.0
    long_park = parked & (run_dur[run_id] >= brk_min)         # จอดยาว ≥ parked_break_minutes → ตัดทิ้ง + ตัด segment
    n_long = int(long_park.sum())
    cab_chg = df["cab"].values[1:] != df["cab"].values[:-1]
    gap_brk = df["dt"].values[1:] > gap
    park_chg = long_park[1:] != long_park[:-1]
    brk = np.r_[True, cab_chg | gap_brk | park_chg]
    seg = np.cumsum(brk) - 1
    df = df.assign(seg=seg)
    df = df[~long_park]
    log(f"  จอดสั้น (< {brk_min:.0f} นาที) เก็บไว้ในเส้นทาง · จอดยาวตัดทิ้ง {n_long:,} จุด")
    g = df.groupby("seg")
    span = g["t"].max() - g["t"].min()
    keep = span.index[span >= min_len]
    n_seg_all = df["seg"].nunique()
    df = df[df["seg"].isin(keep)]
    log(f"  segment ทั้งหมด {n_seg_all:,} → ยาวพอ (≥ {min_len/60:.0f} นาที) {len(keep):,}")
    # resample ต่อ segment (vectorized ต่อ segment ด้วย np.interp)
    cols = {k: [] for k in ("cab", "seg", "t", "lat", "lon", "occ", "gap_s", "parked")}
    t_all = df["t"].values; lat_all = df["lat"].values; lon_all = df["lon"].values; occ_all = df["occ"].values; park_all = df["parked"].values
    cab_all = df["cab"].values; seg_all = df["seg"].values
    starts = np.flatnonzero(np.r_[True, seg_all[1:] != seg_all[:-1]]); ends = np.r_[starts[1:], len(df)]
    for s0, s1 in zip(starts, ends):
        t = t_all[s0:s1].astype(np.int64)
        t0 = int(np.ceil(t[0] / dt_s) * dt_s); t1 = int(np.floor(t[-1] / dt_s) * dt_s)
        if t1 - t0 < min_len: continue
        grid = np.arange(t0, t1 + 1, dt_s, dtype=np.int64)
        lat = np.interp(grid, t, lat_all[s0:s1]); lon = np.interp(grid, t, lon_all[s0:s1])
        idx = np.searchsorted(t, grid, side="right") - 1          # จุดดิบล่าสุดก่อนหน้า
        occ = occ_all[s0:s1][np.clip(idx, 0, len(t) - 1)]
        nxt = np.clip(idx + 1, 0, len(t) - 1)
        gap_s = (t[nxt] - t[np.clip(idx, 0, len(t) - 1)]).astype(np.float32)   # ระยะห่างจุดดิบสองข้าง
        n = len(grid)
        cols["cab"].append(np.full(n, cab_all[s0], dtype=object)); cols["seg"].append(np.full(n, seg_all[s0], dtype=np.int64))
        cols["t"].append(grid); cols["lat"].append(lat); cols["lon"].append(lon); cols["occ"].append(occ.astype(np.int8)); cols["gap_s"].append(gap_s)
        cols["parked"].append(park_all[s0:s1][np.clip(idx, 0, len(t) - 1)])
    rs = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
    seg_len = rs.groupby("seg").size() * dt_s / 60
    rep = pd.DataFrame([{
        "segments": int(rs["seg"].nunique()), "steps": len(rs), "cabs": int(rs["cab"].nunique()),
        "seg_len_min_p25": float(seg_len.quantile(.25)), "seg_len_min_p50": float(seg_len.quantile(.5)), "seg_len_min_p75": float(seg_len.quantile(.75)),
        "pct_steps_gap_gt_2dt": float((rs["gap_s"] > 2 * dt_s).mean() * 100),
        "pct_steps_parked_short": float(rs["parked"].mean() * 100),
        "hours_total": len(rs) * dt_s / 3600,
    }])
    log(f"  resample → {len(rs):,} ก้าว · {rs['seg'].nunique():,} segment · ก้าวที่ interpolate ไกล (>2Δτ) {rep['pct_steps_gap_gt_2dt'].iloc[0]:.1f}%")
    return rs, rep

# ----------------------------------------------------------------------------- D3 features
def d3_features(rs: pd.DataFrame, cfg: dict, log) -> pd.DataFrame:
    lat0, lon0 = cfg["origin"]; dt_s = int(cfg["dt_s"])
    project(rs, lat0, lon0)
    same = rs["seg"].values[1:] == rs["seg"].values[:-1]
    dx = np.r_[0.0, np.where(same, np.diff(rs["x"].values), 0.0)].astype(np.float32)
    dy = np.r_[0.0, np.where(same, np.diff(rs["y"].values), 0.0)].astype(np.float32)
    rs["dx"], rs["dy"] = dx, dy
    rs["v"] = (np.hypot(dx, dy) / dt_s).astype(np.float32)          # m/s
    loc = pd.to_datetime(rs["t"], unit="s", utc=True).dt.tz_convert(local_tz(cfg))
    dow = loc.dt.dayofweek.values; minute = (loc.dt.hour * 60 + loc.dt.minute).values
    rs["dow_sin"] = np.sin(2 * np.pi * dow / 7).astype(np.float32); rs["dow_cos"] = np.cos(2 * np.pi * dow / 7).astype(np.float32)
    rs["slot_sin"] = np.sin(2 * np.pi * minute / 1440).astype(np.float32); rs["slot_cos"] = np.cos(2 * np.pi * minute / 1440).astype(np.float32)
    rs["date"] = loc.dt.strftime("%Y-%m-%d").values; rs["hour"] = loc.dt.hour.values.astype(np.int8); rs["minute"] = loc.dt.minute.values.astype(np.int8)
    if not cfg.get("occ_feature", True): rs["occ"] = np.int8(0)
    rs["occ"] = rs["occ"].astype(np.float32)
    log(f"  ฟีเจอร์ {cfg['features']}")
    return rs

# ----------------------------------------------------------------------------- D4 split + windows
def assign_split(rs: pd.DataFrame, cfg: dict) -> pd.Series:
    d = cfg["dates"]; date = rs["date"]
    split = pd.Series("unused", index=rs.index)
    for name in ("train", "val", "test"):
        a, b = [str(x) for x in d[name]]
        split[(date >= a) & (date <= b)] = name
    return split

def make_windows(rs: pd.DataFrame, cfg: dict, log):
    M, H, dt_s = int(cfg["M_in"]), int(cfg["H"]), int(cfg["dt_s"]); F = cfg["features"]
    clean_mult = float(cfg.get("clean_gap_mult", 2))          # หน้าต่าง "clean" = ไม่มีช่องว่างดิบ > clean_gap_mult·Δτ
    # ตัดเฉพาะที่รอยต่อ split (train/val/test) — ภายใน split เดียวกันให้ segment ข้ามเที่ยงคืนได้ (ไม่งั้น cycle 23:00 หาย)
    piece = (rs["seg"].astype(str) + "_" + rs["split"].astype(str)).values
    starts = np.flatnonzero(np.r_[True, piece[1:] != piece[:-1]]); ends = np.r_[starts[1:], len(rs)]
    feat = rs[F].values.astype(np.float32); X_m = rs[["x", "y"]].values.astype(np.float64)
    latlon = rs[["lat", "lon"]].values; gap = rs["gap_s"].values; t = rs["t"].values
    split = rs["split"].values; cab = rs["cab"].values; seg = rs["seg"].values; hour = rs["hour"].values; minute = rs["minute"].values; date = rs["date"].values
    stride = {"train": cfg["train_stride"], "val": cfg["val_stride"], "test": cfg["test_stride"]}
    Xs, Ys, metas, cyc_paths = {k: [] for k in stride}, {k: [] for k in stride}, {k: [] for k in stride}, []
    for s0, s1 in zip(starts, ends):
        sp = split[s0]
        if sp not in stride: continue
        n = s1 - s0
        if n < M + H: continue
        # ตำแหน่ง T (ก้าวสุดท้ายของอินพุต): T ∈ [s0+M−1, s1−H−1]
        T_all = np.arange(s0 + M - 1, s1 - H)
        cand = T_all[::stride[sp]]
        if sp == "test":   # เพิ่มหน้าต่างต้น cycle ชั่วโมงเต็มเสมอ
            cyc = T_all[(minute[T_all] == 0)]
            cand = np.union1d(cand, cyc)
        for T in cand:
            xin = feat[T - M + 1:T + 1]
            tgt = X_m[T + 1:T + H + 1] - X_m[T]
            mg = float(gap[T - M + 1:T + H + 1].max())
            is_cyc = bool(sp == "test" and minute[T] == 0)          # ต่อเนื่องอยู่แล้ว (segment ไม่มี gap > GAP_MAX)
            is_clean = bool(is_cyc and mg <= clean_mult * dt_s)       # ใช้เป็น label reachability / ตัวกรองเสริม
            Xs[sp].append(xin); Ys[sp].append(tgt.astype(np.float32))
            metas[sp].append((cab[T], int(seg[T]), int(t[T]), float(latlon[T, 0]), float(latlon[T, 1]), date[T], int(hour[T]), mg, is_cyc, is_clean))
            if is_cyc: cyc_paths.append(latlon[T + 1:T + H + 1])
    out = {}
    for sp in stride:
        X = np.stack(Xs[sp]) if Xs[sp] else np.zeros((0, M, len(F)), np.float32)
        Y = np.stack(Ys[sp]) if Ys[sp] else np.zeros((0, H, 2), np.float32)
        meta = pd.DataFrame(metas[sp], columns=["cab", "seg", "t_T", "lat_T", "lon_T", "date", "hour", "max_gap_s", "is_cycle_start", "is_cycle_clean"])
        out[sp] = (X, Y, meta)
        log(f"  {sp:<5} X {X.shape}  Y {Y.shape}  cycle-windows {int(meta['is_cycle_start'].sum()) if len(meta) else 0} (clean {int(meta['is_cycle_clean'].sum()) if len(meta) else 0})"),
    paths = np.stack(cyc_paths) if cyc_paths else np.zeros((0, H, 2))
    return out, paths

def fit_scaler(X_train: np.ndarray, F: list) -> dict:
    scale_cols = [i for i, f in enumerate(F) if f in ("x", "y", "v", "dx", "dy")]
    flat = X_train.reshape(-1, X_train.shape[-1])
    return {"cols": scale_cols, "names": [F[i] for i in scale_cols],
            "mean": [float(flat[:, i].mean()) for i in scale_cols], "std": [float(flat[:, i].std() + 1e-9) for i in scale_cols]}

def apply_scaler(X: np.ndarray, sc: dict) -> np.ndarray:
    X = X.copy()
    for j, i in enumerate(sc["cols"]): X[..., i] = (X[..., i] - sc["mean"][j]) / sc["std"][j]
    return X

# ----------------------------------------------------------------------------- main
def run(city: str, cfg_path: str, raw: str, n_cabs, out_root: str, resume: bool = False):
    t_start = time.time(); log = lambda *m: print(f"[{time.time()-t_start:6.1f}s]", *m, flush=True)
    cfg = load_config(cfg_path, city)
    out = Path(out_root) / city
    for sub in ("interim", "processed", "reports"): (out / sub).mkdir(parents=True, exist_ok=True)
    log(f"config {cfg_path} · Δτ={cfg['dt_s']}s H={cfg['H']} M_in={cfg['M_in']} R={cfg['R_m']}m pt={cfg['pt_s']}s")

    d3_file = next((f for f in (out / "interim" / "d3_features.parquet", out / "interim" / "d3_features.pkl") if f.exists()), None)
    if resume and d3_file:
        log(f"resume: อ่าน {d3_file.name} (ข้าม D1–D3)")
        rs = pd.read_parquet(d3_file) if d3_file.suffix == ".parquet" else pd.read_pickle(d3_file)
        rs["split"] = assign_split(rs, cfg)
        t_clean = pd.read_csv(out / "reports" / "D1_cleaning.csv"); t_seg = pd.read_csv(out / "reports" / "D2_segments.csv")
    else:
        # ---- load
        if city == "sf": df = load_sf(raw, n_cabs)
        elif city == "rome": df = load_rome(_rome_file(raw), n_cabs)
        else: df = load_tdrive(raw, n_cabs)
        log(f"D0 โหลด {len(df):,} จุด · {df['cab'].nunique()} คัน")

        # ---- D1
        log("D1 clean"); df, t_clean = d1_clean(df, cfg, log)
        t_clean.to_csv(out / "reports" / "D1_cleaning.csv", index=False)
        save_table(df[["cab", "t", "lat", "lon", "occ", "parked"]], out / "interim" / "d1_clean.parquet")

        # ---- D2
        log("D2 segment + resample"); rs, t_seg = d2_segment_resample(df, cfg, log); del df
        t_seg.to_csv(out / "reports" / "D2_segments.csv", index=False)
        save_table(rs, out / "interim" / "d2_resampled.parquet")

        # ---- D3
        log("D3 features"); rs = d3_features(rs, cfg, log)
        rs["split"] = assign_split(rs, cfg)
        save_table(rs, out / "interim" / "d3_features.parquet")

    # ---- D4
    log("D4 split + windows"); wins, paths = make_windows(rs, cfg, log)
    F = cfg["features"]
    sc = fit_scaler(wins["train"][0], F); json.dump(sc, open(out / "scaler.json", "w"), indent=2)
    rows = []
    for sp, (X, Y, meta) in wins.items():
        np.save(out / "processed" / f"{sp}_X.npy", apply_scaler(X, sc)); np.save(out / "processed" / f"{sp}_Y.npy", Y)
        save_table(meta, out / "processed" / f"{sp}_meta.parquet")
        a, b = cfg["dates"][sp]
        rows.append({"split": sp, "date_from": str(a), "date_to": str(b), "cabs": int(meta["cab"].nunique()) if len(meta) else 0,
                     "segments": int(meta["seg"].nunique()) if len(meta) else 0, "windows": len(meta),
                     "cycle_windows": int(meta["is_cycle_start"].sum()) if len(meta) else 0})
    np.save(out / "processed" / "test_cycle_paths.npy", paths)
    t_split = pd.DataFrame(rows); t_split.to_csv(out / "reports" / "D4_split.csv", index=False)
    # cycle ที่มีคันพร้อม ≥ W (จาก cycle-windows จริงหลัง clean)
    tm = wins["test"][2]; tm = tm[tm["is_cycle_start"]]
    per_cycle = tm.groupby(["date", "hour"])["cab"].nunique() if len(tm) else pd.Series(dtype=int)
    W = int(cfg.get("W", 60)); n_ok = int((per_cycle >= W).sum())
    tmc = wins["test"][2]; tmc = tmc[tmc["is_cycle_clean"]]
    per_cycle_clean = tmc.groupby(["date", "hour"])["cab"].nunique() if len(tmc) else pd.Series(dtype=int)
    n_ok_clean = int((per_cycle_clean >= W).sum())
    log(f"  test cycles ที่มีคันพร้อม ≥ {W}: {n_ok} (median คัน/cycle = {per_cycle.median() if len(per_cycle) else 0:.0f}) · เฉพาะ clean (gap ≤ 2Δτ): {n_ok_clean}")
    cyc_tbl = per_cycle.rename("cabs_ready").reset_index() if len(per_cycle) else pd.DataFrame(columns=["date", "hour", "cabs_ready"])
    cyc_tbl.to_csv(out / "reports" / "D4_test_cycles.csv", index=False)

    # ---- report + manifest
    md = [f"# D1–D4 — {city}", "", f"config: `{cfg_path}` · Δτ={cfg['dt_s']}s · H={cfg['H']} · M_in={cfg['M_in']} · R={cfg['R_m']}m · pt={cfg['pt_s']}s", "",
          "## D1 cleaning (จุดดิบ → เหลือ)", md_table(t_clean), "",
          "## D2 segments", md_table(t_seg.round(2)), "",
          "## D4 split", md_table(t_split), "",
          f"test cycles with ≥ {W} cabs ready: **{n_ok}** (clean-only: {n_ok_clean}) · median cabs/cycle: {per_cycle.median() if len(per_cycle) else 0:.0f} · scaler: {json.dumps({k: [round(v, 3) for v in sc[k]] for k in ('mean', 'std')})}"]
    (out / "reports" / "D1_D4_report.md").write_text("\n".join(md), encoding="utf-8")
    manifest = {"city": city, "config": cfg_path, "config_sha1": hashlib.sha1(open(cfg_path, "rb").read()).hexdigest(), "raw": raw, "n_cabs": n_cabs,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "outputs": sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())}
    json.dump(manifest, open(out / "manifest.json", "w"), indent=2, ensure_ascii=False)
    log(f"เสร็จ → {out}/reports/D1_D4_report.md")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", choices=["sf", "rome", "beijing"])
    ap.add_argument("--config"); ap.add_argument("--raw"); ap.add_argument("--n-cabs", type=int, default=None); ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true", help="ข้าม D1–D3 ถ้ามี interim/d3_features อยู่แล้ว (รัน D4 ใหม่อย่างเดียว)")
    a = ap.parse_args()
    city = a.city or PYCHARM["CITY"]
    cfg_path = a.config or PYCHARM["CONFIG"] or str(HERE / "config" / f"data_{city}.yaml")
    folder = {"sf": "DataTaxiSanFran", "rome": "DataTaxiRome", "beijing": "DataTaxiBejin"}[city]
    raw = a.raw or PYCHARM["RAW"] or _find_folder(folder) or (_find_folder("DataTaxiBeijing") if city == "beijing" else None)
    if not raw: sys.exit(f"ไม่พบโฟลเดอร์ข้อมูล {folder} — ใส่ path ใน PYCHARM['RAW']")
    n_cabs = a.n_cabs if a.n_cabs is not None else PYCHARM["N_CABS"]
    out = a.out or str(HERE / PYCHARM["OUT"])
    run(city, cfg_path, raw, n_cabs, out, resume=a.resume or PYCHARM.get("RESUME", False))

if __name__ == "__main__":
    main()
