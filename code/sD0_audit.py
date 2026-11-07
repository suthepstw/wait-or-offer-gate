#!/usr/bin/env python3
"""
sD0_audit.py — ขั้น D0: ตรวจข้อมูลดิบแท็กซี่ก่อนตัดสินใจ Δτ, R, pt, H, หน้าต่างวัน

รองรับ 2 รูปแบบ
  --city sf    : CRAWDAD cabspotting  (โฟลเดอร์ที่มี new_<id>.txt  บรรทัด: lat lon occ unix_time)
  --city rome  : CRAWDAD roma/taxi    (ไฟล์เดียว taxi_february.txt  บรรทัด: id;timestamp;POINT(lat lon))
  --city beijing : T-Drive            (โฟลเดอร์ taxi_log_2008_by_id/ ไฟล์ <id>.txt  บรรทัด: id,datetime,lon,lat)

ตัวอย่าง
  python sD0_audit.py --city sf   --raw "D:/data/cabspottingdata" --n-cabs 20 --out reports
  python sD0_audit.py --city sf   --raw "D:/data/cabspottingdata"             --out reports   # เต็ม
  python sD0_audit.py --city rome --raw "D:/data/taxi_february.txt" --n-cabs 20 --out reports
  python sD0_audit.py --demo --out reports        # ข้อมูลสังเคราะห์ ทดสอบว่าสคริปต์รันได้

ผลลัพธ์ใน --out/D0_<city>/
  ① interval.csv  ② speed.csv  ③ residence_R<R>.csv (+ residence_summary.csv)  ④ cycles.csv
  D0_report.md (รวมทุกตาราง + ค่าที่แนะนำ)  ·  suggestions.json  ·  *.png
ต้องการ: Python >= 3.7 · numpy pandas scipy matplotlib   (pip install numpy pandas scipy matplotlib)
"""
from __future__ import annotations
import argparse, json, os, sys, time, glob, math
from pathlib import Path
import numpy as np
import pandas as pd

R_EARTH = 6_371_000.0

# ============================================================================= PyCharm
# กด Run ใน PyCharm โดยไม่ต้องใส่ argument: สคริปต์จะใช้ค่าในบล็อกนี้
# แก้ path ให้ตรงกับเครื่อง (r"..." เพื่อไม่ต้อง escape backslash) — ถ้าเว้นว่างจะค้นหาโฟลเดอร์ชื่อ
# DataTaxiSanFran / DataTaxiRome ในโฟลเดอร์ของสคริปต์, โฟลเดอร์แม่, และ Desktop ให้อัตโนมัติ
PYCHARM = dict(
    RUN=["beijing"],           # จะรันเมืองไหนบ้าง: ["sf"], ["rome"], ["beijing"], ["sf","rome","beijing"]
    RAW_SF=r"",                # เช่น r"C:\Users\suthe\Desktop\DataTaxiSanFran"
    RAW_ROME=r"",              # เช่น r"C:\Users\suthe\Desktop\DataTaxiRome"  (โฟลเดอร์ หรือไฟล์ .txt ก็ได้)
    RAW_BEIJING=r"",           # เช่น r"C:\Users\suthe\Desktop\DataTaxiBejin"  (โฟลเดอร์ที่มี <id>.txt หลายพันไฟล์)
    GAP_MAX_BY_CITY={"beijing": 300},   # ใช้ 300 เมื่อกรองเฉพาะคันความถี่สูง; ถ้าใช้ทุกคัน (Δτ=180) ตั้ง 600
    CAB_MEDIAN_INTERVAL_MAX_BY_CITY={"beijing": 60},   # เก็บเฉพาะคันที่ median interval ≤ ค่านี้ (วินาที); None = ทุกคัน
    N_CABS=None,                 # None = ทุกคัน
    OUT="reports",             # โฟลเดอร์ผลลัพธ์ (สร้างข้างสคริปต์)
    N_REF=2000, GAP_MAX=300, W_MIN=60, M_IN=8,   # รันเต็ม: ถ้า RAM ≤ 16 GB ลด N_REF=1000
)

def _find_folder(name: str) -> str | None:
    here = Path(__file__).resolve().parent
    cands = [here / name, here.parent / name, Path.home() / "Desktop" / name, Path.home() / name]
    for c in cands:
        if c.exists(): return str(c)
    return None

def _rome_file(path: str) -> str:
    """รับโฟลเดอร์หรือไฟล์ → คืน path ไฟล์ .txt/.csv ที่ใหญ่ที่สุด"""
    p = Path(path)
    if p.is_file(): return str(p)
    files = [f for f in p.rglob("*") if f.suffix.lower() in (".txt", ".csv")]
    if not files: sys.exit(f"ไม่พบไฟล์ .txt/.csv ใน {path}")
    return str(max(files, key=lambda f: f.stat().st_size))

# ----------------------------------------------------------------------------- config
CITY = {
    "sf":   dict(tz="America/Los_Angeles", pattern="new_*.txt"),
    "rome": dict(tz="Europe/Rome",         pattern=None),
    "beijing": dict(tz="Asia/Shanghai",    pattern="*.txt"),
    "demo": dict(tz="America/Los_Angeles", pattern="new_*.txt"),
}
R_LIST   = [100, 150, 200, 250]          # เมตร
PT_LIST  = [10, 30, 60, 120, 180]        # วินาที
DT_CHOICES = [30, 60, 120, 180]          # Δτ ที่อนุญาต

# ----------------------------------------------------------------------------- loaders
def load_sf(raw_dir: str, n_cabs: int | None) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(raw_dir, "**", "new_*.txt"), recursive=True))
    if not files:
        sys.exit(f"ไม่พบไฟล์ new_*.txt ใน {raw_dir} (ค้นหาในโฟลเดอร์ย่อยด้วยแล้ว)")
    if n_cabs:
        # เลือกคันที่ไฟล์ใหญ่สุด (ข้อมูลหนาแน่น) เพื่อให้ทดสอบมีความหมาย
        files = sorted(files, key=os.path.getsize, reverse=True)[:n_cabs]
    parts = []
    for f in files:
        cab = Path(f).stem.replace("new_", "")
        d = pd.read_csv(f, sep=r"\s+", header=None, names=["lat", "lon", "occ", "t"],
                        dtype={"lat": float, "lon": float, "occ": "int8", "t": "int64"})
        d["cab"] = cab
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    return df[["cab", "t", "lat", "lon", "occ"]]

def load_rome(raw_file: str, n_cabs: int | None, chunksize: int = 2_000_000, seed: int = 0) -> pd.DataFrame:
    parts = []
    keep = None
    if n_cabs:
        # เลือกคันแบบสุ่มจาก id ที่พบใน 2 ล้านบรรทัดแรก แล้วอ่าน "ทั้งไฟล์" เฉพาะคันเหล่านั้น → ได้ครบ 30 วัน
        head = pd.read_csv(raw_file, sep=";", header=None, names=["cab", "ts", "pt"], dtype=str, nrows=chunksize)
        ids = head["cab"].value_counts()
        ids = ids[ids >= ids.quantile(0.5)].index          # ตัดคันที่บางมากออกจากการสุ่ม
        keep = set(np.random.default_rng(seed).choice(ids, size=min(n_cabs, len(ids)), replace=False))
    for chunk in pd.read_csv(raw_file, sep=";", header=None, names=["cab", "ts", "pt"],
                             dtype={"cab": str, "ts": str, "pt": str}, chunksize=chunksize):
        if keep is not None:
            chunk = chunk[chunk["cab"].isin(keep)]
            if chunk.empty: continue
        xy = chunk["pt"].str.extract(r"POINT\(\s*([-\d.]+)\s+([-\d.]+)\s*\)")
        chunk["lat"] = xy[0].astype(float)       # ชุดนี้ lat มาก่อน lon
        chunk["lon"] = xy[1].astype(float)
        try:
            ts = pd.to_datetime(chunk["ts"], utc=True, format="mixed", errors="coerce")   # pandas >= 2.0
        except (TypeError, ValueError):
            ts = pd.to_datetime(chunk["ts"], utc=True, errors="coerce")                   # pandas 1.x
        chunk["t"] = (ts - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)
        chunk["occ"] = np.int8(0)
        chunk = chunk.dropna(subset=["lat", "lon", "t"])
        chunk["t"] = chunk["t"].astype("int64")
        parts.append(chunk[["cab", "t", "lat", "lon", "occ"]])
    df = pd.concat(parts, ignore_index=True)
    assert df["lat"].between(41.0, 43.0).mean() > 0.95, "lat ไม่ใช่ ~41.9 — ตรวจลำดับ lat/lon ใน POINT()"
    return df.reset_index(drop=True)

def load_tdrive(raw_dir: str, n_cabs: int | None, seed: int = 0) -> pd.DataFrame:
    """T-Drive: ไฟล์ <id>.txt ต่อคัน (~10,357 ไฟล์) บรรทัด: id,YYYY-MM-DD HH:MM:SS,lon,lat (เวลาท้องถิ่นปักกิ่ง ไม่มี offset)"""
    files = [f for f in glob.glob(os.path.join(raw_dir, "**", "*.txt"), recursive=True) if Path(f).stem.isdigit()]
    if not files:
        sys.exit(f"ไม่พบไฟล์ <id>.txt ใน {raw_dir}")
    files = [f for f in files if os.path.getsize(f) > 2000]      # ตัดไฟล์ว่าง/เกือบว่าง (< ~50 จุด)
    if n_cabs:
        # T-Drive ความถี่ต่างกันมากต่อคัน (บางคัน 5 s บางคัน 5 นาที) — ห้ามเลือกไฟล์ใหญ่สุด ให้สุ่ม
        rng = np.random.default_rng(seed)
        files = list(rng.choice(files, size=min(n_cabs, len(files)), replace=False))
    parts = []
    for i, f in enumerate(files):
        try:
            d = pd.read_csv(f, header=None, names=["cab", "ts", "lon", "lat"], dtype={"cab": str, "ts": str, "lon": float, "lat": float})
        except Exception:
            continue
        if d.empty: continue
        parts.append(d)
        if (i + 1) % 1000 == 0: print(f"   อ่านแล้ว {i+1}/{len(files)} ไฟล์", flush=True)
    df = pd.concat(parts, ignore_index=True)
    ts = pd.to_datetime(df["ts"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    ts = ts.dt.tz_localize("Asia/Shanghai", ambiguous="NaT", nonexistent="NaT")
    df["t"] = (ts - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)
    df["occ"] = np.int8(0)
    df = df.dropna(subset=["lat", "lon", "t"]); df["t"] = df["t"].astype("int64")
    df = df[(df["lat"] != 0) & (df["lon"] != 0)]
    assert df["lat"].between(39.0, 41.5).mean() > 0.9 and df["lon"].between(115.0, 118.0).mean() > 0.9, "lat/lon ไม่ใช่ปักกิ่ง — ตรวจลำดับคอลัมน์ (T-Drive: lon มาก่อน lat)"
    return df[["cab", "t", "lat", "lon", "occ"]].reset_index(drop=True)

def make_demo(out_dir: str, n_cabs: int = 20, days: int = 3, seed: int = 0) -> str:
    """สร้างข้อมูลสังเคราะห์รูปแบบ cabspotting เพื่อทดสอบสคริปต์ (ไม่ใช่ข้อมูลจริง)"""
    rng = np.random.default_rng(seed)
    d = Path(out_dir) / "demo_raw"; d.mkdir(parents=True, exist_ok=True)
    lat0, lon0 = 37.7749, -122.4194
    t0 = 1211011200  # 2008-05-17 08:00 UTC
    for c in range(n_cabs):
        rows = []
        t = t0 + rng.integers(0, 3600)
        x = y = 0.0
        heading = rng.uniform(0, 2 * np.pi)
        while t < t0 + days * 86400:
            # จอด 5% ของเวลา, วิ่ง 95%
            if rng.random() < 0.02:
                dwell = rng.integers(600, 3000)
                for k in range(0, dwell, 60):
                    rows.append((x, y, 0, t + k))
                t += dwell
                continue
            dt = int(np.clip(rng.lognormal(np.log(45), 0.45), 5, 600))
            v = np.clip(rng.normal(9, 4), 0, 30)                      # m/s
            heading += rng.normal(0, 0.6)
            if rng.random() < 0.3: heading = np.round(heading / (np.pi/2)) * (np.pi/2)  # ตารางถนน
            x += v * dt * np.cos(heading); y += v * dt * np.sin(heading)
            x = np.clip(x, -6000, 6000); y = np.clip(y, -8000, 8000)
            rows.append((x, y, int(rng.random() < 0.5), t)); t += dt
        arr = np.array(rows, dtype=float)
        lat = lat0 + arr[:, 1] / R_EARTH * 180 / np.pi
        lon = lon0 + arr[:, 0] / (R_EARTH * np.cos(np.radians(lat0))) * 180 / np.pi
        out = np.column_stack([lat, lon, arr[:, 2], arr[:, 3]])
        out = out[::-1]  # เลียนแบบไฟล์จริงที่เรียงใหม่→เก่า
        np.savetxt(d / f"new_demo{c:02d}.txt", out, fmt=["%.5f", "%.5f", "%d", "%d"])
    return str(d)

# ----------------------------------------------------------------------------- geometry
def project(df: pd.DataFrame, lat0: float, lon0: float) -> None:
    df["x"] = ((df["lon"] - lon0) * np.pi / 180 * R_EARTH * np.cos(np.radians(lat0))).astype(np.float32)
    df["y"] = ((df["lat"] - lat0) * np.pi / 180 * R_EARTH).astype(np.float32)

def per_cab_steps(df: pd.DataFrame) -> pd.DataFrame:
    """เพิ่ม dt, dist, v (km/h) ระหว่างจุดติดกันของคันเดียวกัน"""
    df.sort_values(["cab", "t"], inplace=True); df.reset_index(drop=True, inplace=True)
    same = df["cab"].values[1:] == df["cab"].values[:-1]
    dt = np.diff(df["t"].values).astype(float); dx = np.diff(df["x"].values); dy = np.diff(df["y"].values)
    dist = np.hypot(dx, dy)
    df["dt"] = np.r_[np.nan, np.where(same, dt, np.nan)].astype(np.float32)
    df["dist"] = np.r_[np.nan, np.where(same, dist, np.nan)].astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["v"] = (df["dist"] / df["dt"] * 3.6).astype(np.float32)
    return df

def flag_parked(df: pd.DataFrame, radius_m: float = 30.0, min_minutes: float = 30.0) -> None:
    """flag ช่วงที่รถนิ่ง (จุดติดกันห่าง < radius) ต่อเนื่องนาน > min_minutes — แบบ vectorized"""
    still = np.array((df["dist"].fillna(np.inf) < radius_m).values, dtype=bool)   # copy ที่เขียนได้
    cab_change = np.r_[True, df["cab"].values[1:] != df["cab"].values[:-1]]
    still &= ~cab_change                      # ก้าวแรกของคันไม่ใช่ "นิ่ง"
    # run-length ของ still
    brk = np.r_[True, still[1:] != still[:-1]]
    run_id = np.cumsum(brk) - 1
    t = df["t"].values
    run_start = np.flatnonzero(brk)
    run_end = np.r_[run_start[1:], len(df)] - 1
    is_still_run = still[run_start]
    # ระยะเวลาของ run = t[end] − t[start−1] (จุดก่อนหน้าคือจุดเริ่มนิ่ง)
    start_prev = np.maximum(run_start - 1, 0)
    dur = t[run_end] - t[start_prev]
    long_run = is_still_run & (dur >= min_minutes * 60)
    parked = np.array(long_run[run_id], dtype=bool)
    # รวมจุดก่อนหน้าของ run ที่ parked ด้วย
    prev_idx = start_prev[long_run]
    parked[prev_idx] = True
    df["parked"] = parked

# ----------------------------------------------------------------------------- tables ① ②
def table_interval(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["dt"].dropna()
    q = dt.quantile([.10, .25, .50, .75, .90, .95, .99])
    row = {f"p{int(k*100)}": v for k, v in q.items()}
    row["pct_gt_300s"] = (dt > 300).mean() * 100
    row["pct_gt_600s"] = (dt > 600).mean() * 100
    row["n_points"] = len(df); row["n_cabs"] = df["cab"].nunique()
    # ความถี่ต่อคัน: median interval ของแต่ละคัน → ดูว่ามีกี่คันที่ละเอียดพอสำหรับ Δτ = 60 / 180 s
    cab_med = df.groupby("cab")["dt"].median()
    row["cab_median_interval_p25"] = float(cab_med.quantile(.25)); row["cab_median_interval_p50"] = float(cab_med.quantile(.5)); row["cab_median_interval_p75"] = float(cab_med.quantile(.75))
    row["pct_cabs_median_le_60s"] = float((cab_med <= 60).mean() * 100)
    row["pct_cabs_median_le_180s"] = float((cab_med <= 180).mean() * 100)
    days = (df["t"].max() - df["t"].min()) / 86400
    row["hours_logged_per_cab_per_day"] = df.loc[df["dt"] <= 300, "dt"].sum() / 3600 / df["cab"].nunique() / max(days, 1e-9)
    return pd.DataFrame([row])

def table_speed(df: pd.DataFrame) -> pd.DataFrame:
    v = df.loc[df["dt"] >= 3, "v"].dropna()          # ข้าม dt เล็กมาก (Rome)
    q = v.quantile([.10, .50, .75, .90, .95, .99])
    row = {f"p{int(k*100)}": val for k, val in q.items()}
    row["pct_gt_150"] = (v > 150).mean() * 100
    row["pct_lt_1"] = (v < 1).mean() * 100
    row["pct_parked_time"] = df.loc[df["parked"], "dt"].sum() / df["dt"].sum() * 100
    return pd.DataFrame([row])

# ----------------------------------------------------------------------------- table ③ residence
def segments_for_residence(df: pd.DataFrame, gap_max: float, exclude_parked: bool, v_max: float = 150.0):
    """คู่จุดติดกัน (p0→p1) ที่ใช้เป็น segment: dt ≤ gap_max, ความเร็ว ≤ v_max (ตัด GPS กระโดด),
    ไม่ใช่ parked (ถ้าเลือก) — segment ความยาว 0 (รถนิ่ง) ต้องเก็บไว้ ไม่งั้นเวลาจอดรอในรัศมีจะหายไป"""
    ok = (df["dt"] <= gap_max) & (df["dt"] > 0) & (df["v"].fillna(0) <= v_max)
    if exclude_parked: ok &= ~(df["parked"] | df["parked"].shift(1, fill_value=False))
    idx = np.where(ok.values)[0]
    x1, y1, t1 = df["x"].values[idx].astype(np.float64), df["y"].values[idx].astype(np.float64), df["t"].values[idx].astype(float)
    x0, y0, t0 = df["x"].values[idx-1].astype(np.float64), df["y"].values[idx-1].astype(np.float64), df["t"].values[idx-1].astype(float)
    cab = pd.factorize(df["cab"])[0][idx]
    return dict(x0=x0, y0=y0, t0=t0, x1=x1, y1=y1, t1=t1, cab=cab)

def sample_reference_points(df: pd.DataFrame, n_ref: int, seed: int) -> np.ndarray:
    """สุ่มจุดอ้างอิงจากจุด GPS ดิบขณะวิ่ง (ไม่ใช่จอด) → ถ่วงน้ำหนักตามความหนาแน่นถนนโดยธรรมชาติ"""
    rng = np.random.default_rng(seed)
    pool = df.loc[~df["parked"] & (df["v"] > 3), ["x", "y"]].values
    pick = rng.choice(len(pool), size=min(n_ref, len(pool)), replace=False)
    return pool[pick]

def residence_visits(seg: dict, refs: np.ndarray, R: float, chunk: int = 400_000):
    """คืน (durations[s], visits_per_ref) ของ 'การเข้า' แต่ละครั้ง — segment–circle + รวมช่วงต่อเนื่อง
    ประมวลผลเป็น chunk (ตัดที่ขอบเขตคัน) เพื่อไม่ให้หน่วยความจำระเบิดกับข้อมูลหลายสิบล้าน segment"""
    from scipy.spatial import cKDTree
    tree = cKDTree(refs)
    n = len(seg["x0"])
    cab = seg["cab"]
    # ขอบเขต chunk: เดินไปทีละ chunk แล้วขยับให้จบที่รอยต่อคัน
    bounds = [0]
    while bounds[-1] < n:
        e = min(bounds[-1] + chunk, n)
        if e < n:
            while e < n and cab[e] == cab[e - 1]: e += 1
        bounds.append(e)
    durs, counts = [], np.zeros(len(refs), dtype=np.int64)
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        x0, y0, t0 = seg["x0"][b0:b1], seg["y0"][b0:b1], seg["t0"][b0:b1]
        x1, y1, t1 = seg["x1"][b0:b1], seg["y1"][b0:b1], seg["t1"][b0:b1]
        cb = cab[b0:b1]
        mx = (x0 + x1) / 2; my = (y0 + y1) / 2
        half = np.hypot(x1 - x0, y1 - y0) / 2
        cand = tree.query_ball_point(np.column_stack([mx, my]), r=half + R)
        lens = np.fromiter((len(c) for c in cand), dtype=np.int64, count=len(cand))
        if lens.sum() == 0: continue
        si = np.repeat(np.arange(len(cand)), lens)
        ri = np.concatenate([np.asarray(c, dtype=np.int64) for c in cand if len(c)])
        del cand
        dx = x1[si] - x0[si]; dy = y1[si] - y0[si]
        fx = x0[si] - refs[ri, 0]; fy = y0[si] - refs[ri, 1]
        a = dx*dx + dy*dy; b = 2*(fx*dx + fy*dy); c = fx*fx + fy*fy - R*R
        zero = a < 1e-9
        a_safe = np.where(zero, 1.0, a)
        disc = b*b - 4*a_safe*c
        ok = (disc >= 0) & ~zero
        sq = np.sqrt(np.where(ok, disc, 0))
        u_in = np.where(zero, 0.0, (-b - sq) / (2*a_safe)); u_out = np.where(zero, 1.0, (-b + sq) / (2*a_safe))
        u_in = np.clip(u_in, 0, 1); u_out = np.clip(u_out, 0, 1)
        ok = (ok & (u_out > u_in)) | (zero & (c <= 0))
        dt = t1[si] - t0[si]
        v = pd.DataFrame({"ref": ri[ok], "cab": cb[si][ok], "t_in": t0[si][ok] + u_in[ok] * dt[ok], "t_out": t0[si][ok] + u_out[ok] * dt[ok]})
        del si, ri, dx, dy, fx, fy, a, b, c, disc, sq, u_in, u_out, ok, dt
        if v.empty: continue
        v.sort_values(["ref", "cab", "t_in"], inplace=True)
        same = (v["ref"].values[1:] == v["ref"].values[:-1]) & (v["cab"].values[1:] == v["cab"].values[:-1])
        cont = same & (v["t_in"].values[1:] <= v["t_out"].values[:-1] + 1.0)
        grp = np.r_[0, np.cumsum(~cont)]
        g = v.groupby(grp)
        d = (g["t_out"].max() - g["t_in"].min()).values.astype(np.float32)
        r_first = g["ref"].first().values
        durs.append(d); counts += np.bincount(r_first, minlength=len(refs))
        del v, g
    dur = np.concatenate(durs) if durs else np.zeros(0, np.float32)
    return dur, counts

def table_residence(seg: dict, refs: np.ndarray, out_dir: Path, tag: str) -> pd.DataFrame:
    rows = []
    for R in R_LIST:
        dur, counts = residence_visits(seg, refs, R)
        row = {"R_m": R, "n_visits": len(dur), "median_s": float(np.median(dur)) if len(dur) else np.nan,
               "p75_s": float(np.percentile(dur, 75)) if len(dur) else np.nan}
        for pt in PT_LIST:
            row[f"pct_ge_{pt}s"] = float((dur >= pt).mean() * 100) if len(dur) else np.nan
        # จำนวนครั้งที่เข้าต่อจุดต่อชั่วโมง (ตัวประมาณ |C_j| หยาบ)
        span_h = (seg["t1"].max() - seg["t0"].min()) / 3600
        row["visits_per_ref_per_hour"] = len(dur) / max(len(refs), 1) / max(span_h, 1e-9)
        row["pct_refs_with_lt5_visits"] = float((counts < 5).mean() * 100)
        rows.append(row)
        np.save(out_dir / f"residence_durations_R{R}_{tag}.npy", dur)
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------------- table ④ usable cycles
def table_cycles(df: pd.DataFrame, tz: str, gap_max: float, dt_s: int, m_in: int, cycle_min: int, w_min: int):
    """นับต่อ (วันท้องถิ่น, ชั่วโมง): จำนวนคันที่มี segment ต่อเนื่อง (ไม่มี gap > gap_max) ครอบ [start − m_in·dt, end]"""
    # สร้างช่วงต่อเนื่องต่อคัน
    brk = (df["dt"].isna()) | (df["dt"] > gap_max)
    seg_id = brk.cumsum().values
    g = df.groupby(seg_id)
    spans = pd.DataFrame({"cab": g["cab"].first(), "t0": g["t"].min(), "t1": g["t"].max()})
    t_min, t_max = df["t"].min(), df["t"].max()
    first_day = pd.Timestamp(t_min, unit="s", tz="UTC").tz_convert(tz).normalize()
    last_day = pd.Timestamp(t_max, unit="s", tz="UTC").tz_convert(tz).normalize()
    rows = []
    day = first_day
    pre = m_in * dt_s
    while day <= last_day:
        for h in range(0, 24 * 60 // cycle_min):
            cs = int((day + pd.Timedelta(minutes=h * cycle_min)).timestamp()); ce = cs + cycle_min * 60
            ok = (spans["t0"] <= cs - pre) & (spans["t1"] >= ce)
            n = spans.loc[ok, "cab"].nunique()
            rows.append({"date": day.date().isoformat(), "cycle_start_local": (day + pd.Timedelta(minutes=h*cycle_min)).strftime("%H:%M"), "n_cabs_ready": n})
        day += pd.Timedelta(days=1)
    cyc = pd.DataFrame(rows)
    per_day = cyc.groupby("date").agg(cycles_ge_wmin=("n_cabs_ready", lambda s: int((s >= w_min).sum())),
                                      median_cabs=("n_cabs_ready", "median"), max_cabs=("n_cabs_ready", "max"))
    return cyc, per_day

# ----------------------------------------------------------------------------- suggestions
def suggest(interval: pd.DataFrame, residence: pd.DataFrame, per_day: pd.DataFrame, w_min: int, n_cabs_loaded: int) -> dict:
    p75 = float(interval["p75"].iloc[0])
    dt_s = next((c for c in DT_CHOICES if c * 1.10 >= p75), DT_CHOICES[-1])   # ยอมให้ p75 เกินได้ 10% (61 s → 60 s)
    H = 60 * 60 // dt_s
    # (R, pt) ที่สัดส่วน ≥pt อยู่ใน 30–50%
    cands = []
    for _, r in residence.iterrows():
        for pt in PT_LIST:
            p = r[f"pct_ge_{pt}s"]
            if 30 <= p <= 50:
                cands.append({"R_m": int(r["R_m"]), "pt_s": pt, "pct_ge_pt": round(float(p), 1)})
    cands.sort(key=lambda c: (c["R_m"], -c["pt_s"]))
    total_cycles = int(per_day["cycles_ge_wmin"].sum())
    return {
        "delta_tau_s": dt_s, "H": H, "M_IN": max(2, round(8 * 60 / dt_s)),
        "delta_tau_60s_usable": bool(p75 <= 66),   # True = ใช้ Δτ=60 s ร่วมกับเมืองอื่นได้ (downsample)
        "n_cabs_median_interval_le_60s": int(round(interval["pct_cabs_median_le_60s"].iloc[0] / 100 * n_cabs_loaded)),
        "n_cabs_median_interval_le_180s": int(round(interval["pct_cabs_median_le_180s"].iloc[0] / 100 * n_cabs_loaded)),
        "R_pt_candidates_30_50pct": cands,
        "usable_cycles_total_at_Wmin": total_cycles,
        "days_in_data": int(len(per_day)),
        "note": ("จำนวน cycle ที่ใช้ได้คำนวณจากคันที่โหลดเท่านั้น (%d คัน) — ถ้าทดสอบ 20 คัน ตัวเลขนี้ไม่มีความหมาย ให้ดูคอลัมน์ median_cabs เทียบสัดส่วนแทน" % n_cabs_loaded),
    }

def md_table(df: pd.DataFrame, index: bool = False) -> str:
    """ตาราง markdown โดยไม่ต้องพึ่ง tabulate"""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    fmt = lambda v: (f"{v:.1f}" if isinstance(v, (float, np.floating)) else str(v))
    rows = [" | ".join(fmt(v) for v in r) for r in d.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + r + " |" for r in rows])

# ----------------------------------------------------------------------------- plots
def plots(df: pd.DataFrame, out_dir: Path, tag: str):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
    dt = df["dt"].dropna(); axs[0].hist(dt[dt < 400], bins=80, color="#1f5fa8"); axs[0].set_title("① sampling interval (s)")
    axs[0].axvline(dt.quantile(.75), color="k", ls="--"); axs[0].text(dt.quantile(.75), axs[0].get_ylim()[1]*.9, f" p75={dt.quantile(.75):.0f}s")
    v = df.loc[df["dt"] >= 3, "v"].dropna(); axs[1].hist(v[v < 120], bins=80, color="#1f5fa8"); axs[1].set_title("② speed between points (km/h)")
    for R, c in zip(R_LIST, ["#cde2fb", "#86b6ef", "#3987e5", "#184f95"]):
        f = out_dir / f"residence_durations_R{R}_{tag}.npy"
        if f.exists():
            d = np.load(f); d = d[d < 400]
            if len(d): axs[2].hist(d, bins=60, histtype="step", lw=2, color=c, label=f"R={R} m", density=True)
    for pt in [30, 60, 120]: axs[2].axvline(pt, color="#c9761c", lw=.8, ls=":")
    axs[2].set_title("③ residence time per visit (s)"); axs[2].legend()
    for a in axs: a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_dir / f"D0_hist_{tag}.png", dpi=150); plt.close(fig)

# ----------------------------------------------------------------------------- main
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", choices=["sf", "rome", "beijing"], default="sf")
    ap.add_argument("--raw", help="โฟลเดอร์ (sf) หรือไฟล์ (rome)")
    ap.add_argument("--n-cabs", type=int, default=None, help="จำกัดจำนวนคันสำหรับทดสอบ (เช่น 20)")
    ap.add_argument("--n-ref", type=int, default=2000, help="จำนวนจุดอ้างอิงสำหรับ residence")
    ap.add_argument("--gap-max", type=float, default=300, help="วินาที")
    ap.add_argument("--w-min", type=int, default=60, help="จำนวนคันขั้นต่ำต่อ cycle (|W|)")
    ap.add_argument("--dt", type=int, default=60, help="Δτ ที่ใช้นับ cycle (จะถูกแนะนำใหม่จากตาราง ①)")
    ap.add_argument("--m-in", type=int, default=8)
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"), default=None,
                    help="ถ้าไม่ระบุ ใช้ percentile 2.5–97.5 ของข้อมูล (รายงาน %% ที่ตัด)")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--demo", action="store_true", help="สร้างข้อมูลสังเคราะห์แล้วรัน (ทดสอบสคริปต์)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-cab-median-interval", type=float, default=None,
                    help="เก็บเฉพาะคันที่ median sampling interval ≤ ค่านี้ (วินาที) — ใช้กับ T-Drive ที่ความถี่ต่างกันมากต่อคัน")
    return ap

def run_audit(a):
    t_start = time.time()
    city = "demo" if a.demo else a.city
    tag = f"{city}_{a.n_cabs or 'all'}"
    out_dir = Path(a.out) / (f"D0_{city}" + (f"_le{int(a.max_cab_median_interval)}s" if a.max_cab_median_interval else "")); out_dir.mkdir(parents=True, exist_ok=True)
    log = lambda *m: print(f"[{time.time()-t_start:6.1f}s]", *m, flush=True)

    # ---- load
    if a.demo:
        raw = make_demo(a.out, n_cabs=a.n_cabs or 20); df = load_sf(raw, None)
    elif a.city == "sf":
        df = load_sf(a.raw, a.n_cabs)
    elif a.city == "beijing":
        df = load_tdrive(a.raw, a.n_cabs, seed=a.seed)
    else:
        df = load_rome(a.raw, a.n_cabs, seed=a.seed)
    log(f"โหลด {len(df):,} จุด · {df['cab'].nunique()} คัน")

    # ---- minimal prep: sort, dedupe, bbox
    df = df.drop_duplicates(["cab", "t"]).sort_values(["cab", "t"]).reset_index(drop=True)
    if a.max_cab_median_interval:
        dt_all = df.groupby("cab")["t"].diff()
        cab_med = dt_all.groupby(df["cab"]).median()
        keep = cab_med.index[cab_med <= a.max_cab_median_interval]
        n0 = df["cab"].nunique(); df = df[df["cab"].isin(keep)].reset_index(drop=True)
        log(f"กรองคันที่ median interval ≤ {a.max_cab_median_interval:.0f} s: เหลือ {df['cab'].nunique()} จาก {n0} คัน · {len(df):,} จุด")
        tag += f"_le{int(a.max_cab_median_interval)}s"
    if a.bbox:
        lat_min, lat_max, lon_min, lon_max = a.bbox
    else:
        lat_min, lat_max = df["lat"].quantile([.025, .975]); lon_min, lon_max = df["lon"].quantile([.025, .975])
        pad_lat = 1000 / R_EARTH * 180 / np.pi; pad_lon = pad_lat / np.cos(np.radians(df["lat"].median()))
        lat_min -= pad_lat; lat_max += pad_lat; lon_min -= pad_lon; lon_max += pad_lon
    inside = df["lat"].between(lat_min, lat_max) & df["lon"].between(lon_min, lon_max)
    pct_cut = (1 - inside.mean()) * 100
    df = df[inside].reset_index(drop=True)
    lat0, lon0 = float(df["lat"].median()), float(df["lon"].median())
    project(df, lat0, lon0); df = per_cab_steps(df); flag_parked(df)
    log(f"BBOX [{lat_min:.4f},{lat_max:.4f}]×[{lon_min:.4f},{lon_max:.4f}] ตัด {pct_cut:.1f}% · origin ({lat0:.4f},{lon0:.4f})")

    # ---- ① ②
    t1 = table_interval(df); t2 = table_speed(df)
    t1.to_csv(out_dir / "interval.csv", index=False); t2.to_csv(out_dir / "speed.csv", index=False)
    log("ตาราง ① interval:\n" + t1.round(1).to_string(index=False))
    log("ตาราง ② speed:\n" + t2.round(1).to_string(index=False))

    # ---- ③ residence (สองเวอร์ชัน: ไม่รวม parked / รวม parked)
    refs = sample_reference_points(df, a.n_ref, a.seed)
    seg = segments_for_residence(df, a.gap_max, exclude_parked=True)
    t3 = table_residence(seg, refs, out_dir, tag); t3.to_csv(out_dir / "residence_summary.csv", index=False)
    log("ตาราง ③ residence (ไม่รวม parked):\n" + t3.round(1).to_string(index=False))
    seg_p = segments_for_residence(df, a.gap_max, exclude_parked=False)
    t3p = table_residence(seg_p, refs, out_dir, tag + "_withparked"); t3p.to_csv(out_dir / "residence_summary_withparked.csv", index=False)

    # ---- ④ cycles
    dt_guess = next((c for c in DT_CHOICES if c * 1.10 >= float(t1["p75"].iloc[0])), DT_CHOICES[-1])
    m_in_eff = max(2, round(8 * 60 / dt_guess))          # ประวัติ 8 นาทีเสมอ → จำนวนก้าวขึ้นกับ Δτ
    cyc, per_day = table_cycles(df, CITY[city]["tz"], a.gap_max, dt_guess, m_in_eff, 60, a.w_min)
    cyc.to_csv(out_dir / "cycles.csv", index=False); per_day.to_csv(out_dir / "cycles_per_day.csv")
    log("ตาราง ④ cycle ต่อวัน:\n" + per_day.to_string())

    # ---- suggestions + report
    sug = suggest(t1, t3, per_day, a.w_min, df["cab"].nunique())
    sug.update(bbox=[float(lat_min), float(lat_max), float(lon_min), float(lon_max)], pct_points_outside_bbox=round(pct_cut, 2),
               origin=[lat0, lon0], gap_max_s=a.gap_max, n_cabs_loaded=int(df["cab"].nunique()), n_points=int(len(df)))
    (out_dir / "suggestions.json").write_text(json.dumps(sug, indent=2, ensure_ascii=False), encoding="utf-8")
    plots(df, out_dir, tag)

    md = [f"# D0 audit — {city} ({tag})", "",
          f"จุด {len(df):,} · คัน {df['cab'].nunique()} · BBOX ตัด {pct_cut:.1f}% · origin ({lat0:.4f}, {lon0:.4f})", "",
          "## ① sampling interval (s)", md_table(t1.round(1)), "",
          "## ② speed between points (km/h)", md_table(t2.round(1)), "",
          "## ③ residence time per visit — ไม่รวม parked (ค่า = % ของการเข้ารัศมีที่อยู่นาน ≥ pt)", md_table(t3.round(1)), "",
          "## ③′ residence — รวม parked", md_table(t3p.round(1)), "",
          f"## ④ cycle ที่ใช้ได้ต่อวัน (Δτ={dt_guess}s, M_IN={m_in_eff}, GAP_MAX={a.gap_max:.0f}s, |W|≥{a.w_min})", md_table(per_day, index=True), "",
          "## ค่าที่แนะนำ", "```json", json.dumps(sug, indent=2, ensure_ascii=False), "```",
          "", f"![hist](D0_hist_{tag}.png)"]
    (out_dir / "D0_report.md").write_text("\n".join(md), encoding="utf-8")
    log(f"เสร็จ → {out_dir}/D0_report.md")
    log("แนะนำ: " + json.dumps({k: sug[k] for k in ["delta_tau_s", "H", "M_IN", "R_pt_candidates_30_50pct"]}, ensure_ascii=False))

def main():
    ap = build_parser()
    if len(sys.argv) > 1:                       # โหมด command line
        run_audit(ap.parse_args()); return
    # โหมด PyCharm (ไม่มี argument): ใช้บล็อก PYCHARM ด้านบน
    out = str(Path(__file__).resolve().parent / PYCHARM["OUT"])
    for city in PYCHARM["RUN"]:
        if city == "sf":
            raw = PYCHARM["RAW_SF"] or _find_folder("DataTaxiSanFran")
        elif city == "beijing":
            raw = PYCHARM["RAW_BEIJING"] or _find_folder("DataTaxiBejin") or _find_folder("DataTaxiBeijing")
        else:
            raw = PYCHARM["RAW_ROME"] or _find_folder("DataTaxiRome")
            raw = _rome_file(raw) if raw else None
        if not raw:
            print(f"!! ไม่พบข้อมูลของ {city} — ใส่ path ในบล็อก PYCHARM ที่หัวไฟล์"); continue
        print(f"\n===== {city.upper()}  raw = {raw} =====")
        gap = PYCHARM.get("GAP_MAX_BY_CITY", {}).get(city, PYCHARM["GAP_MAX"])
        argv = ["--city", city, "--raw", raw, "--out", out, "--n-ref", str(PYCHARM["N_REF"]),
                "--gap-max", str(gap), "--w-min", str(PYCHARM["W_MIN"]), "--m-in", str(PYCHARM["M_IN"])]
        mci = PYCHARM.get("CAB_MEDIAN_INTERVAL_MAX_BY_CITY", {}).get(city)
        if mci: argv += ["--max-cab-median-interval", str(mci)]
        if PYCHARM["N_CABS"]: argv += ["--n-cabs", str(PYCHARM["N_CABS"])]
        run_audit(ap.parse_args(argv))

if __name__ == "__main__":
    main()
