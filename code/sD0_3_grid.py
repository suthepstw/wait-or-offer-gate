#!/usr/bin/env python3
"""
sD0_3_grid.py — D0.3 ฉบับกริด: นับ residence บน "จุดบันทึกทุก Δτ" ให้ตรงกับนิยาม (7) ใหม่ของ Def 5

ทำไมต้องมี: D0.3 เดิม (sD0_audit.py) นับ residence แบบต่อเนื่อง (segment–circle บนจุดดิบ) = (7) เดิม
            แต่ label ของ D7/B8 และ (9a) นับบนกริด Δτ ("มีจุดบันทึกในวง ≥ P จุดติดกัน") = (7) ใหม่
            สคริปต์นี้ใช้ df / refs / segment ชุดเดียวกับ sD0_audit (seed เดียวกัน) แล้วรายงานทั้งสองแบบคู่กัน

วิธี: ① visit ต่อเนื่อง (ref, cab, t_in, t_out) จาก segment–circle เหมือน D0.3 เดิม
      ② จุดบันทึกกริด τ_h = k·Δτ (epoch, floor) — ตำแหน่ง ณ τ_h อยู่ในวง ⇔ τ_h ∈ [t_in, t_out] ของ visit ใดวิสิตหนึ่ง
         (จริงเป๊ะ เพราะ visit มาจาก interpolation เส้นตรงชุดเดียวกับที่ D2 ใช้สร้างกริด)
      ③ ต่อ visit ต่อเนื่อง: n_rep = จำนวนกริดใน [t_in, t_out]  → % visit ที่ n_rep ≥ 1 / 2 / 3
      ④ grid-visit = run ของกริดในวงติดกัน (รวมข้าม visit ต่อเนื่องที่ออก-เข้าระหว่างสองกริด) → % r ≥ 2 / 3

ผลลัพธ์ --out/D0_<city>/residence_grid_R<R>.csv + residence_grid_summary.md

ใช้:  python sD0_3_grid.py --city sf --raw <dir> [--n-cabs 20] [--R 200] [--dt 60] [--out reports]
      python sD0_3_grid.py --demo
ต้องมี sD0_audit.py อยู่โฟลเดอร์เดียวกัน (import loaders/geometry จากที่นั่น — ไม่แก้ไฟล์เดิม)
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sD0_audit as D0                                           # noqa: E402


# ----------------------------------------------------------------------------- ① visit table (เหมือน residence_visits แต่คืนช่วงเวลา)
def residence_visit_table(seg: dict, refs: np.ndarray, R: float, chunk: int = 400_000) -> pd.DataFrame:
    from scipy.spatial import cKDTree
    tree = cKDTree(refs)
    n = len(seg["x0"]); cab = seg["cab"]
    bounds = [0]
    while bounds[-1] < n:
        e = min(bounds[-1] + chunk, n)
        if e < n:
            while e < n and cab[e] == cab[e - 1]: e += 1
        bounds.append(e)
    out = []
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
        v = pd.DataFrame({"ref": ri[ok], "cab": cb[si][ok],
                          "t_in": t0[si][ok] + u_in[ok] * dt[ok], "t_out": t0[si][ok] + u_out[ok] * dt[ok]})
        if v.empty: continue
        v.sort_values(["ref", "cab", "t_in"], inplace=True)
        same = (v["ref"].values[1:] == v["ref"].values[:-1]) & (v["cab"].values[1:] == v["cab"].values[:-1])
        cont = same & (v["t_in"].values[1:] <= v["t_out"].values[:-1] + 1.0)
        grp = np.r_[0, np.cumsum(~cont)]
        g = v.groupby(grp)
        out.append(pd.DataFrame({"ref": g["ref"].first().values, "cab": g["cab"].first().values,
                                 "t_in": g["t_in"].min().values, "t_out": g["t_out"].max().values}))
    if not out:
        return pd.DataFrame(columns=["ref", "cab", "t_in", "t_out"])
    vt = pd.concat(out, ignore_index=True)
    # chunk ตัดที่รอยต่อคัน จึงไม่มี visit ข้าม chunk — แต่กัน merge ซ้ำอีกชั้นเพื่อความปลอดภัย
    vt.sort_values(["ref", "cab", "t_in"], inplace=True); vt.reset_index(drop=True, inplace=True)
    same = (vt["ref"].values[1:] == vt["ref"].values[:-1]) & (vt["cab"].values[1:] == vt["cab"].values[:-1])
    cont = same & (vt["t_in"].values[1:] <= vt["t_out"].values[:-1] + 1.0)
    if cont.any():
        grp = np.r_[0, np.cumsum(~cont)]; g = vt.groupby(grp)
        vt = pd.DataFrame({"ref": g["ref"].first().values, "cab": g["cab"].first().values,
                           "t_in": g["t_in"].min().values, "t_out": g["t_out"].max().values})
    vt["dur"] = vt["t_out"] - vt["t_in"]
    return vt


# ----------------------------------------------------------------------------- ②③④ grid counting
def grid_stats(vt: pd.DataFrame, dt_s: int, pt_list: list[int]) -> tuple[dict, pd.DataFrame]:
    """vt: visit ต่อเนื่อง · คืน (summary dict, ตาราง run ของกริด)"""
    if vt.empty:
        return {"n_visits_cont": 0}, pd.DataFrame(columns=["ref", "cab", "h0", "r"])
    # ③ จำนวนกริดใน [t_in, t_out]: h_first = ceil(t_in/Δτ), h_last = floor(t_out/Δτ)
    h_first = np.ceil(vt["t_in"].values / dt_s).astype(np.int64)
    h_last = np.floor(vt["t_out"].values / dt_s).astype(np.int64)
    n_rep = np.maximum(h_last - h_first + 1, 0)
    vt = vt.assign(h_first=h_first, h_last=h_last, n_rep=n_rep)
    s = {"n_visits_cont": int(len(vt)), "median_dur_s": float(np.median(vt["dur"]))}
    for pt in pt_list:
        s[f"cont_pct_ge_{pt}s"] = float((vt["dur"] >= pt).mean() * 100)          # = D0.3 เดิม
    for P in (1, 2, 3):
        s[f"cont_pct_with_ge{P}_reports"] = float((n_rep >= P).mean() * 100)      # ตัวเชื่อม เดิม→ใหม่
    # ④ grid-visit: รวมกริดในวงของ (ref, cab) ทุก visit แล้วหา run ติดกัน
    has = vt[vt["n_rep"] > 0]
    reps = np.repeat(has["h_first"].values, has["n_rep"].values) + \
           (np.arange(has["n_rep"].sum()) - np.repeat(np.cumsum(has["n_rep"].values) - has["n_rep"].values, has["n_rep"].values))
    key_ref = np.repeat(has["ref"].values, has["n_rep"].values)
    key_cab = np.repeat(has["cab"].values, has["n_rep"].values)
    g = pd.DataFrame({"ref": key_ref, "cab": key_cab, "h": reps}).drop_duplicates().sort_values(["ref", "cab", "h"])
    same = (g["ref"].values[1:] == g["ref"].values[:-1]) & (g["cab"].values[1:] == g["cab"].values[:-1])
    cont = same & (g["h"].values[1:] == g["h"].values[:-1] + 1)
    grp = np.r_[0, np.cumsum(~cont)]
    gg = g.groupby(grp)
    runs = pd.DataFrame({"ref": gg["ref"].first().values, "cab": gg["cab"].first().values,
                         "h0": gg["h"].min().values, "r": gg["h"].size().values})
    s["n_visits_grid"] = int(len(runs))
    for P in (1, 2, 3):
        s[f"grid_pct_r_ge{P}"] = float((runs["r"] >= P).mean() * 100) if len(runs) else np.nan
    s["grid_median_r"] = float(np.median(runs["r"])) if len(runs) else np.nan
    return s, runs


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", choices=["sf", "rome", "beijing"], default="sf")
    ap.add_argument("--raw"); ap.add_argument("--n-cabs", type=int, default=None)
    ap.add_argument("--n-ref", type=int, default=2000); ap.add_argument("--gap-max", type=float, default=300)
    ap.add_argument("--R", type=int, nargs="+", default=[200]); ap.add_argument("--dt", type=int, default=60)
    ap.add_argument("--out", default="reports"); ap.add_argument("--demo", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-cab-median-interval", type=float, default=None)
    ap.add_argument("--with-parked", action="store_true", help="รวม segment ที่จอด (ตาราง ③′ ของ D0)")
    a = ap.parse_args()

    t_start = time.time(); log = lambda *m: print(f"[{time.time()-t_start:6.1f}s]", *m, flush=True)
    city = "demo" if a.demo else a.city
    out_dir = Path(a.out) / (f"D0_{city}" + (f"_le{int(a.max_cab_median_interval)}s" if a.max_cab_median_interval else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- โหลด + prep เหมือน sD0_audit.run_audit ทุกขั้น (เพื่อให้ refs/segment ตรงกัน)
    if a.demo:
        raw = D0.make_demo(a.out, n_cabs=a.n_cabs or 20); df = D0.load_sf(raw, None)
    elif a.city == "sf":
        df = D0.load_sf(a.raw, a.n_cabs)
    elif a.city == "beijing":
        df = D0.load_tdrive(a.raw, a.n_cabs, seed=a.seed)
    else:
        df = D0.load_rome(a.raw, a.n_cabs, seed=a.seed)
    df = df.drop_duplicates(["cab", "t"]).sort_values(["cab", "t"]).reset_index(drop=True)
    if a.max_cab_median_interval:
        dt_all = df.groupby("cab")["t"].diff(); cab_med = dt_all.groupby(df["cab"]).median()
        df = df[df["cab"].isin(cab_med.index[cab_med <= a.max_cab_median_interval])].reset_index(drop=True)
    lat_min, lat_max = df["lat"].quantile([.025, .975]); lon_min, lon_max = df["lon"].quantile([.025, .975])
    pad_lat = 1000 / D0.R_EARTH * 180 / np.pi; pad_lon = pad_lat / np.cos(np.radians(df["lat"].median()))
    lat_min -= pad_lat; lat_max += pad_lat; lon_min -= pad_lon; lon_max += pad_lon
    df = df[df["lat"].between(lat_min, lat_max) & df["lon"].between(lon_min, lon_max)].reset_index(drop=True)
    lat0, lon0 = float(df["lat"].median()), float(df["lon"].median())
    D0.project(df, lat0, lon0); df = D0.per_cab_steps(df); D0.flag_parked(df)
    log(f"โหลด {len(df):,} จุด · {df['cab'].nunique()} คัน")

    refs = D0.sample_reference_points(df, a.n_ref, a.seed)          # seed เดียวกับ D0.3
    seg = D0.segments_for_residence(df, a.gap_max, exclude_parked=not a.with_parked)
    log(f"refs {len(refs)} · segments {len(seg['x0']):,} · Δτ = {a.dt} s")

    rows = []
    for R in a.R:
        vt = residence_visit_table(seg, refs, float(R))
        s, runs = grid_stats(vt, a.dt, D0.PT_LIST)
        s = {"R_m": R, **s}; rows.append(s)
        vt.to_csv(out_dir / f"residence_visits_R{R}.csv", index=False)
        runs.to_csv(out_dir / f"residence_gridruns_R{R}.csv", index=False)
        log(f"R={R}: visits ต่อเนื่อง {s['n_visits_cont']:,} · ≥60 s {s.get('cont_pct_ge_60s', float('nan')):.1f}% (= D0.3 เดิม) · "
            f"มีจุดบันทึก ≥1 {s['cont_pct_with_ge1_reports']:.1f}% ≥2 {s['cont_pct_with_ge2_reports']:.1f}% ≥3 {s['cont_pct_with_ge3_reports']:.1f}% · "
            f"grid-visits {s['n_visits_grid']:,} · r≥2 {s['grid_pct_r_ge2']:.1f}% r≥3 {s['grid_pct_r_ge3']:.1f}%")
    summ = pd.DataFrame(rows); summ.to_csv(out_dir / "residence_grid_summary.csv", index=False)

    md = [f"# D0.3 ฉบับกริด — {city} (Δτ = {a.dt} s, {'รวม' if a.with_parked else 'ไม่รวม'} parked, refs {len(refs)}, seed {a.seed})", "",
          "คอลัมน์ `cont_*` = visit ต่อเนื่อง (segment–circle, = D0.3 เดิม) · `cont_pct_with_geP_reports` = % ของ visit ต่อเนื่องที่มีจุดบันทึกกริดในวง ≥ P จุด (ตัวเชื่อมนิยามเดิม→ใหม่) · "
          "`grid_*` = visit บนกริด = run ของจุดบันทึกในวงติดกัน (= นิยาม (7) ใหม่ · r ≥ P ⇔ ส่งงานได้ที่ pt = PΔτ)", "",
          D0.md_table(summ.round(1)), "",
          "อ่านผล: `cont_pct_ge_60s` คือตัวเลข 41.4 % (SF) ของ D0 เดิม · `cont_pct_with_ge1_reports` ≥ ค่านั้นเสมอ (ช่วง ≥ 60 s ต้องมีกริดอย่างน้อย 1) · "
          "ส่วนต่างคือ visit สั้นกว่า 60 s ที่ยังถูก 'เห็น' 1 ครั้ง (รถวิ่งผ่าน) · `grid_pct_r_ge2/3` อธิบายการลดของ positive rate ใน B8 เมื่อ P = 2, 3"]
    (out_dir / "residence_grid_summary.md").write_text("\n".join(md), encoding="utf-8")
    log(f"เสร็จ → {out_dir}/residence_grid_summary.md")


if __name__ == "__main__":
    main()
