# E1 report — rome (dev: hold-out cycles) · E1 v3.2 (2026-10-22: v3.1 + --dev N = สร้าง instance ชุด dev จาก cycle hold-out (ไม่ซ้อน 30 ตัวทดสอบ) สำหรับออกแบบ/จูน MOPSO-IQ v2) · 2026-10-22 16:56

**qm_model = (9a′) reach_tfm/seed2** (D7reach_dev, sha ตรง) · qm_lstm_reach = reach_lstm/seed0 · qm_tfm_ub = (9a) ของ reach_tfm (qm_calib.json) · qm_trajgeom = (9a) base_M3/seed3 (D7_final) · sD7 v7 sha `0fe82c5d0600` · split test_final 2014-02-26..2014-02-28 · cycle ใช้ได้ 32 → เลือก 30 · สร้าง 2

พารามิเตอร์ (เหมือน v2.2): |W| 60 · |T| 200 · cycle 60 นาที · R [150.0, 200.0, 250.0] (หลัก 200) · placement clustered (centre mixed 0.85, jitter 500 m, α 1) · cell_m 500 · β 0.65 · M_i 2 · Q_max 3 · K rule density · master seed 12345

## Gate G2 (ตรวจใน E1)

| ข้อ | ผล | ค่า |
|---|---|---|
| G2.1 (% worker ไม่มีงานเอื้อม < 40) | ✅ | 7.5 % |
| G2.2 (% task E_i < 0.5 < 20) | ❌ | 27.5 % |
| G2.2b (% task E_i < K_i ≤ 70) | ✅ | 42.5 % |
| frame check (median |Y − Δx_T| ≤ 5 m) | ✅ | 1.14 m (p90 6.7, n=20033) |
| sanity A: guarded (9a′) ≥ (9a) same encoder, overall | ✅ | model 0.804 · tfm_ub 0.793 · nog 0.730 · in-domain 59 % ของคู่ |
| sanity B: in-domain (9a′) > (9a) (ตรงกับ D7) | ✅ | in-dom (9a′) 0.718 vs (9a) 0.706 |
| sanity C: model > null > persistence | ✅ | qm_model 0.804 · qm_model_nog 0.730 · qm_lstm_reach 0.813 · qm_tfm_ub 0.793 · qm_lstm_ub 0.792 · qm_trajgeom 0.789 · qm_null_dv 0.761 · qm_persist_cont 0.746 |

## ต่อ instance

|   k | date       |   hour |   n_cand |   W |   T |   area_km2 |   n_centres |   K_mean |     B |   qm_mean |   E_mean |   pos_oracle |   n_i(θ=0.02) |   n_i(θ=0.05) |   n_i(θ=0.1) |   G21_pct |   G22_pct |   G22b_pct |   in_dom_pct |   AUC_qm_model |   AUC_qm_model_nog |   AUC_qm_lstm_reach |   AUC_qm_tfm_ub |   AUC_qm_lstm_ub |   AUC_qm_trajgeom |   AUC_qm_null_dv |   AUC_qm_persist_cont |   AUC_in_dom_model_nog |   AUC_in_dom_tfm_ub |
|----:|:-----------|-------:|---------:|----:|----:|-----------:|------------:|---------:|------:|----------:|---------:|-------------:|--------------:|--------------:|-------------:|----------:|----------:|-----------:|-------------:|---------------:|-------------------:|--------------------:|----------------:|-----------------:|------------------:|-----------------:|----------------------:|-----------------------:|--------------------:|
|   0 | 2014-02-27 |     19 |       80 |  60 | 200 |      459.6 |           8 |     1.33 | 425.9 |    0.0244 |     1.47 |       0.0233 |          19.5 |           8.4 |          3.8 |       6.7 |        48 |       64.5 |         44.2 |          0.838 |              0.712 |               0.857 |           0.849 |            0.854 |             0.849 |            0.831 |                 0.818 |                  0.665 |               0.692 |
|   1 | 2014-02-26 |     17 |       85 |  60 | 200 |      498.4 |           9 |     1    | 397.9 |    0.0432 |     2.59 |       0.0387 |          35.2 |          14.6 |          6.5 |       8.3 |         7 |       20.5 |         74.4 |          0.771 |              0.748 |               0.769 |           0.738 |            0.73  |             0.729 |            0.692 |                 0.673 |                  0.771 |               0.719 |

## หมายเหตุ
- [v3.1] qm_model = (9a′) guarded: ใน domain |l| ≤ max(p75|y_et|, 4R_max) ใช้หัว, นอก domain ใช้ (9a) ของ run เดียวกัน · qm_model_nog = ไม่ guard (diag) · in_domain mask อยู่ใน inst
- E_i, G2.* คำนวณจาก qm_model = (9a′) guarded ของ encoder หลัก · AUC_* = AUC ของแต่ละ arm เทียบ oracle บนคู่ทั้งหมดของ instance (sanity vs D7 final: model ≈ 0.85 SF / 0.82 Rome · trajgeom ≈ 0.81 / 0.79 · null ≈ 0.70 / 0.72)
- โจทย์ (งาน, K_i, B, เวลา, รางวัล, worker ที่สุ่ม) เหมือน v2.2 ทุกบิตเพราะ seed เดียวกัน → เทียบกับ instances/ เดิมได้โดยตรง
- arms สำหรับ E4-lite: qm_model (9a′ tfm) · qm_lstm_reach (9a′ lstm) · qm_tfm_ub (9a tfm) · qm_trajgeom (9a base_M3) · qm_null_dv · qm_persist_* · oracle (replay เท่านั้น)
