# E1 report — sf (pilot) · E1 v3.1 (2026-10-20, estimator (9a') + support guard) · 2026-10-20 14:46

**qm_model = (9a′) reach_tfm/seed2** (D7reach_dev, sha ตรง) · qm_lstm_reach = reach_lstm/seed0 · qm_tfm_ub = (9a) ของ reach_tfm (qm_calib.json) · qm_trajgeom = (9a) base_M3/seed1 (D7_final) · sD7 v7 sha `0fe82c5d0600` · split test_final 2008-06-07..2008-06-08 · cycle ใช้ได้ 46 → เลือก 30 · สร้าง 3

พารามิเตอร์ (เหมือน v2.2): |W| 60 · |T| 200 · cycle 60 นาที · R [150.0, 200.0, 250.0] (หลัก 200) · placement clustered (centre mixed 0.85, jitter 500 m, α 1) · cell_m 500 · β 0.65 · M_i 2 · Q_max 3 · K rule density · master seed 12345

## Gate G2 (ตรวจใน E1)

| ข้อ | ผล | ค่า |
|---|---|---|
| G2.1 (% worker ไม่มีงานเอื้อม < 40) | ✅ | 0.6 % |
| G2.2 (% task E_i < 0.5 < 20) | ❌ | 28.3 % |
| G2.2b (% task E_i < K_i ≤ 70) | ✅ | 40.0 % |
| frame check (median |Y − Δx_T| ≤ 5 m) | ✅ | 2.78 m (p90 9.4, n=20151) |
| sanity A: guarded (9a′) ≥ (9a) same encoder, overall | ✅ | model 0.804 · tfm_ub 0.785 · nog 0.763 · in-domain 65 % ของคู่ |
| sanity B: in-domain (9a′) > (9a) (ตรงกับ D7) | ✅ | in-dom (9a′) 0.735 vs (9a) 0.707 |
| sanity C: model > null > persistence | ✅ | qm_model 0.804 · qm_model_nog 0.763 · qm_lstm_reach 0.797 · qm_tfm_ub 0.785 · qm_lstm_ub 0.782 · qm_trajgeom 0.783 · qm_null_dv 0.754 · qm_persist_cont 0.703 |

## ต่อ instance

|   k | date       |   hour |   n_cand |   W |   T |   area_km2 |   n_centres |   K_mean |     B |   qm_mean |   E_mean |   pos_oracle |   n_i(θ=0.02) |   n_i(θ=0.05) |   n_i(θ=0.1) |   G21_pct |   G22_pct |   G22b_pct |   in_dom_pct |   AUC_qm_model |   AUC_qm_model_nog |   AUC_qm_lstm_reach |   AUC_qm_tfm_ub |   AUC_qm_lstm_ub |   AUC_qm_trajgeom |   AUC_qm_null_dv |   AUC_qm_persist_cont |   AUC_in_dom_model_nog |   AUC_in_dom_tfm_ub |
|----:|:-----------|-------:|---------:|----:|----:|-----------:|------------:|---------:|------:|----------:|---------:|-------------:|--------------:|--------------:|-------------:|----------:|----------:|-----------:|-------------:|---------------:|-------------------:|--------------------:|----------------:|-----------------:|------------------:|-----------------:|----------------------:|-----------------------:|--------------------:|
|   0 | 2008-06-08 |     18 |      202 |  60 | 200 |      399.5 |           4 |     1.36 | 440.8 |    0.0314 |     1.88 |       0.0301 |          30   |          14.6 |          4.8 |       0   |      32.5 |       51.5 |         61.1 |          0.8   |              0.773 |               0.791 |           0.774 |            0.782 |             0.784 |            0.765 |                 0.733 |                  0.726 |               0.686 |
|   1 | 2008-06-07 |     21 |      258 |  60 | 200 |      317.7 |           9 |     1.11 | 399.2 |    0.0435 |     2.61 |       0.052  |          31.7 |          20.4 |         10.1 |       1.7 |      35   |       42   |         61.2 |          0.842 |              0.807 |               0.829 |           0.824 |            0.817 |             0.82  |            0.772 |                 0.687 |                  0.77  |               0.739 |
|   2 | 2008-06-08 |     13 |      207 |  60 | 200 |      363.9 |           9 |     1.08 | 393.4 |    0.0457 |     2.74 |       0.0453 |          37.9 |          21.5 |          8.7 |       0   |      17.5 |       26.5 |         73.8 |          0.769 |              0.71  |               0.77  |           0.756 |            0.748 |             0.746 |            0.725 |                 0.688 |                  0.708 |               0.696 |

## หมายเหตุ
- [v3.1] qm_model = (9a′) guarded: ใน domain |l| ≤ max(p75|y_et|, 4R_max) ใช้หัว, นอก domain ใช้ (9a) ของ run เดียวกัน · qm_model_nog = ไม่ guard (diag) · in_domain mask อยู่ใน inst
- E_i, G2.* คำนวณจาก qm_model = (9a′) guarded ของ encoder หลัก · AUC_* = AUC ของแต่ละ arm เทียบ oracle บนคู่ทั้งหมดของ instance (sanity vs D7 final: model ≈ 0.85 SF / 0.82 Rome · trajgeom ≈ 0.81 / 0.79 · null ≈ 0.70 / 0.72)
- โจทย์ (งาน, K_i, B, เวลา, รางวัล, worker ที่สุ่ม) เหมือน v2.2 ทุกบิตเพราะ seed เดียวกัน → เทียบกับ instances/ เดิมได้โดยตรง
- arms สำหรับ E4-lite: qm_model (9a′ tfm) · qm_lstm_reach (9a′ lstm) · qm_tfm_ub (9a tfm) · qm_trajgeom (9a base_M3) · qm_null_dv · qm_persist_* · oracle (replay เท่านั้น)
