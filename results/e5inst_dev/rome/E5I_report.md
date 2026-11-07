# E5 instances — rome · E5-instances v0.3 (2026-10-22: + arm qm_model_dg duration guard ≥5, บันทึก qm_model_nog/in_domain_dg) · 2026-10-22 16:57 · sD7 v7 sha 0fe82c5d0600

instances 2 · t [0, 10, 20, 30, 40, 50] นาที · R 200 · arms ('qm_model', 'qm_model_dg', 'qm_lstm_reach', 'qm_tfm_ub', 'qm_null_dv', 'qm_persist_cont') + oracle · state จาก e5state_dev/rome/{reach_tfm,reach_lstm} (sE5_encode)

## Sanity

- (1) ΔC(qm_model, t=0) vs instances_v3: max 8.02e-04 (เกณฑ์ ≤ 0.005) → ✅
- (2) diagnostic กรอบพิกัด: ตำแหน่งจาก state (Frame ของ E1) vs path_true (กรอบ D3) median 4.53 m, max ของ median 7.93 m — เรขาคณิต replay ใช้ path_true กรอบเดียวตลอด (ไม่ใช่ gate; E1 frame check เดิม SF 2.8 m / p90 9.4 m)
- (3) oracle(t=0) == instances_v3: 2/2 → ✅

## AUC ต่อ arm ตาม t (mean over instances · เฉพาะงานที่ยังเปิด · ขอบฟ้าสั้นลงเมื่อ t เพิ่ม)

| t_min | k | n_open | pos_rate | in_dom | in_dom_dg | AUC_qm_model | AUC_qm_model_dg | AUC_qm_lstm_reach | AUC_qm_tfm_ub | AUC_qm_null_dv | AUC_qm_persist_cont |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.000 | 0.500 | 200.000 | 0.031 | 0.593 | 0.593 | 0.804 | 0.804 | 0.813 | 0.794 | 0.762 | 0.746 |
| 10.000 | 0.500 | 198.000 | 0.029 | 0.518 | 0.515 | 0.841 | 0.842 | 0.842 | 0.844 | 0.812 | 0.790 |
| 20.000 | 0.500 | 175.500 | 0.026 | 0.453 | 0.441 | 0.869 | 0.872 | 0.872 | 0.869 | 0.859 | 0.832 |
| 30.000 | 0.500 | 139.500 | 0.024 | 0.386 | 0.373 | 0.893 | 0.894 | 0.894 | 0.911 | 0.894 | 0.870 |
| 40.000 | 0.500 | 95.000 | 0.018 | 0.306 | 0.278 | 0.948 | 0.951 | 0.948 | 0.954 | 0.936 | 0.919 |
| 50.000 | 0.500 | 43.000 | 0.014 | 0.203 | 0.164 | 0.977 | 0.976 | 0.977 | 0.976 | 0.960 | 0.954 |

**✅ ผ่าน** · ไฟล์ inst_k_tMM.npz: open (T,), st_rel_all/et_rel_all (T,), w_xy/w_v (W,2), oracle (W,T) uint8, qm_* (W,T) NaN = งานปิด
