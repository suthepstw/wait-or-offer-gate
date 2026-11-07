# E5 instances — sf · E5-instances v0.3 (2026-10-22: + arm qm_model_dg duration guard ≥5, บันทึก qm_model_nog/in_domain_dg) · 2026-10-22 08:44 · sD7 v7 sha 0fe82c5d0600

instances 30 · t [0, 10, 20, 30, 40, 50] นาที · R 200 · arms ('qm_model', 'qm_model_dg', 'qm_lstm_reach', 'qm_tfm_ub', 'qm_null_dv', 'qm_persist_cont') + oracle · state จาก e5state/sf/{reach_tfm,reach_lstm} (sE5_encode)

## Sanity

- (1) ΔC(qm_model, t=0) vs instances_v3: max 4.92e-03 (เกณฑ์ ≤ 0.005) → ✅
- (2) diagnostic กรอบพิกัด: ตำแหน่งจาก state (Frame ของ E1) vs path_true (กรอบ D3) median 7.63 m, max ของ median 14.06 m — เรขาคณิต replay ใช้ path_true กรอบเดียวตลอด (ไม่ใช่ gate; E1 frame check เดิม SF 2.8 m / p90 9.4 m)
- (3) oracle(t=0) == instances_v3: 30/30 → ✅

## AUC ต่อ arm ตาม t (mean over instances · เฉพาะงานที่ยังเปิด · ขอบฟ้าสั้นลงเมื่อ t เพิ่ม)

| t_min | k | n_open | pos_rate | in_dom | in_dom_dg | AUC_qm_model | AUC_qm_model_dg | AUC_qm_lstm_reach | AUC_qm_tfm_ub | AUC_qm_null_dv | AUC_qm_persist_cont |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.000 | 14.500 | 200.000 | 0.046 | 0.679 | 0.679 | 0.771 | 0.771 | 0.761 | 0.757 | 0.709 | 0.660 |
| 10.000 | 14.500 | 196.200 | 0.044 | 0.625 | 0.622 | 0.801 | 0.801 | 0.791 | 0.795 | 0.749 | 0.702 |
| 20.000 | 14.500 | 176.700 | 0.039 | 0.557 | 0.548 | 0.832 | 0.833 | 0.823 | 0.827 | 0.789 | 0.738 |
| 30.000 | 14.500 | 142.833 | 0.033 | 0.480 | 0.463 | 0.859 | 0.862 | 0.852 | 0.859 | 0.828 | 0.773 |
| 40.000 | 14.500 | 95.233 | 0.027 | 0.389 | 0.366 | 0.893 | 0.895 | 0.890 | 0.899 | 0.874 | 0.825 |
| 50.000 | 14.500 | 46.700 | 0.017 | 0.249 | 0.201 | 0.937 | 0.940 | 0.937 | 0.942 | 0.931 | 0.892 |

**✅ ผ่าน** · ไฟล์ inst_k_tMM.npz: open (T,), st_rel_all/et_rel_all (T,), w_xy/w_v (W,2), oracle (W,T) uint8, qm_* (W,T) NaN = งานปิด
