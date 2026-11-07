# E5 instances — sf · E5-instances v0.3 (2026-10-22: + arm qm_model_dg duration guard ≥5, บันทึก qm_model_nog/in_domain_dg) · 2026-10-22 19:20 · sD7 v7 sha 0fe82c5d0600

instances 30 · t [0, 10, 20, 30, 40, 50] นาที · R 200 · arms ('qm_model', 'qm_model_dg', 'qm_lstm_reach', 'qm_tfm_ub', 'qm_null_dv', 'qm_persist_cont') + oracle · state จาก e5state_devsplit/sf/{reach_tfm,reach_lstm} (sE5_encode)

## Sanity

- (1) ΔC(qm_model, t=0) vs instances_v3: max 6.23e-03 (เกณฑ์ ≤ 0.007) → ✅
- (2) diagnostic กรอบพิกัด: ตำแหน่งจาก state (Frame ของ E1) vs path_true (กรอบ D3) median 7.32 m, max ของ median 15.77 m — เรขาคณิต replay ใช้ path_true กรอบเดียวตลอด (ไม่ใช่ gate; E1 frame check เดิม SF 2.8 m / p90 9.4 m)
- (3) oracle(t=0) == instances_v3: 30/30 → ✅

## AUC ต่อ arm ตาม t (mean over instances · เฉพาะงานที่ยังเปิด · ขอบฟ้าสั้นลงเมื่อ t เพิ่ม)

| t_min | k | n_open | pos_rate | in_dom | in_dom_dg | AUC_qm_model | AUC_qm_model_dg | AUC_qm_lstm_reach | AUC_qm_tfm_ub | AUC_qm_null_dv | AUC_qm_persist_cont |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.000 | 14.500 | 200.000 | 0.050 | 0.645 | 0.645 | 0.776 | 0.776 | 0.768 | 0.754 | 0.704 | 0.658 |
| 10.000 | 14.500 | 196.200 | 0.048 | 0.599 | 0.594 | 0.803 | 0.804 | 0.796 | 0.790 | 0.744 | 0.693 |
| 20.000 | 14.500 | 176.700 | 0.043 | 0.550 | 0.539 | 0.836 | 0.837 | 0.830 | 0.827 | 0.788 | 0.738 |
| 30.000 | 14.500 | 142.833 | 0.036 | 0.481 | 0.462 | 0.867 | 0.868 | 0.861 | 0.861 | 0.828 | 0.774 |
| 40.000 | 14.500 | 95.233 | 0.029 | 0.403 | 0.377 | 0.895 | 0.898 | 0.892 | 0.897 | 0.877 | 0.832 |
| 50.000 | 14.500 | 46.700 | 0.018 | 0.262 | 0.207 | 0.941 | 0.945 | 0.941 | 0.949 | 0.943 | 0.908 |

**✅ ผ่าน** · ไฟล์ inst_k_tMM.npz: open (T,), st_rel_all/et_rel_all (T,), w_xy/w_v (W,2), oracle (W,T) uint8, qm_* (W,T) NaN = งานปิด
