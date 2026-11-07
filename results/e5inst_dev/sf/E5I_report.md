# E5 instances — sf · E5-instances v0.3 (2026-10-22: + arm qm_model_dg duration guard ≥5, บันทึก qm_model_nog/in_domain_dg) · 2026-10-22 16:57 · sD7 v7 sha 0fe82c5d0600

instances 16 · t [0, 10, 20, 30, 40, 50] นาที · R 200 · arms ('qm_model', 'qm_model_dg', 'qm_lstm_reach', 'qm_tfm_ub', 'qm_null_dv', 'qm_persist_cont') + oracle · state จาก e5state_dev/sf/{reach_tfm,reach_lstm} (sE5_encode)

## Sanity

- (1) ΔC(qm_model, t=0) vs instances_v3: max 4.47e-03 (เกณฑ์ ≤ 0.005) → ✅
- (2) diagnostic กรอบพิกัด: ตำแหน่งจาก state (Frame ของ E1) vs path_true (กรอบ D3) median 7.16 m, max ของ median 13.35 m — เรขาคณิต replay ใช้ path_true กรอบเดียวตลอด (ไม่ใช่ gate; E1 frame check เดิม SF 2.8 m / p90 9.4 m)
- (3) oracle(t=0) == instances_v3: 16/16 → ✅

## AUC ต่อ arm ตาม t (mean over instances · เฉพาะงานที่ยังเปิด · ขอบฟ้าสั้นลงเมื่อ t เพิ่ม)

| t_min | k | n_open | pos_rate | in_dom | in_dom_dg | AUC_qm_model | AUC_qm_model_dg | AUC_qm_lstm_reach | AUC_qm_tfm_ub | AUC_qm_null_dv | AUC_qm_persist_cont |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.000 | 7.500 | 200.000 | 0.046 | 0.667 | 0.667 | 0.773 | 0.773 | 0.766 | 0.759 | 0.716 | 0.681 |
| 10.000 | 7.500 | 196.688 | 0.043 | 0.608 | 0.604 | 0.798 | 0.799 | 0.791 | 0.795 | 0.748 | 0.711 |
| 20.000 | 7.500 | 176.375 | 0.039 | 0.553 | 0.542 | 0.826 | 0.827 | 0.820 | 0.824 | 0.787 | 0.739 |
| 30.000 | 7.500 | 143.000 | 0.034 | 0.482 | 0.465 | 0.860 | 0.862 | 0.857 | 0.863 | 0.834 | 0.785 |
| 40.000 | 7.500 | 94.812 | 0.028 | 0.391 | 0.367 | 0.891 | 0.894 | 0.890 | 0.900 | 0.877 | 0.839 |
| 50.000 | 7.500 | 46.125 | 0.020 | 0.271 | 0.220 | 0.932 | 0.935 | 0.928 | 0.939 | 0.927 | 0.895 |

**✅ ผ่าน** · ไฟล์ inst_k_tMM.npz: open (T,), st_rel_all/et_rel_all (T,), w_xy/w_v (W,2), oracle (W,T) uint8, qm_* (W,T) NaN = งานปิด
