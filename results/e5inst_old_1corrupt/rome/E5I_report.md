# E5 instances — rome · E5-instances v0.2 (2026-10-21: x_j(t) จาก path_true กรอบเดียว) · 2026-10-21 20:44 · sD7 v7 sha 0fe82c5d0600

instances 30 · t [0, 10, 20, 30, 40, 50] นาที · R 200 · arms ('qm_model', 'qm_lstm_reach', 'qm_tfm_ub', 'qm_null_dv', 'qm_persist_cont') + oracle · state จาก e5state/rome/{reach_tfm,reach_lstm} (sE5_encode)

## Sanity

- (1) ΔC(qm_model, t=0) vs instances_v3: max 2.99e-03 (เกณฑ์ ≤ 0.005) → ✅
- (2) diagnostic กรอบพิกัด: ตำแหน่งจาก state (Frame ของ E1) vs path_true (กรอบ D3) median 5.12 m, max ของ median 10.25 m — เรขาคณิต replay ใช้ path_true กรอบเดียวตลอด (ไม่ใช่ gate; E1 frame check เดิม SF 2.8 m / p90 9.4 m)
- (3) oracle(t=0) == instances_v3: 30/30 → ✅

## AUC ต่อ arm ตาม t (mean over instances · เฉพาะงานที่ยังเปิด · ขอบฟ้าสั้นลงเมื่อ t เพิ่ม)

| t_min | k | n_open | pos_rate | in_dom | AUC_qm_model | AUC_qm_lstm_reach | AUC_qm_tfm_ub | AUC_qm_null_dv | AUC_qm_persist_cont |
|---|---|---|---|---|---|---|---|---|---|
| 0.000 | 14.500 | 200.000 | 0.029 | 0.506 | 0.806 | 0.805 | 0.799 | 0.769 | 0.745 |
| 10.000 | 14.500 | 196.500 | 0.028 | 0.465 | 0.841 | 0.840 | 0.839 | 0.813 | 0.786 |
| 20.000 | 14.500 | 178.033 | 0.025 | 0.405 | 0.868 | 0.867 | 0.872 | 0.850 | 0.817 |
| 30.000 | 14.500 | 142.633 | 0.022 | 0.341 | 0.896 | 0.895 | 0.901 | 0.880 | 0.853 |
| 40.000 | 14.500 | 94.500 | 0.017 | 0.275 | 0.924 | 0.922 | 0.935 | 0.921 | 0.896 |
| 50.000 | 14.500 | 47.200 | 0.011 | 0.165 | 0.966 | 0.965 | 0.971 | 0.963 | 0.949 |

**✅ ผ่าน** · ไฟล์ inst_k_tMM.npz: open (T,), st_rel_all/et_rel_all (T,), w_xy/w_v (W,2), oracle (W,T) uint8, qm_* (W,T) NaN = งานปิด
