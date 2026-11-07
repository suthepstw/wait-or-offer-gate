# E5 instances — rome · E5-instances v0.3 (2026-10-22: + arm qm_model_dg duration guard ≥5, บันทึก qm_model_nog/in_domain_dg) · 2026-10-22 17:24 · sD7 v7 sha 0fe82c5d0600

instances 20 · t [0, 10, 20, 30, 40, 50] นาที · R 200 · arms ('qm_model', 'qm_model_dg', 'qm_lstm_reach', 'qm_tfm_ub', 'qm_null_dv', 'qm_persist_cont') + oracle · state จาก e5state_devsplit/rome/{reach_tfm,reach_lstm} (sE5_encode)

## Sanity

- (1) ΔC(qm_model, t=0) vs instances_v3: max 1.64e-03 (เกณฑ์ ≤ 0.005) → ✅
- (2) diagnostic กรอบพิกัด: ตำแหน่งจาก state (Frame ของ E1) vs path_true (กรอบ D3) median 4.18 m, max ของ median 8.97 m — เรขาคณิต replay ใช้ path_true กรอบเดียวตลอด (ไม่ใช่ gate; E1 frame check เดิม SF 2.8 m / p90 9.4 m)
- (3) oracle(t=0) == instances_v3: 20/20 → ✅

## AUC ต่อ arm ตาม t (mean over instances · เฉพาะงานที่ยังเปิด · ขอบฟ้าสั้นลงเมื่อ t เพิ่ม)

| t_min | k | n_open | pos_rate | in_dom | in_dom_dg | AUC_qm_model | AUC_qm_model_dg | AUC_qm_lstm_reach | AUC_qm_tfm_ub | AUC_qm_null_dv | AUC_qm_persist_cont |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.000 | 9.500 | 200.000 | 0.029 | 0.533 | 0.533 | 0.817 | 0.817 | 0.815 | 0.807 | 0.769 | 0.753 |
| 10.000 | 9.500 | 196.400 | 0.028 | 0.485 | 0.481 | 0.846 | 0.846 | 0.847 | 0.844 | 0.811 | 0.787 |
| 20.000 | 9.500 | 178.650 | 0.025 | 0.422 | 0.413 | 0.876 | 0.877 | 0.874 | 0.876 | 0.847 | 0.820 |
| 30.000 | 9.500 | 143.200 | 0.022 | 0.357 | 0.344 | 0.899 | 0.900 | 0.898 | 0.902 | 0.878 | 0.853 |
| 40.000 | 9.500 | 96.200 | 0.018 | 0.293 | 0.271 | 0.925 | 0.926 | 0.925 | 0.930 | 0.910 | 0.882 |
| 50.000 | 9.500 | 48.350 | 0.012 | 0.180 | 0.137 | 0.960 | 0.962 | 0.960 | 0.961 | 0.950 | 0.933 |

**✅ ผ่าน** · ไฟล์ inst_k_tMM.npz: open (T,), st_rel_all/et_rel_all (T,), w_xy/w_v (W,2), oracle (W,T) uint8, qm_* (W,T) NaN = งานปิด
