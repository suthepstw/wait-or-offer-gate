# reach-eval รวม — rome · split dev · R = 200 m · ref = reach_lstm · คู่ชุดเดียวกัน (CRN_SEED 12345) → ΔAUC แบบ paired, CI = cluster bootstrap 300 รอบ (✔ = CI > 0 ทั้งช่วง)

| run | variant | seeds | AUC@200 | cluster-CI | AUPRC | ECE | AUC hard | ΔAUC vs reach_lstm/qm_reach_cal (paired) |
|---|---|---|---|---|---|---|---|---|
| reach_tfm | qm_ub_cal | 3 | 0.7994 ± 0.0005 | [0.7856, 0.8134] | 0.098 | 0.0026 | 0.509 | -0.0302 [-0.0427, -0.0192] ✘ |
| reach_tfm | qm_reach_cal | 3 | 0.8306 ± 0.0009 | [0.8210, 0.8476] | 0.176 | 0.0016 | 0.636 | +0.0035 [+0.0002, +0.0071] ✔ |
| reach_lstm | qm_ub_cal | 3 | 0.7927 ± 0.0007 | [0.7778, 0.8078] | 0.095 | 0.0023 | 0.514 | -0.0363 [-0.0494, -0.0237] ✘ |
| reach_lstm | qm_reach_cal | 3 | 0.8264 ± 0.0008 | [0.8181, 0.8453] | 0.171 | 0.0018 | 0.627 | — |

อ่านผล: (9a′) ชนะ (9a) อย่างมีนัยเมื่อ ΔAUC paired ของ qm_reach_cal มี CI ไม่คร่อม 0 · ECE ต้อง ≤ ของ qm_ub_cal + 0.001 · ดู AUC hard ประกอบ (งานใกล้เส้นทาง = สิ่งที่ optimizer ต้องแยกจริง)