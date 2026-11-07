# reach-eval รวม — sf · split dev · R = 200 m · ref = reach_lstm · คู่ชุดเดียวกัน (CRN_SEED 12345) → ΔAUC แบบ paired, CI = cluster bootstrap 300 รอบ (✔ = CI > 0 ทั้งช่วง)

| run | variant | seeds | AUC@200 | cluster-CI | AUPRC | ECE | AUC hard | ΔAUC vs reach_lstm/qm_reach_cal (paired) |
|---|---|---|---|---|---|---|---|---|
| reach_tfm | qm_ub_cal | 3 | 0.8369 ± 0.0003 | [0.8244, 0.8468] | 0.101 | 0.0017 | 0.574 | -0.0244 [-0.0315, -0.0178] ✘ |
| reach_tfm | qm_reach_cal | 3 | 0.8643 ± 0.0006 | [0.8587, 0.8745] | 0.113 | 0.0022 | 0.619 | +0.0050 [+0.0011, +0.0085] ✔ |
| reach_lstm | qm_ub_cal | 3 | 0.8297 ± 0.0015 | [0.8174, 0.8402] | 0.097 | 0.0013 | 0.575 | -0.0315 [-0.0390, -0.0248] ✘ |
| reach_lstm | qm_reach_cal | 3 | 0.8588 ± 0.0002 | [0.8525, 0.8691] | 0.104 | 0.0015 | 0.599 | — |

อ่านผล: (9a′) ชนะ (9a) อย่างมีนัยเมื่อ ΔAUC paired ของ qm_reach_cal มี CI ไม่คร่อม 0 · ECE ต้อง ≤ ของ qm_ub_cal + 0.001 · ดู AUC hard ประกอบ (งานใกล้เส้นทาง = สิ่งที่ optimizer ต้องแยกจริง)