# reach-eval รวม — sf · split dev · R = 200 m · ref = reach_tfm · คู่ชุดเดียวกัน (CRN_SEED 12345) → ΔAUC แบบ paired, CI = cluster bootstrap 300 รอบ (✔ = CI > 0 ทั้งช่วง)

| run | variant | seeds | AUC@200 | cluster-CI | AUPRC | ECE | AUC hard | ΔAUC vs reach_tfm/qm_reach_cal (paired) |
|---|---|---|---|---|---|---|---|---|
| reach_tfm | qm_ub_cal | 3 | 0.8369 ± 0.0003 | [0.8244, 0.8468] | 0.101 | 0.0017 | 0.574 | -0.0294 [-0.0371, -0.0226] ✘ |
| reach_tfm | qm_reach_cal | 3 | 0.8643 ± 0.0006 | [0.8587, 0.8745] | 0.113 | 0.0022 | 0.619 | — |
| reach_tfm_d64 | qm_ub_cal | 3 | 0.8369 ± 0.0011 | [0.8242, 0.8466] | 0.100 | 0.0017 | 0.575 | -0.0296 [-0.0374, -0.0231] ✘ |
| reach_tfm_d64 | qm_reach_cal | 3 | 0.8648 ± 0.0013 | [0.8587, 0.8741] | 0.113 | 0.0020 | 0.616 | +0.0001 [-0.0020, +0.0021] |

อ่านผล: (9a′) ชนะ (9a) อย่างมีนัยเมื่อ ΔAUC paired ของ qm_reach_cal มี CI ไม่คร่อม 0 · ECE ต้อง ≤ ของ qm_ub_cal + 0.001 · ดู AUC hard ประกอบ (งานใกล้เส้นทาง = สิ่งที่ optimizer ต้องแยกจริง)