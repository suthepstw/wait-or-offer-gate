# reach-eval รวม — sf · split dev · R = 200 m · ref = reach_tfm · คู่ชุดเดียวกัน (CRN_SEED 12345) → ΔAUC แบบ paired, CI = cluster bootstrap 300 รอบ (✔ = CI > 0 ทั้งช่วง)

| run | variant | seeds | AUC@200 | cluster-CI | AUPRC | ECE | AUC hard | ΔAUC vs reach_tfm/qm_reach_cal (paired) |
|---|---|---|---|---|---|---|---|---|
| reach_tfm | qm_ub_cal | 3 | 0.8369 ± 0.0003 | [0.8244, 0.8468] | 0.101 | 0.0017 | 0.574 | -0.0294 [-0.0383, -0.0215] ✘ |
| reach_tfm | qm_reach_cal | 3 | 0.8643 ± 0.0006 | [0.8587, 0.8745] | 0.113 | 0.0022 | 0.619 | — |
| reach_tfm | qm_reach_cal_g | 3 | 0.8652 ± 0.0006 | [0.8591, 0.8760] | 0.114 | 0.0021 | 0.612 | +0.0008 [-0.0004, +0.0020] |
| reach_tfm_occstop | qm_ub_cal | 3 | 0.8402 ± 0.0010 | [0.8279, 0.8500] | 0.103 | 0.0013 | 0.573 | -0.0260 [-0.0340, -0.0185] ✘ |
| reach_tfm_occstop | qm_reach_cal | 3 | 0.8649 ± 0.0009 | [0.8594, 0.8755] | 0.114 | 0.0016 | 0.618 | +0.0011 [-0.0012, +0.0034] |
| reach_tfm_occstop | qm_reach_cal_g | 3 | 0.8656 ± 0.0009 | [0.8599, 0.8768] | 0.114 | 0.0014 | 0.611 | +0.0017 [-0.0013, +0.0044] |

อ่านผล: (9a′) ชนะ (9a) อย่างมีนัยเมื่อ ΔAUC paired ของ qm_reach_cal มี CI ไม่คร่อม 0 · ECE ต้อง ≤ ของ qm_ub_cal + 0.001 · ดู AUC hard ประกอบ (งานใกล้เส้นทาง = สิ่งที่ optimizer ต้องแยกจริง)