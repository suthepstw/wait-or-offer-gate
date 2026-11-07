# E5 rolling re-plan — sf · E5-replan v0.6.1 (2026-10-22: v0.6 + ProblemT.decode vectorised — ผลเท่าเดิมทุกบิต) · 2026-10-22 15:45

instances 30 · arms ['qm_model', 'qm_tfm_ub', 'qm_null_dv', 'oracle'] · Δ = 10 นาที (6 จุดตัดสินใจ) · optimizer **nsga2** pop 80 × gen 480/จุด (per-step) · warm-start True · withdraw sticky · R 200 · L_i = 1 · Q_max 3 · θ 0.05/0.05 · cand 8
X* = ใกล้ ideal (1,1,1) ที่แต่ละจุด · replay ด้วย path_true · f₃* ใช้ Φ^max/T⁺ ของ oracle ที่ t=0

## timeline (adaptive) — งานสำเร็จสะสม (mean) หลัง replay ช่วง (t, t+Δ] · static = ค่าสุดท้ายในตาราง f₁* × |T|

| t | oracle | qm_model | qm_null_dv | qm_tfm_ub |
|---|---|---|---|---|
| 0.0000 | 11.7300 | 3.3300 | 3.1700 | 3.8000 |
| 10.0000 | 34.6300 | 7.0300 | 8.0000 | 9.2300 |
| 20.0000 | 59.8700 | 9.4700 | 10.7300 | 11.4700 |
| 30.0000 | 85.0000 | 12.1000 | 12.0300 | 13.1300 |
| 40.0000 | 108.1300 | 13.6700 | 12.9700 | 14.2000 |
| 50.0000 | 118.4000 | 14.2700 | 13.4000 | 14.6300 |
