# E5 rolling re-plan — sf · E5-replan v0.6.1 (2026-10-22: v0.6 + ProblemT.decode vectorised — ผลเท่าเดิมทุกบิต) · 2026-10-22 15:44

instances 30 · arms ['qm_model', 'qm_tfm_ub', 'qm_null_dv', 'oracle'] · Δ = 10 นาที (6 จุดตัดสินใจ) · optimizer **nsga2** pop 80 × gen 480/จุด (per-step) · warm-start False · withdraw free · R 200 · L_i = 1 · Q_max 3 · θ 0.05/0.05 · cand 8
X* = ใกล้ ideal (1,1,1) ที่แต่ละจุด · replay ด้วย path_true · f₃* ใช้ Φ^max/T⁺ ของ oracle ที่ t=0

## timeline (adaptive) — งานสำเร็จสะสม (mean) หลัง replay ช่วง (t, t+Δ] · static = ค่าสุดท้ายในตาราง f₁* × |T|

| t | oracle | qm_model | qm_null_dv | qm_tfm_ub |
|---|---|---|---|---|
| 0.0000 | 11.7300 | 3.3300 | 3.1700 | 3.8000 |
| 10.0000 | 34.5700 | 12.9300 | 13.7700 | 13.4000 |
| 20.0000 | 60.1300 | 25.1700 | 27.7000 | 25.3300 |
| 30.0000 | 84.8700 | 36.9700 | 39.0300 | 36.1700 |
| 40.0000 | 107.8300 | 47.5700 | 50.8700 | 45.9300 |
| 50.0000 | 117.9700 | 53.6300 | 56.6700 | 51.1700 |
