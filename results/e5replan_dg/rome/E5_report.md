# E5 rolling re-plan — rome · E5-replan v0.4 (2026-10-22: sticky = offer ค้างเก็บฟรี / ถอนไม่คืนโควตา / ใหม่กินโควตา) · 2026-10-22 11:42

instances 30 · arms ['qm_model_dg'] · Δ = 10 นาที (6 จุดตัดสินใจ) · NSGA-II pop 80 × gen 480/จุด (per-step) · warm-start True · withdraw free · R 200 · L_i = 1 · Q_max 3 · θ 0.05/0.05 · cand 8
X* = ใกล้ ideal (1,1,1) ที่แต่ละจุด · replay ด้วย path_true · f₃* ใช้ Φ^max/T⁺ ของ oracle ที่ t=0

## timeline (adaptive) — งานสำเร็จสะสม (mean) หลัง replay ช่วง (t, t+Δ] · static = ค่าสุดท้ายในตาราง f₁* × |T|

| t | qm_model_dg |
|---|---|
| 0.0000 | 3.7700 |
| 10.0000 | 14.2000 |
| 20.0000 | 25.9000 |
| 30.0000 | 37.2300 |
| 40.0000 | 46.6000 |
| 50.0000 | 51.4000 |
