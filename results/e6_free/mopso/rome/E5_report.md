# E5 rolling re-plan — rome · E5-replan v0.6.1 (2026-10-22: v0.6 + ProblemT.decode vectorised — ผลเท่าเดิมทุกบิต) · 2026-10-22 15:38

instances 30 · arms ['qm_model', 'oracle'] · Δ = 10 นาที (6 จุดตัดสินใจ) · optimizer **mopso** pop 80 × gen 480/จุด (per-step) · warm-start True · withdraw free · R 200 · L_i = 1 · Q_max 3 · θ 0.05/0.05 · cand 8
X* = ใกล้ ideal (1,1,1) ที่แต่ละจุด · replay ด้วย path_true · f₃* ใช้ Φ^max/T⁺ ของ oracle ที่ t=0

## f₁* static vs adaptive (mean over instances · Δ paired · bootstrap 95 % CI · Wilcoxon)

| arm | label | n | f1_static | f1_adaptive | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| qm_model | (9a′) Transformer [main] | 30 | 0.0672 | 0.2313 | 0.1642 | 0.1488 | 0.1793 | 0.0000 | 30 | 0 |
| oracle | oracle (y จริง) | 30 | 0.4583 | 0.4698 | 0.0115 | 0.0063 | 0.0167 | 0.0004 | 21 | 2 |

## Gates

- G5.1 adaptive ≥ static สำหรับ qm_model (ตัวประมาณที่มีสารสนเทศ — ถ้าได้ข้อมูลเพิ่มแล้วแย่ลง = ตรวจ replay/โควตา): ✅ · baseline ไร้สารสนเทศ (persistence/null) อาจแย่ลงได้จากการถอน offer — รายงานตามจริง ไม่ใช่ gate
- G5.2 |Δ oracle| ≤ 0.02: ✅ (Δ = +0.0115)

## Value of Prediction ภายใต้ adaptive — Δf₁* vs qm_model (paired)

| arm | n | mean | ref_mean | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|
| oracle | 30 | 0.4698 | 0.2313 | 0.2385 | 0.2175 | 0.2588 | 0.0000 | 30 | 0 |

## timeline (adaptive) — งานสำเร็จสะสม (mean) หลัง replay ช่วง (t, t+Δ] · static = ค่าสุดท้ายในตาราง f₁* × |T|

| t | oracle | qm_model |
|---|---|---|
| 0.0000 | 10.4700 | 3.3700 |
| 10.0000 | 28.1700 | 12.1300 |
| 20.0000 | 49.3700 | 22.8000 |
| 30.0000 | 70.2300 | 33.4000 |
| 40.0000 | 86.6000 | 41.9300 |
| 50.0000 | 93.9700 | 46.2700 |
