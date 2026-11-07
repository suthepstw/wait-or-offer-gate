# E5 rolling re-plan — rome · E5-replan v0.6.1 (2026-10-22: v0.6 + ProblemT.decode vectorised — ผลเท่าเดิมทุกบิต) · 2026-10-22 15:52

instances 30 · arms ['qm_model', 'oracle'] · Δ = 10 นาที (6 จุดตัดสินใจ) · optimizer **nsga2** pop 80 × gen 480/จุด (per-step) · warm-start True · withdraw free · R 200 · L_i = 1 · Q_max 3 · θ 0.05/0.05 · cand 8
X* = ใกล้ ideal (1,1,1) ที่แต่ละจุด · replay ด้วย path_true · f₃* ใช้ Φ^max/T⁺ ของ oracle ที่ t=0

## f₁* static vs adaptive (mean over instances · Δ paired · bootstrap 95 % CI · Wilcoxon)

| arm | label | n | f1_static | f1_adaptive | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| qm_model | (9a′) Transformer [main] | 30 | 0.0738 | 0.2468 | 0.1730 | 0.1567 | 0.1898 | 0.0000 | 30 | 0 |
| oracle | oracle (y จริง) | 30 | 0.4653 | 0.4818 | 0.0165 | 0.0127 | 0.0203 | 0.0000 | 27 | 0 |

## Gates

- G5.1 adaptive ≥ static สำหรับ qm_model (ตัวประมาณที่มีสารสนเทศ — ถ้าได้ข้อมูลเพิ่มแล้วแย่ลง = ตรวจ replay/โควตา): ✅ · baseline ไร้สารสนเทศ (persistence/null) อาจแย่ลงได้จากการถอน offer — รายงานตามจริง ไม่ใช่ gate
- G5.2 |Δ oracle| ≤ 0.02: ✅ (Δ = +0.0165)

## Value of Prediction ภายใต้ adaptive — Δf₁* vs qm_model (paired)

| arm | n | mean | ref_mean | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|
| oracle | 30 | 0.4818 | 0.2468 | 0.2350 | 0.2125 | 0.2572 | 0.0000 | 30 | 0 |

## timeline (adaptive) — งานสำเร็จสะสม (mean) หลัง replay ช่วง (t, t+Δ] · static = ค่าสุดท้ายในตาราง f₁* × |T|

| t | oracle | qm_model |
|---|---|---|
| 0.0000 | 10.3700 | 3.7700 |
| 10.0000 | 28.7000 | 13.9000 |
| 20.0000 | 50.6700 | 25.2700 |
| 30.0000 | 72.2000 | 35.9000 |
| 40.0000 | 89.0300 | 44.9700 |
| 50.0000 | 96.3700 | 49.3700 |
