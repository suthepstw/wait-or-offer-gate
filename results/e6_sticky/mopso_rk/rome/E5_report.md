# E5 rolling re-plan — rome · E5-replan v0.6.1 (2026-10-22: v0.6 + ProblemT.decode vectorised — ผลเท่าเดิมทุกบิต) · 2026-10-22 14:44

instances 30 · arms ['qm_model', 'oracle'] · Δ = 10 นาที (6 จุดตัดสินใจ) · optimizer **mopso_rk** pop 80 × gen 480/จุด (per-step) · warm-start True · withdraw sticky · R 200 · L_i = 1 · Q_max 3 · θ 0.05/0.05 · cand 8
X* = ใกล้ ideal (1,1,1) ที่แต่ละจุด · replay ด้วย path_true · f₃* ใช้ Φ^max/T⁺ ของ oracle ที่ t=0

## f₁* static vs adaptive (mean over instances · Δ paired · bootstrap 95 % CI · Wilcoxon)

| arm | label | n | f1_static | f1_adaptive | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| qm_model | (9a′) Transformer [main] | 30 | 0.0743 | 0.0722 | -0.0022 | -0.0063 | 0.0023 | 0.3377 | 12 | 16 |
| oracle | oracle (y จริง) | 30 | 0.4563 | 0.4700 | 0.0137 | 0.0095 | 0.0183 | 0.0000 | 24 | 0 |

## Gates

- G5.1 adaptive ≥ static สำหรับ qm_model (ตัวประมาณที่มีสารสนเทศ — ถ้าได้ข้อมูลเพิ่มแล้วแย่ลง = ตรวจ replay/โควตา): ❌ · baseline ไร้สารสนเทศ (persistence/null) อาจแย่ลงได้จากการถอน offer — รายงานตามจริง ไม่ใช่ gate
- G5.2 |Δ oracle| ≤ 0.02: ✅ (Δ = +0.0137)

## Value of Prediction ภายใต้ adaptive — Δf₁* vs qm_model (paired)

| arm | n | mean | ref_mean | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|
| oracle | 30 | 0.4700 | 0.0722 | 0.3978 | 0.3635 | 0.4315 | 0.0000 | 30 | 0 |

## timeline (adaptive) — งานสำเร็จสะสม (mean) หลัง replay ช่วง (t, t+Δ] · static = ค่าสุดท้ายในตาราง f₁* × |T|

| t | oracle | qm_model |
|---|---|---|
| 0.0000 | 9.9700 | 3.9300 |
| 10.0000 | 28.0300 | 7.6300 |
| 20.0000 | 50.6700 | 9.9300 |
| 30.0000 | 71.3000 | 12.3000 |
| 40.0000 | 87.1300 | 13.7700 |
| 50.0000 | 94.0000 | 14.4300 |
