# E5 rolling re-plan — sf · E5-replan v0.6.1 (2026-10-22: v0.6 + ProblemT.decode vectorised — ผลเท่าเดิมทุกบิต) · 2026-10-22 15:23

instances 30 · arms ['qm_model', 'oracle'] · Δ = 10 นาที (6 จุดตัดสินใจ) · optimizer **mopso_q** pop 80 × gen 480/จุด (per-step) · warm-start True · withdraw free · R 200 · L_i = 1 · Q_max 3 · θ 0.05/0.05 · cand 8
X* = ใกล้ ideal (1,1,1) ที่แต่ละจุด · replay ด้วย path_true · f₃* ใช้ Φ^max/T⁺ ของ oracle ที่ t=0

## f₁* static vs adaptive (mean over instances · Δ paired · bootstrap 95 % CI · Wilcoxon)

| arm | label | n | f1_static | f1_adaptive | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| qm_model | (9a′) Transformer [main] | 30 | 0.0670 | 0.2408 | 0.1738 | 0.1592 | 0.1873 | 0.0000 | 30 | 0 |
| oracle | oracle (y จริง) | 30 | 0.5367 | 0.5702 | 0.0335 | 0.0247 | 0.0433 | 0.0000 | 27 | 2 |

## Gates

- G5.1 adaptive ≥ static สำหรับ qm_model (ตัวประมาณที่มีสารสนเทศ — ถ้าได้ข้อมูลเพิ่มแล้วแย่ลง = ตรวจ replay/โควตา): ✅ · baseline ไร้สารสนเทศ (persistence/null) อาจแย่ลงได้จากการถอน offer — รายงานตามจริง ไม่ใช่ gate
- G5.2 |Δ oracle| ≤ 0.02: ❌ (Δ = +0.0335)

## Value of Prediction ภายใต้ adaptive — Δf₁* vs qm_model (paired)

| arm | n | mean | ref_mean | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|
| oracle | 30 | 0.5702 | 0.2408 | 0.3293 | 0.3070 | 0.3523 | 0.0000 | 30 | 0 |

## timeline (adaptive) — งานสำเร็จสะสม (mean) หลัง replay ช่วง (t, t+Δ] · static = ค่าสุดท้ายในตาราง f₁* × |T|

| t | oracle | qm_model |
|---|---|---|
| 0.0000 | 11.6000 | 3.0000 |
| 10.0000 | 33.9700 | 10.7000 |
| 20.0000 | 57.7300 | 22.0000 |
| 30.0000 | 82.6300 | 32.3700 |
| 40.0000 | 103.5700 | 42.4000 |
| 50.0000 | 114.0300 | 48.1700 |
