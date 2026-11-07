# E5 rolling re-plan — sf · E5-replan v0.7.1 (2026-10-23: --shrink → C̃ = min(1, ρ(h)·C) ตาม rho_used ของ rho.json (A1 แก้ไข: ρ ไม่ตัดที่ 1) + --inf-frac (A2) · ไม่ใส่ flag = เดิมทุกบิต) · 2026-10-23 09:05

instances 30 · arms ['qm_model', 'oracle'] · Δ = 10 นาที (6 จุดตัดสินใจ) · optimizer **nsga2** pop 80 × gen 480/จุด (per-step) · warm-start True · withdraw free · R 200 · L_i = 1 · Q_max 3 · θ 0.05/0.05 · cand 8
X* = ใกล้ ideal (1,1,1) ที่แต่ละจุด · replay ด้วย path_true · f₃* ใช้ Φ^max/T⁺ ของ oracle ที่ t=0

## f₁* static vs adaptive (mean over instances · Δ paired · bootstrap 95 % CI · Wilcoxon)

| arm | label | n | f1_static | f1_adaptive | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| qm_model | (9a′) Transformer [main] | 30 | 0.0770 | 0.2857 | 0.2087 | 0.1938 | 0.2250 | 0.0000 | 30 | 0 |
| oracle | oracle (y จริง) | 30 | 0.5915 | 0.6145 | 0.0230 | 0.0197 | 0.0267 | 0.0000 | 30 | 0 |

## Gates

- G5.1 adaptive ≥ static สำหรับ qm_model (ตัวประมาณที่มีสารสนเทศ — ถ้าได้ข้อมูลเพิ่มแล้วแย่ลง = ตรวจ replay/โควตา): ✅ · baseline ไร้สารสนเทศ (persistence/null) อาจแย่ลงได้จากการถอน offer — รายงานตามจริง ไม่ใช่ gate
- G5.2 |Δ oracle| ≤ 0.02: ❌ (Δ = +0.0230)

## Value of Prediction ภายใต้ adaptive — Δf₁* vs qm_model (paired)

| arm | n | mean | ref_mean | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|
| oracle | 30 | 0.6145 | 0.2857 | 0.3288 | 0.3023 | 0.3557 | 0.0000 | 30 | 0 |

## timeline (adaptive) — งานสำเร็จสะสม (mean) หลัง replay ช่วง (t, t+Δ] · static = ค่าสุดท้ายในตาราง f₁* × |T|

| t | oracle | qm_model |
|---|---|---|
| 0.0000 | 12.7300 | 3.7700 |
| 10.0000 | 35.4000 | 13.0300 |
| 20.0000 | 63.8300 | 26.5700 |
| 30.0000 | 89.4300 | 41.2000 |
| 40.0000 | 112.4000 | 51.3300 |
| 50.0000 | 122.9000 | 57.1300 |
