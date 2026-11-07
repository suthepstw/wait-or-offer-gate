# E5 rolling re-plan — rome · E5-replan v0.7.1 (2026-10-23: --shrink → C̃ = min(1, ρ(h)·C) ตาม rho_used ของ rho.json (A1 แก้ไข: ρ ไม่ตัดที่ 1) + --inf-frac (A2) · ไม่ใส่ flag = เดิมทุกบิต) · 2026-10-23 09:20

instances 20 · arms ['qm_model', 'oracle'] · Δ = 10 นาที (6 จุดตัดสินใจ) · optimizer **nsga2** pop 80 × gen 480/จุด (per-step) · warm-start True · withdraw sticky · R 200 · L_i = 1 · Q_max 3 · θ 0.05/0.05 · cand 8
X* = ใกล้ ideal (1,1,1) ที่แต่ละจุด · replay ด้วย path_true · f₃* ใช้ Φ^max/T⁺ ของ oracle ที่ t=0

## f₁* static vs adaptive (mean over instances · Δ paired · bootstrap 95 % CI · Wilcoxon)

| arm | label | n | f1_static | f1_adaptive | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| qm_model | (9a′) Transformer [main] | 20 | 0.0800 | 0.0788 | -0.0013 | -0.0083 | 0.0058 | 0.6466 | 7 | 11 |
| oracle | oracle (y จริง) | 20 | 0.4542 | 0.4628 | 0.0085 | 0.0040 | 0.0135 | 0.0042 | 15 | 1 |

## Gates

- G5.1 adaptive ≥ static สำหรับ qm_model (ตัวประมาณที่มีสารสนเทศ — ถ้าได้ข้อมูลเพิ่มแล้วแย่ลง = ตรวจ replay/โควตา): ❌ · baseline ไร้สารสนเทศ (persistence/null) อาจแย่ลงได้จากการถอน offer — รายงานตามจริง ไม่ใช่ gate
- G5.2 |Δ oracle| ≤ 0.02: ✅ (Δ = +0.0085)

## Value of Prediction ภายใต้ adaptive — Δf₁* vs qm_model (paired)

| arm | n | mean | ref_mean | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|
| oracle | 20 | 0.4628 | 0.0788 | 0.3840 | 0.3375 | 0.4263 | 0.0001 | 20 | 0 |

## timeline (adaptive) — งานสำเร็จสะสม (mean) หลัง replay ช่วง (t, t+Δ] · static = ค่าสุดท้ายในตาราง f₁* × |T|

| t | oracle | qm_model |
|---|---|---|
| 0.0000 | 9.6500 | 3.9000 |
| 10.0000 | 28.2000 | 8.8000 |
| 20.0000 | 48.6500 | 11.9500 |
| 30.0000 | 69.3000 | 13.8500 |
| 40.0000 | 85.2000 | 15.1500 |
| 50.0000 | 92.5500 | 15.7500 |
