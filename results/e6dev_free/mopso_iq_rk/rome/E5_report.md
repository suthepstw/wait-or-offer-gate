# E5 rolling re-plan — rome · E5-replan v0.7.1 (2026-10-23: --shrink → C̃ = min(1, ρ(h)·C) ตาม rho_used ของ rho.json (A1 แก้ไข: ρ ไม่ตัดที่ 1) + --inf-frac (A2) · ไม่ใส่ flag = เดิมทุกบิต) · 2026-10-23 09:27

instances 20 · arms ['qm_model', 'oracle'] · Δ = 10 นาที (6 จุดตัดสินใจ) · optimizer **mopso_iq_rk** pop 80 × gen 480/จุด (per-step) · warm-start True · withdraw free · R 200 · L_i = 1 · Q_max 3 · θ 0.05/0.05 · cand 8
X* = ใกล้ ideal (1,1,1) ที่แต่ละจุด · replay ด้วย path_true · f₃* ใช้ Φ^max/T⁺ ของ oracle ที่ t=0

## f₁* static vs adaptive (mean over instances · Δ paired · bootstrap 95 % CI · Wilcoxon)

| arm | label | n | f1_static | f1_adaptive | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| qm_model | (9a′) Transformer [main] | 20 | 0.0815 | 0.2455 | 0.1640 | 0.1355 | 0.1925 | 0.0000 | 20 | 0 |
| oracle | oracle (y จริง) | 20 | 0.4550 | 0.4702 | 0.0152 | 0.0090 | 0.0222 | 0.0006 | 15 | 0 |

## Gates

- G5.1 adaptive ≥ static สำหรับ qm_model (ตัวประมาณที่มีสารสนเทศ — ถ้าได้ข้อมูลเพิ่มแล้วแย่ลง = ตรวจ replay/โควตา): ✅ · baseline ไร้สารสนเทศ (persistence/null) อาจแย่ลงได้จากการถอน offer — รายงานตามจริง ไม่ใช่ gate
- G5.2 |Δ oracle| ≤ 0.02: ✅ (Δ = +0.0152)

## Value of Prediction ภายใต้ adaptive — Δf₁* vs qm_model (paired)

| arm | n | mean | ref_mean | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|
| oracle | 20 | 0.4702 | 0.2455 | 0.2248 | 0.2005 | 0.2510 | 0.0001 | 20 | 0 |

## timeline (adaptive) — งานสำเร็จสะสม (mean) หลัง replay ช่วง (t, t+Δ] · static = ค่าสุดท้ายในตาราง f₁* × |T|

| t | oracle | qm_model |
|---|---|---|
| 0.0000 | 9.5500 | 4.4500 |
| 10.0000 | 27.8000 | 13.4000 |
| 20.0000 | 49.6000 | 24.7000 |
| 30.0000 | 71.2500 | 36.1500 |
| 40.0000 | 86.6000 | 44.8500 |
| 50.0000 | 94.0500 | 49.1000 |
