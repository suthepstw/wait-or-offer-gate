# E5 rolling re-plan — sf · E5-replan v0.7.1 (2026-10-23: --shrink → C̃ = min(1, ρ(h)·C) ตาม rho_used ของ rho.json (A1 แก้ไข: ρ ไม่ตัดที่ 1) + --inf-frac (A2) · ไม่ใส่ flag = เดิมทุกบิต) · 2026-10-23 09:10

instances 30 · arms ['qm_model', 'oracle'] · Δ = 10 นาที (6 จุดตัดสินใจ) · optimizer **mopso_iq_rk** pop 80 × gen 480/จุด (per-step) · warm-start True · withdraw sticky · R 200 · L_i = 1 · Q_max 3 · θ 0.05/0.05 · cand 8
X* = ใกล้ ideal (1,1,1) ที่แต่ละจุด · replay ด้วย path_true · f₃* ใช้ Φ^max/T⁺ ของ oracle ที่ t=0

## f₁* static vs adaptive (mean over instances · Δ paired · bootstrap 95 % CI · Wilcoxon)

| arm | label | n | f1_static | f1_adaptive | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| qm_model | (9a′) Transformer [main] | 30 | 0.0793 | 0.0760 | -0.0033 | -0.0073 | 0.0005 | 0.1232 | 12 | 15 |
| oracle | oracle (y จริง) | 30 | 0.5790 | 0.6005 | 0.0215 | 0.0162 | 0.0275 | 0.0000 | 28 | 0 |

## Gates

- G5.1 adaptive ≥ static สำหรับ qm_model (ตัวประมาณที่มีสารสนเทศ — ถ้าได้ข้อมูลเพิ่มแล้วแย่ลง = ตรวจ replay/โควตา): ❌ · baseline ไร้สารสนเทศ (persistence/null) อาจแย่ลงได้จากการถอน offer — รายงานตามจริง ไม่ใช่ gate
- G5.2 |Δ oracle| ≤ 0.02: ❌ (Δ = +0.0215)

## Value of Prediction ภายใต้ adaptive — Δf₁* vs qm_model (paired)

| arm | n | mean | ref_mean | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|
| oracle | 30 | 0.6005 | 0.0760 | 0.5245 | 0.4968 | 0.5515 | 0.0000 | 30 | 0 |

## timeline (adaptive) — งานสำเร็จสะสม (mean) หลัง replay ช่วง (t, t+Δ] · static = ค่าสุดท้ายในตาราง f₁* × |T|

| t | oracle | qm_model |
|---|---|---|
| 0.0000 | 11.8300 | 3.9000 |
| 10.0000 | 34.9700 | 7.2700 |
| 20.0000 | 63.5700 | 10.9000 |
| 30.0000 | 88.4000 | 13.2300 |
| 40.0000 | 110.5000 | 14.7000 |
| 50.0000 | 120.1000 | 15.2000 |
