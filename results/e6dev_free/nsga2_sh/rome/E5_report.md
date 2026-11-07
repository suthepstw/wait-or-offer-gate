# E5 rolling re-plan — rome · E5-replan v0.7.1 (2026-10-23: --shrink → C̃ = min(1, ρ(h)·C) ตาม rho_used ของ rho.json (A1 แก้ไข: ρ ไม่ตัดที่ 1) + --inf-frac (A2) · ไม่ใส่ flag = เดิมทุกบิต) · 2026-10-23 09:20

instances 20 · arms ['qm_model', 'oracle'] · Δ = 10 นาที (6 จุดตัดสินใจ) · optimizer **nsga2** pop 80 × gen 480/จุด (per-step) · warm-start True · withdraw free · R 200 · L_i = 1 · Q_max 3 · θ 0.05/0.05 · cand 8
X* = ใกล้ ideal (1,1,1) ที่แต่ละจุด · replay ด้วย path_true · f₃* ใช้ Φ^max/T⁺ ของ oracle ที่ t=0

## f₁* static vs adaptive (mean over instances · Δ paired · bootstrap 95 % CI · Wilcoxon)

| arm | label | n | f1_static | f1_adaptive | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| qm_model | (9a′) Transformer [main] | 20 | 0.0800 | 0.2530 | 0.1730 | 0.1445 | 0.2015 | 0.0001 | 20 | 0 |
| oracle | oracle (y จริง) | 20 | 0.4542 | 0.4668 | 0.0125 | 0.0077 | 0.0180 | 0.0003 | 17 | 0 |

## Gates

- G5.1 adaptive ≥ static สำหรับ qm_model (ตัวประมาณที่มีสารสนเทศ — ถ้าได้ข้อมูลเพิ่มแล้วแย่ลง = ตรวจ replay/โควตา): ✅ · baseline ไร้สารสนเทศ (persistence/null) อาจแย่ลงได้จากการถอน offer — รายงานตามจริง ไม่ใช่ gate
- G5.2 |Δ oracle| ≤ 0.02: ✅ (Δ = +0.0125)

## Value of Prediction ภายใต้ adaptive — Δf₁* vs qm_model (paired)

| arm | n | mean | ref_mean | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|
| oracle | 20 | 0.4668 | 0.2530 | 0.2138 | 0.1895 | 0.2380 | 0.0001 | 20 | 0 |

## timeline (adaptive) — งานสำเร็จสะสม (mean) หลัง replay ช่วง (t, t+Δ] · static = ค่าสุดท้ายในตาราง f₁* × |T|

| t | oracle | qm_model |
|---|---|---|
| 0.0000 | 9.6500 | 3.9000 |
| 10.0000 | 27.8500 | 13.5500 |
| 20.0000 | 48.8500 | 24.9000 |
| 30.0000 | 70.0000 | 37.2000 |
| 40.0000 | 86.1000 | 46.6500 |
| 50.0000 | 93.3500 | 50.6000 |
