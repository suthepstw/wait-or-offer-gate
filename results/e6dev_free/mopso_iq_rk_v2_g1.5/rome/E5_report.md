# E5 rolling re-plan — rome · E5-replan v0.8 (2026-10-23: v0.7.1 + --gate κ (B0): ตัด candidate ใหม่ที่ C̃_ij(t) < κ·V_k(t), k = Q_rem_j, V จาก rho.json · pending ไม่ถูกตัด · oracle ไม่ใช้ gate · ไม่ใส่ flag = เดิมทุกบิต) · 2026-10-23 10:49

instances 20 · arms ['qm_model', 'oracle'] · Δ = 10 นาที (6 จุดตัดสินใจ) · optimizer **mopso_iq_rk** pop 80 × gen 480/จุด (per-step) · warm-start True · withdraw free · R 200 · L_i = 1 · Q_max 3 · θ 0.05/0.05 · cand 8
X* = ใกล้ ideal (1,1,1) ที่แต่ละจุด · replay ด้วย path_true · f₃* ใช้ Φ^max/T⁺ ของ oracle ที่ t=0

## f₁* static vs adaptive (mean over instances · Δ paired · bootstrap 95 % CI · Wilcoxon)

| arm | label | n | f1_static | f1_adaptive | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| qm_model | (9a′) Transformer [main] | 20 | 0.0550 | 0.2130 | 0.1580 | 0.1303 | 0.1840 | 0.0001 | 20 | 0 |
| oracle | oracle (y จริง) | 20 | 0.4490 | 0.4668 | 0.0177 | 0.0110 | 0.0252 | 0.0003 | 17 | 0 |

## Gates

- G5.1 adaptive ≥ static สำหรับ qm_model (ตัวประมาณที่มีสารสนเทศ — ถ้าได้ข้อมูลเพิ่มแล้วแย่ลง = ตรวจ replay/โควตา): ✅ · baseline ไร้สารสนเทศ (persistence/null) อาจแย่ลงได้จากการถอน offer — รายงานตามจริง ไม่ใช่ gate
- G5.2 |Δ oracle| ≤ 0.02: ✅ (Δ = +0.0177)

## Value of Prediction ภายใต้ adaptive — Δf₁* vs qm_model (paired)

| arm | n | mean | ref_mean | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|
| oracle | 20 | 0.4668 | 0.2130 | 0.2537 | 0.2272 | 0.2825 | 0.0001 | 20 | 0 |

## timeline (adaptive) — งานสำเร็จสะสม (mean) หลัง replay ช่วง (t, t+Δ] · static = ค่าสุดท้ายในตาราง f₁* × |T|

| t | oracle | qm_model |
|---|---|---|
| 0.0000 | 9.6000 | 3.8000 |
| 10.0000 | 27.4000 | 11.4500 |
| 20.0000 | 49.3500 | 20.2500 |
| 30.0000 | 69.9000 | 30.0500 |
| 40.0000 | 86.1000 | 38.8500 |
| 50.0000 | 93.3500 | 42.6000 |
