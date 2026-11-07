# E5 rolling re-plan — sf · E5-replan v0.8 (2026-10-23: v0.7.1 + --gate κ (B0): ตัด candidate ใหม่ที่ C̃_ij(t) < κ·V_k(t), k = Q_rem_j, V จาก rho.json · pending ไม่ถูกตัด · oracle ไม่ใช้ gate · ไม่ใส่ flag = เดิมทุกบิต) · 2026-10-23 10:26

instances 30 · arms ['qm_model', 'oracle'] · Δ = 10 นาที (6 จุดตัดสินใจ) · optimizer **mopso_iq_rk** pop 80 × gen 480/จุด (per-step) · warm-start True · withdraw sticky · R 200 · L_i = 1 · Q_max 3 · θ 0.05/0.05 · cand 8
X* = ใกล้ ideal (1,1,1) ที่แต่ละจุด · replay ด้วย path_true · f₃* ใช้ Φ^max/T⁺ ของ oracle ที่ t=0

## f₁* static vs adaptive (mean over instances · Δ paired · bootstrap 95 % CI · Wilcoxon)

| arm | label | n | f1_static | f1_adaptive | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| qm_model | (9a′) Transformer [main] | 30 | 0.0347 | 0.1627 | 0.1280 | 0.1202 | 0.1357 | 0.0000 | 30 | 0 |
| oracle | oracle (y จริง) | 30 | 0.5772 | 0.6003 | 0.0232 | 0.0165 | 0.0313 | 0.0000 | 28 | 0 |

## Gates

- G5.1 adaptive ≥ static สำหรับ qm_model (ตัวประมาณที่มีสารสนเทศ — ถ้าได้ข้อมูลเพิ่มแล้วแย่ลง = ตรวจ replay/โควตา): ✅ · baseline ไร้สารสนเทศ (persistence/null) อาจแย่ลงได้จากการถอน offer — รายงานตามจริง ไม่ใช่ gate
- G5.2 |Δ oracle| ≤ 0.02: ❌ (Δ = +0.0232)

## Value of Prediction ภายใต้ adaptive — Δf₁* vs qm_model (paired)

| arm | n | mean | ref_mean | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|
| oracle | 30 | 0.6003 | 0.1627 | 0.4377 | 0.4122 | 0.4627 | 0.0000 | 30 | 0 |

## timeline (adaptive) — งานสำเร็จสะสม (mean) หลัง replay ช่วง (t, t+Δ] · static = ค่าสุดท้ายในตาราง f₁* × |T|

| t | oracle | qm_model |
|---|---|---|
| 0.0000 | 11.7300 | 3.0700 |
| 10.0000 | 34.8300 | 9.9000 |
| 20.0000 | 64.0000 | 18.9700 |
| 30.0000 | 88.0300 | 25.4300 |
| 40.0000 | 110.6000 | 30.5000 |
| 50.0000 | 120.0700 | 32.5300 |
