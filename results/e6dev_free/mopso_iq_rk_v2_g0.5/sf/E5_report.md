# E5 rolling re-plan — sf · E5-replan v0.8 (2026-10-23: v0.7.1 + --gate κ (B0): ตัด candidate ใหม่ที่ C̃_ij(t) < κ·V_k(t), k = Q_rem_j, V จาก rho.json · pending ไม่ถูกตัด · oracle ไม่ใช้ gate · ไม่ใส่ flag = เดิมทุกบิต) · 2026-10-23 10:25

instances 30 · arms ['qm_model', 'oracle'] · Δ = 10 นาที (6 จุดตัดสินใจ) · optimizer **mopso_iq_rk** pop 80 × gen 480/จุด (per-step) · warm-start True · withdraw free · R 200 · L_i = 1 · Q_max 3 · θ 0.05/0.05 · cand 8
X* = ใกล้ ideal (1,1,1) ที่แต่ละจุด · replay ด้วย path_true · f₃* ใช้ Φ^max/T⁺ ของ oracle ที่ t=0

## f₁* static vs adaptive (mean over instances · Δ paired · bootstrap 95 % CI · Wilcoxon)

| arm | label | n | f1_static | f1_adaptive | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| qm_model | (9a′) Transformer [main] | 30 | 0.0672 | 0.2825 | 0.2153 | 0.1997 | 0.2318 | 0.0000 | 30 | 0 |
| oracle | oracle (y จริง) | 30 | 0.5772 | 0.6088 | 0.0317 | 0.0240 | 0.0403 | 0.0000 | 29 | 0 |

## Gates

- G5.1 adaptive ≥ static สำหรับ qm_model (ตัวประมาณที่มีสารสนเทศ — ถ้าได้ข้อมูลเพิ่มแล้วแย่ลง = ตรวจ replay/โควตา): ✅ · baseline ไร้สารสนเทศ (persistence/null) อาจแย่ลงได้จากการถอน offer — รายงานตามจริง ไม่ใช่ gate
- G5.2 |Δ oracle| ≤ 0.02: ❌ (Δ = +0.0317)

## Value of Prediction ภายใต้ adaptive — Δf₁* vs qm_model (paired)

| arm | n | mean | ref_mean | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|
| oracle | 30 | 0.6088 | 0.2825 | 0.3263 | 0.3010 | 0.3517 | 0.0000 | 30 | 0 |

## timeline (adaptive) — งานสำเร็จสะสม (mean) หลัง replay ช่วง (t, t+Δ] · static = ค่าสุดท้ายในตาราง f₁* × |T|

| t | oracle | qm_model |
|---|---|---|
| 0.0000 | 11.7300 | 3.8300 |
| 10.0000 | 34.3300 | 12.9300 |
| 20.0000 | 63.5700 | 26.2300 |
| 30.0000 | 89.2000 | 40.4700 |
| 40.0000 | 112.1300 | 50.9700 |
| 50.0000 | 121.7700 | 56.5000 |
