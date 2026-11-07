# E5 rolling re-plan — rome · E5-replan v0.9 (2026-10-24: v0.8 + --qgate qgate.json (B1): ต่อคนขับ เดินคู่เรียง C̃ มาก→น้อย เสนอเมื่อ Q_offer ≥ Q_wait → m_j ช่องใหม่ · ตัดคู่ที่ C̃ ต่ำกว่าคู่ที่ m_j · โควตาใหม่ ≤ m_j · --qgate-cv fold = k mod 5 · sticky เท่านั้น · ไม่ใส่ flag = เดิมทุกบิต) · 2026-10-24 09:59

instances 30 · arms ['qm_model', 'oracle'] · Δ = 10 นาที (6 จุดตัดสินใจ) · optimizer **nsga2** pop 80 × gen 480/จุด (per-step) · warm-start True · withdraw sticky · R 200 · L_i = 1 · Q_max 3 · θ 0.05/0.05 · cand 8
X* = ใกล้ ideal (1,1,1) ที่แต่ละจุด · replay ด้วย path_true · f₃* ใช้ Φ^max/T⁺ ของ oracle ที่ t=0

## f₁* static vs adaptive (mean over instances · Δ paired · bootstrap 95 % CI · Wilcoxon)

| arm | label | n | f1_static | f1_adaptive | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| qm_model | (9a′) Transformer [main] | 30 | 0.0740 | 0.0765 | 0.0025 | -0.0020 | 0.0070 | 0.1542 | 16 | 10 |
| oracle | oracle (y จริง) | 30 | 0.4653 | 0.4743 | 0.0090 | 0.0058 | 0.0123 | 0.0001 | 20 | 0 |

## Gates

- G5.1 adaptive ≥ static สำหรับ qm_model (ตัวประมาณที่มีสารสนเทศ — ถ้าได้ข้อมูลเพิ่มแล้วแย่ลง = ตรวจ replay/โควตา): ✅ · baseline ไร้สารสนเทศ (persistence/null) อาจแย่ลงได้จากการถอน offer — รายงานตามจริง ไม่ใช่ gate
- G5.2 |Δ oracle| ≤ 0.02: ✅ (Δ = +0.0090)

## Value of Prediction ภายใต้ adaptive — Δf₁* vs qm_model (paired)

| arm | n | mean | ref_mean | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|
| oracle | 30 | 0.4743 | 0.0765 | 0.3978 | 0.3640 | 0.4300 | 0.0000 | 30 | 0 |

## timeline (adaptive) — งานสำเร็จสะสม (mean) หลัง replay ช่วง (t, t+Δ] · static = ค่าสุดท้ายในตาราง f₁* × |T|

| t | oracle | qm_model |
|---|---|---|
| 0.0000 | 10.3700 | 3.7700 |
| 10.0000 | 28.7700 | 8.4000 |
| 20.0000 | 50.5700 | 11.3000 |
| 30.0000 | 71.9300 | 13.4700 |
| 40.0000 | 87.6300 | 14.9700 |
| 50.0000 | 94.8700 | 15.3000 |
