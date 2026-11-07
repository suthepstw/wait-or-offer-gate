# E5 rolling re-plan — sf · E5-replan v1.0 (2026-10-24: v0.9 + ชุดป้องกัน gate (SPEC #4): --delay t_d · --gate-fixed τ · --gate-raw (gate บน C ดิบ, V จาก rho.json) · --drop-zero (ไม่ commit offer ใหม่ที่ C = 0) · log n_zero_new · ไม่ใส่ flag = v0.9 ทุกบิต) · v0.9 (2026-10-24: v0.8 + --qgate qgate.json (B1): ต่อคนขับ เดินคู่เรียง C̃ มาก→น้อย เสนอเมื่อ Q_offer ≥ Q_wait → m_j ช่องใหม่ · ตัดคู่ที่ C̃ ต่ำกว่าคู่ที่ m_j · โควตาใหม่ ≤ m_j · --qgate-cv fold = k mod 5 · sticky เท่านั้น · ไม่ใส่ flag = เดิมทุกบิต) · 2026-10-24 12:05

instances 30 · arms ['qm_model'] · Δ = 10 นาที (6 จุดตัดสินใจ) · optimizer **nsga2** pop 80 × gen 480/จุด (per-step) · warm-start True · withdraw sticky · R 200 · L_i = 1 · Q_max 3 · θ 0.05/0.05 · cand 8
X* = ใกล้ ideal (1,1,1) ที่แต่ละจุด · replay ด้วย path_true · f₃* ใช้ Φ^max/T⁺ ของ oracle ที่ t=0

## timeline (adaptive) — งานสำเร็จสะสม (mean) หลัง replay ช่วง (t, t+Δ] · static = ค่าสุดท้ายในตาราง f₁* × |T|

| t | qm_model |
|---|---|
| 0.0000 | 2.8000 |
| 10.0000 | 10.2000 |
| 20.0000 | 19.8700 |
| 30.0000 | 27.9300 |
| 40.0000 | 33.3300 |
| 50.0000 | 35.2700 |
