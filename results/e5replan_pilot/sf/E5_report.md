# E5 rolling re-plan — sf · E5-replan v0.2 (2026-10-21: gate G5.1 เฉพาะ qm_model, timeline แยก mode, S1 vs E4) · 2026-10-21 21:20

instances 10 · arms ['qm_model', 'qm_persist_cont', 'oracle'] · Δ = 10 นาที (6 จุดตัดสินใจ) · NSGA-II pop 80 × gen 120/จุด (per-step) · warm-start True · withdraw free · R 200 · L_i = 1 · Q_max 3 · θ 0.05/0.05 · cand 8
X* = ใกล้ ideal (1,1,1) ที่แต่ละจุด · replay ด้วย path_true · f₃* ใช้ Φ^max/T⁺ ของ oracle ที่ t=0

## f₁* static vs adaptive (mean over instances · Δ paired · bootstrap 95 % CI · Wilcoxon)

| arm | label | n | f1_static | f1_adaptive | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| qm_model | (9a′) Transformer [main] | 10 | 0.0820 | 0.2590 | 0.1770 | 0.1440 | 0.2135 | 0.0020 | 10 | 0 |
| qm_persist_cont | persistence | 10 | 0.0600 | 0.2165 | 0.1565 | 0.1245 | 0.1905 | 0.0020 | 10 | 0 |
| oracle | oracle (y จริง) | 10 | 0.5490 | 0.5680 | 0.0190 | 0.0120 | 0.0260 | 0.0020 | 10 | 0 |

## Gates

- G5.1 adaptive ≥ static สำหรับ qm_model (ตัวประมาณที่มีสารสนเทศ — ถ้าได้ข้อมูลเพิ่มแล้วแย่ลง = ตรวจ replay/โควตา): ✅ · baseline ไร้สารสนเทศ (persistence/null) อาจแย่ลงได้จากการถอน offer — รายงานตามจริง ไม่ใช่ gate
- G5.2 |Δ oracle| ≤ 0.02: ✅ (Δ = +0.0190)

## Value of Prediction ภายใต้ adaptive — Δf₁* vs qm_model (paired)

| arm | n | mean | ref_mean | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|
| oracle | 10 | 0.5680 | 0.2590 | 0.3090 | 0.2800 | 0.3355 | 0.0020 | 10 | 0 |
| qm_persist_cont | 10 | 0.2165 | 0.2590 | -0.0425 | -0.0570 | -0.0265 | 0.0039 | 1 | 9 |

## timeline (adaptive) — งานสำเร็จสะสม (mean) หลัง replay ช่วง (t, t+Δ] · static = ค่าสุดท้ายในตาราง f₁* × |T|

| t | oracle | qm_model | qm_persist_cont |
|---|---|---|---|
| 0.0000 | 10.5000 | 3.7000 | 2.7000 |
| 10.0000 | 35.1000 | 13.8000 | 10.4000 |
| 20.0000 | 57.7000 | 26.1000 | 20.7000 |
| 30.0000 | 81.5000 | 35.7000 | 29.8000 |
| 40.0000 | 102.2000 | 45.4000 | 39.5000 |
| 50.0000 | 113.6000 | 51.8000 | 43.3000 |

## S1 — static ของ E5 vs E4 (instance/arm เดียวกัน)

- n 30 · mean Δ +0.0032 · max|Δ| 0.0400 · เท่ากันเป๊ะ 10/30 (ต่างได้จาก ΔC ≤ 5e-3 ของ sE5 และ gen ที่ต่างกัน: E4 ใช้ 480)
