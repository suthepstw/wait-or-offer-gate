# E5 rolling re-plan — sf · E5-replan v0.2 (2026-10-21: gate G5.1 เฉพาะ qm_model, timeline แยก mode, S1 vs E4) · 2026-10-22 06:40

instances 30 · arms ['qm_model', 'qm_lstm_reach', 'qm_tfm_ub', 'qm_null_dv', 'qm_persist_cont', 'oracle'] · Δ = 10 นาที (6 จุดตัดสินใจ) · NSGA-II pop 80 × gen 480/จุด (per-step) · warm-start True · withdraw free · R 200 · L_i = 1 · Q_max 3 · θ 0.05/0.05 · cand 8
X* = ใกล้ ideal (1,1,1) ที่แต่ละจุด · replay ด้วย path_true · f₃* ใช้ Φ^max/T⁺ ของ oracle ที่ t=0

## f₁* static vs adaptive (mean over instances · Δ paired · bootstrap 95 % CI · Wilcoxon)

| arm | label | n | f1_static | f1_adaptive | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| qm_model | (9a′) Transformer [main] | 30 | 0.0737 | 0.2658 | 0.1922 | 0.1782 | 0.2065 | 0.0000 | 30 | 0 |
| qm_lstm_reach | (9a′) Bi-LSTM | 30 | 0.0675 | 0.2533 | 0.1858 | 0.1672 | 0.2055 | 0.0000 | 30 | 0 |
| qm_tfm_ub | (9a) Transformer | 30 | 0.0593 | 0.2635 | 0.2042 | 0.1868 | 0.2228 | 0.0000 | 30 | 0 |
| qm_null_dv | physics null | 30 | 0.0583 | 0.2840 | 0.2257 | 0.2067 | 0.2448 | 0.0000 | 30 | 0 |
| qm_persist_cont | persistence | 30 | 0.0503 | 0.2272 | 0.1768 | 0.1588 | 0.1958 | 0.0000 | 30 | 0 |
| oracle | oracle (y จริง) | 30 | 0.5773 | 0.5960 | 0.0187 | 0.0145 | 0.0230 | 0.0000 | 29 | 0 |

## Gates

- G5.1 adaptive ≥ static สำหรับ qm_model (ตัวประมาณที่มีสารสนเทศ — ถ้าได้ข้อมูลเพิ่มแล้วแย่ลง = ตรวจ replay/โควตา): ✅ · baseline ไร้สารสนเทศ (persistence/null) อาจแย่ลงได้จากการถอน offer — รายงานตามจริง ไม่ใช่ gate
- G5.2 |Δ oracle| ≤ 0.02: ✅ (Δ = +0.0187)

## Value of Prediction ภายใต้ adaptive — Δf₁* vs qm_model (paired)

| arm | n | mean | ref_mean | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|
| oracle | 30 | 0.5960 | 0.2658 | 0.3302 | 0.3030 | 0.3575 | 0.0000 | 30 | 0 |
| qm_lstm_reach | 30 | 0.2533 | 0.2658 | -0.0125 | -0.0222 | -0.0023 | 0.0173 | 7 | 20 |
| qm_null_dv | 30 | 0.2840 | 0.2658 | 0.0182 | 0.0068 | 0.0282 | 0.0044 | 22 | 7 |
| qm_persist_cont | 30 | 0.2272 | 0.2658 | -0.0387 | -0.0473 | -0.0288 | 0.0000 | 3 | 26 |
| qm_tfm_ub | 30 | 0.2635 | 0.2658 | -0.0023 | -0.0137 | 0.0098 | 0.3945 | 10 | 16 |

## timeline (adaptive) — งานสำเร็จสะสม (mean) หลัง replay ช่วง (t, t+Δ] · static = ค่าสุดท้ายในตาราง f₁* × |T|

| t | oracle | qm_lstm_reach | qm_model | qm_null_dv | qm_persist_cont | qm_tfm_ub |
|---|---|---|---|---|---|---|
| 0.0000 | 11.7300 | 3.1000 | 3.3300 | 3.1700 | 2.3300 | 3.8000 |
| 10.0000 | 34.4000 | 12.0700 | 12.9700 | 14.2000 | 10.0000 | 13.2700 |
| 20.0000 | 59.5300 | 25.0700 | 25.3700 | 27.4300 | 20.3300 | 26.1700 |
| 30.0000 | 85.3700 | 35.3000 | 36.8300 | 39.4700 | 30.5000 | 36.7700 |
| 40.0000 | 108.5300 | 45.0700 | 47.0700 | 50.7000 | 40.0000 | 47.2700 |
| 50.0000 | 119.2000 | 50.6700 | 53.1700 | 56.8000 | 45.4300 | 52.7000 |

## S1 — static ของ E5 vs E4 (instance/arm เดียวกัน)

- n 180 · mean Δ -0.0004 · max|Δ| 0.0500 · เท่ากันเป๊ะ 71/180 (ต่างได้จาก ΔC ≤ 5e-3 ของ sE5 และ gen ที่ต่างกัน: E4 ใช้ 480)
