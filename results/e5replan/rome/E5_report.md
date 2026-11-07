# E5 rolling re-plan — rome · E5-replan v0.2 (2026-10-21: gate G5.1 เฉพาะ qm_model, timeline แยก mode, S1 vs E4) · 2026-10-22 06:13

instances 30 · arms ['qm_model', 'qm_lstm_reach', 'qm_tfm_ub', 'qm_null_dv', 'qm_persist_cont', 'oracle'] · Δ = 10 นาที (6 จุดตัดสินใจ) · NSGA-II pop 80 × gen 480/จุด (per-step) · warm-start True · withdraw free · R 200 · L_i = 1 · Q_max 3 · θ 0.05/0.05 · cand 8
X* = ใกล้ ideal (1,1,1) ที่แต่ละจุด · replay ด้วย path_true · f₃* ใช้ Φ^max/T⁺ ของ oracle ที่ t=0

## f₁* static vs adaptive (mean over instances · Δ paired · bootstrap 95 % CI · Wilcoxon)

| arm | label | n | f1_static | f1_adaptive | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| qm_model | (9a′) Transformer [main] | 30 | 0.0738 | 0.2468 | 0.1730 | 0.1567 | 0.1898 | 0.0000 | 30 | 0 |
| qm_lstm_reach | (9a′) Bi-LSTM | 30 | 0.0727 | 0.2448 | 0.1722 | 0.1542 | 0.1893 | 0.0000 | 30 | 0 |
| qm_tfm_ub | (9a) Transformer | 30 | 0.0608 | 0.2643 | 0.2035 | 0.1837 | 0.2225 | 0.0000 | 30 | 0 |
| qm_null_dv | physics null | 30 | 0.0572 | 0.2462 | 0.1890 | 0.1733 | 0.2042 | 0.0000 | 30 | 0 |
| qm_persist_cont | persistence | 30 | 0.0688 | 0.2352 | 0.1663 | 0.1490 | 0.1838 | 0.0000 | 30 | 0 |
| oracle | oracle (y จริง) | 30 | 0.4653 | 0.4818 | 0.0165 | 0.0127 | 0.0203 | 0.0000 | 27 | 0 |

## Gates

- G5.1 adaptive ≥ static สำหรับ qm_model (ตัวประมาณที่มีสารสนเทศ — ถ้าได้ข้อมูลเพิ่มแล้วแย่ลง = ตรวจ replay/โควตา): ✅ · baseline ไร้สารสนเทศ (persistence/null) อาจแย่ลงได้จากการถอน offer — รายงานตามจริง ไม่ใช่ gate
- G5.2 |Δ oracle| ≤ 0.02: ✅ (Δ = +0.0165)

## Value of Prediction ภายใต้ adaptive — Δf₁* vs qm_model (paired)

| arm | n | mean | ref_mean | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|
| oracle | 30 | 0.4818 | 0.2468 | 0.2350 | 0.2125 | 0.2572 | 0.0000 | 30 | 0 |
| qm_lstm_reach | 30 | 0.2448 | 0.2468 | -0.0020 | -0.0092 | 0.0055 | 0.6415 | 14 | 15 |
| qm_null_dv | 30 | 0.2462 | 0.2468 | -0.0007 | -0.0098 | 0.0080 | 0.8787 | 13 | 13 |
| qm_persist_cont | 30 | 0.2352 | 0.2468 | -0.0117 | -0.0188 | -0.0045 | 0.0042 | 4 | 22 |
| qm_tfm_ub | 30 | 0.2643 | 0.2468 | 0.0175 | 0.0052 | 0.0293 | 0.0128 | 20 | 10 |

## timeline (adaptive) — งานสำเร็จสะสม (mean) หลัง replay ช่วง (t, t+Δ] · static = ค่าสุดท้ายในตาราง f₁* × |T|

| t | oracle | qm_lstm_reach | qm_model | qm_null_dv | qm_persist_cont | qm_tfm_ub |
|---|---|---|---|---|---|---|
| 0.0000 | 10.3700 | 3.6300 | 3.7700 | 3.0300 | 3.8300 | 4.1000 |
| 10.0000 | 28.7000 | 13.1700 | 13.9000 | 12.7300 | 13.2300 | 14.9700 |
| 20.0000 | 50.6700 | 24.4000 | 25.2700 | 24.2300 | 23.8700 | 26.9700 |
| 30.0000 | 72.2000 | 35.7700 | 35.9000 | 35.0000 | 33.8700 | 38.8300 |
| 40.0000 | 89.0300 | 44.3700 | 44.9700 | 44.3700 | 42.5300 | 47.7700 |
| 50.0000 | 96.3700 | 48.9700 | 49.3700 | 49.2300 | 47.0300 | 52.8700 |

## S1 — static ของ E5 vs E4 (instance/arm เดียวกัน)

- n 180 · mean Δ +0.0001 · max|Δ| 0.0350 · เท่ากันเป๊ะ 76/180 (ต่างได้จาก ΔC ≤ 5e-3 ของ sE5 และ gen ที่ต่างกัน: E4 ใช้ 480)
