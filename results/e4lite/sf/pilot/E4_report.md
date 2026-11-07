# E4-lite — sf (pilot) · E4-lite v1.0 (2026-10-20, Def 6 v5) · 2026-10-21 12:45

instances 3 (instances_v3/sf/pilot) · arms 7 · NSGA-II pop 80 × gen 120 (seed 12345+k) · R 200 m · L_i = 1 · Q_max 3 · battery off · θ_T/θ_W 0.05/0.05 · candidates/task 8
X* = จุดบน front ใกล้ ideal (1,1,1) ที่สุด (L2 บนค่าที่ optimizer เห็น) · *_real = replay ด้วย y จริง (เงื่อนไข ฎ) · f₃* ใช้ Φ^max และ T⁺ ของ oracle เป็นอ้างอิงร่วม · HV ref (0,0,0)

## สรุปต่อ arm (mean over instances)

| arm | label | n | f1_plan | f1_real | f2_real | f3_real | full_real | hv_real | corr_f1f3 | n_offer | f1_real_maxf1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qm_model | (9a′) Transformer [main] | 3 | 0.0780 | 0.0700 | 0.2047 | 0.1148 | 0.0700 | 0.0030 | 0.6483 | 156.3333 | 0.0650 |
| qm_lstm_reach | (9a′) Bi-LSTM | 3 | 0.0937 | 0.0750 | 0.1981 | 0.1236 | 0.0750 | 0.0025 | 0.8367 | 161.6667 | 0.0700 |
| qm_tfm_ub | (9a) Transformer | 3 | 0.0511 | 0.0600 | 0.1615 | 0.0972 | 0.0600 | 0.0013 | -0.3040 | 130.6667 | 0.0583 |
| qm_trajgeom | (9a) base_M3 | 3 | 0.0521 | 0.0750 | 0.1808 | 0.1199 | 0.0750 | 0.0029 | 0.5633 | 134.0000 | 0.0800 |
| qm_null_dv | physics null | 3 | 0.0271 | 0.0617 | 0.1879 | 0.1003 | 0.0617 | 0.0029 | 0.8970 | 139.0000 | 0.0683 |
| qm_persist_cont | persistence | 3 | 0.1312 | 0.0333 | 0.1262 | 0.0557 | 0.0333 | 0.0010 | 0.5860 | 111.0000 | 0.0350 |
| oracle | oracle (upper bound) | 3 | 0.5383 | 0.5383 | 0.8754 | 0.8674 | 0.5383 | 0.4190 | 1.0000 | 120.6667 | 0.5383 |

## Value of Prediction — Δ vs qm_model (paired ต่อ instance · bootstrap 95 % CI · Wilcoxon)

### f₁* (realised fulfilment)

| arm | n | mean | ref_mean | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|
| oracle | 3 | 0.5383 | 0.0700 | 0.4683 | 0.4050 | 0.5250 | 0.2500 | 3 | 0 |
| qm_lstm_reach | 3 | 0.0750 | 0.0700 | 0.0050 | -0.0150 | 0.0200 | 0.7500 | 2 | 1 |
| qm_null_dv | 3 | 0.0617 | 0.0700 | -0.0083 | -0.0450 | 0.0150 | 1.0000 | 2 | 1 |
| qm_persist_cont | 3 | 0.0333 | 0.0700 | -0.0367 | -0.0700 | -0.0150 | 0.2500 | 0 | 3 |
| qm_tfm_ub | 3 | 0.0600 | 0.0700 | -0.0100 | -0.0450 | 0.0150 | 1.0000 | 1 | 1 |
| qm_trajgeom | 3 | 0.0750 | 0.0700 | 0.0050 | -0.0500 | 0.0500 | 1.0000 | 2 | 1 |

### f₃* (realised service equity)

| arm | n | mean | ref_mean | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|
| oracle | 3 | 0.8674 | 0.1148 | 0.7526 | 0.7292 | 0.7787 | 0.2500 | 3 | 0 |
| qm_lstm_reach | 3 | 0.1236 | 0.1148 | 0.0088 | -0.0246 | 0.0370 | 0.7500 | 2 | 1 |
| qm_null_dv | 3 | 0.1003 | 0.1148 | -0.0146 | -0.0738 | 0.0208 | 1.0000 | 2 | 1 |
| qm_persist_cont | 3 | 0.0557 | 0.1148 | -0.0591 | -0.1148 | -0.0278 | 0.2500 | 0 | 3 |
| qm_tfm_ub | 3 | 0.0972 | 0.1148 | -0.0176 | -0.0738 | 0.0208 | 1.0000 | 1 | 1 |
| qm_trajgeom | 3 | 0.1199 | 0.1148 | 0.0051 | -0.0820 | 0.0694 | 1.0000 | 2 | 1 |

### HV ของ front จริง

| arm | n | mean | ref_mean | delta | lo | hi | wilcoxon_p | wins | losses |
|---|---|---|---|---|---|---|---|---|---|
| oracle | 3 | 0.4190 | 0.0030 | 0.4160 | 0.3580 | 0.4855 | 0.2500 | 3 | 0 |
| qm_lstm_reach | 3 | 0.0025 | 0.0030 | -0.0004 | -0.0035 | 0.0014 | 1.0000 | 2 | 1 |
| qm_null_dv | 3 | 0.0029 | 0.0030 | -0.0001 | -0.0049 | 0.0027 | 1.0000 | 2 | 1 |
| qm_persist_cont | 3 | 0.0010 | 0.0030 | -0.0020 | -0.0058 | 0.0004 | 0.5000 | 1 | 2 |
| qm_tfm_ub | 3 | 0.0013 | 0.0030 | -0.0017 | -0.0053 | 0.0008 | 0.7500 | 1 | 2 |
| qm_trajgeom | 3 | 0.0029 | 0.0030 | -0.0000 | -0.0059 | 0.0035 | 1.0000 | 2 | 1 |

## Gates

- G2.4 corr(f₁,f₃) บน front ของ qm_model: +0.648 → ✅ < 0.9
- G2.6 oracle > model > persistence (f₁*): 0.538 > 0.070 > 0.033 → ✅

## อ่าน
- Δ ลบ = arm นั้นแย่กว่า (9a′) Transformer · ตัวเลขหลักของ paper: Δf₁*(qm_tfm_ub) = ผลของ estimator บน encoder เดียวกัน · Δf₁*(qm_lstm_reach) = ผลของ encoder บน estimator เดียวกัน · Δ(qm_trajgeom) = เทียบ control เดิม
- f1_plan vs f1_real ต่างกันมาก = C_ij ปรับเทียบไม่ตรงกับคู่ที่ถูกเลือก (optimizer เลือกคู่ที่ C สูงเกินจริง = winner's curse) — ดู oracle gap
