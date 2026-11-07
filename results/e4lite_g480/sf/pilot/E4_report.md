# E4-lite — sf (pilot) · E4-lite v1.0 (2026-10-20, Def 6 v5) · 2026-10-21 13:07

instances 1 (instances_v3/sf/pilot) · arms 1 · NSGA-II pop 80 × gen 480 (seed 12345+k) · R 200 m · L_i = 1 · Q_max 3 · battery off · θ_T/θ_W 0.05/0.05 · candidates/task 8
X* = จุดบน front ใกล้ ideal (1,1,1) ที่สุด (L2 บนค่าที่ optimizer เห็น) · *_real = replay ด้วย y จริง (เงื่อนไข ฎ) · f₃* ใช้ Φ^max และ T⁺ ของ oracle เป็นอ้างอิงร่วม · HV ref (0,0,0)

## สรุปต่อ arm (mean over instances)

| arm | label | n | f1_plan | f1_real | f2_real | f3_real | full_real | hv_real | corr_f1f3 | n_offer | f1_real_maxf1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qm_model | (9a′) Transformer [main] | 1 | 0.0714 | 0.0500 | 0.1375 | 0.0926 | 0.0500 | 0.0011 | 0.6150 | 162.0000 | 0.0500 |

## Value of Prediction — Δ vs qm_model (paired ต่อ instance · bootstrap 95 % CI · Wilcoxon)

### f₁* (realised fulfilment)

—

### f₃* (realised service equity)

—

### HV ของ front จริง

—

## Gates

- G2.4 corr(f₁,f₃) บน front ของ qm_model: +0.615 → ✅ < 0.9
- G2.6 oracle > model > persistence (f₁*): nan > 0.050 > nan → ✅

## อ่าน
- Δ ลบ = arm นั้นแย่กว่า (9a′) Transformer · ตัวเลขหลักของ paper: Δf₁*(qm_tfm_ub) = ผลของ estimator บน encoder เดียวกัน · Δf₁*(qm_lstm_reach) = ผลของ encoder บน estimator เดียวกัน · Δ(qm_trajgeom) = เทียบ control เดิม
- f1_plan vs f1_real ต่างกันมาก = C_ij ปรับเทียบไม่ตรงกับคู่ที่ถูกเลือก (optimizer เลือกคู่ที่ C สูงเกินจริง = winner's curse) — ดู oracle gap
