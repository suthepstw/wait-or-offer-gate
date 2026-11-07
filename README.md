# Calibrated wait-or-offer gating — code and results

Code, configuration files, instance generator and result files for the manuscript
*"Calibrated Wait-or-Offer Gating for Irrevocable Task Offers in Prediction-Driven Mobile Crowdsourcing"*.

The repository lets you (a) recompute every table and figure of the paper from the stored result files without
re-running any experiment, and (b) re-run the full pipeline from the raw taxi traces.

> Code comments are partly in Thai. Identifiers, command-line options and all outputs used in the paper are in English.

## Contents

| Folder | What it holds |
|---|---|
| `code/` | All scripts, flat (they import each other by file name). Run every command from inside `code/`. |
| `code/config/` | Data configuration per city (cleaning thresholds, date splits, bounding box). |
| `results/` | Per-instance result files (CSV/JSON/Markdown) of every run reported in the paper. No trajectories. |
| `models/` | Calibration files (dispersion factors, isotonic maps, run configurations) and, where available, the trained weights of the frozen predictors. |
| `MANIFEST.sha256` | SHA-256 of every file in the repository. |

Raw trajectories, processed windows, instance files (`inst_*.npz`, `oracle_*.npz`) and encoder states are **not** included,
because they are derived from the CRAWDAD traces, which must be obtained from the archive under its own terms (see *Data*).

## Environment

Python 3 (version in `ENVIRONMENT.txt`), packages pinned in `requirements.txt`:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

The predictor is trained with TensorFlow on one GPU; everything from instance generation onwards runs on CPU
(about 3 s per NSGA-II solve; the allocation experiments were run in parallel on a 24-core machine).

## Data

* San Francisco: CRAWDAD `epfl/mobility` (cabspotting), version 2009-02-24.
* Rome: CRAWDAD `roma/taxi`, version 2014-07-17.

Place the raw files under `code/DataTaxiSanFran/` and `code/DataTaxiRome/` respectively, as expected by `code/config/data_<city>.yaml`.

## Reproducing the tables and figures from the stored results

```bash
cd code
ln -s ../results/* .           # the analysis scripts read result folders relative to code/
python sE6_paperfigs.py        # all figures of the paper -> paper_figs/  (+ paper_figs/paperfigs_check.md with every plotted value)
python sE6_gatedef.py --final e6gd_final --select e6gd_dev/GATEDEF_select.json --b0-final e6final_sticky/nsga2_sh_g1.5 --nsga-sticky e6final_sticky/nsga2 --nsga-free e6final_free/nsga2
python sE6_nullgate.py --final e6ng_final --select e6ng_dev/NULLGATE_select.json --model-gate e6gd_final/b0dz_k1.5 --model-none e6final_sticky/nsga2
```

The two analysis scripts rewrite `GATEDEF_final.md` and `NULLGATE_final.md` (paired bootstrap with 2,000 resamples and a fixed seed, Wilcoxon signed-rank tests, Holm correction, win/loss counts).

### Where each result of the paper comes from

| Paper item | Result folder(s) | Script |
|---|---|---|
| Prediction quality (Table *pred*), predictor ablations (Appendix A) | `results/reports/`, `models/<city>/*/D7*` | `sD7_pred_eval.py`, `sD7_reach_eval.py` |
| AUC over the cycle (first figure, left) | `results/e5inst_v3/` | `sE5_instances.py` |
| Static vs. rolling re-plan, free vs. sticky (first figure, right; Table *replan*) | `results/e5replan/`, `results/e5replan_sticky3/` | `sE5_replan.py` |
| Optimiser comparison (Table *opt*, hypervolume figure) | `results/e6_sticky/`, `results/e6_free/`, `results/e6_figs/` | `sE6_algos.py`, `sE5_replan.py --algo`, `sE6_figs.py` |
| Gate estimates ρ(h), V_k(t) | `results/e6_rho/`, `results/e6_rho_null/` | `sE6_rho.py` |
| Development-day grids (Appendix C) | `results/e6gd_dev/`, `results/e6ng_dev/`, `results/e6dev_sticky/`, `results/e6dev_free/` | `run_gatedef_v2.sh`, `run_nullgate.sh` |
| Main results, capacity timelines, paired and distribution plots, ablation, greedy allocator (Tables *main*, *abl*) | `results/e6final_sticky/`, `results/e6final_free/`, `results/e6gd_final/`, `results/e6ng_final/` | `sE5_replan.py`, `sE6_gatedef.py`, `sE6_nullgate.py`, `sE6_paperfigs.py` |
| MOPSO-IQ (random key) with the final gate (Section *Across allocators*; Appendix D, amendment 7) | `results/e6c5_final/` | `sE5_replan.py --algo mopso_iq_rk --gate 1.5 --drop-zero` |
| Learned offer-or-wait policy (Section *Q-gate*) | `results/e6_qgate/` | `sE6_qgate.py` |

Each run folder has the layout `<variant>/<city>/` with `e5_runs.csv` (one row per instance, arm and mode: realised completion `f1*`, fairness `f2*`, `f3*`, capacity used, offers made), `e5_timeline.csv` (per decision point) and, for the optimiser runs, `e5_trace.csv` (per generation).

## Re-running the pipeline

Every script prints its options with `-h`. The order is:

| Step | Script | Output |
|---|---|---|
| 1. Trace audit and task-radius choice | `sD0_audit.py`, `sD0_3_grid.py` | `data/<city>/reports/` |
| 2. Cleaning, 60 s resampling, segments, day splits, windows | `sD1_D4_prepare_v2.py` (uses `sD1_D4_prepare.py`) | `data/<city>/processed/` |
| 3. Predictor training (Transformer encoder, mixture heads, reachability head) | `sD5_train.py`, `sD5_train_v4.py` (queue: `run_reach.sh <city> tfm`) | `models/<city>/reach_tfm/` |
| 4. Dispersion calibration per horizon | `sD6_calibrate.py` | `models/<city>/<run>/seed*/calib.json` |
| 5. Prediction evaluation and isotonic calibration | `sD7_pred_eval.py`, `sD7_reach_eval.py` | `models/<city>/<run>/D7*/` |
| 6. Instances (60-min cycles, 60 workers, 200 tasks) | `sE1_instances_v3.py` (uses `sE1_instances.py`) | `instances_v3/` (test), `instances_devsplit/` (development) |
| 7. Encoder states at t = 0,10,…,50 min | `sE5_encode.py` | `e5state/` |
| 8. Success probabilities C_ij(t) for every predictor | `sE5_instances.py` | `e5inst_v3/`, `e5inst_devsplit/` |
| 9. Allocation and replay (NSGA-II, swarms, greedy; free/sticky; gate) | `sE5_replan.py`, `sE4_lite.py`, `sE6_algos.py` | `e5replan*/`, `e6*/` |
| 10. Gate estimates on development days | `sE6_rho.py` | `e6_rho*/` |
| 11. Development grids, then locked test runs | `run_gatedef_v2.sh dev|g2ext|gdev|final`, `run_nullgate.sh dev|final` | `e6gd_*/`, `e6ng_*/` |
| 12. Figures and statistics | `sE6_paperfigs.py`, `sE6_figs.py`, `sE6_gatedef.py`, `sE6_nullgate.py` | `paper_figs/`, reports |

Example (step 9, one development run of the gate with the learned predictor):

```bash
python sE5_replan.py --city sf --inst instances_devsplit --e5 e5inst_devsplit --out e6gd_dev/b0dz_k1.5 \
  --algo nsga2 --arms qm_model --adaptive-only --withdraw sticky --gen 480 --e4 none \
  --shrink e6_rho/sf/rho.json --gate 1.5 --drop-zero
```

Main flags of `sE5_replan.py`: `--withdraw free|sticky` (commitment regime), `--gate κ` (wait-or-offer gate),
`--shrink rho.json` (horizon factors and slot values), `--drop-zero` (zero-probability filter),
`--delay t_d` (fixed waiting), `--gate-fixed τ` (fixed threshold), `--gate-raw` (gate on uncalibrated C),
`--algo nsga2|greedy|mopso*` (allocator). Without the gate flags the script reproduces the ungated runs exactly.

### Protocol

All parameters were chosen on development days; test instances (`instances_v3`) were used only for the locked final runs,
each executed once. The `final` modes of `run_gatedef_v2.sh` and `run_nullgate.sh` refuse to run when the final folder already exists. The specification and its amendments are listed in Appendix D of the paper.

### Frozen files

`MANIFEST.sha256` records the hash of every file. The prediction evaluation script `sD7_pred_eval.py` used for all reported numbers has a SHA-256 starting with `0fe82c5d`; `run_reach.sh` checks this before running.

## Licence

MIT (see `LICENSE`). The CRAWDAD traces are subject to the archive's own terms and are not redistributed here.
