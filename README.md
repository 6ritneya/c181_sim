# C181 project: response delay, disturbance growth and a basic ramp-metering TMC

Author: Britney Aguirre. Python 3.10+.

## Run

```
pip install numpy scipy matplotlib
python run_all.py        # verification, Experiments A1/A2/B, figures 1-5, 7, 8 (~6-8 min on 2 cores)
python model.py          # equilibrium speeds, stability margin, capacity (prints Table 2)
python hdm.py            # Experiment A3: anticipation / multi-anticipation (~1 min)
python exp_tmc_sweep.py  # Experiment B2: ALINEA target sweep, 30 paired seeds (~9 min)
python fig_new.py        # figures 9 and 10 (run after the two scripts above)
```

### Which command makes each mid-report figure and table

| Mid-report item | Command | Output in `results/` |
| --- | --- | --- |
| Table 2 (equilibrium), Fig. 1 | `python model.py`, `python run_all.py` | printout, `fig1_fundamental_diagram.png` |
| Table 4 (verification) | `python run_all.py` | `verification.csv` |
| Fig. 2, Fig. 3, Fig. 4 | `python run_all.py` | `fig2_amplification_vs_delay.png`, `fig4_time_space.png`, `fig3_metastability_map.png` |
| Fig. 5 | `python run_all.py` | `fig5_mixed_traffic.png` |
| Fig. 6 | `python hdm.py && python fig_new.py` | `fig10_hdm_anticipation.png`, `expA3_hdm.csv` |
| Fig. 7, Table 5 | `python run_all.py` | `fig8_tmc_timeseries.png`, `tmc_decision_log_alinea.csv` |
| Fig. 8, Table 6 | `python exp_tmc_sweep.py && python fig_new.py` | `fig9_alinea_target_sweep.png`, `expB2_alinea_sweep_summary.csv` |

All outputs go to `results/`.

## Files

| File | What it does |
| --- | --- |
| `model.py` | IDM acceleration (Eq. 3), equilibrium speed, linear string-stability test, fundamental diagram |
| `platoon.py` | Experiment A. A lead vehicle brakes and 150 followers react with delay τ (Eq. 4). Measures amplification, stops, gaps and collisions |
| `freeway.py` | Experiment B. A 4 km freeway with an on-ramp merge at 2.5–2.8 km, ramp queue (30 veh storage), detectors D1–D5 and the TMC loop |
| `run_all.py` | Runs verification, Experiments A1/A2, the metastability map, the TMC comparison and the detector-dropout test, and makes the figures |
| `hdm.py` | Experiment A3. Adds Human Driver Model anticipation (temporal extrapolation by τ, reaction to n_a = 1, 3, 5 leaders) to the delayed platoon |
| `exp_tmc_sweep.py` | Experiment B2. ALINEA occupancy target 12–17 % vs fixed vs no control, 30 paired seeds, τ = 0.3 s, bootstrap CIs |
| `fig_new.py` | Figures 9 and 10 for Experiments A3 and B2 |

## Inputs (defaults)

- IDM parameters: v0 = 33.3 m/s, T = 1.0 s, s0 = 2 m, a = 1.0 m/s², b = 1.5 m/s², δ = 4, vehicle length 5 m.
- Braking bound: b_max = 9 m/s². This is added on top of IDM. Without it, delayed IDM produced decelerations above 30 m/s².
- Time step: dt = 0.1 s. Delay buffer: τ/dt steps.
- Demand (Experiment B): mainline 1600 / 2000 / 1500 veh/h and ramp 300 / 600 / 300 veh/h for 0–5, 5–35 and 35–60 min, then zero until queues clear (75 min total). Arrivals are Poisson, 10 seeds.
- TMC: control interval 60 s. Observes occupancy at D4 (2.8 km). ALINEA with K_R = 70 veh/h per %, target 17 %, bounds 200–1800 veh/h. The queue override keeps the ramp queue at or below 25 veh. If the detector is missing, it falls back to 520 veh/h. The fixed plan is 520 veh/h (capacity 2,519 − 2,000).

## Outputs

- `verification.csv`: pass/fail checks
- `expA1_delay_sweep.csv`, `expA_metastability_map.csv`, `expA2_mixed_traffic.csv`
- `expB_tmc_runs.csv`: one row per controller × τ × seed
- `tmc_decision_log_alinea.csv`: control-command format (issue_time_s, apply_time_s, actuator_id, action, value, units, plus observation, forecast, queue and reason)
- `detector_observations_alinea.csv`: observation format (sensor_id, interval, count, mean speed, occupancy, quality_flag)
- `tmc_decision_log_detector_dropout.csv`: the degraded-information test
- `expA3_hdm.csv`, `expB2_alinea_sweep_runs.csv`, `expB2_alinea_sweep_summary.csv`
- `summary.json` and figures `fig1`–`fig5`, `fig7`–`fig10` (there is no fig6)

## Known limitations

- Single lane, no lane changing. The merge is gap acceptance: polite at first (follower brakes no harder than 2 m/s²), then forced after 4 s of waiting (up to 5.4 m/s²).
- Pure-delay IDM collides at τ ≥ 0.8–0.9 s. Temporal anticipation alone still collides at τ ≥ 1.0 s; reacting to 3–5 leaders (`hdm.py`) removes collisions up to τ = 1.2 s.
- The stop-and-go wave speed is −22 to −32 km/h at collision-free delays (τ = 0.6–0.7 s), faster than the empirical ≈ −15 km/h. It only matches at τ = 1.0 s, where collisions occur.
- The forecast (linear trend, 60 s ahead) is advisory only. It is logged but not used by the controller.
- With τ = 0, the merge has no capacity drop, so metering cannot reduce delay.
