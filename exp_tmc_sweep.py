"""Experiment B2: ALINEA target sweep with 30 paired seeds (tau = 0.3 s).

Every controller sees the same 30 Poisson demand seeds, so each seed gives a paired difference
dTTS = TTS(controller) - TTS(no control). This removes the seed-to-seed spread in demand.

    python exp_tmc_sweep.py      # ~9 min on 2 cores -> results/expB2_alinea_sweep*.csv, fig9
"""
import os, csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from freeway import simulate

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
TAU = 0.3
SEEDS = range(30)
TARGETS = [12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
CONFIGS = [("none", None), ("fixed", None)] + [("alinea", o) for o in TARGETS]


def job(args):
    ctrl, target, seed = args
    kw = dict(alinea_kwargs=dict(o_target=target)) if ctrl == "alinea" else {}
    r = simulate(ctrl, tau=TAU, seed=seed, **kw)
    return dict(controller=ctrl, target_pct=target if target else "", seed=seed, tts_vehh=r["tts_vehh"],
                delay_main_vehh=r["delay_main_vehh"], delay_ramp_vehh=r["delay_ramp_vehh"],
                max_ramp_queue=r["max_ramp_queue"], max_spillback=r["max_overflow"],
                max_upstream_queue=r["max_up_queue"], collisions=r["collisions"],
                conservation_ok=r["conservation_ok"])


def bootstrap_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return np.percentile(m, [2.5, 97.5])


if __name__ == "__main__":
    jobs = [(c, o, s) for c, o in CONFIGS for s in SEEDS]
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(job, jobs))
    with open(os.path.join(OUT, "expB2_alinea_sweep_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    base = {r["seed"]: r["tts_vehh"] for r in rows if r["controller"] == "none"}
    summ = []
    for c, o in CONFIGS:
        rr = sorted([r for r in rows if r["controller"] == c and (r["target_pct"] or None) == o], key=lambda r: r["seed"])
        tts = np.array([r["tts_vehh"] for r in rr])
        d = np.array([r["tts_vehh"] - base[r["seed"]] for r in rr])
        pct = 100 * d / np.array([base[r["seed"]] for r in rr])
        lo, hi = bootstrap_ci(pct) if c != "none" else (0.0, 0.0)
        summ.append(dict(controller=c, target_pct=o if o else "", mean_tts_vehh=tts.mean(), sd_tts_vehh=tts.std(ddof=1),
                         mean_change_pct=pct.mean(), ci95_lo_pct=lo, ci95_hi_pct=hi,
                         seeds_better=int((d < 0).sum()), n=len(d),
                         mean_max_ramp_queue=np.mean([r["max_ramp_queue"] for r in rr]),
                         mean_max_spillback=np.mean([r["max_spillback"] for r in rr]),
                         collisions=int(sum(r["collisions"] for r in rr)),
                         all_conserved=all(r["conservation_ok"] for r in rr)))
    with open(os.path.join(OUT, "expB2_alinea_sweep_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)
    for s in summ:
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()})
