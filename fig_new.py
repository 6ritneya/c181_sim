"""Figures for the new experiments: fig9 (ALINEA target sweep, 30 paired seeds) and fig10 (HDM anticipation).

    python fig_new.py     # needs results/expB2_alinea_sweep_runs.csv and results/expA3_hdm.csv
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import run_all  # noqa: F401  (applies the shared plotting style)
from run_all import C1, C2, C3, C4, INK, INK2, OUT, save


def read(name):
    with open(os.path.join(OUT, name)) as f:
        return list(csv.DictReader(f))


def fig9():
    rows = read("expB2_alinea_sweep_runs.csv")
    base = {r["seed"]: float(r["tts_vehh"]) for r in rows if r["controller"] == "none"}
    labels, data = [], []
    for c, o in [("fixed", "")] + [("alinea", f"{t:.1f}") for t in [12, 13, 14, 15, 16, 17]]:
        rr = [r for r in rows if r["controller"] == c and r["target_pct"] == o]
        data.append(np.array([100 * (float(r["tts_vehh"]) - base[r["seed"]]) / base[r["seed"]] for r in rr]))
        labels.append("Fixed\n520 veh/h" if c == "fixed" else f"ALINEA\n{float(o):.0f}%")
    fig, ax = plt.subplots(figsize=(9, 4.3))
    rng = np.random.default_rng(0)
    for i, d in enumerate(data):
        c = C3 if i == 0 else C1
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(d)), d, s=10, color=c, alpha=0.45, zorder=2)
        m = d.mean()
        bs = rng.choice(d, size=(5000, len(d))).mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="o", color=INK, ms=7, capsize=5, lw=2, zorder=3)
        ax.text(i + 0.22, m, f"{m:+.1f}%", va="center", fontsize=9.5, color=INK)
    ax.axhline(0, color=INK2, lw=1, ls="--")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Change in total time spent\nvs no control, same seed (%)")
    ax.set_title("τ = 0.3 s, 30 paired seeds: mean and 95% bootstrap CI (black), each seed (dots)")
    ax.grid(axis="x", visible=False)
    save(fig, "fig9_alinea_target_sweep.png")


def fig10():
    rows = read("expA3_hdm.csv")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    styles = [((1, 0), INK2, "Pure delay (Exp. A)"), ((1, 1), C2, "+ temporal anticipation"),
              ((3, 1), C3, "+ 3 leaders"), ((5, 1), C1, "+ 5 leaders")]
    for ax, k in zip(axs, [25, 30]):
        for (na, ant), c, lab in styles:
            rr = sorted([r for r in rows if int(r["k"]) == k and int(r["n_a"]) == na and int(r["anticipation"]) == ant],
                        key=lambda r: float(r["tau_s"]))
            t = [float(r["tau_s"]) for r in rr]; A = [float(r["A100"]) for r in rr]
            ax.plot(t, A, "-o", color=c, ms=6, mec="#fcfcfb", mew=1.5, label=lab)
            for r in rr:
                if int(r["collisions"]) > 0:
                    ax.plot(float(r["tau_s"]), float(r["A100"]), "x", color=INK, ms=10, mew=2.2, zorder=4)
        ax.axhline(1, color=INK2, lw=1, ls="--")
        ax.set_title(f"{k} veh/km"); ax.set_xlabel("Response delay τ (s)")
    axs[0].set_ylabel("Amplification A100")
    axs[0].plot([], [], "x", color=INK, ms=10, mew=2.2, label="collisions occurred")
    axs[0].legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), fontsize=9.5)
    fig.suptitle("Temporal anticipation alone still collides at τ ≥ 1.0 s; reacting to 3–5 leaders does not",
                 fontweight="bold", y=1.02)
    save(fig, "fig10_hdm_anticipation.png")


if __name__ == "__main__":
    fig10()
    if os.path.exists(os.path.join(OUT, "expB2_alinea_sweep_runs.csv")):
        fig9()
