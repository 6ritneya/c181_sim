"""Run every check and experiment, write CSV results and figures to results/.

    python run_all.py            # full run (~6-8 min on 2 cores); see README for hdm.py, exp_tmc_sweep.py
"""
import os, json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from concurrent.futures import ProcessPoolExecutor

from model import (P, idm_acc, equilibrium_from_density, string_stability_margin, critical_density,
                   fundamental_diagram, wave_speed_dqdk)
from platoon import run_platoon, wave_speed
import freeway
from freeway import simulate

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

# ---- plotting style (reference palette, slots 1-3; recessive grid) ----
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
    "lines.linewidth": 2, "legend.frameon": False, "font.family": "DejaVu Sans"})
DIV = LinearSegmentedColormap.from_list("div", ["#1f5fae", "#8db6e8", "#e9e8e4", "#f2a27f", "#c8441b"])
N_PL, T_PL, N_MEAS = 150, 900.0, 100
TAU_HUMAN = 0.6   # human delay in mixed traffic: largest tested tau that stays collision-free in pure-delay IDM


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_csv(rows, name):
    if not rows:
        return
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


# ------------------------------------------------------------------ jobs (top level for multiprocessing)
def job_platoon(args):
    k, tau, v_drop = args
    v_e = equilibrium_from_density(k)[1]
    r = run_platoon(k, np.full(N_PL, tau), N=N_PL, t_end=T_PL, v_low=max(v_e - v_drop, 0.0))
    return dict(k=k, tau_s=tau, drop_mps=v_drop, A10=r["A"][10], A40=r["A"][40], A100=r["A"][N_MEAS],
                stopped=int(r["stopped"].sum()), min_gap_m=float(r["min_gap"].min()),
                hard_brake_veh=int(r["hard_brake"].sum()), collisions=int(r["collided"].sum()))


def job_mixed(args):
    share, tau_av, seed = args
    rng = np.random.default_rng(seed)
    is_av = rng.random(N_PL) < share
    taus = np.where(is_av, tau_av, TAU_HUMAN)
    r = run_platoon(30, taus, N=N_PL, t_end=T_PL)
    return dict(av_share=share, tau_av_s=tau_av, tau_human_s=TAU_HUMAN, seed=seed, A100=r["A"][N_MEAS], stopped=int(r["stopped"].sum()),
                min_gap_m=float(r["min_gap"].min()), collisions=int(r["collided"].sum()))


def job_tmc(args):
    tau, ctrl, seed = args
    r = simulate(ctrl, tau=tau, seed=seed)
    return dict(tau_s=tau, controller=ctrl, seed=seed, tts_vehh=r["tts_vehh"], delay_vehh=r["total_delay_vehh"],
                delay_main_vehh=r["delay_main_vehh"], delay_ramp_vehh=r["delay_ramp_vehh"],
                max_ramp_queue=r["max_ramp_queue"], max_spillback=r["max_overflow"],
                max_upstream_queue=r["max_up_queue"], collisions=r["collisions"], served=r["exited"],
                conservation_ok=r["conservation_ok"])


# ------------------------------------------------------------------ verification
def verification():
    rows = []
    # 1 free motion
    v, t = 0.0, 0.0
    while v < 0.99 * P["v0"] and t < 600:
        v += float(idm_acc(1e4, v, 0.0)) * 0.1; t += 0.1
    rows.append(dict(check="Free motion", test="one vehicle, empty road, from 0 m/s",
                     result=f"reaches 0.99 v0 = {0.99*P['v0']:.1f} m/s after {t:.0f} s", passed=v >= 0.99 * P["v0"]))
    # 2 equilibrium
    errs = []
    for k in [20, 30, 40]:
        g, ve, _ = equilibrium_from_density(k)
        r = run_platoon(k, np.zeros(40), N=40, t_end=300, v_low=ve)
        errs.append(abs(r["V"][-1, 1:].mean() - ve) / ve * 100)
    rows.append(dict(check="Equilibrium", test="uniform platoon at 20/30/40 veh/km, no disturbance",
                     result=f"max speed error {max(errs):.3f}%", passed=max(errs) < 1.0))
    # 3 conservation + detectors (one freeway run)
    r = simulate("alinea", tau=0.0, seed=1)
    d1 = len(r["crossings"]["D1"]); d5 = len(r["crossings"]["D5"])
    n_main = int((r["travel"][:, 0] == 0).sum())
    rows.append(dict(check="Conservation", test="entered = exited + on road, every step; queues empty at end",
                     result=f"entered {r['entered']}, exited {r['exited']}, left {r['on_road_end']+r['queues_end']}",
                     passed=r["conservation_ok"] and r["on_road_end"] + r["queues_end"] == 0))
    rows.append(dict(check="Detector audit", test="D1 count = mainline vehicles served; D5 count = all exits",
                     result=f"D1 {d1} vs {n_main}; D5 {d5} vs {r['exited']}", passed=(d1 == n_main and d5 == r["exited"])))
    # 4 boundary: inflow reaches downstream
    r2 = simulate("none", tau=0.0, seed=2, demand=lambda t: (1500.0, 0.0) if t < 1800 else (0.0, 0.0), t_end=2100)
    rec = [o for o in r2["obs"] if o["sensor_id"] == "D5" and 600 <= o["interval_end_s"] <= 1800]
    rec1 = [o for o in r2["obs"] if o["sensor_id"] == "D1" and 600 <= o["interval_end_s"] <= 1800]
    q5 = sum(o["count"] for o in rec) * 3600 / (1200); q1 = sum(o["count"] for o in rec1) * 3600 / 1200
    rows.append(dict(check="Boundaries", test="1,500 veh/h inflow, no ramp, 600-1800 s",
                     result=f"D1 {q1:.0f} veh/h, D5 {q5:.0f} veh/h", passed=abs(q5 - q1) / q1 < 0.02))
    # 5 time step
    res = []
    for tau in [0.4, 1.0]:
        a = run_platoon(30, np.full(N_PL, tau), N=N_PL, t_end=T_PL, dt=0.1)["A"][N_MEAS]
        b = run_platoon(30, np.full(N_PL, tau), N=N_PL, t_end=T_PL, dt=0.05)["A"][N_MEAS]
        res.append((tau, a, b, abs(a - b) / a * 100))
    rows.append(dict(check="Time step", test="A100 at 30 veh/km, dt 0.1 vs 0.05 s",
                     result="; ".join(f"tau {t}: {a:.2f} vs {b:.2f} ({e:.1f}%)" for t, a, b, e in res),
                     passed=all(e < 5 for *_, e in res)))
    write_csv(rows, "verification.csv")
    return rows


# ------------------------------------------------------------------ figures
def fig_fd():
    ks, qs = fundamental_diagram(kmax=130)
    kc = critical_density()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    st = ks < kc
    ax.plot(ks[st], qs[st], color=C1, label="string-stable (τ = 0)")
    ax.plot(ks[~st], qs[~st], color=C2, label="string-unstable (τ = 0)")
    ax.axvline(kc, color=INK2, lw=1, ls="--")
    ax.text(kc + 2, 600, f"k_c ≈ {kc:.1f} veh/km\ncapacity ≈ {qs.max():.0f} veh/h", color=INK2, fontsize=10)
    for k in [25, 30, 40]:
        q = equilibrium_from_density(k)[2]
        ax.plot(k, q, "o", ms=8, color=INK, mec="#fcfcfb", mew=2)
        ax.annotate(f"{k}", (k, q), textcoords="offset points", xytext=(0, -18), ha="center", fontsize=10, color=INK2)
    ax.set_xlabel("Density k (veh/km)"); ax.set_ylabel("Flow q (veh/h)")
    ax.set_title("IDM fundamental diagram and stability (test densities marked)")
    ax.legend(loc="upper right"); ax.set_ylim(0, 2800)
    save(fig, "fig1_fundamental_diagram.png")


def fig_A_vs_tau(rows):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for k, c in zip([25, 30, 40], [C1, C3, C2]):
        rr = sorted([r for r in rows if r["k"] == k and r["drop_mps"] == 12.95], key=lambda r: r["tau_s"])
        if not rr:
            rr = sorted([r for r in rows if r["k"] == k], key=lambda r: r["tau_s"])
        ax.plot([r["tau_s"] for r in rr], [r["A100"] for r in rr], "-o", ms=6, color=c, label=f"{k} veh/km",
                mec="#fcfcfb", mew=1.5)
    ax.axhline(1.0, color=INK2, lw=1, ls="--")
    ax.text(0.01, 1.05, "A = 1   (above: grows, below: decays)", color=INK2, fontsize=10)
    ax.axvspan(0.1, 0.3, color=C1, alpha=0.08); ax.text(0.12, 2.6, "AV sensing\n0.1–0.3 s", fontsize=9, color=INK2)
    ax.axvline(1.0, color=INK2, lw=1, ls=":"); ax.text(1.02, 2.6, "human\n~1 s", fontsize=9, color=INK2)
    ax.set_xlabel("Response delay τ (s)"); ax.set_ylabel(f"Amplification A{N_MEAS} (vehicle {N_MEAS} / vehicle 1)")
    ax.set_title("Braking disturbance (lead brakes to 10 m/s): decay vs growth")
    ax.legend(loc="upper left", bbox_to_anchor=(0.33, 1.0), title="Density"); ax.set_ylim(0, 2.9)
    save(fig, "fig2_amplification_vs_delay.png")


def fig_map(rows):
    taus = sorted(set(r["tau_s"] for r in rows)); drops = sorted(set(r["drop_mps"] for r in rows))
    Z = np.full((len(drops), len(taus)), np.nan)
    for r in rows:
        Z[drops.index(r["drop_mps"]), taus.index(r["tau_s"])] = r["A100"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    im = ax.pcolormesh(np.arange(len(taus) + 1), np.arange(len(drops) + 1), np.log10(Z), cmap=DIV,
                       norm=TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1), edgecolors="#fcfcfb", linewidth=2)
    for i in range(len(drops)):
        for j in range(len(taus)):
            ax.text(j + 0.5, i + 0.5, f"{Z[i, j]:.2f}" if Z[i, j] < 10 else f"{Z[i, j]:.0f}", ha="center",
                    va="center", fontsize=8.5, color=INK)
    ax.set_xticks(np.arange(len(taus)) + 0.5); ax.set_xticklabels([f"{t:.1f}" for t in taus])
    ax.set_yticks(np.arange(len(drops)) + 0.5); ax.set_yticklabels([f"{d:g}" for d in drops])
    ax.set_xlabel("Response delay τ (s)"); ax.set_ylabel("Lead speed drop (m/s)")
    ax.set_title(f"A{N_MEAS} at 30 veh/km: bigger disturbances need less delay to grow")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, ticks=[-1, 0, 1]); cb.ax.set_yticklabels(["0.1 (decays)", "1", "10 (grows)"])
    cb.outline.set_visible(False)
    save(fig, "fig3_metastability_map.png")


def fig_traj():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, tau in zip(axs, [0.2, 0.6]):
        r = run_platoon(30, np.full(N_PL, tau), N=N_PL, t_end=400, record_every=0.5)
        T, X, V = r["T"], r["X"], r["V"]
        for i in range(0, N_PL + 1, 3):
            pts = ax.scatter(T, X[:, i] / 1000, c=V[:, i], cmap="viridis_r", vmin=0, vmax=25, s=0.4, rasterized=True)
        ax.set_title(f"τ = {tau} s\nA{N_MEAS} = {r['A'][N_MEAS]:.2f}, {int(r['stopped'].sum())} vehicles stop", fontsize=12)
        ax.set_xlabel("Time (s)"); ax.grid(False)
    axs[0].set_ylabel("Position (km)")
    cb = fig.colorbar(pts, ax=axs, shrink=0.9); cb.set_label("Speed (m/s)"); cb.outline.set_visible(False)
    fig.suptitle("Time–space diagram, 30 veh/km (every 3rd vehicle shown)", fontweight="bold", y=1.04)
    save(fig, "fig4_time_space.png")


def fig_mixed(rows):
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for tau_av, c, lab in [(0.2, C1, "AV τ = 0.2 s (best case)"), (0.5, C2, "AV τ = 0.5 s (with actuation lag)")]:
        shares = sorted(set(r["av_share"] for r in rows))
        med = []; lo = []; hi = []
        for s in shares:
            vals = [r["A100"] for r in rows if r["av_share"] == s and r["tau_av_s"] == tau_av]
            med.append(np.median(vals)); lo.append(min(vals)); hi.append(max(vals))
        x = np.array(shares) * 100
        ax.errorbar(x, med, yerr=[np.array(med) - lo, np.array(hi) - np.array(med)], fmt="-o", color=c, ms=7,
                    capsize=4, lw=2, mec="#fcfcfb", mew=1.5, label=lab)
    ax.axhline(1.0, color=INK2, lw=1, ls="--")
    ax.set_xlabel("Automated-vehicle share (%)  [humans τ = 0.6 s]"); ax.set_ylabel(f"A{N_MEAS} (median, min–max of 5 seeds)")
    ax.set_title("Mixed traffic at 30 veh/km"); ax.legend(loc="lower left"); ax.set_xticks([0, 25, 50, 75, 100])
    save(fig, "fig5_mixed_traffic.png")


def fig_tmc_bars(rows):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    names = {"none": "No control", "fixed": "Fixed 520 veh/h", "alinea": "ALINEA"}
    for ax, tau in zip(axs, [0.0, 0.3]):
        for i, (ctrl, c) in enumerate([("none", INK2), ("fixed", C3), ("alinea", C1)]):
            vals = np.array([r["tts_vehh"] for r in rows if r["tau_s"] == tau and r["controller"] == ctrl])
            ax.bar(i, vals.mean(), color=c, width=0.6)
            ax.scatter(np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals)), vals, s=14, color=INK, zorder=3)
            ax.text(i, vals.mean() * 0.06, f"{vals.mean():.0f}", ha="center", fontsize=12, color="white", fontweight="bold")
        ax.set_xticks(range(3)); ax.set_xticklabels(names.values())
        ax.set_title(f"τ = {tau} s" + ("  (standard IDM)" if tau == 0 else "  (delayed response)"))
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("Total time spent (veh·h), incl. all queues")
    fig.suptitle("Basic TMC: mean of 10 seeds (bars) and each seed (dots)", fontweight="bold")
    save(fig, "fig7_tmc_comparison.png")


def fig_tmc_timeseries(seed):
    fig, axs = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    for ctrl, c, lab in [("none", INK2, "No control"), ("alinea", C1, "ALINEA")]:
        r = simulate(ctrl, tau=0.3, seed=seed)
        d2 = [o for o in r["obs"] if o["sensor_id"] == "D3"]
        axs[0].plot([o["interval_end_s"] / 60 for o in d2], [o["mean_speed_mps"] for o in d2], color=c, label=lab, lw=1.6)
        s = r["series"]
        axs[1].plot([q["t"] / 60 for q in s], [q["ramp_queue"] + q["overflow"] for q in s], color=c, lw=1.6)
        if ctrl == "alinea":
            axs[2].step([q["issue_time_s"] / 60 for q in r["ctrl"]], [q["value"] for q in r["ctrl"]], where="post", color=c, lw=1.6)
            axs[2].axhline(520, color=C3, lw=1.2, ls="--"); axs[2].text(1, 560, "fixed-rate plan 520 veh/h", color=INK2, fontsize=9)
            write_csv(r["ctrl"], "tmc_decision_log_alinea.csv")
            write_csv([o for o in r["obs"]], "detector_observations_alinea.csv")
    axs[0].set_ylabel("Speed at D3,\n2.4 km (m/s)"); axs[0].legend(loc="upper right", ncol=2)
    axs[1].set_ylabel("Ramp queue +\nspillback (veh)"); axs[1].axhline(30, color=INK2, lw=1, ls=":")
    axs[1].text(1, 33, "storage 30 veh", color=INK2, fontsize=9)
    axs[2].set_ylabel("ALINEA release\nrate (veh/h)"); axs[2].set_xlabel("Time (min)")
    for a in axs:
        a.axvspan(5, 35, color=C2, alpha=0.06)
    axs[0].text(5.5, 33, "peak demand 2,000 + 600 veh/h", color=INK2, fontsize=9)
    fig.suptitle(f"One run (τ = 0.3 s, seed {seed}): observe → decide → act → response", fontweight="bold")
    save(fig, "fig8_tmc_timeseries.png")


def degraded_test(seed):
    ok = simulate("alinea", tau=0.3, seed=seed)
    bad = simulate("alinea", tau=0.3, seed=seed, detector_dropout=(1200.0, 1800.0))
    fb = [c for c in bad["ctrl"] if "fallback" in c["reason"]]
    write_csv(bad["ctrl"], "tmc_decision_log_detector_dropout.csv")
    return dict(tts_normal=ok["tts_vehh"], tts_dropout=bad["tts_vehh"], fallback_steps=len(fb),
                fallback_first=fb[0]["issue_time_s"] if fb else None, fallback_last=fb[-1]["issue_time_s"] if fb else None,
                collisions=bad["collisions"])


if __name__ == "__main__":
    summary = {}
    print("== verification"); ver = verification()
    for r in ver: print(r)
    fig_fd()
    ks = [25, 30, 40]; taus = [round(x, 1) for x in np.arange(0, 1.21, 0.1)]
    jobsA = [(k, t, 12.95) for k in ks for t in taus]
    jobsM = [(30, t, d) for t in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0] for d in [2, 4, 6, 8, 10, 12.95]]
    jobsX = [(s, ta, seed) for s in [0, 0.25, 0.5, 0.75, 1.0] for ta in [0.2, 0.5] for seed in range(5)]
    seeds = range(10)
    jobsT = [(tau, c, s) for tau in [0.0, 0.3] for c in ["none", "fixed", "alinea"] for s in seeds]
    with ProcessPoolExecutor() as ex:
        A = list(ex.map(job_platoon, jobsA)); M = list(ex.map(job_platoon, jobsM))
        X = list(ex.map(job_mixed, jobsX)); TM = list(ex.map(job_tmc, jobsT))
    write_csv(A, "expA1_delay_sweep.csv"); write_csv(M, "expA_metastability_map.csv")
    write_csv(X, "expA2_mixed_traffic.csv"); write_csv(TM, "expB_tmc_runs.csv")
    fig_A_vs_tau(A); fig_map(M); fig_traj(); fig_mixed(X); fig_tmc_bars(TM)
    # representative seed = median no-control TTS at tau 0.3
    nc = sorted([r for r in TM if r["tau_s"] == 0.3 and r["controller"] == "none"], key=lambda r: r["tts_vehh"])
    rep = nc[len(nc) // 2]["seed"]
    fig_tmc_timeseries(rep)
    summary["degraded"] = degraded_test(rep); summary["rep_seed"] = rep
    # wave speed for stop-and-go case vs theory
    r = run_platoon(30, np.full(N_PL, 1.0), N=N_PL, t_end=T_PL)
    summary["wave_speed_tau1_kmh"] = wave_speed(r, 20, 100)
    summary["dqdk_30_kmh"] = wave_speed_dqdk(30)
    summary["critical_density"] = critical_density()
    summary["capacity"] = float(fundamental_diagram()[1].max())
    tm = {}
    for tau in [0.0, 0.3]:
        for c in ["none", "fixed", "alinea"]:
            rr = [x for x in TM if x["tau_s"] == tau and x["controller"] == c]
            tm[f"{tau}_{c}"] = {k: float(np.mean([x[k] for x in rr])) for k in
                                ["tts_vehh", "delay_vehh", "delay_main_vehh", "delay_ramp_vehh", "max_ramp_queue",
                                 "max_spillback", "max_upstream_queue", "collisions"]}
            tm[f"{tau}_{c}"]["tts_min"] = min(x["tts_vehh"] for x in rr); tm[f"{tau}_{c}"]["tts_max"] = max(x["tts_vehh"] for x in rr)
    summary["tmc"] = tm
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))
