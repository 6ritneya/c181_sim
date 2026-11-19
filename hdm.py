"""Experiment A3: does anticipation (Human Driver Model ideas) remove collisions at human-like delays?

Follower i observes the state tau seconds ago and compensates in two ways (Treiber, Kesting & Helbing, 2006):
  1. Temporal anticipation: extrapolate the delayed state forward by tau
        s_est = s(t-tau) - tau * dv(t-tau)        (constant approach rate)
        v_est = v(t-tau) + tau * a(t-tau)          (constant acceleration, own vehicle)
  2. Spatial multi-anticipation: react to the n_a vehicles ahead
        a = a_max [1 - (v/v0)^delta] - (a_max / c_na) * sum_{j=1..n_a} ( s*(v, dv_j) / s_j )^2
     with s_j = x_{i-j} - x_i - j L (so s_j = j s in equilibrium) and c_na = sum_j 1/j^2,
     which keeps the equilibrium speed-gap relation identical to IDM.
No estimation noise is added (deterministic, so results compare directly with Experiment A).
Leader disturbance and metrics are the same as platoon.py.

    python hdm.py      # ~1 min -> results/expA3_hdm.csv
"""
import os, csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from model import P, s_star, equilibrium_from_density
from platoon import leader_speed_profile

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def hdm_acc(gaps, v, dvs, avail, p=P):
    """gaps, dvs: arrays (n_a, N) for the j-th leader; v: (N,); avail[j, i] True if leader j exists."""
    na = gaps.shape[0]
    w = 1.0 / np.arange(1, na + 1)[:, None] ** 2
    c = np.sum(w * avail, axis=0)          # per-vehicle normalisation over the leaders it actually has
    inter = np.zeros_like(v)
    for j in range(na):
        sj = np.maximum(gaps[j], 0.01)
        inter += np.where(avail[j], (s_star(v, dvs[j], p) / sj) ** 2, 0.0)
    acc = p["a"] * (1.0 - (np.maximum(v, 0) / p["v0"]) ** p["delta"]) - p["a"] * inter / c
    return np.maximum(acc, -p["b_max"])


def run_platoon_hdm(k, tau, n_a=1, anticipate=True, dt=0.1, N=150, t_end=900.0, v_low=10.0, p=P):
    gap_e, v_e, _ = equilibrium_from_density(k, p)
    L = p["L"]; n = N + 1
    x = -np.arange(n) * (gap_e + L); v = np.full(n, v_e); a_prev = np.zeros(n)
    d = int(round(tau / dt)); H = d + 1
    hx = np.tile(x, (H, 1)); hv = np.full((H, n), v_e); ha = np.zeros((H, n))
    # history starts at equilibrium (only position differences are used, so a constant shift is harmless)
    steps = int(round(t_end / dt))
    vmin = np.full(n, np.inf); min_gap = np.full(N, np.inf); collided = np.zeros(N, bool)
    fol = np.arange(1, n)
    for it in range(steps):
        t = it * dt
        slot = it % H
        hx[slot], hv[slot], ha[slot] = x, v, a_prev
        ds = (it - d) % H
        X, V, A = hx[ds], hv[ds], ha[ds]
        gaps = np.empty((n_a, N)); dvs = np.empty((n_a, N)); avail = np.empty((n_a, N), bool)
        for j in range(1, n_a + 1):
            lead = np.maximum(fol - j, 0)
            jj = fol - lead                      # actual number of vehicles ahead used (< j near the front)
            gaps[j - 1] = X[lead] - X[fol] - jj * L
            dvs[j - 1] = V[fol] - V[lead]
            avail[j - 1] = jj == j                # vehicles near the front have fewer than n_a leaders
        v_own = V[fol]
        if anticipate and d > 0:
            gaps = gaps - tau * dvs
            v_own = np.maximum(v_own + tau * A[fol], 0.0)
        acc = hdm_acc(gaps, v_own, dvs, avail, p)
        v_lead_new = leader_speed_profile(t + dt, v_e, v_low=v_low)
        v_new = np.maximum(v[1:] + acc * dt, 0.0)
        x[1:] += 0.5 * (v[1:] + v_new) * dt
        x[0] += 0.5 * (v[0] + v_lead_new) * dt
        a_prev = np.concatenate([[(v_lead_new - v[0]) / dt], acc])
        v[1:] = v_new; v[0] = v_lead_new
        g = x[:-1] - x[1:] - L
        coll = g <= 0
        if coll.any():
            collided |= coll
            for i in np.where(coll)[0]:
                x[i + 1] = x[i] - L - 0.01; v[i + 1] = min(v[i + 1], v[i])
            g = x[:-1] - x[1:] - L
        min_gap = np.minimum(min_gap, g); vmin = np.minimum(vmin, v)
    drop = v_e - vmin
    A = drop / drop[1]
    return dict(A100=float(A[100]), stopped=int((vmin[1:] < 1.0).sum()), min_gap=float(min_gap.min()),
                collisions=int(collided.sum()), v_end_err=float(abs(v[1:].mean() - v_e) / v_e * 100))


def job(args):
    k, tau, n_a, ant = args
    r = run_platoon_hdm(k, tau, n_a=n_a, anticipate=ant)
    return dict(k=k, tau_s=tau, n_a=n_a, anticipation=int(ant), **r)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    taus = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    variants = [(1, False), (1, True), (3, True), (5, True)]
    jobs = [(k, t, na, ant) for k in [25, 30] for t in taus for na, ant in variants]
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(job, jobs))
    with open(os.path.join(OUT, "expA3_hdm.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for r in rows:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
