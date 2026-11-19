"""Experiment A: braking disturbance in a platoon with response delay tau.

Vehicle 0 is an externally driven leader (the 'disturber'). Followers 1..N use IDM evaluated on
inputs (own speed, gap, approach rate) observed tau seconds earlier (Eq. 4 of the proposal).
Positions and speeds are updated with the CURRENT state (ballistic update, v >= 0).
"""
import numpy as np
from model import P, idm_acc, equilibrium_from_density


def leader_speed_profile(t, v_e, t0=60.0, v_low=10.0, decel=3.0, hold=5.0, acc=1.0):
    """Lead vehicle: cruise at v_e, brake at -decel to v_low, hold, accelerate back at +acc."""
    t_brake = (v_e - v_low) / decel
    t1, t2 = t0 + t_brake, t0 + t_brake + hold
    t3 = t2 + (v_e - v_low) / acc
    if t < t0:
        return v_e
    if t < t1:
        return v_e - decel * (t - t0)
    if t < t2:
        return v_low
    if t < t3:
        return v_low + acc * (t - t2)
    return v_e


def run_platoon(k, taus, dt=0.1, N=80, t_end=600.0, v_low=10.0, p=P, record_every=1.0):
    """taus: array of length N (delay of follower i, s). Returns dict of results."""
    gap_e, v_e, q_e = equilibrium_from_density(k, p)
    L = p["L"]
    n = N + 1
    x = -np.arange(n) * (gap_e + L)
    v = np.full(n, v_e)
    d_steps = np.rint(np.asarray(taus) / dt).astype(int)
    H = int(d_steps.max()) + 1
    # history ring buffers for followers: gap, v, dv (initialised at equilibrium)
    hs = np.full((H, N), gap_e)
    hv = np.full((H, N), v_e)
    hd = np.zeros((H, N))
    steps = int(round(t_end / dt))
    rec_k = max(1, int(round(record_every / dt)))
    T_rec, X_rec, V_rec = [], [], []
    vmin = np.full(n, np.inf)
    min_gap = np.full(N, np.inf)
    hard_brake = np.zeros(N, dtype=bool)
    collided = np.zeros(N, dtype=bool)
    idx = np.arange(N)
    for it in range(steps):
        t = it * dt
        gap = x[:-1] - x[1:] - L
        dv = v[1:] - v[:-1]
        slot = it % H
        hs[slot], hv[slot], hd[slot] = gap, v[1:], dv
        # delayed inputs (if not enough history yet, the buffer holds equilibrium values)
        dslot = (it - d_steps) % H
        acc = idm_acc(hs[dslot, idx], hv[dslot, idx], hd[dslot, idx], p)
        # leader
        v_lead_new = leader_speed_profile(t + dt, v_e, v_low=v_low)
        # ballistic update for followers, no negative speeds
        v_new = np.maximum(v[1:] + acc * dt, 0.0)
        x[1:] += 0.5 * (v[1:] + v_new) * dt
        x[0] += 0.5 * (v[0] + v_lead_new) * dt
        v[1:] = v_new
        v[0] = v_lead_new
        gap_new = x[:-1] - x[1:] - L
        coll = gap_new <= 0
        if coll.any():
            collided |= coll
            # resolve overlap so simulation can continue; collision is recorded as an outcome
            for i in np.where(coll)[0]:
                x[i + 1] = x[i] - L - 0.01
                v[i + 1] = min(v[i + 1], v[i])
            gap_new = x[:-1] - x[1:] - L
        min_gap = np.minimum(min_gap, gap_new)
        hard_brake |= (acc < -4.0)
        vmin = np.minimum(vmin, v)
        if it % rec_k == 0:
            T_rec.append(t); X_rec.append(x.copy()); V_rec.append(v.copy())
    drop = v_e - vmin  # speed drop per vehicle
    A = drop / drop[1]  # amplification relative to first follower
    return dict(k=k, v_e=v_e, gap_e=gap_e, q_e=q_e, A=A, drop=drop, vmin=vmin,
                min_gap=min_gap, hard_brake=hard_brake, stopped=(vmin[1:] < 1.0), collided=collided,
                T=np.array(T_rec), X=np.array(X_rec), V=np.array(V_rec), taus=np.asarray(taus))


def wave_speed(res, n_from=5, n_to=40):
    """Ground-frame speed (km/h) of the speed-minimum front: fit x vs t of each vehicle's minimum."""
    T, X, V = res["T"], res["X"], res["V"]
    ts, xs = [], []
    for i in range(n_from, n_to + 1):
        j = np.argmin(V[:, i])
        ts.append(T[j]); xs.append(X[j, i])
    slope = np.polyfit(ts, xs, 1)[0]
    return slope * 3.6
