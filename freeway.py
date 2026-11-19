"""Experiment B: 4 km single-lane freeway, on-ramp merge bottleneck, virtual detectors and a basic TMC.

Loop (every control interval): observe -> estimate -> predict -> decide -> act -> evaluate (logged).
Controllers: 'none' (no metering), 'fixed' (constant release), 'alinea' (Papageorgiou et al., 1991).
Units: m, s, m/s internally; veh/h and % occupancy only at the detector/controller boundary.
"""
import numpy as np
from model import P, idm_acc, v_eq

SEG_LEN = 4000.0
MERGE_START, MERGE_END = 2500.0, 2800.0
DETECTORS = {"D1": 1000.0, "D2": 2000.0, "D3": 2400.0, "D4": 2800.0, "D5": 3500.0}
OBS_INTERVAL = 30.0
V_RAMP = 33.3        # m/s, max merge speed; ramp vehicles match the slower of leader and follower
FORCE_AFTER = 4.0    # s, waiting time after which a ramp vehicle forces into a smaller gap

# lookup table for equilibrium speed vs gap (used only for inserting vehicles at the boundary)
_G = np.linspace(0, 400, 4001)
_VE = np.array([v_eq(g) for g in _G])


def v_eq_fast(gap):
    return np.interp(gap, _G, _VE)


def default_demand(t):
    """(mainline veh/h, ramp veh/h). Warm-up, 30-min peak, shoulder, then zero (clearance)."""
    if t < 300:
        return 1600.0, 300.0
    if t < 2100:
        return 2000.0, 600.0
    if t < 3600:
        return 1500.0, 300.0
    return 0.0, 0.0


class ALINEA:
    def __init__(self, o_target=17.0, K_R=70.0, r_min=200.0, r_max=1800.0, r_init=600.0,
                 fallback_rate=520.0):
        self.o_target, self.K_R = o_target, K_R
        self.r_min, self.r_max, self.r = r_min, r_max, r_init
        self.fallback_rate = fallback_rate

    def decide(self, o_hat):
        if o_hat is None:  # missing observation -> fallback
            return self.fallback_rate, "fallback: detector missing"
        self.r = float(np.clip(self.r + self.K_R * (self.o_target - o_hat), self.r_min, self.r_max))
        return self.r, f"ALINEA: o_hat={o_hat:.1f}% vs target {self.o_target:.1f}%"


def simulate(controller="none", dt=0.1, t_end=4500.0, demand=default_demand, tau=0.0,
             ramp_storage=30, fixed_rate=520.0, control_interval=60.0, alinea_kwargs=None,
             detector_dropout=None, queue_override=25, p=P, seed=0, record_traj=False, poisson=True):
    rng = np.random.default_rng(seed)
    L = p["L"]
    d_tau = int(round(tau / dt))
    H = d_tau + 1
    MAXV = 20000
    hist_s = np.zeros((H, MAXV)); hist_v = np.zeros((H, MAXV)); hist_d = np.zeros((H, MAXV))

    x = np.zeros(0); v = np.zeros(0); vid = np.zeros(0, dtype=int)
    t_in = {}; origin = {}
    next_id = 0
    up_queue = []      # mainline vehicles waiting at x=0 (boundary queue): arrival times
    ramp_queue = []    # vehicles stored on the ramp (<= ramp_storage)
    overflow = []      # ramp vehicles that could not enter storage (spillback = unmet ramp demand)
    ready = []         # released by the meter, waiting to merge
    acc_main = acc_ramp = 0.0
    ramp_arrivals = 0
    release_credit = 0.0
    rate = np.inf if controller == "none" else (fixed_rate if controller == "fixed" else None)
    alinea = ALINEA(**(alinea_kwargs or {})) if controller == "alinea" else None
    if alinea:
        rate = alinea.r

    # detector raw crossings: name -> list of (t, speed)
    crossings = {k: [] for k in DETECTORS}
    obs_rows, ctrl_log, series = [], [], []
    occ_hist = []
    entered = exited = 0
    n_ramp_overflow_total = 0
    tts = 0.0  # total time spent, veh*s (road + all queues)
    travel_records = []  # (origin, enter_demand_time, exit_time)
    arr_time = {}
    traj = [] if record_traj else None
    collisions = 0
    conservation_ok = True
    last_ctrl = 0.0
    last_obs = 0.0
    steps = int(round(t_end / dt))

    for it in range(steps):
        t = it * dt
        qm, qr = demand(t)
        # ---- arrivals (deterministic, fractional accumulation) ----
        if poisson:
            n_m, n_r = rng.poisson(qm * dt / 3600.0), rng.poisson(qr * dt / 3600.0)
        else:
            acc_main += qm * dt / 3600.0; acc_ramp += qr * dt / 3600.0
            n_m, n_r = int(acc_main), int(acc_ramp)
            acc_main -= n_m; acc_ramp -= n_r
        up_queue.extend([t] * n_m)
        overflow.extend([t] * n_r)
        ramp_arrivals += n_r
        # overflow moves into ramp storage when space is available (FIFO)
        while overflow and len(ramp_queue) < ramp_storage:
            ramp_queue.append(overflow.pop(0))

        # ---- meter release (token bucket) ----
        if controller == "none":
            while ramp_queue and len(ready) < 2:
                ready.append((ramp_queue.pop(0), t))
        else:
            release_credit = min(release_credit + rate * dt / 3600.0, 1.0)
            if release_credit >= 1.0 and ramp_queue and len(ready) < 2:
                ready.append((ramp_queue.pop(0), t)); release_credit -= 1.0

        # ---- insertion at upstream boundary ----
        if up_queue:
            gap_last = (x[-1] - L) if len(x) else 1e4
            if gap_last >= 10.0:
                v_ins = min(p["v0"], v_eq_fast(min(gap_last, 400.0)), v[-1] + 2.0 if len(v) else p["v0"])
                arr = up_queue.pop(0)
                x = np.append(x, 0.0); v = np.append(v, v_ins); vid = np.append(vid, next_id)
                t_in[next_id] = t; origin[next_id] = 0; arr_time[next_id] = arr
                hist_s[:, next_id] = gap_last; hist_v[:, next_id] = v_ins; hist_d[:, next_id] = 0
                next_id += 1; entered += 1

        # ---- merge from ramp ----
        if ready:
            in_zone = np.where((x >= MERGE_START) & (x <= MERGE_END))[0]
            best = None
            # candidate gaps: between consecutive vehicles where the insertion point is in the zone
            cand = []
            if len(x) == 0:
                cand = [(None, None, MERGE_START + 50.0)]
            else:
                # leader index j, follower j+1
                near = np.where((x >= MERGE_START - 150.0) & (x <= MERGE_END + 150.0))[0]
                jlo = (near.min() - 1) if len(near) else max(int(np.searchsorted(-x, -MERGE_END)) - 1, -1)
                jhi = (near.max() + 1) if len(near) else jlo + 1
                for j in range(max(jlo, -1), min(jhi, len(x) - 1) + 1):
                    xl = x[j] if j >= 0 else 1e9
                    xf = x[j + 1] if j + 1 < len(x) else -1e9
                    lo, hi = max(xf + L, MERGE_START), min(xl - L, MERGE_END)
                    if hi - lo > 2 * p["s0"]:
                        cand.append((j if j >= 0 else None, j + 1 if j + 1 < len(x) else None, 0.5 * (lo + hi)))
            for (jl, jf, xi) in cand:
                vl = v[jl] if jl is not None else p["v0"]
                vf = v[jf] if jf is not None else vl
                v_ins = min(vl, max(vf, 0.0), V_RAMP)
                gl = (x[jl] - xi - L) if jl is not None else 1e4
                gf = (xi - x[jf] - L) if jf is not None else 1e4
                if gl < p["s0"] + 1.0 or gf < p["s0"] + 1.0:
                    continue
                a_new = idm_acc(gl, v_ins, v_ins - vl)
                a_fol = idm_acc(gf, vf, vf - v_ins) if jf is not None else 0.0
                # gap acceptance: polite (-2 m/s^2) at first, forced (-b_max/2) after FORCE_AFTER s
                b_acc = 2.0 if (t - ready[0][1]) < FORCE_AFTER else 0.6 * p["b_max"]
                if a_new < -b_acc or a_fol < -b_acc:
                    continue
                score = min(gl, gf)
                if best is None or score > best[0]:
                    best = (score, jl, jf, xi, v_ins, gl)
            if best is not None:
                _, jl, jf, xi, v_ins, gl = best
                pos = (jl + 1) if jl is not None else 0
                x = np.insert(x, pos, xi); v = np.insert(v, pos, v_ins); vid = np.insert(vid, pos, next_id)
                arr = ready.pop(0)[0]
                t_in[next_id] = t; origin[next_id] = 1; arr_time[next_id] = arr
                hist_s[:, next_id] = gl; hist_v[:, next_id] = v_ins; hist_d[:, next_id] = 0
                next_id += 1; entered += 1

        # ---- car following ----
        n = len(x)
        if n:
            gap = np.empty(n); dv = np.empty(n)
            gap[0] = 1e4; dv[0] = 0.0
            gap[1:] = x[:-1] - x[1:] - L
            dv[1:] = v[1:] - v[:-1]
            slot = it % H
            hist_s[slot, vid] = gap; hist_v[slot, vid] = v; hist_d[slot, vid] = dv
            dslot = (it - d_tau) % H
            acc = idm_acc(hist_s[dslot, vid], hist_v[dslot, vid], hist_d[dslot, vid], p)
            v_new = np.maximum(v + acc * dt, 0.0)
            x_new = x + 0.5 * (v + v_new) * dt
            # collision check / resolve
            g_new = x_new[:-1] - x_new[1:] - L
            if (g_new <= 0).any():
                for i in np.where(g_new <= 0)[0]:
                    collisions += 1
                    x_new[i + 1] = x_new[i] - L - 0.01
                    v_new[i + 1] = min(v_new[i + 1], v_new[i])
            # detectors
            for name, xd in DETECTORS.items():
                crossed = np.where((x < xd) & (x_new >= xd))[0]
                for i in crossed:
                    crossings[name].append((t + dt, 0.5 * (v[i] + v_new[i]), int(vid[i])))
            x, v = x_new, v_new
            # exits
            out = x > SEG_LEN
            if out.any():
                for i in np.where(out)[0]:
                    travel_records.append((origin[int(vid[i])], arr_time[int(vid[i])], t + dt))
                exited += int(out.sum())
                x, v, vid = x[~out], v[~out], vid[~out]

        n_queue = len(up_queue) + len(ramp_queue) + len(overflow) + len(ready)
        tts += (len(x) + n_queue) * dt
        if entered - exited != len(x):
            conservation_ok = False
        if record_traj and it % 10 == 0:
            traj.append((t, x.copy(), v.copy()))

        # ---- observations every OBS_INTERVAL ----
        if t + dt - last_obs >= OBS_INTERVAL - 1e-9:
            t0, t1 = last_obs, t + dt
            for name in DETECTORS:
                cs = [c for c in crossings[name] if t0 < c[0] <= t1]
                cnt = len(cs)
                ms = float(np.mean([c[1] for c in cs])) if cnt else np.nan
                occ = 100.0 * sum(L / max(c[1], 0.5) for c in cs) / (t1 - t0)
                qflag = "ok"
                if detector_dropout and name == "D4" and detector_dropout[0] <= t1 <= detector_dropout[1]:
                    cnt, ms, occ, qflag = None, np.nan, None, "missing"
                obs_rows.append(dict(sensor_id=name, interval_start_s=round(t0, 1), interval_end_s=round(t1, 1), count=cnt,
                                     mean_speed_mps=ms, occupancy_pct=occ, quality_flag=qflag))
            series.append(dict(t=t1, ramp_queue=len(ramp_queue), overflow=len(overflow),
                               up_queue=len(up_queue), rate=(np.nan if rate == np.inf else rate), on_road=len(x)))
            last_obs = t1

        # ---- TMC control loop ----
        if controller != "none" and t + dt - last_ctrl >= control_interval - 1e-9:
            t0, t1 = last_ctrl, t + dt
            # observe: D4 occupancy over the last control interval (two 30 s records)
            recs = [r for r in obs_rows if r["sensor_id"] == "D4" and r["interval_end_s"] > t0 + 1e-9]
            if any(r["quality_flag"] != "ok" for r in recs) or not recs:
                o_hat = None
            else:
                o_hat = float(np.mean([r["occupancy_pct"] for r in recs]))  # estimate
            occ_hist.append(o_hat)
            # predict (advisory only): linear trend of last 3 estimates, 60 s ahead
            valid = [o for o in occ_hist[-3:] if o is not None]
            o_pred = (valid[-1] + (valid[-1] - valid[0]) / max(len(valid) - 1, 1)) if len(valid) >= 2 else (valid[-1] if valid else None)
            if controller == "alinea":
                new_rate, reason = alinea.decide(o_hat)
            else:
                new_rate, reason = fixed_rate, "fixed rate"
            # queue control (storage constraint): release enough to keep the queue below the limit
            # over the next interval: r_q = d_ramp + (w - w_limit) / interval   (veh/h)
            d_ramp = ramp_arrivals * 3600.0 / control_interval
            w = len(ramp_queue) + len(overflow)
            r_q = d_ramp + (w - queue_override) * 3600.0 / control_interval
            if r_q > new_rate:
                new_rate = min(r_q, 1800.0)
                reason += f"; queue override (w={w} veh, limit {queue_override})"
            ramp_arrivals = 0
            rate = new_rate
            ctrl_log.append(dict(issue_time_s=round(t1, 1), apply_time_s=round(t1 + dt, 1), actuator_id="RM1",
                                 action="set_release_rate", value=round(rate, 1), units="veh/h",
                                 bounds="[200, 1800]", observed_occ_pct=None if o_hat is None else round(o_hat, 2),
                                 forecast_occ_pct_60s=None if o_pred is None else round(o_pred, 2),
                                 ramp_queue_veh=len(ramp_queue), overflow_veh=len(overflow), reason=reason))
            last_ctrl = t1

    # ---- evaluation ----
    tr = np.array(travel_records) if travel_records else np.zeros((0, 3))
    ff_main = SEG_LEN / p["v0"]
    ff_ramp = (SEG_LEN - MERGE_START) / p["v0"]
    ff = np.where(tr[:, 0] == 0, ff_main, ff_ramp) if len(tr) else np.zeros(0)
    tt = tr[:, 2] - tr[:, 1] if len(tr) else np.zeros(0)
    delay = tt - ff
    res = dict(controller=controller, tau=tau, entered=entered, exited=exited,
               on_road_end=len(x), queues_end=len(up_queue) + len(ramp_queue) + len(overflow) + len(ready),
               tts_vehh=tts / 3600.0, total_delay_vehh=float(delay.sum()) / 3600.0,
               delay_main_vehh=float(delay[tr[:, 0] == 0].sum()) / 3600.0 if len(tr) else 0,
               delay_ramp_vehh=float(delay[tr[:, 0] == 1].sum()) / 3600.0 if len(tr) else 0,
               collisions=collisions, conservation_ok=conservation_ok,
               obs=obs_rows, ctrl=ctrl_log, series=series, crossings=crossings, traj=traj, travel=tr)
    res["max_ramp_queue"] = max(r["ramp_queue"] for r in series)
    res["max_overflow"] = max(r["overflow"] for r in series)
    res["max_up_queue"] = max(r["up_queue"] for r in series)
    return res
