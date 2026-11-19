"""IDM car-following model, equilibrium and linear string-stability tools.

Units: SI internally (m, s, m/s, m/s^2). Convert to km/h and veh/h only for reporting.
Source: Treiber, Hennecke & Helbing (2000); Treiber & Kesting (2013), Ch. 11 (parameters), Ch. 15 (stability).
"""
import numpy as np
from scipy.optimize import brentq

# Parameters from Treiber & Kesting (2013), Ch. 11 example set
P = dict(v0=120 / 3.6, T=1.0, s0=2.0, a=1.0, b=1.5, delta=4.0, L=5.0,
         b_max=9.0)  # b_max: physical braking limit (dry road), added bound, not part of IDM


def s_star(v, dv, p=P):
    """Desired gap s* (m). dv = v - v_leader (approach rate, + when closing in)."""
    return p["s0"] + np.maximum(0.0, v * p["T"] + v * dv / (2.0 * np.sqrt(p["a"] * p["b"])))


def idm_acc(s, v, dv, p=P):
    """IDM acceleration (m/s^2). s = bumper-to-bumper gap (m)."""
    s = np.maximum(s, 0.01)
    acc = p["a"] * (1.0 - (v / p["v0"]) ** p["delta"] - (s_star(v, dv, p) / s) ** 2)
    return np.maximum(acc, -p["b_max"])


def v_eq(gap, p=P):
    """Equilibrium speed (m/s) for a given gap (m)."""
    if gap <= p["s0"]:
        return 0.0
    return brentq(lambda v: idm_acc(gap, v, 0.0, p), 0.0, p["v0"] * (1 - 1e-12))


def equilibrium_from_density(k_veh_per_km, p=P):
    gap = 1000.0 / k_veh_per_km - p["L"]
    v = v_eq(gap, p)
    return gap, v, k_veh_per_km * v * 3.6  # gap m, speed m/s, flow veh/h


def string_stability_margin(k_veh_per_km, p=P):
    """>=0 string stable, <0 unstable.  Condition f_v^2/2 - f_dvr*f_v - f_s >= 0
    (dvr = v_lead - v), Treiber & Kesting (2013) Ch. 15, tau = 0."""
    gap, v, _ = equilibrium_from_density(k_veh_per_km, p)
    h = 1e-5
    f = lambda s, vv, dvr: idm_acc(s, vv, -dvr, p)
    fs = (f(gap + h, v, 0) - f(gap - h, v, 0)) / (2 * h)
    fv = (f(gap, v + h, 0) - f(gap, v - h, 0)) / (2 * h)
    fd = (f(gap, v, h) - f(gap, v, -h)) / (2 * h)
    return fv ** 2 / 2 - fd * fv - fs


def critical_density(p=P):
    return brentq(lambda k: string_stability_margin(k, p), 25, 45)


def fundamental_diagram(p=P, kmax=140):
    ks = np.linspace(1, kmax, 700)
    qs = np.array([equilibrium_from_density(k, p)[2] for k in ks])
    return ks, qs


def wave_speed_dqdk(k, p=P):
    """Linear (kinematic) propagation speed of small perturbations dQ/dk, km/h."""
    h = 0.05
    return (equilibrium_from_density(k + h, p)[2] - equilibrium_from_density(k - h, p)[2]) / (2 * h)


if __name__ == "__main__":
    for k in [20, 25, 30, 40, 50]:
        g, v, q = equilibrium_from_density(k)
        print(f"k={k} gap={g:.1f} v={v:.2f} m/s ({v*3.6:.1f} km/h) q={q:.0f} "
              f"margin={string_stability_margin(k):+.4f} dQ/dk={wave_speed_dqdk(k):+.1f} km/h")
    ks, qs = fundamental_diagram()
    print("critical density", critical_density(), "capacity", qs.max(), "at", ks[qs.argmax()])
