"""
Current Controller Design and calibration Wizard (dq current PI controllers).
Author: Masoud Bakhshi

Added:
A) Robustness sweep (Monte Carlo) over Rs, Ld/Lq, total delay Td
B) Voltage saturation + anti-windup time-domain check (limit-cycle/ripple risk indicator)

Dependencies:
- streamlit, numpy, matplotlib
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# Helpers (freq-domain)
# ============================================================

def db20(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def hz_to_rad_per_s(f_hz: float) -> float:
    return 2.0 * math.pi * f_hz


def rad_per_s_to_hz(w: float) -> float:
    return w / (2.0 * math.pi)


def pade11_delay_jw(Td_s: float, w: np.ndarray) -> np.ndarray:
    """Pade(1,1): e^{-sTd} ≈ (1 - sTd/2)/(1 + sTd/2), evaluated at s=jw."""
    if Td_s <= 0:
        return np.ones_like(w, dtype=np.complex128)
    jw = 1j * w
    num = 1.0 - jw * (Td_s / 2.0)
    den = 1.0 + jw * (Td_s / 2.0)
    return num / den


def exact_delay_jw(Td_s: float, w: np.ndarray) -> np.ndarray:
    if Td_s <= 0:
        return np.ones_like(w, dtype=np.complex128)
    return np.exp(-1j * w * Td_s)


def rl_plant_jw(R: float, L: float, w: np.ndarray) -> np.ndarray:
    """G(jw)=1/(R + jwL)."""
    return 1.0 / (R + 1j * w * L)


def pi_controller_jw(Kp: float, Ki: float, w: np.ndarray) -> np.ndarray:
    """C(jw)=Kp + Ki/(jw)."""
    return Kp + Ki / (1j * w)


def unwrap_phase_deg(angle_rad: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.unwrap(angle_rad))


def find_gain_crossover(w: np.ndarray, Ljw: np.ndarray) -> Optional[float]:
    """Find gain crossover w_gc where |L| crosses 1 (0 dB). Uses log-frequency interpolation."""
    mag_db = db20(Ljw)
    idx = np.where(np.diff(np.signbit(mag_db)))[0]
    if idx.size == 0:
        return None
    i = idx[0]
    w1, w2 = w[i], w[i + 1]
    y1, y2 = mag_db[i], mag_db[i + 1]
    if abs(y2 - y1) < 1e-12:
        return float(w1)
    lw1, lw2 = np.log10(w1), np.log10(w2)
    lwc = lw1 + (0.0 - y1) * (lw2 - lw1) / (y2 - y1)
    return float(10.0 ** lwc)


def interp_log_at(w: np.ndarray, y: np.ndarray, w0: float) -> float:
    """Interpolate y(w) at w0 using log-frequency interpolation (consistent with log-spaced grids)."""
    if w0 <= w[0]:
        return float(y[0])
    if w0 >= w[-1]:
        return float(y[-1])
    lw = np.log10(w)
    return float(np.interp(np.log10(w0), lw, y))


def find_phase_crossings(w: np.ndarray, phase_deg_unwrapped: np.ndarray, target_deg: float = -180.0) -> List[float]:
    """Find w where phase crosses target_deg (unwrapped phase). Uses log-frequency interpolation."""
    y = phase_deg_unwrapped - target_deg
    idx = np.where(np.diff(np.signbit(y)))[0]
    out: List[float] = []
    for i in idx:
        w1, w2 = w[i], w[i + 1]
        y1, y2 = y[i], y[i + 1]
        if abs(y2 - y1) < 1e-12:
            out.append(float(w1))
            continue
        lw1, lw2 = np.log10(w1), np.log10(w2)
        lwc = lw1 + (0.0 - y1) * (lw2 - lw1) / (y2 - y1)
        out.append(float(10.0 ** lwc))
    return out


def compute_margins(w: np.ndarray, Ljw: np.ndarray) -> Dict[str, float]:
    """
    PM computed at gain crossover (|L|=1).
    GM computed at phase crossover(s) (phase=-180deg), worst-case (smallest GM).
    Uses log-frequency interpolation for consistency.
    """
    ph = unwrap_phase_deg(np.angle(Ljw))

    w_gc = find_gain_crossover(w, Ljw)
    if w_gc is None:
        PM = float("nan")
    else:
        ph_gc = interp_log_at(w, ph, w_gc)
        PM = 180.0 + ph_gc

    w_pcs = find_phase_crossings(w, ph, target_deg=-180.0)
    GM_candidates: List[Tuple[float, float]] = []
    mag = np.abs(Ljw)

    for w_pc in w_pcs:
        mag_pc = float(interp_log_at(w, mag, w_pc))
        if mag_pc <= 0:
            continue
        GM = 1.0 / mag_pc
        GM_candidates.append((GM, w_pc))

    if len(GM_candidates) == 0:
        GM = float("inf")
        w_pc_dom = float("nan")
    else:
        GM, w_pc_dom = min(GM_candidates, key=lambda t: t[0])

    return {
        "PM_deg": float(PM),
        "GM": float(GM),
        "w_gc": float("nan") if w_gc is None else float(w_gc),
        "w_pc": float(w_pc_dom),
    }


def sensitivity_peaks_db(Ljw: np.ndarray) -> Dict[str, float]:
    S = 1.0 / (1.0 + Ljw)
    T = Ljw / (1.0 + Ljw)
    Ms = float(np.max(np.abs(S)))
    Mt = float(np.max(np.abs(T)))
    return {
        "Ms_db": 20.0 * math.log10(max(Ms, 1e-30)),
        "Mt_db": 20.0 * math.log10(max(Mt, 1e-30)),
        "Ms": Ms,
        "Mt": Mt,
    }


def tustin_incremental_pi_coeffs(Kp: float, Ki: float, Ts: float) -> Dict[str, float]:
    """
    Incremental (velocity) form:
      u[k] = u[k-1] + a1*e[k] + b1*e[k-1]
    with Tustin (bilinear) integrator.
    """
    a1 = Kp + Ki * Ts / 2.0
    b1 = -(Kp - Ki * Ts / 2.0)
    return {"a1": float(a1), "b1": float(b1), "Ts_s": float(Ts)}


def delay_limited_fc_max(Td_s: float, PM_target_deg: float) -> float:
    """
    Heuristic for RL bandwidth PI:
      PM ≈ 90° - (wc*Td)*(180/pi)
      => wc_max = (pi/2 - PM_target_rad)/Td
    """
    if Td_s <= 0:
        return float("inf")
    PM_target_rad = math.radians(PM_target_deg)
    if PM_target_rad >= (math.pi / 2.0):
        return 0.0
    wc_max = (math.pi / 2.0 - PM_target_rad) / Td_s
    return rad_per_s_to_hz(wc_max)


def bandwidth_pi_gains(R: float, L: float, fc_hz: float) -> Tuple[float, float]:
    """Bandwidth PI for RL: Kp=L*wc, Ki=R*wc."""
    wc = hz_to_rad_per_s(fc_hz)
    return float(L * wc), float(R * wc)


# ============================================================
# Helpers (time-domain saturation + AW) 
# ============================================================

@dataclass
class SimCfg:
    t_end_s: float = 0.02
    i_ref_step_A: float = 200.0
    i0_A: float = 0.0
    Vdc_V: float = 700.0
    m_max: float = 0.95
    pwm: str = "SVPWM"
    td_s: float = 150e-6
    Ts_s: float = 100e-6

    # Delay placement for Td:
    # - "measurement": apply Td to measured current (controller sees delayed i)
    # - "actuation": apply Td to applied voltage (plant sees delayed v)
    # - "split": half on measurement + half on actuation
    delay_placement: str = "split"

    # Saturation model: PI output is assumed phase (line-neutral) voltage peak
    sat_model: str = "V_phase_peak"


def v_phase_peak_max(Vdc: float, m_max: float, pwm: str) -> float:
    """
    Practical-ish voltage headroom model (conservative).
    If controller output is "phase (line-neutral) voltage peak":
      SPWM: v_phase_peak_max ≈ m_max * Vdc/2
      SVPWM: v_phase_peak_max ≈ m_max * Vdc/√3
    """
    if pwm.upper() == "SVPWM":
        return m_max * Vdc / math.sqrt(3.0)
    return m_max * Vdc / 2.0


def delayed_sample_linear(x: np.ndarray, k: int, d_samp: float) -> float:
    """
    Fractional sample delay using linear interpolation.
    Returns x[k - d_samp]. If index is <0, clamps to x[0].
    """
    if d_samp <= 0:
        return float(x[k])
    n = int(math.floor(d_samp))
    a = d_samp - n

    k0 = k - n
    k1 = k - n - 1

    if k0 < 0:
        return float(x[0])
    if k1 < 0:
        return float(x[k0])
    return float((1.0 - a) * x[k0] + a * x[k1])


def simulate_current_loop_rl(
    R: float, L: float,
    Kp: float, Ki: float,
    cfg: SimCfg,
    aw_mode: str = "clamp",   # "none", "clamp", "backcalc"
    kaw: float = 1.0          # back-calc gain (dimensionless in this simplified discrete form)
) -> Dict[str, np.ndarray]:
    """
    Discrete-time simulation of a single-axis current loop (dq axis):
      Plant (exact RL under ZOH): i[k+1] = a*i[k] + b*v_applied[k]
      Controller (position PI with Tustin integrator):
         xI[k] = xI[k-1] + Ki*Ts/2*(e[k]+e[k-1])  (+ AW correction)
         v_cmd = Kp*e + xI
      Saturation: v_sat = clip(v_cmd, ±Vmax)

    Delay model:
      Uses fractional-sample delay (linear interpolation) with selectable placement:
        measurement / actuation / split
      Total delay Td is applied accordingly.
    """
    Ts = float(cfg.Ts_s)
    N = int(max(2, math.ceil(cfg.t_end_s / Ts)))
    t = np.arange(N) * Ts

    Vmax = float(v_phase_peak_max(cfg.Vdc_V, cfg.m_max, cfg.pwm))

    # fractional delays (samples)
    d_total = float(cfg.td_s / Ts)
    placement = str(cfg.delay_placement).lower().strip()
    if placement == "measurement":
        d_meas, d_act = d_total, 0.0
    elif placement == "actuation":
        d_meas, d_act = 0.0, d_total
    else:  # "split" default
        d_meas, d_act = 0.5 * d_total, 0.5 * d_total

    # signals
    i = np.zeros(N, dtype=float)
    i[:] = float(cfg.i0_A)

    i_meas = np.zeros(N, dtype=float)
    e = np.zeros(N, dtype=float)
    v_cmd = np.zeros(N, dtype=float)
    v_sat = np.zeros(N, dtype=float)
    v_applied = np.zeros(N, dtype=float)
    sat_flag = np.zeros(N, dtype=np.int32)

    # PI integrator state (true state, so AW comparison is meaningful)
    xI = 0.0
    xI_hist = np.zeros(N, dtype=float)
    e_prev = 0.0

    i_ref = np.ones(N, dtype=float) * float(cfg.i_ref_step_A)

    # exact RL discretization
    if R > 0:
        a = math.exp(-R * Ts / L)
        b = (1.0 - a) / R
    else:
        # If R ~ 0, fall back to integrator: di = (Ts/L)*v
        a = 1.0
        b = Ts / L

    for k in range(N):
        # measurement (possibly delayed)
        y = delayed_sample_linear(i, k, d_meas)
        i_meas[k] = y

        # error
        e_k = i_ref[k] - y
        e[k] = e_k

        # Tustin integrator candidate
        xI_cand = xI + (Ki * Ts / 2.0) * (e_k + e_prev)
        v_cand = Kp * e_k + xI_cand

        # saturation based on candidate
        v_cand_sat = max(-Vmax, min(Vmax, v_cand))
        sat = 1 if abs(v_cand - v_cand_sat) > 1e-12 else 0

        # Anti-windup
        if aw_mode == "none":
            xI = xI_cand
            v_use = v_cand

        elif aw_mode == "clamp":
            if sat == 0:
                xI = xI_cand
                v_use = v_cand
            else:
                # Only allow integration if it helps recover from saturation
                # If saturated high, allow if error is negative; if saturated low, allow if error is positive
                if (v_cand_sat >= Vmax and e_k < 0) or (v_cand_sat <= -Vmax and e_k > 0):
                    xI = xI_cand
                    v_use = v_cand
                else:
                    # freeze integrator
                    xI = xI
                    v_use = Kp * e_k + xI

        elif aw_mode == "backcalc":
            # back-calculation: push integrator to reduce (v_cmd - v_sat)
            # Here implemented as: xI = xI_cand + kaw*(v_sat - v_cand)
            xI = xI_cand + kaw * (v_cand_sat - v_cand)
            v_use = Kp * e_k + xI

        else:
            xI = xI_cand
            v_use = v_cand

        # final saturation after AW decision
        v_use_sat = max(-Vmax, min(Vmax, v_use))
        sat_flag[k] = 1 if abs(v_use - v_use_sat) > 1e-12 else 0

        v_cmd[k] = v_use
        v_sat[k] = v_use_sat
        xI_hist[k] = xI

        # actuation delay (possibly delayed)
        v_applied_k = delayed_sample_linear(v_sat, k, d_act)
        v_applied[k] = v_applied_k

        # plant update
        if k < N - 1:
            i[k + 1] = a * i[k] + b * v_applied_k

        e_prev = e_k

    return {
        "t": t,
        "i_ref": i_ref,
        "i": i.copy(),
        "i_meas": i_meas,
        "e": e,
        "v_cmd": v_cmd,
        "v_sat": v_sat,
        "v_applied": v_applied,
        "sat_flag": sat_flag.astype(float),
        "Vmax": np.array([Vmax]),
        "d_total": np.array([d_total]),
        "d_meas": np.array([d_meas]),
        "d_act": np.array([d_act]),
        "xI": xI_hist,
    }


def step_metrics(t: np.ndarray, y: np.ndarray, yref: np.ndarray, settle_band: float = 0.02) -> Dict[str, float]:
    """Simple step metrics: overshoot (%), settling time, final values."""
    y_final = float(np.mean(y[int(0.9 * len(y)):]))
    ref_final = float(np.mean(yref[int(0.9 * len(yref)):]))

    if abs(ref_final) < 1e-12:
        overshoot = float("nan")
    else:
        overshoot = (float(np.max(y)) - ref_final) / abs(ref_final) * 100.0

    band = settle_band * max(abs(ref_final), 1e-12)
    err = np.abs(y - ref_final)

    ts = float("nan")
    for k in range(len(y)):
        if np.all(err[k:] <= band):
            ts = float(t[k])
            break

    return {"y_final": y_final, "ref_final": ref_final, "overshoot_pct": overshoot, "settling_s": ts}


def ripple_risk_indicator(t: np.ndarray, y: np.ndarray, yref: np.ndarray, sat_frac: float) -> Dict[str, object]:
    """
    Heuristic risk flag:
      - Check last 20% window peak-to-peak vs reference magnitude
      - Combine with saturation fraction
    """
    n0 = int(0.8 * len(y))
    yy = y[n0:]
    ypp = float(np.max(yy) - np.min(yy))
    ref_mag = float(max(abs(np.mean(yref[n0:])), 1e-12))
    ypp_pct = 100.0 * ypp / ref_mag

    risk = (sat_frac > 0.15) or (ypp_pct > 1.0)
    level = "HIGH" if risk else "LOW"
    reason = []
    if sat_frac > 0.15:
        reason.append(f"saturation fraction {sat_frac*100:.1f}%")
    if ypp_pct > 1.0:
        reason.append(f"steady ripple {ypp_pct:.2f}% p-p (tail)")
    if not reason:
        reason.append("low saturation and low tail ripple in this check")

    return {"risk": risk, "level": level, "tail_ripple_pp_pct": ypp_pct, "reason": ", ".join(reason)}


# ============================================================
# UI models
# ============================================================

@dataclass
class ProjectConfig:
    project_id: str = "IPMSM_CurrentCtrl_v1"
    owner: str = "Masoud Bakhshi"
    fs_hz: float = 10_000.0
    pwm_method: str = "SVPWM"
    units_current: str = "A_peak"
    units_voltage: str = "V_phase"
    delay_pwm_zoh_s: float = 50e-6
    delay_compute_s: float = 50e-6
    delay_measure_s: float = 50e-6

    @property
    def Ts_s(self) -> float:
        return 1.0 / self.fs_hz

    @property
    def Td_s(self) -> float:
        return float(self.delay_pwm_zoh_s + self.delay_compute_s + self.delay_measure_s)


@dataclass
class PlantConfig:
    Rs_ohm: float = 0.002
    Ld_H: float = 0.0003
    Lq_H: float = 0.00045
    psi_f_Wb: float = 0.0  # optional (currently unused)

    def tau_d(self) -> float:
        return self.Ld_H / self.Rs_ohm

    def tau_q(self) -> float:
        return self.Lq_H / self.Rs_ohm


@dataclass
class TuningConfig:
    fc_target_hz: float = 500.0
    PM_target_deg: float = 60.0


# ============================================================
# Plot helpers
# ============================================================

def plot_bode_mag(w: np.ndarray, curves: Dict[str, np.ndarray], ylabel: str, title: str):
    fig, ax = plt.subplots()
    for name, H in curves.items():
        ax.semilogx(w, db20(H), label=name)
    ax.set_xlabel("Frequency [rad/s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both")
    ax.legend()
    st.pyplot(fig)


def plot_time(t: np.ndarray, curves: Dict[str, np.ndarray], title: str, xlabel: str = "Time [s]"):
    fig, ax = plt.subplots()
    for name, y in curves.items():
        ax.plot(t, y, label=name)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)


def as_config_summary(proj: ProjectConfig) -> Dict[str, object]:
    return {
        "fs_hz": proj.fs_hz,
        "pwm_method": proj.pwm_method,
        "delay_total_s": proj.Td_s,
        "units_current": proj.units_current,
        "units_voltage": proj.units_voltage,
        "note_units": "Internal math assumes A_peak and V_phase_peak (units selection is labeling unless you extend conversions).",
    }


# ============================================================
# Streamlit pages
# ============================================================

def init_state():
    if "proj" not in st.session_state:
        st.session_state.proj = ProjectConfig()
    if "plant" not in st.session_state:
        st.session_state.plant = PlantConfig()
    if "tune" not in st.session_state:
        st.session_state.tune = TuningConfig()
    if "gains" not in st.session_state:
        st.session_state.gains = {}


def page_header():
    st.markdown("## Current Controller Design Wizard")
    st.caption("Guided workflow for dq current control design (robust, traceable, defensible).")
    st.markdown("**Author:** Masoud Bakhshi")
    st.markdown("---")


def page_project_setup():
    proj: ProjectConfig = st.session_state.proj

    st.markdown("### 1) Project Setup")
    c1, c2, c3 = st.columns([2.2, 1.2, 1.2])
    with c1:
        proj.project_id = st.text_input("Project ID", proj.project_id)
    with c2:
        proj.fs_hz = st.number_input("Sampling frequency fs [Hz]", min_value=100.0, value=float(proj.fs_hz), step=100.0)
    with c3:
        proj.units_current = st.selectbox("Current unit", ["A_peak", "A_rms"], index=0 if proj.units_current == "A_peak" else 1)

    c4, c5, c6 = st.columns([2.2, 1.2, 1.2])
    with c4:
        proj.owner = st.text_input("Owner", proj.owner)
    with c5:
        proj.pwm_method = st.selectbox("PWM method", ["SVPWM", "SPWM"], index=0 if proj.pwm_method == "SVPWM" else 1)
    with c6:
        proj.units_voltage = st.selectbox("Voltage unit", ["V_phase", "V_line"], index=0 if proj.units_voltage == "V_phase" else 1)

    st.markdown("### Delays (implementation reality)")
    d1, d2, d3, d4 = st.columns([1.2, 1.2, 1.2, 1.0])
    with d1:
        proj.delay_pwm_zoh_s = st.number_input("PWM/ZOH delay [s]", min_value=0.0, value=float(proj.delay_pwm_zoh_s), step=10e-6, format="%.8f")
    with d2:
        proj.delay_compute_s = st.number_input("Computation delay [s]", min_value=0.0, value=float(proj.delay_compute_s), step=10e-6, format="%.8f")
    with d3:
        proj.delay_measure_s = st.number_input("Measurement delay [s]", min_value=0.0, value=float(proj.delay_measure_s), step=10e-6, format="%.8f")
    with d4:
        st.metric("Total delay Td [s]", f"{proj.Td_s:.6e}")

    st.markdown("### Run validation checks")
    if st.button("Run Checks"):
        errs = []
        if proj.fs_hz <= 0:
            errs.append("fs must be > 0.")
        if proj.Td_s < 0:
            errs.append("Total delay must be >= 0.")
        if errs:
            st.error("FAIL: " + " ".join(errs))
        else:
            st.success("PASS: Configuration is consistent.")
            st.info("Tip: Keep Td realistic (PWM + compute + filtering). Too-optimistic delay assumptions are a top cause of unstable/rippled current loops.")

    st.sidebar.markdown("### Active Configuration (Summary)")
    st.sidebar.json(as_config_summary(proj))


def page_plant_model():
    plant: PlantConfig = st.session_state.plant

    st.markdown("### 2) Plant / Model (dq RL)")
    c1, c2, c3 = st.columns(3)
    with c1:
        plant.Rs_ohm = st.number_input("Rs [ohm]", min_value=1e-9, value=float(plant.Rs_ohm), format="%.9f")
    with c2:
        plant.Ld_H = st.number_input("Ld [H]", min_value=1e-12, value=float(plant.Ld_H), format="%.9f")
    with c3:
        plant.Lq_H = st.number_input("Lq [H]", min_value=1e-12, value=float(plant.Lq_H), format="%.9f")

    plant.psi_f_Wb = st.number_input("psi_f [Wb] (optional)", min_value=0.0, value=float(plant.psi_f_Wb), format="%.9f")

    st.markdown("#### Validation + quick plant insights")
    c4, c5 = st.columns(2)
    with c4:
        st.metric("Tau_d = Ld/Rs [s]", f"{plant.tau_d():.6e}")
    with c5:
        st.metric("Tau_q = Lq/Rs [s]", f"{plant.tau_q():.6e}")

    w = np.logspace(0, 5, 800)  # rad/s
    Gd = rl_plant_jw(plant.Rs_ohm, plant.Ld_H, w)
    Gq = rl_plant_jw(plant.Rs_ohm, plant.Lq_H, w)
    plot_bode_mag(w, {"|Gd| [d]": Gd, "|Gq| [q]": Gq}, ylabel="Magnitude [dB]", title="Plant magnitude: dq voltage → dq current")


def page_tuning():
    proj: ProjectConfig = st.session_state.proj
    plant: PlantConfig = st.session_state.plant
    tune: TuningConfig = st.session_state.tune

    st.markdown("### 5) Tuning (Preview): Delay-aware bandwidth PI")

    c1, c2, c3 = st.columns([2.0, 2.0, 1.2])
    with c1:
        tune.fc_target_hz = st.number_input("Target crossover fc [Hz]", min_value=1.0, value=float(tune.fc_target_hz), step=10.0)
    with c2:
        tune.PM_target_deg = st.number_input("PM target (heuristic) [deg]", min_value=10.0, max_value=89.0, value=float(tune.PM_target_deg), step=1.0)
    with c3:
        fc_max = delay_limited_fc_max(proj.Td_s, tune.PM_target_deg)
        st.metric("Delay-limited fc_max [Hz]", f"{fc_max:.1f}")

    fc_used = min(tune.fc_target_hz, fc_max)
    if tune.fc_target_hz > fc_max:
        st.warning(f"Target fc={tune.fc_target_hz:.1f} Hz exceeds delay limit. Using fc_used={fc_used:.1f} Hz.")
    else:
        st.info(f"Using fc_used={fc_used:.1f} Hz.")

    Kp_d, Ki_d = bandwidth_pi_gains(plant.Rs_ohm, plant.Ld_H, fc_used)
    Kp_q, Ki_q = bandwidth_pi_gains(plant.Rs_ohm, plant.Lq_H, fc_used)

    st.session_state.gains = {
        "Kp_d": Kp_d, "Ki_d": Ki_d,
        "Kp_q": Kp_q, "Ki_q": Ki_q,
        "fc_used_hz": fc_used, "fc_max_hz": fc_max
    }

    st.markdown("#### Proposed nominal continuous-time gains (bandwidth PI)")
    c4, c5 = st.columns(2)
    with c4:
        st.code(f'{{\n  "Kp_d": {Kp_d:.15g},\n  "Ki_d": {Ki_d:.15g}\n}}', language="json")
    with c5:
        st.code(f'{{\n  "Kp_q": {Kp_q:.15g},\n  "Ki_q": {Ki_q:.15g}\n}}', language="json")


def build_open_loop(
    w: np.ndarray,
    R: float, L: float,
    Kp: float, Ki: float,
    Td_s: float,
    delay_model: str = "pade11",
) -> np.ndarray:
    G = rl_plant_jw(R, L, w)
    C = pi_controller_jw(Kp, Ki, w)
    if delay_model == "exact":
        D = exact_delay_jw(Td_s, w)
    else:
        D = pade11_delay_jw(Td_s, w)
    return C * G * D


def page_stability():
    proj: ProjectConfig = st.session_state.proj
    plant: PlantConfig = st.session_state.plant

    st.markdown("### 7) Stability & Margins (Nominal)")

    g = st.session_state.get("gains", {})
    Kp_d0 = float(g.get("Kp_d", 0.0))
    Ki_d0 = float(g.get("Ki_d", 0.0))
    Kp_q0 = float(g.get("Kp_q", 0.0))
    Ki_q0 = float(g.get("Ki_q", 0.0))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        Kp_d = st.number_input("Kp_d", value=Kp_d0, step=0.01, format="%.6f")
    with c2:
        Ki_d = st.number_input("Ki_d", value=Ki_d0, step=0.1, format="%.6f")
    with c3:
        Kp_q = st.number_input("Kp_q", value=Kp_q0, step=0.01, format="%.6f")
    with c4:
        Ki_q = st.number_input("Ki_q", value=Ki_q0, step=0.1, format="%.6f")

    st.markdown("#### Discrete incremental PI coefficients: Tustin (bilinear)")
    coeff_d = tustin_incremental_pi_coeffs(Kp_d, Ki_d, proj.Ts_s)
    coeff_q = tustin_incremental_pi_coeffs(Kp_q, Ki_q, proj.Ts_s)
    c5, c6 = st.columns(2)
    with c5:
        st.code(f'{{\n  "d-axis a1": {coeff_d["a1"]:.15g},\n  "d-axis b1": {coeff_d["b1"]:.15g},\n  "Ts [s]": {coeff_d["Ts_s"]:.15g}\n}}', language="json")
    with c6:
        st.code(f'{{\n  "q-axis a1": {coeff_q["a1"]:.15g},\n  "q-axis b1": {coeff_q["b1"]:.15g},\n  "Ts [s]": {coeff_q["Ts_s"]:.15g}\n}}', language="json")

    st.caption("Delta form: u[k] = u[k-1] + a1·e[k] + b1·e[k-1] (Tustin discretization)")

    st.markdown("#### Loop assumptions")
    delay_model = st.selectbox("Delay model used in loop", ["pade11", "exact"], index=0)
    st.code(
        '{\n'
        '  "Plant": "G(s)=1/(L s + R) per axis (dq decoupled RL)",\n'
        '  "Controller": "C(s)=Kp + Ki/s (continuous)",\n'
        f'  "Delay model": "{delay_model}",\n'
        f'  "Td [s]": {proj.Td_s:.6e}\n'
        '}',
        language="json",
    )

    w = np.logspace(0, 5, 1600)

    Ld_loop = build_open_loop(w, plant.Rs_ohm, plant.Ld_H, Kp_d, Ki_d, proj.Td_s, delay_model=delay_model)
    Lq_loop = build_open_loop(w, plant.Rs_ohm, plant.Lq_H, Kp_q, Ki_q, proj.Td_s, delay_model=delay_model)

    plot_bode_mag(w, {"|L| [d]": Ld_loop, "|L| [q]": Lq_loop}, ylabel="Magnitude [dB]", title="Open-loop magnitude |L(jω)|")

    Sd = 1.0 / (1.0 + Ld_loop)
    Sq = 1.0 / (1.0 + Lq_loop)
    plot_bode_mag(w, {"|S| [d]": Sd, "|S| [q]": Sq}, ylabel="|S| [dB]", title="Sensitivity magnitude |S(jω)|")

    m_d = compute_margins(w, Ld_loop)
    m_q = compute_margins(w, Lq_loop)
    sp_d = sensitivity_peaks_db(Ld_loop)
    sp_q = sensitivity_peaks_db(Lq_loop)

    PM_MIN = st.number_input("Gate: PM_min [deg]", value=50.0, step=1.0)
    GM_MIN = st.number_input("Gate: GM_min [-]", value=2.0, step=0.1)
    MS_MAX_DB = st.number_input("Gate: Ms_max [dB]", value=6.0, step=0.1)
    MT_MAX_DB = st.number_input("Gate: Mt_max [dB]", value=2.3, step=0.1)

    def pass_fail(m: Dict[str, float], sp: Dict[str, float]) -> bool:
        PM_ok = (not math.isnan(m["PM_deg"])) and (m["PM_deg"] >= PM_MIN)
        GM_ok = (not math.isnan(m["GM"])) and (m["GM"] >= GM_MIN)  # inf is ok
        Ms_ok = (sp["Ms_db"] <= MS_MAX_DB)
        Mt_ok = (sp["Mt_db"] <= MT_MAX_DB)
        return bool(PM_ok and GM_ok and Ms_ok and Mt_ok)

    ok_d = pass_fail(m_d, sp_d)
    ok_q = pass_fail(m_q, sp_q)

    if st.button("Run Stability Gates"):
        st.write("### Results (Nominal)")
        c7, c8 = st.columns(2)
        with c7:
            st.markdown("**d-axis**")
            st.code(
                "{\n"
                f'  "Phase Margin [deg]": {m_d["PM_deg"]:.12g},\n'
                f'  "Gain Margin [-]": {m_d["GM"]:.12g},\n'
                f'  "Sensitivity peak Ms [dB]": {sp_d["Ms_db"]:.12g},\n'
                f'  "Complementary sensitivity peak Mt [dB]": {sp_d["Mt_db"]:.12g},\n'
                f'  "PASS": {"true" if ok_d else "false"}\n'
                "}",
                language="json",
            )
        with c8:
            st.markdown("**q-axis**")
            st.code(
                "{\n"
                f'  "Phase Margin [deg]": {m_q["PM_deg"]:.12g},\n'
                f'  "Gain Margin [-]": {m_q["GM"]:.12g},\n'
                f'  "Sensitivity peak Ms [dB]": {sp_q["Ms_db"]:.12g},\n'
                f'  "Complementary sensitivity peak Mt [dB]": {sp_q["Mt_db"]:.12g},\n'
                f'  "PASS": {"true" if ok_q else "false"}\n'
                "}",
                language="json",
            )

        if ok_d and ok_q:
            st.success("PASS: Nominal stability gates satisfied.")
        else:
            st.error("FAIL: One or more nominal gates not satisfied. Reduce fc, reduce Td, or revisit assumptions.")


def page_robustness_monte_carlo():
    proj: ProjectConfig = st.session_state.proj
    plant: PlantConfig = st.session_state.plant

    st.markdown("### 8) Robustness Sweep: Monte Carlo (Rs, L, Td)")

    g = st.session_state.get("gains", {})
    if not g:
        st.warning("No tuned gains found yet. Go to **Tuning (Preview)** first.")
        return

    Kp_d = float(g["Kp_d"]); Ki_d = float(g["Ki_d"])
    Kp_q = float(g["Kp_q"]); Ki_q = float(g["Ki_q"])

    delay_model = st.selectbox("Delay model used in Monte Carlo loop", ["pade11", "exact"], index=0)

    st.markdown("#### Uncertainty / variation ranges")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        Rs_min = st.number_input("Rs multiplier min", value=0.7, step=0.05)
        Rs_max = st.number_input("Rs multiplier max", value=1.5, step=0.05)
    with c2:
        Ld_min = st.number_input("Ld multiplier min", value=0.8, step=0.05)
        Ld_max = st.number_input("Ld multiplier max", value=1.2, step=0.05)
    with c3:
        Lq_min = st.number_input("Lq multiplier min", value=0.8, step=0.05)
        Lq_max = st.number_input("Lq multiplier max", value=1.2, step=0.05)
    with c4:
        Td_min = st.number_input("Td multiplier min", value=0.8, step=0.05)
        Td_max = st.number_input("Td multiplier max", value=1.2, step=0.05)

    st.markdown("#### Robust gates (same logic as nominal)")
    PM_MIN = st.number_input("Gate: PM_min [deg] (robust)", value=50.0, step=1.0)
    GM_MIN = st.number_input("Gate: GM_min [-] (robust)", value=2.0, step=0.1)
    MS_MAX_DB = st.number_input("Gate: Ms_max [dB] (robust)", value=6.0, step=0.1)
    MT_MAX_DB = st.number_input("Gate: Mt_max [dB] (robust)", value=2.3, step=0.1)

    Nmc = st.number_input("Monte Carlo samples", min_value=50, max_value=5000, value=400, step=50)
    seed = st.number_input("Random seed", min_value=0, max_value=10_000_000, value=12345, step=1)

    w = np.logspace(0, 5, 1200)

    def gates(m: Dict[str, float], sp: Dict[str, float]) -> bool:
        PM_ok = (not math.isnan(m["PM_deg"])) and (m["PM_deg"] >= PM_MIN)
        GM_ok = (not math.isnan(m["GM"])) and (m["GM"] >= GM_MIN)
        Ms_ok = (sp["Ms_db"] <= MS_MAX_DB)
        Mt_ok = (sp["Mt_db"] <= MT_MAX_DB)
        return bool(PM_ok and GM_ok and Ms_ok and Mt_ok)

    if st.button("Run Monte Carlo Robustness Sweep"):
        rng = np.random.default_rng(int(seed))

        Rs_mult = rng.uniform(Rs_min, Rs_max, int(Nmc))
        Ld_mult = rng.uniform(Ld_min, Ld_max, int(Nmc))
        Lq_mult = rng.uniform(Lq_min, Lq_max, int(Nmc))
        Td_mult = rng.uniform(Td_min, Td_max, int(Nmc))

        PMd = np.zeros(int(Nmc)); GMd = np.zeros(int(Nmc)); Msd = np.zeros(int(Nmc)); Mtd = np.zeros(int(Nmc))
        PMq = np.zeros(int(Nmc)); GMq = np.zeros(int(Nmc)); Msq = np.zeros(int(Nmc)); Mtq = np.zeros(int(Nmc))
        pass_both = np.zeros(int(Nmc), dtype=np.int32)

        for k in range(int(Nmc)):
            Rk = plant.Rs_ohm * Rs_mult[k]
            Ldk = plant.Ld_H * Ld_mult[k]
            Lqk = plant.Lq_H * Lq_mult[k]
            Tdk = proj.Td_s * Td_mult[k]

            Ld_loop = build_open_loop(w, Rk, Ldk, Kp_d, Ki_d, Tdk, delay_model=delay_model)
            Lq_loop = build_open_loop(w, Rk, Lqk, Kp_q, Ki_q, Tdk, delay_model=delay_model)

            md = compute_margins(w, Ld_loop)
            mq = compute_margins(w, Lq_loop)
            sd = sensitivity_peaks_db(Ld_loop)
            sq = sensitivity_peaks_db(Lq_loop)

            PMd[k] = md["PM_deg"]; GMd[k] = md["GM"]; Msd[k] = sd["Ms_db"]; Mtd[k] = sd["Mt_db"]
            PMq[k] = mq["PM_deg"]; GMq[k] = mq["GM"]; Msq[k] = sq["Ms_db"]; Mtq[k] = sq["Mt_db"]

            pass_both[k] = 1 if (gates(md, sd) and gates(mq, sq)) else 0

        pass_rate = float(np.mean(pass_both)) * 100.0

        wc = {
            "PM_d_min": float(np.nanmin(PMd)),
            "PM_q_min": float(np.nanmin(PMq)),
            "GM_d_min": float(np.nanmin(GMd)),
            "GM_q_min": float(np.nanmin(GMq)),
            "Ms_d_max_db": float(np.nanmax(Msd)),
            "Ms_q_max_db": float(np.nanmax(Msq)),
            "Mt_d_max_db": float(np.nanmax(Mtd)),
            "Mt_q_max_db": float(np.nanmax(Mtq)),
            "Pass_rate_%": pass_rate,
        }

        st.markdown("#### Robust summary (worst-case + pass-rate)")
        st.code(str(wc).replace("'", '"'), language="json")

        if pass_rate >= 95.0:
            st.success(f"Robustness: PASS-like behavior (pass rate {pass_rate:.1f}%).")
        else:
            st.warning(f"Robustness: not fully convincing yet (pass rate {pass_rate:.1f}%). Consider reducing fc, improving Td, or revisiting model/filters/decoupling.")

        st.markdown("#### Distributions (histograms)")
        fig1, ax1 = plt.subplots()
        ax1.hist(PMd[~np.isnan(PMd)], bins=30, alpha=0.7, label="PM d [deg]")
        ax1.hist(PMq[~np.isnan(PMq)], bins=30, alpha=0.7, label="PM q [deg]")
        ax1.set_title("Phase Margin distribution")
        ax1.set_xlabel("PM [deg]")
        ax1.set_ylabel("Count")
        ax1.grid(True)
        ax1.legend()
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        ax2.hist(Msd, bins=30, alpha=0.7, label="Ms d [dB]")
        ax2.hist(Msq, bins=30, alpha=0.7, label="Ms q [dB]")
        ax2.set_title("Sensitivity peak Ms distribution")
        ax2.set_xlabel("Ms [dB]")
        ax2.set_ylabel("Count")
        ax2.grid(True)
        ax2.legend()
        st.pyplot(fig2)


def page_saturation_aw_check():
    proj: ProjectConfig = st.session_state.proj
    plant: PlantConfig = st.session_state.plant
    g = st.session_state.get("gains", {})

    st.markdown("### 9) Saturation & Anti-Windup Check (time-domain): (fractional delay + true integrator)")

    if not g:
        st.warning("No tuned gains found yet. Go to **Tuning (Preview)** first.")
        return

    st.info("Note: Internal saturation math assumes PI output is phase voltage peak (V_phase_peak). Units selection is labeling unless you extend conversions.")

    st.markdown("#### Select axis and gains")
    axis = st.selectbox("Axis to simulate", ["d-axis", "q-axis"], index=0)
    if axis == "d-axis":
        L = plant.Ld_H
        Kp = float(g["Kp_d"]); Ki = float(g["Ki_d"])
    else:
        L = plant.Lq_H
        Kp = float(g["Kp_q"]); Ki = float(g["Ki_q"])

    c0, c1 = st.columns(2)
    with c0:
        Kp = st.number_input("Kp (used in sim)", value=float(Kp), step=0.01, format="%.6f")
    with c1:
        Ki = st.number_input("Ki (used in sim)", value=float(Ki), step=0.1, format="%.6f")

    st.markdown("#### Hardware/operating assumptions (voltage limit)")
    c2, c3, c4 = st.columns(3)
    with c2:
        Vdc = st.number_input("DC link Vdc [V]", value=700.0, step=10.0)
    with c3:
        mmax = st.number_input("Max modulation index m_max", value=0.95, step=0.01)
    with c4:
        pwm = st.selectbox("PWM utilization model", ["SVPWM", "SPWM"], index=0 if proj.pwm_method == "SVPWM" else 1)

    Vmax = v_phase_peak_max(float(Vdc), float(mmax), str(pwm))
    st.info(f"Voltage limit model: v_phase_peak_max ≈ {Vmax:.2f} V (used as PI output saturation).")

    st.markdown("#### Simulation setup")
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        i_step = st.number_input("Current step [A]", value=200.0, step=10.0)
    with c6:
        t_end = st.number_input("Sim duration [s]", value=0.02, step=0.005, format="%.4f")
    with c7:
        td_use = st.number_input("Use total delay Td [s]", value=float(proj.Td_s), step=10e-6, format="%.6f")
    with c8:
        settle_band = st.number_input("Settling band (fraction)", value=0.02, step=0.01)

    st.markdown("#### Delay placement (how Td is applied in the discrete sim)")
    delay_placement = st.selectbox("Delay placement", ["split", "measurement", "actuation"], index=0)

    st.caption("Delay uses fractional-sample linear interpolation. This check targets saturation/windup/ripple risk, not a full inverter+FOC simulation.")

    cfg = SimCfg(
        t_end_s=float(t_end),
        i_ref_step_A=float(i_step),
        i0_A=0.0,
        Vdc_V=float(Vdc),
        m_max=float(mmax),
        pwm=str(pwm),
        td_s=float(td_use),
        Ts_s=float(proj.Ts_s),
        delay_placement=str(delay_placement),
    )

    st.markdown("#### Anti-windup options")
    aw_mode = st.selectbox("AW mode", ["none", "clamp", "backcalc"], index=1)
    kaw = st.number_input("Back-calc gain k_aw (only if backcalc)", value=1.0, step=0.1)

    if st.button("Run Saturation/AW Check"):
        out_no = simulate_current_loop_rl(plant.Rs_ohm, L, Kp, Ki, cfg, aw_mode="none", kaw=float(kaw))
        out_aw = simulate_current_loop_rl(plant.Rs_ohm, L, Kp, Ki, cfg, aw_mode=str(aw_mode), kaw=float(kaw))

        sat_frac_no = float(np.mean(out_no["sat_flag"]))
        sat_frac_aw = float(np.mean(out_aw["sat_flag"]))

        met_no = step_metrics(out_no["t"], out_no["i"], out_no["i_ref"], settle_band=float(settle_band))
        met_aw = step_metrics(out_aw["t"], out_aw["i"], out_aw["i_ref"], settle_band=float(settle_band))

        risk_no = ripple_risk_indicator(out_no["t"], out_no["i"], out_no["i_ref"], sat_frac_no)
        risk_aw = ripple_risk_indicator(out_aw["t"], out_aw["i"], out_aw["i_ref"], sat_frac_aw)

        st.markdown("#### Results summary")
        cA, cB = st.columns(2)
        with cA:
            st.markdown("**No anti-windup**")
            st.code(
                "{\n"
                f'  "delay_total_samples": {float(out_no["d_total"][0]):.6g},\n'
                f'  "delay_meas_samples": {float(out_no["d_meas"][0]):.6g},\n'
                f'  "delay_act_samples": {float(out_no["d_act"][0]):.6g},\n'
                f'  "Vmax_V": {float(out_no["Vmax"][0]):.6g},\n'
                f'  "saturation_fraction": {sat_frac_no:.6g},\n'
                f'  "overshoot_pct": {met_no["overshoot_pct"]:.6g},\n'
                f'  "settling_s": {met_no["settling_s"]:.6g},\n'
                f'  "tail_ripple_pp_pct": {risk_no["tail_ripple_pp_pct"]:.6g},\n'
                f'  "risk_level": "{risk_no["level"]}",\n'
                f'  "risk_reason": "{risk_no["reason"]}"\n'
                "}",
                language="json",
            )
        with cB:
            st.markdown(f"**With anti-windup ({aw_mode})**")
            st.code(
                "{\n"
                f'  "delay_total_samples": {float(out_aw["d_total"][0]):.6g},\n'
                f'  "delay_meas_samples": {float(out_aw["d_meas"][0]):.6g},\n'
                f'  "delay_act_samples": {float(out_aw["d_act"][0]):.6g},\n'
                f'  "Vmax_V": {float(out_aw["Vmax"][0]):.6g},\n'
                f'  "saturation_fraction": {sat_frac_aw:.6g},\n'
                f'  "overshoot_pct": {met_aw["overshoot_pct"]:.6g},\n'
                f'  "settling_s": {met_aw["settling_s"]:.6g},\n'
                f'  "tail_ripple_pp_pct": {risk_aw["tail_ripple_pp_pct"]:.6g},\n'
                f'  "risk_level": "{risk_aw["level"]}",\n'
                f'  "risk_reason": "{risk_aw["reason"]}"\n'
                "}",
                language="json",
            )

        plot_time(out_no["t"], {"i_ref": out_no["i_ref"], "i (no AW)": out_no["i"], "i (AW)": out_aw["i"]},
                  title=f"{axis} current step response (compare: no AW vs {aw_mode})")

        plot_time(out_no["t"], {"v_cmd (no AW)": out_no["v_cmd"], "v_sat (no AW)": out_no["v_sat"], "v_applied (no AW)": out_no["v_applied"]},
                  title="Voltage command/saturation/applied (no AW)")

        plot_time(out_aw["t"], {"v_cmd (AW)": out_aw["v_cmd"], "v_sat (AW)": out_aw["v_sat"], "v_applied (AW)": out_aw["v_applied"]},
                  title=f"Voltage command/saturation/applied ({aw_mode})")

        plot_time(out_no["t"], {"xI (no AW)": out_no["xI"], "xI (AW)": out_aw["xI"]},
                  title="Integrator state xI (windup comparison)")

        if risk_aw["risk"]:
            st.warning("This check flags a potential saturation/ripple risk under the chosen step + voltage assumptions. In practice: validate voltage headroom vs speed/flux, decoupling terms, and filter delays.")
        else:
            st.success("This saturation/AW check looks clean under the chosen assumptions (low tail ripple + manageable saturation).")


# ============================================================
# App main
# ============================================================

def main():
    st.set_page_config(page_title="Current Controller Design Wizard", layout="wide")
    init_state()

    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Project Setup",
            "Plant / Model",
            "Tuning (Preview)",
            "Stability & Margins (Nominal)",
            "Robustness Sweep (Monte Carlo)",
            "Saturation & Anti-Windup Check",
        ],
        index=0,
    )

    page_header()

    if page == "Project Setup":
        page_project_setup()
    elif page == "Plant / Model":
        page_plant_model()
    elif page == "Tuning (Preview)":
        page_tuning()
    elif page == "Stability & Margins (Nominal)":
        page_stability()
    elif page == "Robustness Sweep (Monte Carlo)":
        page_robustness_monte_carlo()
    elif page == "Saturation & Anti-Windup Check":
        page_saturation_aw_check()


if __name__ == "__main__":
    main()
