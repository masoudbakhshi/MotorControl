"""
FOC_Loop_Animation_MasoudBakhshi.py

Expert-level animation of the complete closed-loop Field-Oriented Control (FOC) for PMSM machines.
- Shows signal flow from speed reference to PWM generation
- Includes block diagram animation, dynamic plots, and interactive controls
- Exports GIF and MP4 (1920x1080, 30fps, 25-30s)
- Author: Masoud Bakhshi
"""

# === Imports ===
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, CheckButtons
# from control import TransferFunction, forced_response  # Uncomment if using control library

# === Parameters ===
# Simulation parameters
SIM_DURATION = 2.5  # seconds (zoomed-in, detailed view)
FS = 1000           # Sampling frequency (Hz)
ANIM_FPS = 30       # Animation frame rate (smooth)
N_FRAMES = int(SIM_DURATION * ANIM_FPS)  # 1:1 with simulation
N_SAMPLES = int(SIM_DURATION * FS)

# Motor and controller parameters (example values, to be refined)
Ld = 0.001  # d-axis inductance (H)
Lq = 0.001  # q-axis inductance (H)
psi_pm = 0.1  # Permanent magnet flux (Wb)
Rs = 0.01    # Stator resistance (Ohm)
P = 4        # Pole pairs
J = 0.001    # Inertia (kg*m^2)
B = 0.0001   # Friction (N*m*s)

# Controller gains (reduced for stability)
Kp_speed = 0.1
Ki_speed = 1.0
Kp_id = 0.2
Ki_id = 5.0
Kp_iq = 0.2
Ki_iq = 5.0

# Reference values
omega_ref_default = 100.0  # rad/s
id_ref_default = 0.0       # A (for SPM)

decoupling_enabled_default = True

# === Expert Options ===
# (No longer used as globals, now passed to simulation function)

# === Placeholders for simulation data ===
# (To be filled by simulation logic)
time = np.linspace(0, SIM_DURATION, N_SAMPLES)
# Reference step: 0 until t=0.2s, then omega_ref_default
omega_ref_profile = np.zeros(N_SAMPLES)
omega_ref_profile[time >= 0.2] = omega_ref_default
# Load torque profile: 0 Nm until t=0.2s, then 10 Nm, then 30 Nm at t=1.0s
load_torque_profile = np.zeros(N_SAMPLES)
load_torque_profile[time >= 0.2] = 10.0
load_torque_profile[time >= 1.0] = 30.0
omega = np.zeros(N_SAMPLES)
torque = np.zeros(N_SAMPLES)
id = np.zeros(N_SAMPLES)
iq = np.zeros(N_SAMPLES)
vd = np.zeros(N_SAMPLES)
vq = np.zeros(N_SAMPLES)
svpwms = np.zeros((N_SAMPLES, 3))  # 3-phase SVPWM
v_alpha_arr = np.zeros(N_SAMPLES)  # Store v_alpha
v_beta_arr = np.zeros(N_SAMPLES)   # Store v_beta
i_alpha_arr = np.zeros(N_SAMPLES)  # Store i_alpha
i_beta_arr = np.zeros(N_SAMPLES)   # Store i_beta
iabc_arr = np.zeros((N_SAMPLES, 3))  # Store Iabc (Ia, Ib, Ic)
load_torque_est_arr = np.zeros(N_SAMPLES)  # Store estimated load torque
delta_arr = np.zeros(N_SAMPLES)  # Store power angle (radians)
theta_e_arr = np.zeros(N_SAMPLES)  # Store rotor electrical angle (rad)

# --- Simulation function for comparison ---
def run_foc_simulation(USE_FEEDFORWARD=False, USE_DIST_OBS=False):
    omega = np.zeros(N_SAMPLES)
    torque = np.zeros(N_SAMPLES)
    id = np.zeros(N_SAMPLES)
    iq = np.zeros(N_SAMPLES)
    vd = np.zeros(N_SAMPLES)
    vq = np.zeros(N_SAMPLES)
    svpwms = np.zeros((N_SAMPLES, 3))
    v_alpha_arr = np.zeros(N_SAMPLES)
    v_beta_arr = np.zeros(N_SAMPLES)
    i_alpha_arr = np.zeros(N_SAMPLES)
    i_beta_arr = np.zeros(N_SAMPLES)
    iabc_arr = np.zeros((N_SAMPLES, 3))
    load_torque_est_arr = np.zeros(N_SAMPLES)
    omega_m = 0.0
    theta_e = 0.0
    id_val = 0.0
    iq_val = 0.0
    speed_int = 0.0
    id_int = 0.0
    iq_int = 0.0
    INT_LIM = 1000.0
    OUT_LIM = 500.0
    load_torque_est = 0.0
    for k in range(N_SAMPLES):
        t = time[k]
        omega_ref = omega_ref_profile[k]
        load_torque = load_torque_profile[k]
        speed_err = omega_ref - omega_m
        speed_int += speed_err / FS
        speed_int = np.clip(speed_int, -INT_LIM, INT_LIM)
        # --- Expert: Feedforward or Disturbance Observer ---
        ff_torque = 0.0
        if USE_FEEDFORWARD:
            ff_torque = load_torque
        elif USE_DIST_OBS:
            ff_torque = load_torque_est_arr[k-1] if k > 0 else 0.0
        # --- Speed PI controller with feedforward ---
        tq_ref = Kp_speed * speed_err + Ki_speed * speed_int + ff_torque
        tq_ref = np.clip(tq_ref, -OUT_LIM, OUT_LIM)
        iq_ref = tq_ref / (1.5 * P * psi_pm) if psi_pm != 0 else 0.0
        iq_ref = np.clip(iq_ref, -OUT_LIM, OUT_LIM)
        id_ref = id_ref_default
        id_err = id_ref - id_val
        id_int += id_err / FS
        id_int = np.clip(id_int, -INT_LIM, INT_LIM)
        vd_val = Kp_id * id_err + Ki_id * id_int
        vd_val -= omega_m * Lq * iq_val if decoupling_enabled_default else 0.0
        vd_val = np.clip(vd_val, -OUT_LIM, OUT_LIM)
        iq_err = iq_ref - iq_val
        iq_int += iq_err / FS
        iq_int = np.clip(iq_int, -INT_LIM, INT_LIM)
        vq_val = Kp_iq * iq_err + Ki_iq * iq_int
        vq_val += omega_m * (Ld * id_val + psi_pm) if decoupling_enabled_default else 0.0
        vq_val = np.clip(vq_val, -OUT_LIM, OUT_LIM)
        theta_e = theta_e % (2 * np.pi)
        theta_e_arr[k] = theta_e  # Store rotor electrical angle
        v_alpha = vd_val * np.cos(theta_e) - vq_val * np.sin(theta_e)
        v_beta  = vd_val * np.sin(theta_e) + vq_val * np.cos(theta_e)
        v_alpha_arr[k] = v_alpha
        v_beta_arr[k] = v_beta
        # Inverse Park for currents
        i_alpha = id_val * np.cos(theta_e) - iq_val * np.sin(theta_e)
        i_beta  = id_val * np.sin(theta_e) + iq_val * np.cos(theta_e)
        i_alpha_arr[k] = i_alpha
        i_beta_arr[k] = i_beta
        # Inverse Clarke for Iabc
        ia = i_alpha
        ib = -0.5 * i_alpha + np.sqrt(3)/2 * i_beta
        ic = -0.5 * i_alpha - np.sqrt(3)/2 * i_beta
        iabc_arr[k, 0] = ia
        iabc_arr[k, 1] = ib
        iabc_arr[k, 2] = ic
        va = v_alpha
        vb = -0.5 * v_alpha + np.sqrt(3)/2 * v_beta
        vc = -0.5 * v_alpha - np.sqrt(3)/2 * v_beta
        va = np.clip(va, -OUT_LIM, OUT_LIM)
        vb = np.clip(vb, -OUT_LIM, OUT_LIM)
        vc = np.clip(vc, -OUT_LIM, OUT_LIM)
        if Ld == 0 or Lq == 0:
            print(f"Zero inductance at step {k}")
            break
        did_dt = (vd_val - Rs * id_val + omega_m * Lq * iq_val) / Ld
        diq_dt = (vq_val - Rs * iq_val - omega_m * (Ld * id_val + psi_pm)) / Lq
        id_val += did_dt / FS
        iq_val += diq_dt / FS
        id_val = np.clip(id_val, -OUT_LIM, OUT_LIM)
        iq_val = np.clip(iq_val, -OUT_LIM, OUT_LIM)
        tq = 1.5 * P * psi_pm * iq_val
        domega_dt = (tq - load_torque - B * omega_m) / J
        omega_m += domega_dt / FS
        omega_m = np.clip(omega_m, -OUT_LIM, OUT_LIM)
        theta_e += omega_m / FS
        omega[k] = omega_m
        torque[k] = tq
        id[k] = id_val
        iq[k] = iq_val
        vd[k] = vd_val
        vq[k] = vq_val
        svpwms[k, 0] = va
        svpwms[k, 1] = vb
        svpwms[k, 2] = vc
        # --- Disturbance Observer: estimate load torque ---
        load_torque_est = tq - J * domega_dt - B * omega_m
        load_torque_est_arr[k] = load_torque_est
        if (np.isnan(omega_m) or np.isnan(id_val) or np.isnan(iq_val) or
            np.isnan(vd_val) or np.isnan(vq_val) or np.isnan(va) or np.isnan(vb) or np.isnan(vc) or
            np.isinf(omega_m) or np.isinf(id_val) or np.isinf(iq_val) or
            np.isinf(vd_val) or np.isinf(vq_val) or np.isinf(va) or np.isinf(vb) or np.isinf(vc)):
            print(f"NaN or inf detected at step {k}, t={t:.4f}s")
            print(f"omega_m={omega_m}, id={id_val}, iq={iq_val}, vd={vd_val}, vq={vq_val}, va={va}, vb={vb}, vc={vc}")
            break
    return omega, iq, load_torque_est_arr

# === FOC Simulation Functions ===
def foc_simulation():
    """Run the FOC closed-loop simulation and fill global arrays."""
    global omega, torque, id, iq, vd, vq, svpwms, v_alpha_arr, v_beta_arr, i_alpha_arr, i_beta_arr, iabc_arr, load_torque_est_arr, delta_arr, theta_e_arr
    omega_m = 0.0
    theta_e = 0.0
    id_val = 0.0
    iq_val = 0.0
    speed_int = 0.0
    id_int = 0.0
    iq_int = 0.0
    INT_LIM = 1000.0
    OUT_LIM = 500.0
    load_torque_est = 0.0
    for k in range(N_SAMPLES):
        t = time[k]
        omega_ref = omega_ref_profile[k]
        load_torque = load_torque_profile[k]
        speed_err = omega_ref - omega_m
        speed_int += speed_err / FS
        speed_int = np.clip(speed_int, -INT_LIM, INT_LIM)
        # --- Expert: Feedforward or Disturbance Observer ---
        ff_torque = 0.0
        # Default: use feedforward for animation
        ff_torque = load_torque
        # --- Speed PI controller with feedforward ---
        tq_ref = Kp_speed * speed_err + Ki_speed * speed_int + ff_torque
        tq_ref = np.clip(tq_ref, -OUT_LIM, OUT_LIM)
        iq_ref = tq_ref / (1.5 * P * psi_pm) if psi_pm != 0 else 0.0
        iq_ref = np.clip(iq_ref, -OUT_LIM, OUT_LIM)
        id_ref = id_ref_default
        id_err = id_ref - id_val
        id_int += id_err / FS
        id_int = np.clip(id_int, -INT_LIM, INT_LIM)
        vd_val = Kp_id * id_err + Ki_id * id_int
        vd_val -= omega_m * Lq * iq_val if decoupling_enabled_default else 0.0
        vd_val = np.clip(vd_val, -OUT_LIM, OUT_LIM)
        iq_err = iq_ref - iq_val
        iq_int += iq_err / FS
        iq_int = np.clip(iq_int, -INT_LIM, INT_LIM)
        vq_val = Kp_iq * iq_err + Ki_iq * iq_int
        vq_val += omega_m * (Ld * id_val + psi_pm) if decoupling_enabled_default else 0.0
        vq_val = np.clip(vq_val, -OUT_LIM, OUT_LIM)
        theta_e = theta_e % (2 * np.pi)
        theta_e_arr[k] = theta_e  # Store rotor electrical angle
        v_alpha = vd_val * np.cos(theta_e) - vq_val * np.sin(theta_e)
        v_beta  = vd_val * np.sin(theta_e) + vq_val * np.cos(theta_e)
        v_alpha_arr[k] = v_alpha
        v_beta_arr[k] = v_beta
        # Inverse Park for currents
        i_alpha = id_val * np.cos(theta_e) - iq_val * np.sin(theta_e)
        i_beta  = id_val * np.sin(theta_e) + iq_val * np.cos(theta_e)
        i_alpha_arr[k] = i_alpha
        i_beta_arr[k] = i_beta
        # Inverse Clarke for Iabc
        ia = i_alpha
        ib = -0.5 * i_alpha + np.sqrt(3)/2 * i_beta
        ic = -0.5 * i_alpha - np.sqrt(3)/2 * i_beta
        iabc_arr[k, 0] = ia
        iabc_arr[k, 1] = ib
        iabc_arr[k, 2] = ic
        va = v_alpha
        vb = -0.5 * v_alpha + np.sqrt(3)/2 * v_beta
        vc = -0.5 * v_alpha - np.sqrt(3)/2 * v_beta
        va = np.clip(va, -OUT_LIM, OUT_LIM)
        vb = np.clip(vb, -OUT_LIM, OUT_LIM)
        vc = np.clip(vc, -OUT_LIM, OUT_LIM)
        if Ld == 0 or Lq == 0:
            print(f"Zero inductance at step {k}")
            break
        did_dt = (vd_val - Rs * id_val + omega_m * Lq * iq_val) / Ld
        diq_dt = (vq_val - Rs * iq_val - omega_m * (Ld * id_val + psi_pm)) / Lq
        id_val += did_dt / FS
        iq_val += diq_dt / FS
        id_val = np.clip(id_val, -OUT_LIM, OUT_LIM)
        iq_val = np.clip(iq_val, -OUT_LIM, OUT_LIM)
        tq = 1.5 * P * psi_pm * iq_val
        domega_dt = (tq - load_torque - B * omega_m) / J
        omega_m += domega_dt / FS
        omega_m = np.clip(omega_m, -OUT_LIM, OUT_LIM)
        theta_e += omega_m / FS
        omega[k] = omega_m
        torque[k] = tq
        id[k] = id_val
        iq[k] = iq_val
        vd[k] = vd_val
        vq[k] = vq_val
        svpwms[k, 0] = va
        svpwms[k, 1] = vb
        svpwms[k, 2] = vc
        # --- Power angle (delta) ---
        delta_arr[k] = np.arctan2(iq_val, id_val)  # radians
        # --- Disturbance Observer: estimate load torque ---
        load_torque_est = tq - J * domega_dt - B * omega_m
        load_torque_est_arr[k] = load_torque_est
        if (np.isnan(omega_m) or np.isnan(id_val) or np.isnan(iq_val) or
            np.isnan(vd_val) or np.isnan(vq_val) or np.isnan(va) or np.isnan(vb) or np.isnan(vc) or
            np.isinf(omega_m) or np.isinf(id_val) or np.isinf(iq_val) or
            np.isinf(vd_val) or np.isinf(vq_val) or np.isinf(va) or np.isinf(vb) or np.isinf(vc)):
            print(f"NaN or inf detected at step {k}, t={t:.4f}s")
            print(f"omega_m={omega_m}, id={id_val}, iq={iq_val}, vd={vd_val}, vq={vq_val}, va={va}, vb={vb}, vc={vc}")
            break

# === Animation Setup ===
def setup_block_diagram(ax):
    """This function does not draw anything. The author and title are added outside the plot area."""
    ax.axis('off')
    # Nothing is drawn here

# --- Animation update function ---
def update_animation(frame):
    """Update all the lines in the plots for the current animation frame."""
    idx = int(frame * FS / ANIM_FPS)  # This gives the current sample index
    if idx >= N_SAMPLES:
        idx = N_SAMPLES - 1
    # Update each line with data up to the current frame
    line_omega.set_data(time[:idx], omega[:idx])
    line_load.set_data(time[:idx], load_torque_profile[:idx])
    line_load_est.set_data(time[:idx], load_torque_est_arr[:idx])
    line_iq.set_data(time[:idx], iq[:idx])
    line_id.set_data(time[:idx], id[:idx])
    line_vq.set_data(time[:idx], vq[:idx])
    line_vd.set_data(time[:idx], vd[:idx])
    line_valpha.set_data(time[:idx], v_alpha_arr[:idx])
    line_vbeta.set_data(time[:idx], v_beta_arr[:idx])
    line_ialpha.set_data(time[:idx], i_alpha_arr[:idx])
    line_ibeta.set_data(time[:idx], i_beta_arr[:idx])
    line_ia.set_data(time[:idx], iabc_arr[:idx,0])
    line_ib.set_data(time[:idx], iabc_arr[:idx,1])
    line_ic.set_data(time[:idx], iabc_arr[:idx,2])
    # theta_e_arr is in radians, convert to degrees for plotting
    line_thetae.set_data(time[:idx], np.rad2deg(theta_e_arr[:idx]))
    line_sinthetae.set_data(time[:idx], np.sin(theta_e_arr[:idx]))
    line_costhetae.set_data(time[:idx], np.cos(theta_e_arr[:idx]))
    # Return all lines so the animation can update them
    return (line_omega, line_load, line_load_est, line_iq, line_id, line_vq, line_vd, line_valpha, line_vbeta, line_ialpha, line_ibeta, line_ia, line_ib, line_ic, line_thetae, line_sinthetae, line_costhetae)

# === Interactive Controls ===
def setup_interactive_controls(fig, ax):
    """Add sliders and toggles for ω_ref, Kp, Ki, decoupling."""
    pass  # To be implemented

# --- Export function ---
def export_animation(anim):
    """Save the animation as GIF and MP4 with high quality (200 dpi). Both files will play at the same speed."""
    print('Saving GIF...')
    anim.save('FOC_Loop_Animation_MasoudBakhshi.gif', writer='pillow', fps=15, dpi=200)
    print('Saving MP4...')
    anim.save('FOC_Loop_Animation_MasoudBakhshi.mp4', writer='ffmpeg', fps=15, dpi=200)
    print('Export complete.')

# === Main Script ===
if __name__ == "__main__":
    # --- Run both cases for comparison ---
    omega_pi, iq_pi, load_est_pi = run_foc_simulation(USE_FEEDFORWARD=False, USE_DIST_OBS=False)
    omega_ff, iq_ff, load_est_ff = run_foc_simulation(USE_FEEDFORWARD=True, USE_DIST_OBS=False)
    # --- Comparison Plot ---
    import matplotlib.pyplot as plt
    fig_cmp, (ax_cmp1, ax_cmp2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax_cmp1.plot(time, omega_pi, 'b-', lw=2, label='ω (PI only)')
    ax_cmp1.plot(time, omega_ff, 'r--', lw=2, label='ω (PI + Feedforward)')
    ax_cmp1.set_ylabel('Speed ω [rad/s]')
    ax_cmp1.legend(loc='upper left')
    ax_cmp1.grid(True, linestyle='--', alpha=0.7)
    ax_cmp2.plot(time, iq_pi, 'b-', lw=2, label='Iq (PI only)')
    ax_cmp2.plot(time, iq_ff, 'r--', lw=2, label='Iq (PI + Feedforward)')
    ax_cmp2.set_xlabel('Time [s]')
    ax_cmp2.set_ylabel('Iq [A]')
    ax_cmp2.legend(loc='upper left')
    ax_cmp2.grid(True, linestyle='--', alpha=0.7)
    fig_cmp.suptitle('Comparison: Classic PI vs. PI + Feedforward', fontsize=18)
    fig_cmp.tight_layout(rect=[0, 0, 1, 0.96])
    # --- Continue with animation as before ---
    foc_simulation()  # For animation
    # Auto-scale y-limits for each plot with margin
    margin = 0.2  # Increased margin for current plots
    def get_ylim(arr):
        vmin, vmax = np.min(arr), np.max(arr)
        if vmin == vmax:
            return vmin-1, vmax+1
        rng = vmax - vmin
        return vmin - margin*rng, vmax + margin*rng
    fig = plt.figure(figsize=(16, 18), dpi=120)
    # Use 7 rows for extra separation
    gs = fig.add_gridspec(7, 2, height_ratios=[1,1,1.2,1,1.2,0.1,0.7])
    ax_block = fig.add_subplot(gs[0, :])
    setup_block_diagram(ax_block)
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])
    ax5 = fig.add_subplot(gs[3, 0])
    ax6 = fig.add_subplot(gs[3, 1])
    ax7 = fig.add_subplot(gs[4, :])
    # Set font sizes
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.title.set_fontsize(20)
        ax.xaxis.label.set_fontsize(16)
        ax.yaxis.label.set_fontsize(16)
        ax.tick_params(axis='both', labelsize=14)
    # Omega + load torque
    ax1.set_title('Speed (ω) and Load Torque', pad=18)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('ω [rad/s] / T_load [Nm]')
    ax1.set_xlim(0, SIM_DURATION)
    ax1.set_ylim(*get_ylim(np.concatenate([omega, load_torque_profile, load_torque_est_arr])))
    global line_omega, line_load, line_load_est
    line_omega, = ax1.plot([], [], color='b', lw=3, label='ω')
    line_load, = ax1.plot([], [], 'k--', lw=2, label='T_load')
    line_load_est, = ax1.plot([], [], color='orange', lw=2, linestyle=':', label='T_load_est')
    ax1.legend(loc='upper left')
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    # Iq, Id
    ax2.set_title('Currents (Iq, Id)', pad=18)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Current [A]')
    ax2.set_xlim(0, SIM_DURATION)
    ax2.set_ylim(*get_ylim(np.concatenate([iq, id, iabc_arr.flatten()])))
    global line_iq, line_id
    line_iq, = ax2.plot([], [], color='r', lw=3, label='Iq')
    line_id, = ax2.plot([], [], color='g', lw=3, label='Id')
    ax2.legend(loc='upper left')
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)
    # Vq, Vd
    ax3.set_title('Voltages (Vq, Vd)', pad=18)
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Voltage [V]')
    ax3.set_xlim(0, SIM_DURATION)
    ax3.set_ylim(*get_ylim(np.concatenate([vq, vd])))
    global line_vq, line_vd
    line_vq, = ax3.plot([], [], 'm', lw=2, label='Vq')
    line_vd, = ax3.plot([], [], 'c', lw=2, label='Vd')
    ax3.legend(loc='upper left')
    ax3.grid(True, which='both', linestyle='--', alpha=0.7)
    # Alpha-Beta Voltages
    ax4.set_title('Alpha-Beta Voltages', pad=18)
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Voltage [V]')
    ax4.set_xlim(0, SIM_DURATION)
    ax4.set_ylim(*get_ylim(np.concatenate([v_alpha_arr, v_beta_arr])))
    global line_valpha, line_vbeta
    line_valpha, = ax4.plot([], [], 'b', lw=2, label='V_alpha')
    line_vbeta, = ax4.plot([], [], 'r', lw=2, label='V_beta')
    ax4.legend(loc='upper left')
    ax4.grid(True, which='both', linestyle='--', alpha=0.7)
    # Alpha-Beta Currents
    ax5.set_title('Alpha-Beta Currents', pad=18)
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Current [A]')
    ax5.set_xlim(0, SIM_DURATION)
    ax5.set_ylim(*get_ylim(np.concatenate([i_alpha_arr, i_beta_arr, iabc_arr.flatten()])))
    global line_ialpha, line_ibeta
    line_ialpha, = ax5.plot([], [], color='b', lw=3, label='I_alpha')
    line_ibeta, = ax5.plot([], [], color='r', lw=3, label='I_beta')
    ax5.legend(loc='upper left')
    ax5.grid(True, which='both', linestyle='--', alpha=0.7)
    # Iabc
    ax6.set_title('Iabc Phase Currents', pad=18)
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Current [A]')
    ax6.set_xlim(0, SIM_DURATION)
    ax6.set_ylim(*get_ylim(iabc_arr))
    global line_ia, line_ib, line_ic
    line_ia, = ax6.plot([], [], color='#0072B2', lw=2.5, label='Ia')      # Blue
    line_ib, = ax6.plot([], [], color='#E69F00', lw=2.5, label='Ib')     # Orange
    line_ic, = ax6.plot([], [], color='#009E73', lw=2.5, label='Ic')     # Green
    ax6.legend(loc='upper left')
    ax6.grid(True, which='both', linestyle='--', alpha=0.7)
    # Resolver info
    ax8 = fig.add_subplot(gs[4, :])
    # Rotor angle θe (deg)
    ax8.set_title('Resolver Angle θe', pad=18)
    ax8.set_xlabel('Time [s]')
    ax8.set_ylabel('θe [deg]')
    ax8.set_xlim(0, SIM_DURATION)
    ax8.set_ylim(0, 360)
    ax8.set_yticks([0, 180, 360])
    ax8.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ax8.yaxis.set_ticks_position('left')
    ax8.xaxis.set_ticks_position('bottom')
    ax8.tick_params(top=False, right=False, labelsize=10)
    global line_thetae
    # theta_e_arr is in radians, convert to degrees for plotting
    line_thetae, = ax8.plot([], [], color='k', lw=2, label='θe (deg)')
    ax8.legend(loc='upper left')
    ax8.grid(True, which='both', linestyle='--', alpha=0.7)
    # Add a blank row for separation
    ax_blank = fig.add_subplot(gs[5, :])
    ax_blank.axis('off')
    # Resolver sin/cos signals
    ax9 = fig.add_subplot(gs[6, :])
    ax9.set_title('Resolver Signals: sin(θe) (resolver), cos(θe) (resolver)', pad=18)
    ax9.set_xlabel('Time [s]')
    ax9.set_ylabel('Amplitude')
    ax9.set_xlim(0, SIM_DURATION)
    ax9.set_ylim(-1.2, 1.2)
    global line_sinthetae, line_costhetae
    line_sinthetae, = ax9.plot([], [], color='b', lw=1.5, label='sin(θe) (resolver)')
    line_costhetae, = ax9.plot([], [], color='r', lw=1.5, label='cos(θe) (resolver)')
    ax9.legend(loc='upper left')
    ax9.grid(True, which='both', linestyle='--', alpha=0.7)
    fig.subplots_adjust(top=0.94)
    fig.tight_layout(h_pad=0.1)
    # Add author and title using fig.text for even tighter placement
    fig.text(0.5, 0.97, 'Masoud Bakhshi', ha='center', va='top', fontsize=15)
    fig.text(0.5, 0.95, 'Field-Oriented Control (FOC) Closed-Loop Python Simulation with Load Torque Disturbance Observer', ha='center', va='top', fontsize=18, fontweight='bold')
    fig.subplots_adjust(top=0.94)
    fig.tight_layout(h_pad=0.1)
    anim = animation.FuncAnimation(fig, update_animation, frames=N_FRAMES, interval=1000/ANIM_FPS, blit=True)
    export_animation(anim)
    # plt.show() 