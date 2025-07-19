"""
C-1 Decoupling d- & q-Axis Currents
Author – Masoud Bakhshi · July 2025

This script simulates and visualizes the effect of cross-coupling and decoupling (feedforward compensation) in the d-q current control of a Permanent Magnet Synchronous Motor (PMSM).
It uses a physically accurate PMSM model, realistic PI current controllers, and step current references. The results are animated and exported as both MP4 and GIF.
"""
import time
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

# --- PMSM and Control Parameters ---
# These values are typical for a small PMSM and can be adjusted for other machines.
R_s = 0.5      # Stator resistance [Ohm]
L_d = 0.005    # d-axis inductance [H]
dL_q = 0.005   # q-axis inductance [H]
lambda_f = 0.15  # Permanent magnet flux linkage [Vs]
omega = 1000.0   # Electrical speed [rad/s]

# PI controller gains, tuned for a closed-loop bandwidth of ~500 rad/s
# Kp = L * wb, Ki = R * wb, where wb is the desired bandwidth
Kp = 2.5   # Proportional gain
Ki = 250.0 # Integral gain

# --- Simulation Parameters ---
sim_t = 0.05  # Total simulation time [s]
dt = 1e-5     # Simulation time step [s]
t = np.arange(0, sim_t + dt, dt)
n_steps = len(t)

# --- Reference Currents ---
# Step in d-axis current reference, q-axis reference remains zero
# These can be modified for other test scenarios.
i_d_ref = np.ones_like(t) * 10.0  # 0 -> 10A step at t=0
i_q_ref = np.zeros_like(t)         # always 0A

# --- Simulation Function ---
def simulate(decoupling=True):
    """
    Simulate the PMSM current response with or without decoupling (feedforward compensation).
    Returns the d- and q-axis currents and the applied voltages.
    """
    i_d = np.zeros(n_steps)
    i_q = np.zeros(n_steps)
    int_e_d = 0.0
    int_e_q = 0.0
    v_d_hist = np.zeros(n_steps)
    v_q_hist = np.zeros(n_steps)
    for k in range(n_steps - 1):
        # Current control errors
        e_d = i_d_ref[k] - i_d[k]
        e_q = i_q_ref[k] - i_q[k]
        int_e_d += e_d * dt
        int_e_q += e_q * dt
        # PI controller output
        v_d_pi = Kp * e_d + Ki * int_e_d
        v_q_pi = Kp * e_q + Ki * int_e_q
        # Feedforward (decoupling) terms
        if decoupling:
            v_d_ff = -omega * dL_q * i_q[k]
            v_q_ff = omega * L_d * i_d[k] + omega * lambda_f
        else:
            v_d_ff = 0.0
            v_q_ff = 0.0
        v_d = v_d_pi + v_d_ff
        v_q = v_q_pi + v_q_ff
        v_d_hist[k] = v_d
        v_q_hist[k] = v_q
        # PMSM model (Euler integration)
        di_d = (v_d - R_s * i_d[k] + omega * dL_q * i_q[k]) / L_d
        di_q = (v_q - R_s * i_q[k] - omega * L_d * i_d[k] - omega * lambda_f) / dL_q
        i_d[k+1] = i_d[k] + di_d * dt
        i_q[k+1] = i_q[k] + di_q * dt
    return i_d, i_q, v_d_hist, v_q_hist

# --- Run Simulations ---
start = time.time()
i_d_c, i_q_c, _, _ = simulate(decoupling=False)   # Coupled (no decoupling)
i_d_d, i_q_d, _, _ = simulate(decoupling=True)    # Decoupled (with feedforward)
end = time.time()

# --- Overshoot and Deviation Calculation ---
def first_peak(signal, ref_idx=0):
    """
    Find the first peak after the step (ignoring initial value).
    Returns the value and its index.
    """
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(signal[ref_idx:])
    if len(peaks) > 0:
        idx = peaks[0] + ref_idx
        return signal[idx], idx
    else:
        # If no peak, use max value after step
        idx = np.argmax(signal[ref_idx:]) + ref_idx
        return signal[idx], idx

# For i_d: overshoot is (first peak - final)/final * 100%
step_idx = 0  # step at t=0
id_final = i_d_ref[-1]
id_peak, id_peak_idx = first_peak(i_d_c, ref_idx=step_idx+1)
os_id = 100 * (id_peak - id_final) / id_final if id_final != 0 else 0.0

# For i_q: report max absolute deviation from zero
max_iq_dev = np.max(np.abs(i_q_c))

# --- Max Deviation Calculation ---
# For i_d: max absolute deviation from reference
max_id_dev = np.max(np.abs(i_d_c - i_d_ref))
# For i_q: max absolute deviation from zero (already calculated as max_iq_dev)

# --- Plotting and Animation Setup ---
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.8, 7.2), dpi=100, sharex=True)
# Adjust top margin and suptitle position so the title is fully visible and not cut off
fig.subplots_adjust(hspace=0.3, left=0.08, right=0.96, top=0.88, bottom=0.12)

# Add a professional, descriptive title at the top of the figure
fig.suptitle("PMSM d-q Current Control: Cross-Coupling and Decoupling Demonstration\nStep Response with and without Feedforward Compensation", fontsize=16, fontweight='bold', y=0.97)

# --- Top: i_d step response ---
line_id_c, = ax1.plot([], [], label='Coupled', color='#1f77b4', lw=2)
line_id_d, = ax1.plot([], [], label='Decoupled', color='#2ca02c', lw=2, ls='--')
line_id_ref, = ax1.plot([], [], label=r'$i_{d,ref}$', color='gray', lw=2, ls=':')
ax1.set_ylabel(r'$i_d$ (A)')
ax1.set_xlim(0, sim_t)
# Auto-scale y-limits for i_d (with margin)
all_id = np.concatenate([i_d_c, i_d_d, i_d_ref])
id_min, id_max = np.min(all_id), np.max(all_id)
id_margin = 0.1 * (id_max - id_min if id_max != id_min else 1)
ax1.set_ylim(id_min - id_margin, id_max + id_margin)
ax1.legend(loc='lower right', fontsize=10, framealpha=0.95)
ax1.set_title(r'$i_d$ Step Response', fontsize=12)

# --- Bottom: i_q cross-response ---
line_iq_c, = ax2.plot([], [], label='Coupled', color='#ff7f0e', lw=2)
line_iq_d, = ax2.plot([], [], label='Decoupled', color='#d62728', lw=2, ls='--')
line_iq_ref, = ax2.plot([], [], label=r'$i_{q,ref}$', color='gray', lw=2, ls=':')
ax2.set_ylabel(r'$i_q$ (A)')
ax2.set_xlabel('Time (s)')
ax2.set_xlim(0, sim_t)
# Auto-scale y-limits for i_q (with margin)
all_iq = np.concatenate([i_q_c, i_q_d, i_q_ref])
iq_min, iq_max = np.min(all_iq), np.max(all_iq)
iq_margin = 0.1 * (iq_max - iq_min if iq_max != iq_min else 1)
ax2.set_ylim(iq_min - iq_margin, iq_max + iq_margin)
ax2.legend(loc='lower right', fontsize=10, framealpha=0.95)
ax2.set_title(r'$i_q$ Cross-Response', fontsize=12)

# --- Overshoot Badges ---
badge_id = ax1.text(0.98, 0.85, '', transform=ax1.transAxes, ha='right', va='center', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.3', fc='w', ec='k', lw=1))
badge_iq = ax2.text(0.98, 0.85, '', transform=ax2.transAxes, ha='right', va='center', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.3', fc='w', ec='k', lw=1))

# --- Author Credit ---
fig.text(0.99, 0.02, 'Masoud Bakhshi · July 2025', ha='right', va='bottom', fontsize=8, style='italic')

# --- Animation Parameters ---
fps_mp4 = 30
fps_gif = 20
n_frames = 150
frame_indices = np.linspace(0, n_steps - 1, n_frames).astype(int)
t_anim = t[frame_indices]
i_d_c_anim = i_d_c[frame_indices]
i_d_d_anim = i_d_d[frame_indices]
i_q_c_anim = i_q_c[frame_indices]
i_q_d_anim = i_q_d[frame_indices]

def badge_color(frame):
    # Badge flashes green for first 20% of animation
    if frame < n_frames * 0.2:
        return '#7fff7f'
    return 'w'

def init():
    # Initialize all animated lines and badges to empty
    line_id_c.set_data([], [])
    line_id_d.set_data([], [])
    line_id_ref.set_data([], [])
    line_iq_c.set_data([], [])
    line_iq_d.set_data([], [])
    line_iq_ref.set_data([], [])
    badge_id.set_text('')
    badge_iq.set_text('')
    return (line_id_c, line_id_d, line_id_ref, line_iq_c, line_iq_d, line_iq_ref, badge_id, badge_iq)

def animate(i):
    # Update all animated lines for frame i
    line_id_c.set_data(t_anim[:i+1], i_d_c_anim[:i+1])
    line_id_d.set_data(t_anim[:i+1], i_d_d_anim[:i+1])
    line_id_ref.set_data(t_anim[:i+1], i_d_ref[frame_indices][:i+1])
    line_iq_c.set_data(t_anim[:i+1], i_q_c_anim[:i+1])
    line_iq_d.set_data(t_anim[:i+1], i_q_d_anim[:i+1])
    line_iq_ref.set_data(t_anim[:i+1], i_q_ref[frame_indices][:i+1])
    # Show max deviation for i_d and i_q
    badge_id.set_text(f'Max Dev: {max_id_dev:.2f} A')
    badge_id.set_bbox(dict(boxstyle='round,pad=0.3', fc=badge_color(i), ec='k', lw=1))
    badge_iq.set_text(f'Max Dev: {max_iq_dev:.2f} A')
    badge_iq.set_bbox(dict(boxstyle='round,pad=0.3', fc=badge_color(i), ec='k', lw=1))
    return (line_id_c, line_id_d, line_id_ref, line_iq_c, line_iq_d, line_iq_ref, badge_id, badge_iq)

ani = FuncAnimation(
    fig, animate, frames=n_frames, init_func=init, blit=True, interval=1000 / fps_mp4
)

# --- Export Animation ---
print('Rendering MP4...')
ani.save('C1_Decoupling_dq_Currents.mp4', writer=FFMpegWriter(fps=fps_mp4, bitrate=1800))
print('Rendering GIF...')
ani.save('C1_Decoupling_dq_Currents.gif', writer=PillowWriter(fps=fps_gif))

print(f'Runtime: {end - start:.2f} s')
print('MP4 and GIF saved as C1_Decoupling_dq_Currents.mp4 and .gif') 