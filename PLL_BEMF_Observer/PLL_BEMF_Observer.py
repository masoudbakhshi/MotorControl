#PLL‑Based Back‑EMF Observer at Low Speed
#Author: Masoud Bakhshi · July 2025

#This script simulates a phase-locked loop (PLL) tracking the angle of a shrinking back-EMF vector, as would occur in a motor running down from 100 rpm to 0. The animation shows how the PLL phase error increases as the signal fades, and includes a lock indicator and speed panel.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, patches
import matplotlib as mpl

# Simulation parameters
AUTHOR = "Masoud Bakhshi"
TITLE = "PLL‑Based Back‑EMF Observer at Low Speed"
DATE = "July 2025"
DURATION = 3.0  # seconds
FPS = 40
N_FRAMES = int(DURATION * FPS)
T = np.linspace(0, DURATION, N_FRAMES)

# Motor and observer settings
RPM_START = 100
RPM_END = 0
POLE_PAIRS = 4
OMEGA_M = 2 * np.pi * (RPM_START + (RPM_END - RPM_START) * (T / DURATION)) / 60  # Mechanical speed [rad/s]
OMEGA_E = POLE_PAIRS * OMEGA_M  # Electrical speed [rad/s]
BEMF_MAX = 1.0  # Maximum back-EMF amplitude
BEMF = BEMF_MAX * (OMEGA_M / OMEGA_M[0])  # Back-EMF amplitude fades with speed

# PI controller gains for the PLL
KP = 18.0
KI = 400.0
DT = T[1] - T[0]

# Arrays to store simulation results
true_theta = np.zeros(N_FRAMES)      # True electrical angle
pll_theta = np.zeros(N_FRAMES)       # PLL estimated angle
pll_omega = np.zeros(N_FRAMES)       # PLL estimated speed
phase_error = np.zeros(N_FRAMES)     # Phase error in degrees
lock_status = np.zeros(N_FRAMES, dtype=bool)  # PLL lock status

# Run the PLL simulation
int_err = 0.0
for k in range(1, N_FRAMES):
    # Update true angle
    true_theta[k] = true_theta[k-1] + OMEGA_E[k-1] * DT
    # Compute back-EMF vector
    e_alpha = BEMF[k] * np.cos(true_theta[k])
    e_beta = BEMF[k] * np.sin(true_theta[k])
    # PLL phase detector
    est_angle = np.arctan2(e_beta, e_alpha)
    err = np.angle(np.exp(1j*(est_angle - pll_theta[k-1])))  # Error wrapped to [-pi, pi]
    phase_error[k] = np.degrees(err)
    int_err += err * DT
    # PI controller
    omega_est = KP * err + KI * int_err
    pll_omega[k] = omega_est
    pll_theta[k] = pll_theta[k-1] + omega_est * DT
    # PLL is considered locked if error is within ±10 degrees
    lock_status[k] = np.abs(phase_error[k]) < 10

# Set up the figure and axes
mpl.rcParams.update({
    'font.family': "sans-serif",
    'mathtext.fontset': 'cm',
    'axes.titlesize': 15,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11
})
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 0.5])
fig.subplots_adjust(top=0.92, left=0.06, right=0.98, hspace=0.45, wspace=0.22)
fig.suptitle(TITLE, fontsize=22, y=0.98, weight='bold')
fig.text(0.07, 0.96, AUTHOR, fontsize=15, weight='bold')
fig.text(0.07, 0.93, DATE, fontsize=13, style='italic')

# Top panel: Back-EMF waveforms
ax0 = fig.add_subplot(gs[0, :])
ax0.set_title("Back-EMF $e_{\\alpha}$, $e_{\\beta}$ (amplitude shrinks as speed drops)")
ax0.set_xlim(T[0], T[-1])
ax0.set_ylim(-1.1, 1.1)
ax0.set_ylabel("Amplitude")
ax0.set_xlabel("Time [s]")
line_ea, = ax0.plot([], [], color='#287EC7', lw=2, label="$e_{\\alpha}$")
line_eb, = ax0.plot([], [], color='#FFD700', lw=2, label="$e_{\\beta}$")
ax0.legend(loc='upper right', fontsize=12)
ax0.grid(True, linestyle=':', alpha=0.7)

# Middle panel: PLL phase error
ax1 = fig.add_subplot(gs[1, :])
ax1.set_title("PLL Phase Error $(\\theta_{PLL} - \\theta_{true})$")
ax1.set_xlim(T[0], T[-1])
err_margin = 10
err_min = np.min(phase_error) - err_margin
err_max = np.max(phase_error) + err_margin
ax1.set_ylim(err_min, err_max)
ax1.set_ylabel("Error [deg]")
ax1.set_xlabel("Time [s]")
line_err, = ax1.plot([], [], color='navy', lw=2, label="Phase Error")
err_spike, = ax1.plot([], [], 'r.', ms=8, alpha=0.7, label="|Error| > 20°")
ax1.axhline(20, color='red', ls='--', lw=1, alpha=0.5)
ax1.axhline(-20, color='red', ls='--', lw=1, alpha=0.5)
ax1.legend(loc='upper right', fontsize=12)
ax1.grid(True, linestyle=':', alpha=0.7)

# Bottom left: Lock indicator
ax2 = fig.add_subplot(gs[2, 0])
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')
led = patches.Circle((0.5, 0.5), 0.18, color='green', ec='black', lw=2)
ax2.add_patch(led)
ax2.text(0.5, 0.85, "PLL Lock Status", ha='center', va='center', fontsize=14, weight='bold')
lock_txt = ax2.text(0.5, 0.18, "LOCKED", ha='center', va='center', fontsize=13, weight='bold', color='green')

# Bottom right: Speed panel
ax3 = fig.add_subplot(gs[2, 1])
ax3.set_xlim(T[0], T[-1])
ax3.set_ylim(0, RPM_START*1.1)
ax3.set_ylabel("Speed [rpm]")
ax3.set_xlabel("Time [s]")
line_rpm, = ax3.plot([], [], color='#287EC7', lw=2, label="Speed (rpm)")
ax3.legend(loc='upper right', fontsize=12)
ax3.grid(True, linestyle=':', alpha=0.7)

# Animation initialization

def init():
    line_ea.set_data([], [])
    line_eb.set_data([], [])
    line_err.set_data([], [])
    err_spike.set_data([], [])
    line_rpm.set_data([], [])
    led.set_color('green')
    lock_txt.set_text('LOCKED')
    lock_txt.set_color('green')
    return (line_ea, line_eb, line_err, err_spike, led, lock_txt, line_rpm)

# Animation update for each frame

def animate(i):
    # Update back-EMF waveforms
    line_ea.set_data(T[:i+1], BEMF[:i+1]*np.cos(true_theta[:i+1]))
    line_eb.set_data(T[:i+1], BEMF[:i+1]*np.sin(true_theta[:i+1]))
    # Update phase error
    line_err.set_data(T[:i+1], phase_error[:i+1])
    spikes = np.where(np.abs(phase_error[:i+1]) > 20)[0]
    err_spike.set_data(T[spikes], phase_error[spikes])
    # Update lock indicator
    if np.abs(phase_error[i]) > 10:
        led.set_color('red')
        lock_txt.set_text('UNLOCKED')
        lock_txt.set_color('red')
    else:
        led.set_color('green')
        lock_txt.set_text('LOCKED')
        lock_txt.set_color('green')
    # Update speed panel
    line_rpm.set_data(T[:i+1], (OMEGA_M[:i+1]/(2*np.pi))*60)
    return (line_ea, line_eb, line_err, err_spike, led, lock_txt, line_rpm)

# Create and export the animation
ani = animation.FuncAnimation(
    fig, animate, frames=N_FRAMES, init_func=init, blit=True, interval=1000/FPS
)

EXPORT_BASENAME = "PLL_BEMF_Observer_MasoudBakhshi_July2025"
mp4_path = f"{EXPORT_BASENAME}.mp4"
gif_path = f"{EXPORT_BASENAME}.gif"

ani.save(mp4_path, writer=animation.FFMpegWriter(fps=FPS, bitrate=1800), dpi=100)
ani.save(gif_path, writer=animation.PillowWriter(fps=FPS), dpi=100)

print(f"\nExported animation as {mp4_path} and {gif_path} (1280x720, {N_FRAMES} frames, {DURATION:.1f}s)")
print(f"Author: {AUTHOR}  |  Date: {DATE}")
print(f"PI gains: Kp={KP}, Ki={KI}")
print(f"Speed ramp: {RPM_START} → {RPM_END} rpm over {DURATION}s") 
