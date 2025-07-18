# -----------------------------------------------------------------------------
#  SPWM_Principles_SinglePhase.py
#
#  Slow-motion visualisation of single-phase SPWM, including the effect on an RL load.  
#  Written by: Masoud Bakhshi
#
#  Outputs:
#      • SPWM_Principles_SinglePhase.gif
#      • SPWM_Principles_SinglePhase.mp4   (ffmpeg required for MP4)
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter, FFMpegWriter

# --------------------------------------------------------------------------- #
# User settings
# --------------------------------------------------------------------------- #
FREF          = 50.0            # Reference (fundamental) frequency [Hz]
FCARRIER      = 1800.0          # Triangular carrier frequency   [Hz]

M_START, M_END, M_STEP = 0.5, 1.0, 0.1   # Modulation index sweep

PTS_PER_CAR   = 200             # Samples per carrier period
FPS           = 30              # Animation frame-rate
SECONDS_PER_CYCLE = 10.0        # How long (wall-clock) to draw one 20 ms cycle
CYCLES_ON_SCREEN  = 2           # Number of fundamental cycles to display

R_LOAD        = 0.2             # Ω
L_LOAD        = 10e-3           # H    (10 mH)

GIF_FILE      = "SPWM_Principles_SinglePhase.gif"
MP4_FILE      = "SPWM_Principles_SinglePhase.mp4"

# --------------------------------------------------------------------------- #
# Derived numbers – avoid magic literals later on
# --------------------------------------------------------------------------- #
T_CYCLE     = 1.0 / FREF                      # 20 ms
T_WINDOW    = CYCLES_ON_SCREEN * T_CYCLE      # 40 ms window

M_LIST      = np.round(np.arange(M_START, M_END + 1e-6, M_STEP), 2)

TCAR        = 1.0 / FCARRIER
DT          = TCAR / PTS_PER_CAR

PTS_CYCLE   = int(np.ceil(T_CYCLE / DT))
PTS_WINDOW  = PTS_CYCLE * CYCLES_ON_SCREEN

FRM_CYCLE   = int(FPS * SECONDS_PER_CYCLE)
FRM_WINDOW  = FRM_CYCLE * CYCLES_ON_SCREEN    # Frames per 40 ms window

# Build a long time base: one 40 ms “window” per m value
total_pts   = PTS_WINDOW * len(M_LIST)
t           = np.arange(total_pts) * DT
m_timeline  = np.repeat(M_LIST, PTS_WINDOW)   # m value at each sample

# --------------------------------------------------------------------------- #
# Build continuous carrier and reference signals
# --------------------------------------------------------------------------- #
half_tri    = np.linspace(-1.0, 1.0, PTS_PER_CAR // 2, endpoint=False)
tri_period  = np.concatenate((half_tri, half_tri[::-1]))
carrier     = np.tile(tri_period, int(np.ceil(total_pts / tri_period.size)))[:total_pts]

# Reference sine – resets every 40 ms so each window is independent
v_ref       = m_timeline * np.sin(2*np.pi*FREF * (t % T_WINDOW))

# PWM comparator and resulting pole voltage
gate        = (v_ref >= carrier).astype(float)
v_pole      = np.where(gate > 0.5, 1.0, -1.0)

# RL-load current, first-order explicit Euler
i_load = np.zeros_like(v_pole)
rl_tau = L_LOAD / R_LOAD
for k in range(1, total_pts):
    di = (v_pole[k-1] - R_LOAD * i_load[k-1]) / L_LOAD
    i_load[k] = i_load[k-1] + di * DT

# --------------------------------------------------------------------------- #

frames = []
base = 0
for _ in M_LIST:
    frames.extend(np.linspace(base, base + PTS_WINDOW - 1,
                              FRM_WINDOW, dtype=int))
    base += PTS_WINDOW

# --------------------------------------------------------------------------- #
# Figure: four stacked axes
# --------------------------------------------------------------------------- #
plt.close("all")
fig, axes = plt.subplots(
    4, 1,
    figsize=(10, 8),
    sharex=True,
    gridspec_kw=dict(height_ratios=[3, 1, 1, 2], hspace=0.15),
    dpi=150
)

ax_ref, ax_gate, ax_pole, ax_cur = axes

# Fixed x-axis = 0…40 ms
ax_cur.set_xlim(0, T_WINDOW)

ax_ref.set_ylim(-1.2, 1.2)
ax_gate.set_ylim(-0.1, 1.1)
ax_pole.set_ylim(-1.2, 1.2)
ax_cur.set_ylim(-1.5, 1.5)

ax_cur.set_xlabel("Time  [s]")
ax_ref.set_ylabel("Amplitude [p.u.]")
ax_gate.set_ylabel("Gate")
ax_pole.set_ylabel("Pole\nvoltage")
ax_cur.set_ylabel("RL\nload current")

ax_ref.set_title(
    f"Single-phase SPWM  –  {CYCLES_ON_SCREEN} × 20 ms window per m\n"
    f"Sweep m = {M_START:.1f} … {M_END:.1f}   |   "
    f"f_ref = {FREF:.0f} Hz,  f_carrier = {FCARRIER:.0f} Hz   |   "
    f"RL load  R = {R_LOAD} Ω,  L = {L_LOAD*1e3:.0f} mH"
)

# Create the line handles
ln_carrier, = ax_ref.plot([], [], lw=1.5, label="Carrier")
ln_ref,     = ax_ref.plot([], [], lw=1.5, label="Reference")
ax_ref.legend(loc="upper left", ncol=2, fontsize=8)

ln_gate,    = ax_gate.plot([], [], lw=1.5)
ln_pole,    = ax_pole.plot([], [], lw=1.5)
ln_cur,     = ax_cur .plot([], [], lw=1.5, color="tab:red")

m_text = ax_ref.text(0.99, 0.02, "", transform=ax_ref.transAxes,
                     ha="right", va="bottom", fontsize=10, weight="bold")

fig.text(0.5, 0.01, "Author – Masoud Bakhshi",
         ha="center", va="center", fontsize=9, style="italic")

# --------------------------------------------------------------------------- #
# Animation helpers
# --------------------------------------------------------------------------- #
def init():
    for ln in (ln_carrier, ln_ref, ln_gate, ln_pole, ln_cur):
        ln.set_data([], [])
    m_text.set_text("")
    return ln_carrier, ln_ref, ln_gate, ln_pole, ln_cur, m_text

def update(idx):
    seg_idx   = idx // PTS_WINDOW
    seg_start = seg_idx * PTS_WINDOW
    t_local   = t[seg_start:idx] - t[seg_start]   # 0…40 ms span

    ln_carrier.set_data(t_local, carrier[seg_start:idx])
    ln_ref.set_data(t_local, v_ref[seg_start:idx])
    ln_gate.set_data(t_local, gate[seg_start:idx])
    ln_pole.set_data(t_local, v_pole[seg_start:idx])
    ln_cur.set_data(t_local, i_load[seg_start:idx])

    m_text.set_text(f"m = {M_LIST[seg_idx]:.1f}")
    return ln_carrier, ln_ref, ln_gate, ln_pole, ln_cur, m_text

ani = animation.FuncAnimation(
    fig,
    update,
    frames=frames,
    init_func=init,
    blit=True,
    interval=1000 / FPS,
    repeat=False
)

# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
print(f"Saving {GIF_FILE} …")
ani.save(GIF_FILE, writer=PillowWriter(fps=FPS))

print(f"Saving {MP4_FILE} …   (ensure ffmpeg is installed)")
ani.save(MP4_FILE, writer=FFMpegWriter(fps=FPS))

print("Finished:")
print(f"  • {GIF_FILE}")
print(f"  • {MP4_FILE}")
