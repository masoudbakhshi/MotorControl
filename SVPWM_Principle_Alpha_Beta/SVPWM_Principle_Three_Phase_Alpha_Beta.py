# -----------------------------------------------------------------------------
#  SVPWM_Principles_SinglePhase.py
#
#  αβ-driven SVPWM animation with three phase-to-star voltages, large hexagon
#  inset, two composing vectors, and practical modulation sweep 0.5→1.15.
#
#  Author : Masoud Bakhshi   • 15-Jul-2025
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter, FFMpegWriter
from matplotlib.lines import Line2D
from math import pi

# ─── USER SETTINGS ──────────────────────────────────────────────────────────
FREF, FCAR  = 50.0, 1800.0                 # fundamental & carrier [Hz]
M_LIST      = [0.5, 0.8, 1.0, 1.15]        # up to practical six-step onset
PTS_CAR     = 200                          # samples per carrier
FPS, SLOW   = 30, 10.0                     # 20 ms ⇒ 10 s (×500 slow-mo)
CYC_WIN     = 2                            # 40 ms window
R_LOAD, L_LOAD = 1.0, 10e-3                # Ω, H
GIF_FILE, MP4_FILE = ("SVPWM_Principles_SinglePhase.gif",
                      "SVPWM_Principles_SinglePhase.mp4")

# ─── TIME GRID ──────────────────────────────────────────────────────────────
T_CY, TCAR = 1/FREF, 1/FCAR
DT        = TCAR / PTS_CAR
PTS_CY    = int(np.ceil(T_CY/DT))
PTS_WIN   = PTS_CY * CYC_WIN
FRM_WIN   = int(FPS * SLOW * CYC_WIN)
TOTAL     = PTS_WIN * len(M_LIST)
t         = np.arange(TOTAL) * DT
m_vec     = np.repeat(M_LIST, PTS_WIN)

# ─── αβ REFERENCE (pure sine) ───────────────────────────────────────────────
vα = m_vec * np.cos(2*pi*FREF * (t % (CYC_WIN*T_CY)))
vβ = m_vec * np.sin(2*pi*FREF * (t % (CYC_WIN*T_CY)))
theta_deg = np.mod(np.degrees(np.arctan2(vβ, vα)), 360)
sector = (theta_deg // 60).astype(int) % 6              # 0…5
θ_s    = np.deg2rad(theta_deg % 60)

# Duty ratios (linear SVPWM)
T1 = m_vec * np.sin(pi/3 - θ_s)
T2 = m_vec * np.sin(θ_s)

# ─── 7-SEGMENT GATE PATTERN (length-safe) ───────────────────────────────────
U = np.array([[1,0,0],[1,1,0],[0,1,0],
              [0,1,1],[0,0,1],[1,0,1],
              [0,0,0]], dtype=int)          # row 6 = zero vector

Sa = np.zeros(TOTAL, int); Sb = np.zeros_like(Sa); Sc = np.zeros_like(Sa)

for n in range(0, TOTAL, PTS_CAR):
    s   = int(sector[n])                     # current sector (0-5)
    t1n = int(round(T1[n] * PTS_CAR))
    t2n = int(round(T2[n] * PTS_CAR))
    t0n = PTS_CAR - t1n - t2n
    if t0n < 0:                              # clip if rounding overflow
        t1n = max(0, t1n + t0n//2)
        t2n = max(0, t2n + t0n - t0n//2)
        t0n = 0
    t0h1 = t0n // 2
    t0h2 = t0n - t0h1

    pattern = (
        [U[6]]*t0h1 +
        [U[s]]*t1n +
        [U[(s+1)%6]]*t2n +
        [U[6]]*t0h2
    )
    # guarantee exact 200 samples
    if len(pattern) < PTS_CAR:
        pattern += [U[6]] * (PTS_CAR - len(pattern))
    elif len(pattern) > PTS_CAR:
        pattern = pattern[:PTS_CAR]

    pat = np.asarray(pattern, int)
    Sa[n:n+PTS_CAR], Sb[n:n+PTS_CAR], Sc[n:n+PTS_CAR] = pat.T

gateA = Sa.astype(float)

# ─── PHASE-TO-STAR VOLTAGES (carrier-average) ───────────────────────────────
kernel = np.ones(PTS_CAR) / PTS_CAR
Va = np.convolve((2*Sa - Sb - Sc)/3, kernel, mode='same')
Vb = np.convolve((2*Sb - Sc - Sa)/3, kernel, mode='same')
Vc = np.convolve((2*Sc - Sa - Sb)/3, kernel, mode='same')

# RL current for phase-A
iL = np.zeros_like(Va)
for k in range(1, TOTAL):
    iL[k] = iL[k-1] + DT * ((Va[k-1] - R_LOAD*iL[k-1]) / L_LOAD)

# Instantaneous applied vector (blue square)
α_act = (2*Sa - Sb - Sc)/3
β_act = (Sb - Sc)/np.sqrt(3)

# Frame list
frames, base = [], 0
for _ in M_LIST:
    frames += np.linspace(base, base+PTS_WIN-1, FRM_WIN, dtype=int).tolist()
    base   += PTS_WIN

# Hexagon geometry
vec_ang = np.deg2rad(30 + 60*np.arange(6))
α_tip, β_tip = np.cos(vec_ang), np.sin(vec_ang)

# ─── FIGURE & AXES ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10,13), dpi=150)
gs  = fig.add_gridspec(7,1, height_ratios=[9,4,1.6,1.6,1.6,1.6,3], hspace=0.18)

ax_hex = fig.add_subplot(gs[0]); ax_hex.set_aspect("equal","box"); ax_hex.axis("off")
ax_ref  = fig.add_subplot(gs[1])
ax_gate = fig.add_subplot(gs[2], sharex=ax_ref)
ax_phase= fig.add_subplot(gs[3], sharex=ax_ref)
ax_sec  = fig.add_subplot(gs[4], sharex=ax_ref)
ax_cur  = fig.add_subplot(gs[5], sharex=ax_ref)

for ax in (ax_ref,ax_gate,ax_phase,ax_sec,ax_cur):
    ax.set_xlim(0, CYC_WIN / FREF)

ax_ref .set_ylim(-1.2,1.2); ax_ref .set_ylabel("α  /  β")
ax_gate.set_ylim(-.1,1.1);  ax_gate.set_ylabel("Gate A")
ax_phase.set_ylim(-1.2,1.2);ax_phase.set_ylabel("Phase V")
ax_sec .set_ylim(.4,6.6);   ax_sec .set_ylabel("Sector")
ax_cur .set_ylim(-1.6,1.6); ax_cur .set_ylabel("iₐ")
ax_cur .set_xlabel("Time  [s]")

fig.suptitle("SVPWM – Single-Phase Principles (three phase-to-star voltages)",
             fontsize=13, weight="bold", y=0.98)

# Hexagon static
ang = np.deg2rad(np.arange(0,361,60)); ax_hex.plot(np.cos(ang),np.sin(ang),'k',lw=1)
for i,(xt,yt) in enumerate(zip(α_tip,β_tip)):
    ax_hex.arrow(0,0,xt,yt,head_width=.07,head_length=.1,length_includes_head=True,
                 fc='k',ec='k')
    ax_hex.text(1.18*xt, 1.18*yt, f"U{i+1}", ha='center', va='center', fontsize=9)
ax_hex.text(0,0,"U7,8", ha='center', va='center', fontsize=9)
ax_hex.set_xlim(-2.0,2.0); ax_hex.set_ylim(-2.0,2.0)
ax_hex.legend(handles=[
    Line2D([],[],color='k',lw=1.5,label='U1…U6'),
    Line2D([],[],color='red',lw=2,label='Reference'),
    Line2D([],[],color='green',lw=2,label='U_s'),
    Line2D([],[],color='orange',lw=2,label='U_{s+1}'),
    Line2D([],[],marker='s',color='tab:blue',lw=0,label='Applied vec')],
    loc='upper center', bbox_to_anchor=(0.5,-0.09), ncol=3, fontsize=8)

# Dynamic lines
ln_α, = ax_ref.plot([],[],lw=1.3,color='tab:blue',   label='α')
ln_β, = ax_ref.plot([],[],lw=1.3,color='tab:orange', label='β')
ax_ref.legend(loc='upper left', fontsize=8)

ln_gate ,= ax_gate.plot([],[],lw=1.2)
ln_va, = ax_phase.plot([],[],lw=1.2,color='tab:blue',   label='Va')
ln_vb, = ax_phase.plot([],[],lw=1.2,color='tab:orange', label='Vb')
ln_vc, = ax_phase.plot([],[],lw=1.2,color='tab:green',  label='Vc')
ax_phase.legend(loc='upper left', fontsize=8)

ln_sec ,= ax_sec.plot([],[],lw=1.2,drawstyle='steps-post')
ln_cur ,= ax_cur.plot([],[],lw=1.2,color='tab:red')

ref_ln ,= ax_hex.plot([],[],'-',lw=2,color='red')
us_ln  ,= ax_hex.plot([],[],'-',lw=2,color='green')
usp1_ln,= ax_hex.plot([],[],'-',lw=2,color='orange')
ref_dot,= ax_hex.plot([],[],'o',ms=7,color='red')
vec_dot,= ax_hex.plot([],[],'s',ms=8,color='tab:blue')

m_txt = ax_hex.text(0,-1.9,"",ha='center',va='top',fontsize=12,weight='bold')
fig.text(0.5,0.009,"Author – Masoud Bakhshi",
         ha='center', va='center', fontsize=9, style='italic')

# ─── ANIMATION CALLBACKS ────────────────────────────────────────────────────
def init():
    for ln in (ln_α,ln_β,ln_gate,ln_va,ln_vb,ln_vc,ln_sec,ln_cur):
        ln.set_data([], [])
    for ln in (ref_ln,us_ln,usp1_ln): ln.set_data([], [])
    ref_dot.set_data([], []); vec_dot.set_data([], []); m_txt.set_text("")
    return (ln_α,ln_β,ln_gate,ln_va,ln_vb,ln_vc,ln_sec,ln_cur,
            ref_ln,us_ln,usp1_ln,ref_dot,vec_dot,m_txt)

def update(i):
    seg   = i // PTS_WIN
    start = seg * PTS_WIN
    x     = t[start:i] - t[start]

    ln_α.set_data(x, vα[start:i]); ln_β.set_data(x, vβ[start:i])
    ln_gate.set_data(x, Sa[start:i])
    ln_va.set_data(x, Va[start:i]); ln_vb.set_data(x, Vb[start:i]); ln_vc.set_data(x, Vc[start:i])
    ln_sec.set_data(x, sector[start:i] + 1); ln_cur.set_data(x, iL[start:i])

    s = sector[i]
    ref_ln.set_data([0, vα[i]], [0, vβ[i]]); ref_dot.set_data([vα[i]], [vβ[i]])
    us_ln .set_data([0, α_tip[s]], [0, β_tip[s]])
    usp1_ln.set_data([0, α_tip[(s+1)%6]], [0, β_tip[(s+1)%6]])
    vec_dot.set_data([α_act[i]], [β_act[i]])
    m_txt.set_text(f"m = {M_LIST[seg]:.2f}")
    return (ln_α,ln_β,ln_gate,ln_va,ln_vb,ln_vc,ln_sec,ln_cur,
            ref_ln,us_ln,usp1_ln,ref_dot,vec_dot,m_txt)

ani = animation.FuncAnimation(
    fig, update, frames=frames, init_func=init,
    blit=True, interval=1000/FPS, repeat=False)

print("Exporting GIF …"); ani.save(GIF_FILE, writer=PillowWriter(fps=FPS))
print("Exporting MP4 …"); ani.save(MP4_FILE, writer=FFMpegWriter(fps=FPS))
print("✓  Done – files:", GIF_FILE, "and", MP4_FILE)
