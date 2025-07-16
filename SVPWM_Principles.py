import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, patches, gridspec, lines, cm

# ───────── USER SETTINGS ─────────
Vdc        = 1.0                # DC link voltage normalized to 1.0
m_list     = [0.50, 0.80, 0.90, 1.00, 1.15]  # modulation index values
f_e        = 50.0               # electrical frequency in Hz
T_rev      = 1.0 / f_e          # one electrical period in seconds
frames_rev = 180                # frames per revolution
fps        = 25                 # animation frames per second
gif_name   = "svpwm_principle.gif"
mp4_name   = "svpwm_principle.mp4"
dpi_out    = 120
# ────────────────────────────────────

# calculate switching-period equivalent for averaging
Ts = T_rev / frames_rev
# angles for each frame
theta_rev = np.linspace(0, 2*np.pi, frames_rev, endpoint=False)

# ───── SVPWM HELPER FUNCTIONS ─────
def sector_idx(th):
    # return sector number (0 to 5) for angle th
    return int(th // (np.pi/3)) % 6

def duty_linear(m, th):
    # compute t1, t2, t0 in linear region
    th_s = th % (np.pi/3)
    t1 = m * Ts * np.sin(np.pi/3 - th_s) / np.sin(np.pi/3)
    t2 = m * Ts * np.sin(th_s)           / np.sin(np.pi/3)
    t0 = Ts - t1 - t2
    return t1, t2, t0

def duty_overmod(m, th):
    # handle over-modulation I and II
    t1, t2, t0 = duty_linear(m, th)
    if t0 >= 0:
        return t1, t2, t0       # linear
    if t1 + t2 <= Ts:
        return t1, t2, 0.0      # over-mod I
    # over-mod II (one vector only)
    return (Ts,0.0,0.0) if t1>t2 else (0.0,Ts,0.0)

def phase_duties(sec, t1, t2, t0):
    # get duty ratios [d_a, d_b, d_c] for each segment
    seq = [
        [t1+t2+t0/2, t2+t0/2,       t0/2      ],
        [t1+t0/2,    t1+t2+t0/2,    t0/2      ],
        [t0/2,       t1+t2+t0/2,    t2+t0/2   ],
        [t0/2,       t1+t0/2,       t1+t2+t0/2],
        [t2+t0/2,    t0/2,          t1+t2+t0/2],
        [t1+t2+t0/2, t0/2,          t1+t0/2   ],
    ]
    return [d/Ts for d in seq[sec]]

# bit patterns for zero vectors
bit_zero_lo = (0,0,0)  # V0
bit_zero_hi = (1,1,1)  # V7

# bit patterns for active vectors Vk
bits_active = [
    (1,0,0), (1,1,0), (0,1,0),
    (0,1,1), (0,0,1), (1,0,1),
]

def seg_bits_for_sector(sec):
    # return the 7 switch states for sector sec
    V_k  = bits_active[sec]
    V_k1 = bits_active[(sec+1)%6]
    return [
        bit_zero_lo,    # V0
        V_k,            # Vk
        V_k1,           # Vk+1
        bit_zero_hi,    # V7
        V_k1,           # Vk+1
        V_k,            # Vk
        bit_zero_lo     # V0
    ]

# coordinates for each active vector in alpha-beta plane
V_act = (2/3)*Vdc * np.exp(1j*np.deg2rad(np.arange(0,360,60)))
sector_bounds = np.linspace(0, 2*np.pi, 7)

# colors for sector shading
cmap = plt.get_cmap("Pastel1")
sector_cols = [cmap(i) for i in range(6)]

# ───────── FIGURE SETUP ─────────
plt.rcParams.update({
    "font.family":"sans-serif",
    "mathtext.fontset":"cm"
})
fig = plt.figure(figsize=(8,12))
# add extra top margin and space between panels
fig.subplots_adjust(top=0.88, hspace=0.4)
gs = gridspec.GridSpec(3,1)

# main title and author
fig.suptitle("SVPWM Principle: Zero Vectors & 7-Segment Sequence",
             fontsize=18, y=0.95)
fig.text(0.5,0.98,"Masoud Bakhshi", ha='center',
         fontsize=14, weight='bold')

# ─── Panel 1: Alpha-Beta Plane ───
ax1 = fig.add_subplot(gs[0], aspect='equal')
R = 0.8*Vdc
ax1.set_xlim(-R,R); ax1.set_ylim(-R,R)
ax1.set_title("αβ Plane with Zero Vectors")
ax1.set_xlabel("$i_\\alpha$"); ax1.set_ylabel("$i_\\beta$")
ax1.grid(alpha=0.3, ls=':')

# draw shaded sectors and labels
for i in range(6):
    ax1.add_patch(patches.Wedge(
        (0,0), R,
        np.degrees(sector_bounds[i]),
        np.degrees(sector_bounds[i+1]),
        facecolor=sector_cols[i], alpha=0.15, edgecolor='none'
    ))
    mid = 0.5*(sector_bounds[i]+sector_bounds[i+1])
    ax1.text(0.6*R*np.cos(mid), 0.6*R*np.sin(mid),
             f"S{i+1}", ha='center', va='center', fontsize=9)

# draw all active vectors
for k, V in enumerate(V_act):
    ax1.arrow(0,0, V.real, V.imag,
              color='grey', lw=1.2, head_width=0.03,
              length_includes_head=True)
    ax1.text(1.05*V.real, 1.05*V.imag, f"$V_{k+1}$",
             color='grey', ha='center', va='center', fontsize=9)

# legend for reference, applied, and zero vectors
ax1.legend(handles=[
    lines.Line2D([],[],color='k',lw=2, label='$V^*$'),
    lines.Line2D([],[],color='orange',lw=2, label='Applied $V_k$'),
    lines.Line2D([],[],color='grey',lw=1, label='$V_0/V_7$')
], loc='upper right', fontsize=9)

# dynamic arrows
q_ref = ax1.quiver([0],[0],[0],[0],
                   angles='xy', scale_units='xy', scale=1,
                   color='k', width=0.01, zorder=3)
q_act = ax1.quiver([0],[0],[0],[0],
                   angles='xy', scale_units='xy', scale=1,
                   color='orange', width=0.01, zorder=3)
txt_m = ax1.text(0, 0.85*R, "", ha='center', va='center',
                 fontsize=11, color='navy',
                 bbox=dict(facecolor='white', alpha=0.8))

# ─── Panel 2: Average Voltages ───
ax2 = fig.add_subplot(gs[1])
ax2.set_title("Average Phase-Neutral Voltage per $T_s$ (ms)")
ax2.set_xlim(0, T_rev*1e3); ax2.set_ylim(-0.6,0.6)
ax2.set_xlabel("Time [ms]"); ax2.set_ylabel("$v_{xN}$ (norm.)")
ax2.grid(alpha=0.3, ls=':')

t_rev_ms = np.linspace(0, T_rev*1e3, frames_rev)
van_line, = ax2.plot([],[], color='C0', label='$v_{an}$')
vbn_line, = ax2.plot([],[], color='C1', label='$v_{bn}$')
vcn_line, = ax2.plot([],[], color='C2', label='$v_{cn}$')
ax2.legend(loc='upper right')

van_hist = np.zeros(frames_rev)
vbn_hist = np.zeros(frames_rev)
vcn_hist = np.zeros(frames_rev)

# ─── Panel 3: 3-Leg Schematic ───
ax3 = fig.add_subplot(gs[2])
ax3.axis('off')
ax3.set_title("3-Leg Inverter – Switch States")

# draw DC rails
ax3.plot([0,1],[1,1], color='k', lw=2)
ax3.plot([0,1],[0,0], color='k', lw=2)
ax3.text(1.02,1.0,"+V_dc", va='center')
ax3.text(1.02,0.0,"-V_dc", va='center')

# draw switches and phase labels
devices = []
xs = [0.2, 0.5, 0.8]
for xi, ph in zip(xs, ["a","b","c"]):
    ax3.plot([xi, xi], [1.0, 0.90], color='k', lw=1.5)
    up = patches.Rectangle((xi-0.05,0.65),0.10,0.25,
                           ec='k', fc='black', lw=1.2)
    ax3.add_patch(up)
    ax3.plot([xi, xi], [0.65, 0.35], color='k', lw=1.5)
    lo = patches.Rectangle((xi-0.05,0.10),0.10,0.25,
                           ec='k', fc='black', lw=1.2)
    ax3.add_patch(lo)
    ax3.plot([xi, xi], [0.35, 0.00], color='k', lw=1.5)
    ax3.text(xi, 0.50, f"$V_{{{ph}}}$",
             ha='center', va='center', fontsize=12)
    devices += [up, lo]

# ───────── ANIMATION ─────────
total_fr = frames_rev * len(m_list)

def init():
    van_line.set_data([],[])
    vbn_line.set_data([],[])
    vcn_line.set_data([],[])
    return (q_ref, q_act, txt_m,
            van_line, vbn_line, vcn_line,
            *devices)

def animate(fr):
    # find which m and frame we are at
    step = fr // frames_rev
    idx  = fr %  frames_rev
    m    = m_list[step]
    θ    = theta_rev[idx]
    sec  = sector_idx(θ)

    # find segment 0..6 for this frame
    seg = (idx * 7) // frames_rev

    # update reference arrow
    Vref = m*R * np.exp(1j*θ)
    q_ref.set_UVC([Vref.real], [Vref.imag])

    # get switch bits for this sector and segment
    seq = seg_bits_for_sector(sec)
    bits = seq[seg]

    # update applied vector arrow
    if bits == bit_zero_lo or bits == bit_zero_hi:
        q_act.set_UVC([0], [0])  # zero vector
    else:
        k = bits_active.index(bits)
        Vapp = V_act[k]
        q_act.set_UVC([Vapp.real], [Vapp.imag])

    txt_m.set_text(f"m = {m:.2f}   |   Sector {sec+1}")

    # compute average voltages
    t1, t2, t0    = duty_overmod(m, θ)
    d_a, d_b, d_c = phase_duties(sec, t1, t2, t0)
    van = (2*d_a - 1)*Vdc/2
    vbn = (2*d_b - 1)*Vdc/2
    vcn = (2*d_c - 1)*Vdc/2

    if idx == 0:
        van_hist[:] = 0
        vbn_hist[:] = 0
        vcn_hist[:] = 0

    van_hist[idx] = van
    vbn_hist[idx] = vbn
    vcn_hist[idx] = vcn

    van_line.set_data(t_rev_ms[:idx+1], van_hist[:idx+1])
    vbn_line.set_data(t_rev_ms[:idx+1], vbn_hist[:idx+1])
    vcn_line.set_data(t_rev_ms[:idx+1], vcn_hist[:idx+1])

    # update switch colors
    SaU, SaL, SbU, SbL, ScU, ScL = devices
    a, b, c = bits
    SaU.set_facecolor('green' if a else 'black')
    SaL.set_facecolor('green' if not a else 'black')
    SbU.set_facecolor('green' if b else 'black')
    SbL.set_facecolor('green' if not b else 'black')
    ScU.set_facecolor('green' if c else 'black')
    ScL.set_facecolor('green' if not c else 'black')

    return (q_ref, q_act, txt_m,
            van_line, vbn_line, vcn_line,
            *devices)

ani = animation.FuncAnimation(
    fig, animate, frames=total_fr,
    init_func=init, blit=False, interval=30
)

# save files
ani.save(gif_name, writer='pillow', fps=fps, dpi=dpi_out)
ani.save(mp4_name, writer='ffmpeg', fps=fps, dpi=dpi_out)
print("✔ Saved", gif_name, "and", mp4_name)
