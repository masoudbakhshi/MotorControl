import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, patches
from matplotlib.lines import Line2D

# --- PARAMETERS ---
omega_elec = 2 * np.pi / 5
frames = 200
t_max = 2 * np.pi / omega_elec
t = np.linspace(0, t_max, frames)

# abc (balanced sine waves)
Ia = np.sin(omega_elec * t)
Ib = np.sin(omega_elec * t - 2 * np.pi / 3)
Ic = np.sin(omega_elec * t - 4 * np.pi / 3)

# Clarke Transform (abc → alpha Beta)
i_alpha = (2/3) * (Ia - 0.5*Ib - 0.5*Ic)
i_beta  = (2/3) * (np.sqrt(3)/2 * (Ib - Ic))

# dq frame angles
theta_fixed = 0 * t
theta_sync  = omega_elec * t

# dq projections for both frames
idq_fixed = np.zeros((2, frames))
idq_sync  = np.zeros((2, frames))
for k in range(frames):
    # stator-fixed
    c, s = np.cos(theta_fixed[k]), np.sin(theta_fixed[k])
    idq_fixed[0, k] =  i_alpha[k]*c + i_beta[k]*s
    idq_fixed[1, k] = -i_alpha[k]*s + i_beta[k]*c
    # synchronous
    c, s = np.cos(theta_sync[k]), np.sin(theta_sync[k])
    idq_sync[0, k] =  i_alpha[k]*c + i_beta[k]*s
    idq_sync[1, k] = -i_alpha[k]*s + i_beta[k]*c

plt.rcParams.update({
    'font.family': "sans-serif",
    'animation.html': 'jshtml',
    'mathtext.fontset': 'cm',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

fig = plt.figure(figsize=(19, 7))
gs = fig.add_gridspec(2, 3, height_ratios=[1,2])
fig.subplots_adjust(top=0.91, left=0.04, right=0.99, hspace=0.28, wspace=0.23)
fig.suptitle("ABC to dq Transformation: Clarke, Stator-Fixed dq, and Synchronous dq Frames", fontsize=18, y=0.97)
fig.text(0.05, 0.95, "Masoud Bakhshi", fontsize=15, weight='bold')

# === abc currents ===
ax0 = fig.add_subplot(gs[0, :])
ax0.set_title("Three-Phase Currents (abc)")
ax0.set_xlabel("Time [s]")
ax0.set_ylabel("Current")
ax0.set_xlim(t[0], t[-1])
ax0.set_ylim(-1.3, 1.3)
line_Ia, = ax0.plot([], [], color='#FFD700', label="$i_a$")
line_Ib, = ax0.plot([], [], color='#287EC7', label="$i_b$")
line_Ic, = ax0.plot([], [], color='#D62728', label="$i_c$")
ax0.legend(loc='upper right', fontsize=11)
ax0.grid(True, linestyle=':', alpha=0.7)

# === ploting Clarke  ===
axC = fig.add_subplot(gs[1, 0])
axC.set_title("Clarke Transformation ($\\alpha\\beta$ Plane)")
axC.set_xlabel("$i_\\alpha$")
axC.set_ylabel("$i_\\beta$")
axC.set_xlim(-1.3, 1.3)
axC.set_ylim(-1.3, 1.3)
axC.set_aspect('equal')
axC.grid(True, linestyle=':', alpha=0.6)
axC.arrow(0, 0, 1, 0, width=0.01, head_width=0.05, color='black', alpha=0.6, length_includes_head=True)
axC.arrow(0, 0, 0, 1, width=0.01, head_width=0.05, color='black', alpha=0.6, length_includes_head=True)
axC.text(1.09, 0, "$\\alpha$", fontsize=12, color='black')
axC.text(0.05, 1.09, "$\\beta$", fontsize=12, color='black')
# Phase axes
phase_angle = [0, 2*np.pi/3, -2*np.pi/3]
phase_colors = ['#FFD700', '#287EC7', '#D62728']
phase_labels = ['a', 'b', 'c']
for angle, col, label in zip(phase_angle, phase_colors, phase_labels):
    axC.arrow(0, 0, np.cos(angle), np.sin(angle), width=0.01, head_width=0.06, color=col, length_includes_head=True, alpha=0.8)
    axC.text(1.08*np.cos(angle), 1.08*np.sin(angle), label, fontsize=13, color=col, weight='bold')
arc_ab = patches.Arc((0,0), 1.1, 1.1, theta1=0, theta2=120, edgecolor='gray', lw=1.3, linestyle='--', zorder=1)
arc_bc = patches.Arc((0,0), 1.1, 1.1, theta1=120, theta2=240, edgecolor='gray', lw=1.3, linestyle='--', zorder=1)
axC.add_patch(arc_ab)
axC.add_patch(arc_bc)
axC.text(0.55, 0.55*np.tan(np.pi/6), "120°", color='gray', fontsize=11, rotation=32)
axC.text(-0.7, 0.35, "120°", color='gray', fontsize=11, rotation=-32)
axC.clarke_vector = None  
pt_clarkeC, = axC.plot([], [], 'ko', markersize=8, alpha=0.7, label="Tip $i_{\\alpha\\beta}$")
axC.legend(handles=[
    Line2D([0], [0], color='magenta', lw=3, label='$i_{\\alpha\\beta}$'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8, label='Tip $i_{\\alpha\\beta}$'),
], loc='upper left', fontsize=9, frameon=True)

# === dq plots ===
axes_dq = []
dq_labels = []
for col, (theta_arr, idq, title) in enumerate([
    (theta_fixed, idq_fixed, "Stator-Fixed dq Frame ($\\theta=0$)"),
    (theta_sync,  idq_sync,  "Synchronous dq Frame ($\\theta=\\omega_e t$)"),
]):
    ax = fig.add_subplot(gs[1, col+1])
    axes_dq.append(ax)
    ax.set_title(title)
    ax.set_xlabel("$i_\\alpha$")
    ax.set_ylabel("$i_\\beta$")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    # alpha Beta axes
    ax.arrow(0, 0, 1, 0, width=0.01, head_width=0.05, color='black', alpha=0.6, length_includes_head=True)
    ax.arrow(0, 0, 0, 1, width=0.01, head_width=0.05, color='black', alpha=0.6, length_includes_head=True)
    ax.text(1.09, 0, "$\\alpha$", fontsize=12, color='black')
    ax.text(0.05, 1.09, "$\\beta$", fontsize=12, color='black')
    # a phase axis (yellow)
    ax.arrow(0, 0, 1, 0, width=0.01, head_width=0.05, color='#FFD700', alpha=0.4, length_includes_head=True)
    ax.text(1.09, -0.13, "a", fontsize=11, color='#FFD700', weight='bold')
    # Legend
    ax.legend(handles=[
        Line2D([0], [0], color='#90ee90', lw=2, label='d/q-axis'),
        Line2D([0], [0], color='purple', lw=2, label='$i_d$'),
        Line2D([0], [0], color='orange', lw=2, label='$i_q$'),
        Line2D([0], [0], color='navy', lw=2, label='$i_{dq}$'),
        Line2D([0], [0], color='magenta', lw=2, label='$i_{\\alpha\\beta}$'),
    ], loc='upper left', fontsize=9, frameon=True)
    # Placeholders for arrows, tip dots, and text labels
    for key in ["d_axis", "q_axis", "id_proj", "iq_proj", "idq_arrow", "vec_clarke"]:
        setattr(ax, key, None)
    ax.pt_park, = ax.plot([], [], 'mo', markersize=8, label="Tip $i_{dq}$")
    ax.pt_clarke, = ax.plot([], [], 'ko', markersize=7, alpha=0.7)
    dq_labels.append((ax.text(0,0,"",fontsize=11,color='#90ee90',weight='bold'),  # d label
                      ax.text(0,0,"",fontsize=11,color='#90ee90',weight='bold'))) # q label

def init():
    line_Ia.set_data([], [])
    line_Ib.set_data([], [])
    line_Ic.set_data([], [])
    if axC.clarke_vector is not None:
        axC.clarke_vector.remove()
        axC.clarke_vector = None
    pt_clarkeC.set_data([], [])
    for ax in axes_dq:
        for key in ["d_axis", "q_axis", "id_proj", "iq_proj", "idq_arrow", "vec_clarke"]:
            if getattr(ax, key):
                getattr(ax, key).remove()
                setattr(ax, key, None)
        ax.pt_park.set_data([], [])
        ax.pt_clarke.set_data([], [])
    for d_label, q_label in dq_labels:
        d_label.set_text("")
        q_label.set_text("")
    return (line_Ia, line_Ib, line_Ic, pt_clarkeC) + tuple(a for ax in axes_dq for a in (ax.pt_park, ax.pt_clarke))

def animate(i):
    # abc plot
    line_Ia.set_data(t[:i+1], Ia[:i+1])
    line_Ib.set_data(t[:i+1], Ib[:i+1])
    line_Ic.set_data(t[:i+1], Ic[:i+1])
    # Clarke plot
    if axC.clarke_vector is not None:
        axC.clarke_vector.remove()
    axC.clarke_vector = axC.arrow(0, 0, i_alpha[i], i_beta[i], color='magenta', width=0.014, head_width=0.08)
    pt_clarkeC.set_data([i_alpha[i]], [i_beta[i]])
    # dq plots
    for idx, (ax, theta_arr, idq) in enumerate(zip(axes_dq, [theta_fixed, theta_sync], [idq_fixed, idq_sync])):
        for key in ["d_axis", "q_axis", "id_proj", "iq_proj", "idq_arrow", "vec_clarke"]:
            if getattr(ax, key):
                getattr(ax, key).remove()
                setattr(ax, key, None)
        theta = theta_arr[i]
        d_x, d_y = np.cos(theta), np.sin(theta)
        q_x, q_y = -np.sin(theta), np.cos(theta)
        # dq axes
        ax.d_axis = ax.arrow(0, 0, d_x, d_y, color='#90ee90', lw=2, length_includes_head=True)
        ax.q_axis = ax.arrow(0, 0, q_x, q_y, color='#90ee90', lw=2, length_includes_head=True)
        dq_labels[idx][0].set_position((1.13*d_x, 1.13*d_y))
        dq_labels[idx][0].set_text("d")
        dq_labels[idx][1].set_position((1.05*q_x, 1.05*q_y))
        dq_labels[idx][1].set_text("q")
        # Clarke alpha Beta vector (magenta)
        ax.vec_clarke = ax.arrow(0, 0, i_alpha[i], i_beta[i], color='magenta', width=0.012, head_width=0.08, length_includes_head=True)
        ax.pt_clarke.set_data([i_alpha[i]], [i_beta[i]])
        # dq projections
        Id = idq[0, i]
        Iq = idq[1, i]
        id_tip_x = Id * d_x
        id_tip_y = Id * d_y
        iq_tip_x = id_tip_x + Iq * q_x
        iq_tip_y = id_tip_y + Iq * q_y
        ax.id_proj = ax.arrow(0, 0, id_tip_x, id_tip_y, color='purple', width=0.010, head_width=0.045)
        ax.iq_proj = ax.arrow(id_tip_x, id_tip_y, Iq * q_x, Iq * q_y, color='orange', width=0.010, head_width=0.045)
        ax.idq_arrow = ax.arrow(0, 0, iq_tip_x, iq_tip_y, color='navy', width=0.013, head_width=0.07)
        ax.pt_park.set_data([iq_tip_x], [iq_tip_y])
    return (line_Ia, line_Ib, line_Ic, pt_clarkeC) + tuple(a for ax in axes_dq for a in (ax.pt_park, ax.pt_clarke))

ani = animation.FuncAnimation(fig, animate, frames=frames, init_func=init, blit=False, interval=35)
ani.save("abc_to_dq_clarke_comparison_masoud.gif", writer='pillow', fps=25, dpi=120)
#ani.save("abc_to_dq_clarke_comparison_masoud.mp4", writer='ffmpeg', fps=25, dpi=120)
plt.show()
