
"""
Side-by-side animation comparing:
    • Complex-vector synchronous frame PI controller (left column)
    • Conventional (standard) synchronous frame PI controller (right column)
for an RL plant in the stationary α-β frame.

Author: Masoud Bakhshi
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from PIL import Image  # pillow writer
import math

# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── system constants ──────────────────────────────────────────────
    R = 1.1
    L = 3.7e-3
    f_bw = 200                       # Hz
    ωc   = 2 * math.pi * f_bw        # rad/s
    Kp   = L * ωc                    # ≈ 4.65
    Ki   = R * ωc                    # ≈ 1.38e3

    print("\nSystem parameters")
    print(f"  R  = {R} Ω")
    print(f"  L  = {L*1e3:.1f} mH")
    print(f"  f_bw = {f_bw} Hz  → ωc = {ωc:.1f} rad/s")
    print(f"  Kp = {Kp:.3f},  Ki = {Ki:.0f}\n")

    # ── figure layout (3 rows × 2 columns) ───────────────────────────
    fig = plt.figure(figsize=(24, 12))  
    gs  = gridspec.GridSpec(
        3, 2,
        width_ratios=[1, 1],
        height_ratios=[2, 1, 1],
        wspace=0.15, hspace=0.30,
        left=0.04, right=0.96, top=0.88, bottom=0.08
    )

    ax_rl_cpx  = fig.add_subplot(gs[0, 0])           # complex-PI root locus
    ax_mag_cpx = fig.add_subplot(gs[1, 0])
    ax_ph_cpx  = fig.add_subplot(gs[2, 0], sharex=ax_mag_cpx)

    ax_rl_std  = fig.add_subplot(gs[0, 1])           # standard PI root locus
    ax_mag_std = fig.add_subplot(gs[1, 1])
    ax_ph_std  = fig.add_subplot(gs[2, 1], sharex=ax_mag_std)

    # helper function to convert complex numbers from rad/s to Hz
    to_hz = lambda s: (s.real / (2*np.pi), s.imag / (2*np.pi))

    # ── root-locus axes formatting (both columns) ────────────────────
    # Left side: Complex-PI Controller
    ax_rl_cpx.set_xlim(-350, 50)
    ax_rl_cpx.set_ylim(-10, 220)
    ax_rl_cpx.set_xlabel("real (Hz)")
    ax_rl_cpx.set_ylabel("imaginary (Hz)")
    ax_rl_cpx.grid(True, alpha=0.3)
    ax_rl_cpx.set_title("Complex-PI Controller", fontsize=12, fontweight="bold")
    
    # Right side: Standard-PI Controller (different y-axis range)
    ax_rl_std.set_xlim(-350, 50)
    ax_rl_std.set_ylim(-50, 250)  # Changed from (-10, 220) to (-50, 250)
    ax_rl_std.set_xlabel("real (Hz)")
    ax_rl_std.set_ylabel("imaginary (Hz)")
    ax_rl_std.grid(True, alpha=0.3)
    ax_rl_std.set_title("Standard-PI Controller", fontsize=12, fontweight="bold")

    # scatter creators -------------------------------------------------
    def make_scatters(ax):
        # create scatter plots for poles and zeros with different markers
        zero = ax.scatter([], [], marker='o',  s=100, facecolors='none',
                          edgecolors='k', label='Zero')
        intp = ax.scatter([], [], marker='x',  s=100, color='red',
                          label='Integrator pole')
        rlpl = ax.scatter([], [], marker='x',  s=100, color='black',
                          label='RL pole')
        dom  = ax.scatter([], [], marker='*', s=150, color='blue',
                          label='Dominant CL pole')
        sec  = ax.scatter([], [], marker='*', s=150, color='cyan',
                          label='Secondary CL pole')
        ax.legend(title="Markers", loc="lower left")
        return zero, intp, rlpl, dom, sec

    zero_cpx, int_cpx, rl_cpx, dom_cpx, sec_cpx = make_scatters(ax_rl_cpx)
    zero_std, int_std, rl_std, dom_std, sec_std = make_scatters(ax_rl_std)

    # frequency display boxes for both controllers
    freq_txt_cpx = ax_rl_cpx.text(0.02, 0.98, "", transform=ax_rl_cpx.transAxes,
                                  va="top", fontsize=12,
                                  bbox=dict(boxstyle='round',
                                            facecolor='white', alpha=0.8))
    freq_txt_std = ax_rl_std.text(0.02, 0.98, "", transform=ax_rl_std.transAxes,
                                  va="top", fontsize=12,
                                  bbox=dict(boxstyle='round',
                                            facecolor='white', alpha=0.8))

    # Author
    ax_rl_cpx.text(0.97, 0.03, "Masoud Bakhshi", transform=ax_rl_cpx.transAxes,
                   ha="right", va="bottom", fontsize=8)
    ax_rl_std.text(0.97, 0.03, "Masoud Bakhshi", transform=ax_rl_std.transAxes,
                   ha="right", va="bottom", fontsize=8)

    # ── frequency-response (static) ──────────────────────────────────
    f_e_plot = [0, 100, 200]           # Hz
    ω_hz     = np.linspace(-800, 800, 1601)
    ω_rad    = 2 * np.pi * ω_hz
    colours  = ['blue', 'red', 'green']
    styles   = ['-', '--', '-.']

    def T_cpx(jw, ωe):
        # complex-PI controller transfer function in stationary frame
        s = 1j*jw
        num = Kp*(s - 1j*ωe) + Ki
        den = L*(s - 1j*ωe)**2 + (R+Kp)*(s - 1j*ωe) + Ki
        return num / den

    def T_std(jw, ωe):
        # standard-PI controller transfer function in stationary frame
        if abs(jw) < 1e-12:            # DC limit: unity gain
            return 1.0 + 0j
        s = 1j*jw
        C = Kp + Ki/s
        G = 1 / (L*s + R + 1j*ωe*L)
        return (C*G) / (1 + C*G)

    # plot frequency responses for different electrical frequencies
    for col, ls, fe in zip(colours, styles, f_e_plot):
        ωe = 2*np.pi*fe
        mag_c, pha_c = [], []
        mag_s, pha_s = [], []
        for ω in ω_rad:
            Tc = T_cpx(ω, ωe)
            Ts = T_std(ω, ωe)
            mag_c.append(abs(Tc))
            pha_c.append(math.degrees(np.angle(Tc)))
            mag_s.append(abs(Ts))
            pha_s.append(math.degrees(np.angle(Ts)))
        ax_mag_cpx.plot(ω_hz, mag_c, ls, color=col, lw=2,
                        label=f"f_e = {fe} Hz")
        ax_ph_cpx .plot(ω_hz, pha_c, ls, color=col, lw=2)
        ax_mag_std.plot(ω_hz, mag_s, ls, color=col, lw=2,
                        label=f"f_e = {fe} Hz")
        ax_ph_std .plot(ω_hz, pha_s, ls, color=col, lw=2)

    # format the frequency response plots
    for ax_mag, ax_ph in [(ax_mag_cpx, ax_ph_cpx),
                          (ax_mag_std, ax_ph_std)]:
        ax_mag.set_ylabel("|T(jω)|");  ax_mag.set_ylim(0, 1.4)
        ax_mag.set_xlim(-800, 800);    ax_mag.grid(True, alpha=0.3)
        ax_ph.set_ylabel("phase (deg)")
        ax_ph.set_xlabel("frequency (Hz)")
        ax_ph.set_ylim(-90, 90)
        ax_ph.set_yticks([-90, -45, 0, 45, 90])
        ax_ph.grid(True, alpha=0.3)

    ax_mag_cpx.legend(title="Electrical Frequency", loc="upper right")
    ax_mag_std.legend(title="Electrical Frequency", loc="upper right")

    # ── animation settings ───────────────────────────────────────────
    f_e_values = np.linspace(0, 200, 101)  # 0 … 200 Hz (2-Hz steps)

    def animate(frame: int):
        fe  = f_e_values[frame]
        ωe  = 2*np.pi*fe

        # ─ complex-PI (dq-frame to αβ frame) ───────────────────────────────
        s_int    = 0
        s_rl     = -R/L - 1j*ωe
        s_zero   = -Ki/Kp - 1j*ωe
        a, b, c  = L, (R+Kp), Ki
        disc     = b*b - 4*a*c
        root_fast = (-b - math.sqrt(disc)) / (2*a)
        root_slow = (-b + math.sqrt(disc)) / (2*a)

        # map to αβ frame
        zero_cpx.set_offsets(np.column_stack(to_hz(s_zero + 1j*ωe)))
        int_cpx .set_offsets(np.column_stack(to_hz(s_int  + 1j*ωe)))
        rl_cpx  .set_offsets(np.column_stack(to_hz(s_rl   + 1j*ωe)))
        dom_cpx .set_offsets(np.column_stack(to_hz(root_fast + 1j*ωe)))
        sec_cpx .set_offsets(np.column_stack(to_hz(root_slow + 1j*ωe)))

        # title 
        ax_rl_cpx.set_title(
            f"Complex vector root locus (α-β frame) an RL load with a complex vector synchronous frame PI current controller\n\n"
            f"RL plant (R = 1.1 Ω, L = 3.7 mH) "
            f"(ωc = 2π×200 Hz);  Fe = {fe:.0f} Hz",
            pad=20, fontsize=10
        )
        
        freq_txt_cpx.set_text(f"Fe = {fe:.1f} Hz")

        # ─ standard-PI (dq-frame to αβ frame) ───────────────────────────────
        s_zero_std = -Ki / Kp                 # no −jωe term
        s_int_std  = 0
        s_rl_std   = -R/L - 1j*ωe

        # characteristic: L s² + (R+Kp+jωeL)s + Ki = 0
        b_std  = R + Kp + 1j*ωe*L
        disc_s = b_std**2 - 4*L*Ki
        r1, r2 = (-b_std + np.sqrt(disc_s)) / (2*L), (-b_std - np.sqrt(disc_s)) / (2*L)
        root_fast_std, root_slow_std = (r1, r2) if r1.real < r2.real else (r2, r1)

        # map to αβ (add +jωe)
        zero_std.set_offsets(np.column_stack(to_hz(s_zero_std + 1j*ωe)))  # ← fixed
        int_std .set_offsets(np.column_stack(to_hz(s_int_std  + 1j*ωe)))
        rl_std  .set_offsets(np.column_stack(to_hz(s_rl_std   + 1j*ωe)))
        dom_std .set_offsets(np.column_stack(to_hz(root_fast_std + 1j*ωe)))
        sec_std .set_offsets(np.column_stack(to_hz(root_slow_std + 1j*ωe)))

        # title 
        ax_rl_std.set_title(
            f"Root locus (α-β frame) an RL load with a synchronous frame PI current controller\n\n"
            f"RL plant (R = 1.1 Ω, L = 3.7 mH) "
            f"(ωc = 2π×200 Hz);  Fe = {fe:.0f} Hz",
            pad=20, fontsize=10
        )
        
        freq_txt_std.set_text(f"Fe = {fe:.1f} Hz")

        return (zero_cpx, int_cpx, rl_cpx, dom_cpx, sec_cpx, freq_txt_cpx,
                ax_rl_cpx.title,  
                zero_std, int_std, rl_std, dom_std, sec_std, freq_txt_std,
                ax_rl_std.title)  

    anim = animation.FuncAnimation(fig, animate, frames=len(f_e_values),
                                   interval=80, blit=False, repeat=True)

    anim.save("cv_vs_pi_root_locus.gif", writer="pillow", fps=12.5)
    print("GIF saved as cv_vs_pi_root_locus.gif")

    # Save high-quality MP4 (slower speed, higher resolution)
    try:
        anim.save("cv_vs_pi_root_locus.mp4", writer="ffmpeg", fps=15,  
                  extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-vf', 'scale=1920:1080'])
        print("MP4 saved as cv_vs_pi_root_locus.mp4 (high quality, slower speed, 1920x1080)")
    except Exception:
        anim.save("cv_vs_pi_root_locus.mp4", writer="ffmpeg", fps=15)  
        print("MP4 saved as cv_vs_pi_root_locus.mp4 (slower speed)")

    plt.show()


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
