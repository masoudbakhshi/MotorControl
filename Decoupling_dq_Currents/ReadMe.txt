# PMSM d-q Current Control: Cross-Coupling and Decoupling Demonstration

**Author:** Masoud Bakhshi  
**Date:** July 2025

## Overview

This project provides a physically accurate, educational simulation of d-q axis current control in a Permanent Magnet Synchronous Machine (PMSM). The aim is to visualize the effects of cross-coupling between the d- and q-axis current loops, and to demonstrate how feedforward decoupling terms can restore near-independent control.

The simulation is implemented in Python and produces animated visualizations (MP4 and GIF) of the current step responses, making it ideal for teaching, research, or professional reference.

---

## Aim

- **Visualize** the cross-coupling that naturally occurs in PMSM current loops due to the machine’s physical properties.
- **Demonstrate** how adding feedforward (decoupling) terms to the control law can nearly eliminate this coupling, allowing for independent d- and q-axis current control.
- **Compare** the system’s response with and without decoupling, using realistic machine and controller parameters.

---

## What the Script Does

1. **Models the PMSM** using the standard d-q axis equations, including stator resistance, d- and q-axis inductances, and permanent magnet flux linkage.
2. **Implements PI current controllers** for both axes, with gains tuned for a practical closed-loop bandwidth.
3. **Simulates two cases:**
   - **Coupled:** Standard PMSM model, showing natural cross-coupling.
   - **Decoupled:** Adds feedforward compensation to cancel cross-coupling terms.
4. **Applies a step input** to the d-axis current reference (from 0 to 10 A), with the q-axis reference held at zero.
5. **Animates the results:**  
   - Top plot: d-axis current response (coupled, decoupled, and reference).
   - Bottom plot: q-axis cross-response (coupled, decoupled, and reference).
   - Both plots include a badge showing the maximum absolute deviation from the reference.
6. **Exports the animation** as both MP4 and GIF at 1280×720 resolution.

---

## Machine and Controller Specifications

- **Stator resistance (Rs):** 0.5 Ω
- **d-axis inductance (Ld):** 0.005 H
- **q-axis inductance (Lq):** 0.005 H
- **Permanent magnet flux linkage (λf):** 0.15 Vs
- **Electrical speed (ω):** 1000 rad/s

**PI Controller Gains:**
- **Proportional gain (Kp):** 2.5
- **Integral gain (Ki):** 250

**Simulation:**
- **Step in d-axis current reference:** 0 → 10 A
- **q-axis current reference:** 0 A
- **Simulation time:** 0.05 s
- **Time step:** 10 μs

---

## How to Use

1. Run `Decoupling_dq_Currents.py` in your Python environment.
2. The script will generate `C1_Decoupling_dq_Currents.mp4` and `C1_Decoupling_dq_Currents.gif` in the same directory.
3. Open the animation to observe the effect of cross-coupling and the benefit of decoupling in PMSM current control.

---

## Educational Value

- **For students:** See the real impact of cross-coupling and the power of decoupling in vector-controlled drives.
- **For engineers:** Validate control strategies and visualize transient performance with realistic parameters.
- **For educators:** Use the animation in lectures or presentations to illustrate key concepts in modern motor control.

---

## License

This project is provided for educational and research purposes. Please credit the author if you use or adapt the code.
