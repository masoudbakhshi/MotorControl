Field-Oriented Control (FOC) Closed-Loop Python Simulation with Load Torque Disturbance Observer
====================================================================
Author: Masoud Bakhshi
**1. Aim of the Simulation**
The main goal of this simulation is to show, in a clear and visual way, how Field-Oriented Control (FOC) works for a Permanent Magnet Synchronous Machine (PMSM). The simulation demonstrates how the control system responds to changes in speed and load, and how an observer can estimate the load torque in real time. This helps engineers, students, and researchers understand the dynamic behavior of FOC systems.
**2. Why is this Animation Necessary?**
- FOC is a widely used method for controlling modern electric motors, especially in electric vehicles and industrial drives.
- The control process involves many steps and mathematical transformations, which can be hard to understand from equations alone.
- By animating the signals and showing the system’s response to changes, users can see how each part of the control system works and interacts.
- The animation also shows the effect of using a load torque observer, which is an advanced technique for improving performance.

**3. Implementation Method**

- The simulation is written in Python, using the libraries NumPy and Matplotlib.
- The system is modeled in discrete time, with a sampling rate of 1000 Hz.
- The animation is created using Matplotlib’s animation tools and exported as both GIF and MP4 files.
- All signals (speed, currents, voltages, angles, etc.) are calculated step by step in the code, and then plotted frame by frame.
- The observer is implemented as a simple disturbance estimator for the load torque.
**4. Machine Specifications**

The simulated PMSM (Permanent Magnet Synchronous Motor) has the following parameters:
- d-axis inductance (Ld): 0.001 H
- q-axis inductance (Lq): 0.001 H
- Permanent magnet flux (ψpm): 0.1 Wb
- Stator resistance (Rs): 0.01 Ω
- Number of pole pairs (P): 4
- Rotor inertia (J): 0.001 kg·m²
- Friction coefficient (B): 0.0001 N·m·s

**5. Mathematics Used **
- **Speed PI Controller:**
  - Calculates the difference between the reference speed and the actual speed.
  - Uses proportional and integral gains to create a torque command.
  - Formula:  T_ref = Kp_speed * (ω_ref - ω) + Ki_speed * ∫(ω_ref - ω)dt

- **Torque to Current Conversion:**
  - For a surface PMSM, the torque command is converted to a q-axis current reference.
  - Formula:  Iq_ref = T_ref / (1.5 * P * ψpm)

- **Current PI Controllers (with Decoupling):**
  - Separate PI controllers for d-axis and q-axis currents.
  - Decoupling terms compensate for cross-coupling between axes.
  - Formulas:
    - Vd = PI_d(Id_ref - Id) - ω * Lq * Iq
    - Vq = PI_q(Iq_ref - Iq) + ω * (Ld * Id + ψpm)

- **Park and Clarke Transformations:**
  - Convert between three-phase (abc), stationary (αβ), and rotating (dq) reference frames.
  - Used to simplify control and analysis.

- **SVPWM (Space Vector PWM):**
  - Generates three-phase voltage signals for the inverter based on the αβ voltages.

- **PMSM Model:**
  - Electrical dynamics: Calculates how currents change based on applied voltages and motor parameters.
  - Mechanical dynamics: Calculates how speed changes based on torque and load.

- **Load Torque Disturbance Observer:**
  - Estimates the load torque using the difference between the produced torque and the observed acceleration.
  - Formula:  T_load_est = T_em - J * dω/dt - B * ω

- **Resolver Signals:**
  - The rotor electrical angle (θe) is tracked.
  - sin(θe) and cos(θe) are shown as they would be in a real resolver sensor.

---

**Summary**

This simulation provides a step-by-step, visual explanation of how FOC works for a PMSM, including advanced features like load torque estimation. It is designed to be accessible to users with a moderate technical background, and all key signals and transformations are shown in the plots. 
