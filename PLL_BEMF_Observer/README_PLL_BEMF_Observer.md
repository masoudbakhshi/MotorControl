# PLL-Based Back-EMF Observer at Low Speed

**Author:** Masoud Bakhshi  
**Date:** July 2025

---

## Overview
This project demonstrates how a phase-locked loop (PLL) observer tracks the electrical angle of a permanent magnet synchronous machine (PMSM) by following the phase of the back-EMF vector. The simulation focuses on low-speed operation, where the back-EMF amplitude fades as the machine slows down. 
The animation clearly shows how the PLL’s phase error grows and lock is lost as the signal weakens, highlighting the need for advanced sensorless control techniques at low speed.

## Why This Matters
In sensorless motor control, accurate estimation of rotor position is critical. At low speeds, the back-EMF signal becomes very small, making it difficult for traditional PLL-based observers to maintain lock. 
This simulation provides a visual, intuitive understanding of why signal-injection or alternative methods are often required for robust low-speed operation.

## How It Works
- **Speed Ramp:** The simulated PMSM ramps down from 100 rpm to 0 over three seconds, causing the back-EMF amplitude to decrease.
- **PLL Observer:** A PI-based PLL attempts to track the phase of the back-EMF vector throughout the speed ramp.
- **Visualization:** The animation includes:
  - A waveform panel showing the fading back-EMF.
  - A strip chart of the PLL phase error, with excursions above 20° highlighted in red.
  - A lock indicator that turns red when the PLL loses lock.
  - A speed panel for reference.

## Output and Interpretation
- **Back-EMF Panel:** Shows how the signal fades as speed drops.
- **Phase Error Chart:** Watch for the phase error growing as the signal weakens. Red dots indicate moments when the error exceeds ±20°.
- **Lock Indicator:** The LED turns red when the PLL is no longer able to track the phase accurately (error > 10°).
- **Speed Panel:** For reference, shows the speed ramp from 100 rpm to 0.

This animation is a practical tool for engineers, students, and researchers to understand the limitations of sensorless PLL observers at low speed, and to motivate the use of signal-injection or other advanced techniques.

## Attribution
Developed by Masoud Bakhshi, July 2025. For educational and research use. 