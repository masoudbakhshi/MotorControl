# Complex vector synchronous frame PI current controller vs Standard synchronous frame PI current Controller Root Locus Analysis and frequency response.

## Overview

This project provides an animated side-by-side comparison of **Complex vector synchronous frame PI** and **Standard synchronous frame** current controllers for an RL plant in the stationary α-β frame. The visualization clearly demonstrates the fundamental differences between these two control approaches, particularly their speed-dependent vs speed-independent behavior.

## Key Features

### Educational Value
- **Clear visualization** of control theory concepts
- **Side-by-side comparison** makes differences obvious
- **Professional quality** suitable for academic presentations
- **High-resolution output** (1920x1080 Full HD)

### Technical Details
- **System**: RL plant (R = 1.1 Ω, L = 3.7 mH)
- **Bandwidth**: 200 Hz current loop
- **Controllers**: Complex-PI vs Standard-PI with symmetrical-optimum tuning
- **Frame**: Stationary α-β frame analysis

### Animation Features
- **Smooth animation** with 101 frames (0-200 Hz sweep)
- **Live frequency display** showing current electrical frequency
- **titles** with system parameters
- **Author attribution** for proper credit

## Files Generated

- `cv_vs_pi_root_locus.gif` - Animated GIF (12.5 fps)
- `cv_vs_pi_root_locus.mp4` - High-quality MP4 (15 fps, 1920x1080)

## Why This Matters

Understanding the difference between Complex-PI and Standard-PI controllers is crucial for:
- **Motor drive applications** where speed independence is critical
- **Research and education** in advanced control theory
- **Industrial applications** where controller selection impacts system performance

## Technical Background

### Complex-PI Controller
The complex-PI controller provides **speed-independent** behavior by incorporating the electrical frequency in its structure:
```
C(s) = Kp + (Ki + jωe*Kp)/s
```

### Standard-PI Controller
The standard-PI controller shows **speed-dependent** behavior:
```
C(s) = Kp + Ki/s
```

The root locus animation clearly shows how the zero is added by the controller approximately on top of the plant pole, even if the plant pole is complex. In comparison, the standard PI poles shift with the electrical frequency.

## Requirements

- Python 3.10+
- NumPy
- Matplotlib
- Pillow (for GIF export)
- FFmpeg (for MP4 export)

## Usage

Simply run the script:
```bash
python root_locus_frequency_response.py
```

The script will generate both GIF and MP4 files with the animation.

## Author

**Masoud Bakhshi**

This project demonstrates advanced control systems concepts through clear, professional visualizations. Perfect for educational use, research presentations, and understanding the practical differences between complex vector and conventional PI controllers.

## License

This project is provided for educational and research purposes. Feel free to use and modify for academic work, but please maintain proper attribution.

---
