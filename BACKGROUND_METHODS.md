# Background Subtraction Methods

This document describes the background subtraction methods available in `gs_analysis` for gamma spectroscopy peak analysis.

## Overview

Background subtraction is a critical step in gamma spectroscopy analysis to accurately determine the net counts in a peak region. The `calc_bg()` function supports multiple methods for estimating the background under a peak, each with different characteristics suitable for various scenarios.

## Available Methods

### Method 1: Trapezoid (Maestro)

**Usage:** `calc_bg(counts, c1, c2, m=1)` (default)

**Description:**
The trapezoid method is based on the Maestro software's approach. It uses up to two channels before the peak start (`c1`) and up to two channels after the peak end (`c2`) to estimate the background. The background is calculated as:

```
bg = (low_sum + high_sum) * ((c2 - c1 + 1) / 6)
```

**Characteristics:**
- Uses multiple channels for better statistics
- Weights the background regions equally
- Well-tested and widely used in gamma spectroscopy
- Good for peaks with relatively flat backgrounds

**Best for:**
- Standard peak analysis
- Spectra with flat or slowly varying backgrounds
- When compatibility with Maestro software is desired

### Method 2: Linear Interpolation

**Usage:** `calc_bg(counts, c1, c2, m=2)`

**Description:**
This method averages 2 channels on each side of the peak region and performs a linear interpolation between these averages. The background under the peak is calculated as the trapezoidal area under the interpolated line.

```
bg = (low_avg + high_avg) * width / 2
```

**Characteristics:**
- Simple and intuitive approach
- Accounts for linearly varying backgrounds
- Symmetric treatment of both sides
- Good statistical properties with averaging

**Best for:**
- Peaks on sloping backgrounds
- When background varies linearly across the peak region
- Simple, straightforward analysis

### Method 3: Step Function

**Usage:** `calc_bg(counts, c1, c2, m=3)`

**Description:**
The step function method calculates the average of the background regions on both sides of the peak and uses this constant value as the background level under the entire peak region.

```
avg_bg = (low_avg + high_avg) / 2
bg = avg_bg * width
```

**Characteristics:**
- Assumes constant background level
- Simple averaging approach
- Robust to local fluctuations
- Symmetric treatment of edges

**Best for:**
- Peaks on flat backgrounds
- When you want a conservative background estimate
- Low-statistics spectra where averaging helps

### Method 4: Sliding Window Average

**Usage:** `calc_bg(counts, c1, c2, m=4)`

**Description:**
This method uses a moving average window (default 5 channels) in the regions adjacent to the peak to estimate the background. It then performs linear interpolation between these averaged regions. This approach is more robust to local variations.

```
bg = (low_window_avg + high_window_avg) * width / 2
```

**Characteristics:**
- Uses larger windows for better noise rejection
- More robust to statistical fluctuations
- Smooths out local variations
- Good for noisy spectra

**Best for:**
- Noisy or low-statistics spectra
- When adjacent channels have high variability
- Complex background shapes near peaks

## Usage Examples

### Basic Usage

```python
import numpy as np
import gs_analysis as gs

# Load or create your spectrum
counts = np.array([...])  # Your spectrum data

# Define peak region
c1, c2 = 100, 120  # Start and end channels

# Try different methods
bg_trapezoid = gs.calc_bg(counts, c1, c2, m=1)
bg_linear = gs.calc_bg(counts, c1, c2, m=2)
bg_step = gs.calc_bg(counts, c1, c2, m=3)
bg_sliding = gs.calc_bg(counts, c1, c2, m=4)

# Calculate net counts
net_counts = gs.net_counts(counts, c1, c2, m=2)  # Using linear method
```

### Comparing Methods

```python
# Compare all methods for a peak
methods = {
    1: "Trapezoid",
    2: "Linear", 
    3: "Step",
    4: "Sliding"
}

for method_id, method_name in methods.items():
    bg = gs.calc_bg(counts, c1, c2, m=method_id)
    net = gs.net_counts(counts, c1, c2, m=method_id)
    print(f"{method_name}: Background={bg:.1f}, Net={net:.1f}")
```

### Example Script

A comprehensive example script is available in `examples_background_subtraction.py` that demonstrates:
- Creating synthetic spectra
- Comparing all methods
- Visualizing the differences
- Analyzing multiple peaks

Run it with:
```bash
python examples_background_subtraction.py
```

## Choosing the Right Method

| Scenario | Recommended Method |
|----------|-------------------|
| Standard analysis, flat background | Method 1 (Trapezoid) |
| Sloping background | Method 2 (Linear) |
| Very flat background, low stats | Method 3 (Step) |
| Noisy spectrum, high variability | Method 4 (Sliding) |
| Compatibility with Maestro | Method 1 (Trapezoid) |
| Conservative estimate | Method 3 (Step) |

## Method Comparison

Here's a visual comparison of all four methods applied to the same peak:

![Background Methods Comparison](https://github.com/user-attachments/assets/b65d0704-9c70-44a8-8721-f13f9f19a153)

In this example:
- **Trapezoid** gives the lowest background estimate (3507.0 counts)
- **Linear and Step** give identical results for symmetric peaks (5010.0 counts)
- **Sliding Window** provides an intermediate estimate (4126.0 counts)

## Technical Details

### Edge Handling

All methods gracefully handle edge cases:
- Peaks at the start of the spectrum (c1 = 0)
- Peaks at the end of the spectrum (c2 = len(counts))
- Single-channel peaks
- Adjacent peaks

The methods automatically adjust their window sizes to stay within spectrum boundaries.

### Statistical Considerations

1. **Averaging reduces variance:** Methods 2-4 use averaging which reduces statistical uncertainty
2. **Window size trade-off:** Larger windows (Method 4) provide better statistics but may include unwanted features
3. **Symmetry:** All methods treat left and right sides symmetrically for unbiased estimates

### Performance

All methods are computationally efficient:
- O(1) complexity for fixed window sizes
- Vectorized NumPy operations where possible
- Suitable for batch processing of many peaks

## References

- Maestro Software Documentation (ORTEC)
- Knoll, G. F. "Radiation Detection and Measurement" - Chapter on Gamma Spectroscopy
- Gilmore, G. "Practical Gamma-ray Spectrometry" - Background subtraction techniques

## See Also

- `calc_bg()` - Main background calculation function
- `net_counts()` - Calculate net counts with background subtraction
- `estimate_background_trapezoid()` - Direct trapezoid method
- `estimate_background_linear()` - Direct linear method
- `estimate_background_step()` - Direct step method
- `estimate_background_sliding_average()` - Direct sliding window method
