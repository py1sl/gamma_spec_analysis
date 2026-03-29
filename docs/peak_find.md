# Peak Finding Methods

This document describes the peak finding and peak fitting methods available in `gs_analysis` for gamma spectroscopy analysis, including both single-peak and doublet (double-peak) options.

## Overview

Identifying peaks in a gamma spectrum involves two main stages:

1. **Peak detection** — locating channel positions that correspond to photopeak centroids in the smoothed spectrum.
2. **Peak fitting** — fitting an analytical model (Gaussian or double Gaussian) to the region of interest around each detected peak to extract the centroid, width, and area.

The library provides two complementary peak detection algorithms and separate fitting routines for isolated (single) peaks and overlapping pairs (doublets).

---

## Smoothing

Both detection algorithms operate on a smoothed version of the raw counts to reduce the influence of statistical noise. The helper `five_point_smooth()` applies a 5-point weighted moving average. The detection functions call this internally, so you do not normally need to invoke it directly.

---

## Single Peak Detection

### `peak_finder()`

```python
peak_finder(
    counts: npt.NDArray,
    prominence: float,
    wlen: int,
) -> Tuple[npt.NDArray, npt.NDArray]
```

A straightforward peak detector that wraps SciPy's `find_peaks()`. The spectrum is smoothed twice with the 5-point smoother before peak detection is applied.

| Parameter | Description |
|-----------|-------------|
| `counts` | Raw spectrum counts array |
| `prominence` | Minimum peak prominence required for a channel to be classified as a peak. Controls sensitivity — lower values detect more (possibly noisy) peaks; higher values detect only the most prominent features. |
| `wlen` | Window length passed to `find_peaks()`. Limits the region used when computing prominence, useful for spectra with a rising or falling baseline. |

**Returns:** `(smoothed_counts, peaks)`

| Return value | Description |
|--------------|-------------|
| `smoothed_counts` | Doubly-smoothed spectrum array |
| `peaks` | Array of channel indices where peaks were detected |

**Characteristics:**
- Simple and fast
- Directly tunable via `prominence` and `wlen`
- Relies on SciPy's well-tested `find_peaks` implementation
- Good starting point for clean, high-statistics spectra

---

### `mariscotti_peak_finder()`

```python
mariscotti_peak_finder(
    counts: Union[Sequence[float], npt.NDArray],
    threshold: Optional[float] = None,
    smooth_iterations: int = 2,
) -> Tuple[npt.NDArray, npt.NDArray]
```

An implementation of the classical Mariscotti second-difference method. After smoothing, the discrete second difference of the spectrum is computed. Peaks in the original spectrum appear as significant negative excursions in the second difference. A channel is flagged as a peak if its second-difference value falls below the threshold.

| Parameter | Description |
|-----------|-------------|
| `counts` | Raw spectrum counts array |
| `threshold` | Second-difference threshold for peak identification. More negative values require a stronger signal to trigger detection. When `None` (default), the threshold is set automatically to `mean − 1×std` of the negative second differences, providing adaptive noise rejection. |
| `smooth_iterations` | Number of 5-point smoothing passes before computing the second difference. Default is 2. Increasing this suppresses noise further but may broaden the detected peak positions. |

**Returns:** `(smoothed_counts, peaks)`

| Return value | Description |
|--------------|-------------|
| `smoothed_counts` | Smoothed spectrum after the requested number of passes |
| `peaks` | Array of channel indices where peaks were detected |

**Characteristics:**
- Physics-motivated: uses the curvature (second difference) of the spectrum
- Automatic threshold adapts to the noise level of each spectrum
- More robust on low-statistics or noisy spectra
- Reference: M. A. Mariscotti, *Nuclear Instruments and Methods* **50** (1967) 309–320

**Best for:**
- Low-count-rate spectra
- Spectra with varying background or significant statistical noise
- When an automatic, parameter-free detection is preferred

---

## Extracting a Peak Region of Interest

Before fitting, a region of interest (ROI) is extracted around each detected peak centroid using `get_peak_roi()`.

```python
get_peak_roi(
    peak_pos: int,
    counts: npt.NDArray,
    ebins: npt.NDArray,
    offset: int = 10,
) -> Tuple[npt.NDArray, npt.NDArray]
```

Extracts `2 × offset` channels centred on `peak_pos`.

| Parameter | Description |
|-----------|-------------|
| `peak_pos` | Channel index of the peak |
| `counts` | Spectrum counts array |
| `ebins` | Energy bin array |
| `offset` | Half-width of the ROI in channels. Default 10. |

**Returns:** `(roi_counts, roi_ebins)` — the counts and energy values for the extracted window.

---

## Single Peak Fitting

### `fit_peak()`

```python
fit_peak(
    x: npt.NDArray,
    y: npt.NDArray,
) -> npt.NDArray
```

Fits a single Gaussian to the ROI using `scipy.optimize.curve_fit`. Initial parameter estimates are derived from the weighted mean (centroid) and weighted standard deviation (sigma) of the ROI.

The Gaussian model is:

```
f(x) = a · exp( −(x − x₀)² / (2σ²) )
```

| Parameter | Description |
|-----------|-------------|
| `x` | Energy bin values for the ROI |
| `y` | Counts for the ROI |

**Returns:** `[a, x₀, σ]` — amplitude, centroid (energy), and sigma of the fitted Gaussian.

**Use when:** the peak is clearly isolated from neighbouring features.

---

## Doublet (Double Peak) Detection and Fitting

Two peaks that are very close in energy may overlap significantly, making a single-Gaussian fit unreliable. The library handles this case with dedicated doublet identification and fitting routines.

### Identifying Doublets — `identify_doublets()`

```python
identify_doublets(
    peaks: Sequence[int],
    ebins: npt.NDArray,
    max_separation: float = 10.0,
) -> List[Tuple[int, int]]
```

Scans the list of detected peak positions and returns pairs of adjacent peaks whose energy separation is at or below `max_separation`.

| Parameter | Description |
|-----------|-------------|
| `peaks` | Sorted array of peak channel indices (as returned by `peak_finder` or `mariscotti_peak_finder`) |
| `ebins` | Energy bin array mapping channel indices to energy values |
| `max_separation` | Maximum energy separation (in the same units as `ebins`) for two peaks to be considered a doublet. Default 10.0. |

**Returns:** List of `(peak1_index, peak2_index)` channel-index pairs.

---

### Fitting a Doublet — `fit_doublet()`

```python
fit_doublet(
    x: npt.NDArray,
    y: npt.NDArray,
) -> npt.NDArray
```

Fits a sum of two Gaussians (double Gaussian) to the ROI:

```
f(x) = a₁·exp(−(x−x₀₁)²/(2σ₁²)) + a₂·exp(−(x−x₀₂)²/(2σ₂²))
```

Initial estimates for the two centroids are taken from the two largest local maxima in the ROI. If fewer than two local maxima exist, the ROI is split at its midpoint and the maximum of each half is used.

| Parameter | Description |
|-----------|-------------|
| `x` | Energy bin values for the ROI |
| `y` | Counts for the ROI |

**Returns:** `[a₁, x₀₁, σ₁, a₂, x₀₂, σ₂]` — amplitude, centroid, and sigma for each of the two Gaussian components.

**Raises:** `RuntimeError` if the curve fit fails to converge.

**Use when:** `identify_doublets()` has flagged two adjacent peaks as a doublet, or when a visual inspection suggests that a peak has an asymmetric or shouldered shape.

---

## Net Counts

After fitting, the net counts in a peak (after background subtraction) can be retrieved with `peak_counts()`:

```python
peak_counts(
    peaks: Sequence[int],
    index: int,
    smooth_counts: npt.NDArray,
    ebins: npt.NDArray,
) -> Tuple[int, float]
```

`index` is the position within the `peaks` array (e.g. `0` for the first peak), **not** the channel index itself.

**Returns:** `(channel_index, net_counts)`

---

## Usage Examples

### Detecting Peaks with `peak_finder`

```python
import numpy as np
import gs_analysis as gs

# Load spectrum (counts array and energy bin array)
counts = np.array([...])
ebins = np.array([...])

smoothed, peaks = gs.peak_finder(counts, prominence=50, wlen=100)
print(f"Found {len(peaks)} peaks at channels: {peaks}")
```

### Detecting Peaks with `mariscotti_peak_finder`

```python
# Automatic threshold (recommended for noisy spectra)
smoothed, peaks = gs.mariscotti_peak_finder(counts)

# Manual threshold and extra smoothing
smoothed, peaks = gs.mariscotti_peak_finder(
    counts,
    threshold=-200.0,
    smooth_iterations=3,
)
print(f"Found {len(peaks)} peaks at channels: {peaks}")
```

### Fitting a Single Peak

```python
# Extract ROI around the first detected peak
roi_counts, roi_ebins = gs.get_peak_roi(peaks[0], smoothed, ebins, offset=15)

# Fit a Gaussian to the ROI
a, x0, sigma = gs.fit_peak(roi_ebins, roi_counts)
print(f"Peak centroid: {x0:.3f} MeV, sigma: {sigma:.4f} MeV, amplitude: {a:.1f}")
```

### Detecting and Fitting Doublets

```python
# Find pairs of close peaks
doublet_pairs = gs.identify_doublets(peaks, ebins, max_separation=8.0)
print(f"Doublet pairs (channel indices): {doublet_pairs}")

# Fit the first doublet
if doublet_pairs:
    peak1, peak2 = doublet_pairs[0]
    # Use the midpoint between the two peaks as the ROI centre
    roi_centre = (peak1 + peak2) // 2
    roi_counts, roi_ebins = gs.get_peak_roi(roi_centre, smoothed, ebins, offset=20)

    a1, x01, s1, a2, x02, s2 = gs.fit_doublet(roi_ebins, roi_counts)
    print(f"Component 1: centroid={x01:.3f} MeV, sigma={s1:.4f} MeV")
    print(f"Component 2: centroid={x02:.3f} MeV, sigma={s2:.4f} MeV")
```

### Full Workflow Example

```python
import numpy as np
import gs_analysis as gs
import gs_plotting as gsp

# 1. Load data
counts = np.array([...])
ebins = np.array([...])

# 2. Detect peaks
smoothed, peaks = gs.mariscotti_peak_finder(counts)

# 3. Identify doublets
doublets = gs.identify_doublets(peaks, ebins, max_separation=10.0)
doublet_channels = {ch for pair in doublets for ch in pair}

# 4. Fit each peak
for i, peak_ch in enumerate(peaks):
    roi_counts, roi_ebins = gs.get_peak_roi(peak_ch, smoothed, ebins)

    if peak_ch in doublet_channels:
        # Skip individual fit for doublet members; fit the pair together
        continue

    a, x0, sigma = gs.fit_peak(roi_ebins, roi_counts)
    _, net = gs.peak_counts(peaks, i, smoothed, ebins)
    print(f"Peak at {x0:.3f} MeV | net counts: {net:.0f}")

# 5. Plot spectrum with peaks highlighted
gsp.plot_spect_peaks(ebins, counts, peaks)
```

---

## Choosing Between Detection Methods

| Scenario | Recommended Method |
|----------|-------------------|
| Clean, high-statistics spectrum | `peak_finder` |
| Noisy or low-statistics spectrum | `mariscotti_peak_finder` |
| Want automatic sensitivity tuning | `mariscotti_peak_finder` (auto threshold) |
| Need direct control of sensitivity | `peak_finder` (tune `prominence`) |
| Overlapping peaks / doublets | `identify_doublets` + `fit_doublet` |

---

## References

- Mariscotti, M. A. *Nuclear Instruments and Methods* **50** (1967) 309–320
- Knoll, G. F. *Radiation Detection and Measurement*, 4th ed. — Chapter on Gamma Spectroscopy
- Gilmore, G. *Practical Gamma-ray Spectrometry* — Peak analysis techniques

## See Also

- `peak_finder()` — Prominence-based peak detector (wraps SciPy)
- `mariscotti_peak_finder()` — Second-difference peak detector
- `get_peak_roi()` — Extract a region of interest around a peak
- `fit_peak()` — Single Gaussian fit
- `fit_doublet()` — Double Gaussian fit for overlapping peaks
- `identify_doublets()` — Find adjacent peak pairs that may overlap
- `peak_counts()` — Calculate net counts for a detected peak
- `plot_spect_peaks()` — Visualise spectrum with detected peaks
- `plot_peak_roi()` — Visualise a peak ROI with fit overlay
- `plot_doublet_fit()` — Visualise a doublet decomposition
