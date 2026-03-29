# Detector Efficiency Methods

This document describes the detector efficiency fitting methods available in `gs_analysis` for gamma spectroscopy analysis.

## Overview

Detector efficiency quantifies the fraction of gamma-ray emissions that are actually recorded by the detector. It varies with photon energy and must be calibrated experimentally using sources of known activity. Once a calibration curve has been fit, `calc_energy_efficiency()` evaluates the efficiency at any energy of interest, enabling accurate calculation of activity or flux from measured peak areas.

Efficiency values are typically in the range (0, 1] and are strongly energy-dependent — usually peaking at low-to-mid energies and falling off at both ends of the spectrum.

## Fit Types

The `EfficiencyFitType` enum selects between two empirical fitting equations. Both are expressed in the logarithmic domain (`log(eff)`) which linearises the typically smooth, monotonically varying efficiency curve and improves the numerical behaviour of the least-squares fit.

### LOG — Logarithmic Polynomial Fit

**Selector:** `EfficiencyFitType.LOG` (default)

**Equation:**

```
ln(ε) = a₀ + a₁·ln(E) + a₂·ln(E)² + a₃·ln(E)³ + …
```

or equivalently:

```
ε = exp( a₀ + a₁·ln(E) + a₂·ln(E)² + … )
```

where `E` is the photon energy in **MeV** and `a₀, a₁, …` are the fitted coefficients stored in `eff_coeff`.

**Characteristics:**
- Widely used in gamma spectroscopy (e.g., Knoll, Gilmore)
- The polynomial order is controlled by the length of `eff_coeff`; a 3–5 term expansion is typical
- Captures the broad, smooth shape of most HPGe and NaI efficiency curves
- Well-conditioned over a wide energy range (tens of keV to several MeV)

**Best for:**
- High-purity germanium (HPGe) detectors
- Sodium iodide (NaI) detectors
- Any detector where efficiency varies smoothly with log-energy
- General-purpose efficiency calibration

---

### INVERSE_ENERGY — Inverse-Energy Polynomial Fit

**Selector:** `EfficiencyFitType.INVERSE_ENERGY`

**Equation:**

```
ln(ε) = a₀ + a₁·(1/E) + a₂·(1/E)² + a₃·(1/E)³ + …
```

or equivalently:

```
ε = exp( a₀ + a₁/E + a₂/E² + … )
```

where `E` is the photon energy in **MeV**.

**Characteristics:**
- Parameterises efficiency in terms of inverse energy
- Can better describe detectors with a sharper low-energy fall-off
- Polynomial order is again set by the length of `eff_coeff`

**Best for:**
- Detectors with strong low-energy efficiency dependence
- Alternative calibration when the LOG fit gives a poor residual

---

## API Reference

### `calc_energy_efficiency()`

```python
calc_energy_efficiency(
    energy: float,
    eff_coeff: Sequence[float],
    eff_fit: EfficiencyFitType = EfficiencyFitType.LOG,
) -> float
```

Evaluates the fitted efficiency curve at a single energy value.

| Parameter | Description |
|-----------|-------------|
| `energy` | Photon energy in **MeV** |
| `eff_coeff` | Sequence of fit coefficients `[a₀, a₁, a₂, …]`. The length determines the polynomial order. |
| `eff_fit` | Fitting equation to use (`EfficiencyFitType.LOG` or `EfficiencyFitType.INVERSE_ENERGY`). Defaults to `LOG`. |

**Returns:** Efficiency value `ε` (dimensionless, typically in the range (0, 1]).

**Raises:** `ValueError` if `eff_fit` is not a valid `EfficiencyFitType`.

---

### `plot_efficiency_curve()`

```python
plot_efficiency_curve(
    erg_range: Tuple[float, float],
    eff_coeff: Sequence[float],
    fit_type: EfficiencyFitType = EfficiencyFitType.LOG,
    n_points: int = 200,
    erg_points: Optional[npt.NDArray] = None,
    eff_points: Optional[npt.NDArray] = None,
    fname: Optional[str] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    xlabel: str = "Energy (MeV)",
    ylabel: str = "Efficiency",
) -> Axes
```

Plots the fitted efficiency curve over a specified energy range, optionally overlaying the measured calibration data points.

| Parameter | Description |
|-----------|-------------|
| `erg_range` | `(min_energy, max_energy)` in MeV |
| `eff_coeff` | Fit coefficients |
| `fit_type` | Fit equation type. Default `EfficiencyFitType.LOG` |
| `n_points` | Number of curve evaluation points. Default 200 |
| `erg_points` | Measured energy values (MeV) to overlay as scatter points |
| `eff_points` | Measured efficiency values corresponding to `erg_points` |
| `fname` | File path to save the figure (optional) |
| `ax` | Existing `matplotlib.axes.Axes` to plot on; a new figure is created when `None` |
| `title` | Plot title (optional) |
| `xlabel` | x-axis label. Default `"Energy (MeV)"` |
| `ylabel` | y-axis label. Default `"Efficiency"` |

**Returns:** `matplotlib.axes.Axes`

---

## Usage Examples

### Evaluating Efficiency at a Single Energy

```python
import gs_analysis as gs
from gs_analysis import EfficiencyFitType

# Coefficients obtained from a prior efficiency calibration (LOG fit)
eff_coeff = [-3.5, -0.8, -0.1]  # [a0, a1, a2]

# Evaluate at 1.332 MeV (Co-60 peak)
eff = gs.calc_energy_efficiency(1.332, eff_coeff)
print(f"Efficiency at 1.332 MeV: {eff:.4f}")
```

### Using the INVERSE_ENERGY Fit

```python
eff_coeff_inv = [-2.1, 0.05, -0.001]  # coefficients for inverse-energy fit

eff = gs.calc_energy_efficiency(
    0.662,
    eff_coeff_inv,
    eff_fit=EfficiencyFitType.INVERSE_ENERGY
)
print(f"Efficiency at 662 keV: {eff:.4f}")
```

### Plotting the Efficiency Curve

```python
import gs_plotting as gsp

eff_coeff = [-3.5, -0.8, -0.1]

# Measured calibration points (optional overlay)
import numpy as np
erg_measured = np.array([0.122, 0.356, 0.662, 1.173, 1.332])
eff_measured = np.array([0.045, 0.032, 0.021, 0.015, 0.014])

ax = gsp.plot_efficiency_curve(
    erg_range=(0.05, 2.0),
    eff_coeff=eff_coeff,
    erg_points=erg_measured,
    eff_points=eff_measured,
    title="HPGe Detector Efficiency Curve",
    fname="efficiency_curve.png",
)
```

### Correcting Peak Area for Efficiency

```python
# Raw net counts in a peak
net_counts = 15000.0
live_time = 3600.0     # seconds
peak_energy = 1.332    # MeV (Co-60)
source_activity = 37000  # Bq (1 µCi)

eff = gs.calc_energy_efficiency(peak_energy, eff_coeff)

# Activity = net_counts / (efficiency * live_time * branching_ratio)
branching_ratio = 0.9985
calculated_activity = net_counts / (eff * live_time * branching_ratio)
print(f"Calculated activity: {calculated_activity:.1f} Bq")
```

### Using Efficiency in Spectrum Creation

When generating synthetic spectra with `gs_creator`, efficiency coefficients can be supplied to scale peak amplitudes by the energy-dependent efficiency:

```python
import gs_creator as gsc
from gs_analysis import EfficiencyFitType

peak_list = [
    {"energy": 1.173, "intensity": 1.0},
    {"energy": 1.332, "intensity": 1.0},
]

spectrum = gsc.create_spectrum_from_peaks(
    peaks=peak_list,
    efficiency_coefficients=[-3.5, -0.8, -0.1],
    efficiency_fit_type=EfficiencyFitType.LOG,
)
```

---

## Choosing the Right Fit

| Scenario | Recommended Fit |
|----------|----------------|
| General-purpose HPGe or NaI calibration | `LOG` (default) |
| Detector with sharp low-energy dependence | `INVERSE_ENERGY` |
| Broad energy range (keV to MeV) | `LOG` |
| Narrow high-energy range | Either; compare residuals |

A higher-order polynomial (more coefficients) provides more flexibility but can over-fit sparse calibration data. A 3–4 term `LOG` fit is a good starting point for most detectors.

---

## References

- Knoll, G. F. *Radiation Detection and Measurement*, 4th ed. — Chapter on Efficiency
- Gilmore, G. *Practical Gamma-ray Spectrometry* — Efficiency calibration techniques
- ORTEC Maestro Software Documentation

## See Also

- `calc_energy_efficiency()` — Evaluate efficiency at a given energy
- `plot_efficiency_curve()` — Plot the fitted efficiency curve
- `EfficiencyFitType` — Enum selecting the fit equation
- `PhSpectrum.efficiency_fit_coefficients` — Per-spectrum storage of fit coefficients
