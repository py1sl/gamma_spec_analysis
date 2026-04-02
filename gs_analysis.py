# -*- coding: utf-8 -*-
"""
gamma spectrum analysis
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, lfilter, lfilter_zi
from enum import IntEnum

import ph_spectrum

from typing import Callable, List, Optional, Sequence, Tuple, Union, Any
import numpy.typing as npt


class BackgroundMethod(IntEnum):
    """Selector for background estimation methods used in :func:`calc_bg`."""

    TRAPEZOID = 1
    """Simple trapezoid background from Maestro."""
    LINEAR = 2
    """Linear interpolation method."""
    STEP = 3
    """Step function method (average of edges)."""
    SLIDING_AVERAGE = 4
    """Sliding window average method."""


class EfficiencyFitType(IntEnum):
    """Selector for the detector efficiency fitting equation used in
    :func:`calc_energy_efficiency`."""

    LOG = 1
    """Logarithmic fit: ``eff = exp(a0 + a1*ln(E) + a2*ln(E)^2 + ...)``."""
    INVERSE_ENERGY = 2
    """Inverse-energy fit: ``eff = exp(a0 + a1/E + a2/E^2 + ...)``."""


def generate_ebins(spec: "ph_spectrum.PhSpectrum") -> npt.NDArray[Any]:
    """Generate energy bin boundaries from the energy fit coefficients.

    .. deprecated::
        Prefer calling :meth:`~ph_spectrum.PhSpectrum.generate_ebins` directly
        on the spectrum object.  This module-level wrapper is kept for
        backward compatibility.

    Parameters
    ----------
    spec:
        A :class:`~ph_spectrum.PhSpectrum` with ``energy_fit_coefficients``
        set to a two-element sequence ``[a0, a1]``.

    Returns
    -------
    numpy.ndarray
        Energy values for each channel (length equal to
        ``spec.num_channels``).

    Raises
    ------
    ValueError
        If ``spec.energy_fit_coefficients`` is ``None`` or does not contain
        exactly two coefficients.
    """
    return spec.generate_ebins()


def five_point_smooth(
    counts: Union[Sequence[float], npt.NDArray[Any]],
) -> npt.NDArray[Any]:
    """5 point smoothing function.
    Recommended for use in low statistics in
    G.W. Phillips , Nucl. Instrum. Methods 153 (1978), 449
    Parameters
    ----------
    """
    if len(counts) < 5:
        raise ValueError("Input array must have at least 5 elements for smoothing.")

    counts_array = np.asarray(counts)
    smooth_spec = np.empty_like(counts_array, dtype=np.float64)

    # first 2 elements unchanged
    smooth_spec[:2] = counts_array[:2]

    # smooth middle elements using vectorized operations
    # val = (1/9) * (counts[i-2] + counts[i+2] + 2*counts[i+1] + 2*counts[i-1] + 3*counts[i])
    smooth_spec[2:-2] = (1.0 / 9.0) * (
        counts_array[:-4]       # i-2
        + counts_array[4:]      # i+2
        + 2 * counts_array[3:-1]  # 2*counts[i+1]
        + 2 * counts_array[1:-3]  # 2*counts[i-1]
        + 3 * counts_array[2:-2]  # 3*counts[i]
    )

    # last two elements unchanged
    smooth_spec[-2:] = counts_array[-2:]

    return smooth_spec


def three_point_smooth(
    counts: Union[Sequence[float], npt.NDArray[Any]],
) -> npt.NDArray[Any]:
    """3 point smoothing function using a simple moving average.

    This function applies a 3-point smoothing filter where each point
    (except the first and last) is replaced by the average of itself
    and its two neighbors.

    Parameters
    ----------
    counts : array-like
        The data array to be smoothed

    Returns
    -------
    numpy.ndarray
        Smoothed array of the same length as input

    Raises
    ------
    ValueError
        If input array has fewer than 3 elements
    """
    if len(counts) < 3:
        raise ValueError("Input array must have at least 3 elements for smoothing.")

    counts_array = np.asarray(counts, dtype=np.float64)
    smooth_spec = np.empty_like(counts_array)

    # first element unchanged
    smooth_spec[0] = counts_array[0]

    # smooth middle elements using vectorized operations: average of 3 points
    smooth_spec[1:-1] = (counts_array[:-2] + counts_array[1:-1] + counts_array[2:]) / 3.0

    # last element unchanged
    smooth_spec[-1] = counts_array[-1]

    return smooth_spec


def moving_average(
    counts: Union[Sequence[float], npt.NDArray[Any]],
    window: int = 5,
) -> npt.NDArray[Any]:
    """Moving average smoothing function with configurable window size.

    This function applies a moving average filter where each point is
    replaced by the average of points within the window. Edge points are
    handled by using available neighbors.

    Parameters
    ----------
    counts : array-like
        The data array to be smoothed
    window : int, optional
        Size of the moving window (must be odd). Default is 5.

    Returns
    -------
    numpy.ndarray
        Smoothed array of the same length as input

    Raises
    ------
    ValueError
        If window is not a positive odd integer or if input array
        is shorter than the window size
    """
    if window < 1 or window % 2 == 0:
        raise ValueError("Window size must be a positive odd integer.")

    counts_array = np.asarray(counts, dtype=np.float64)

    if len(counts_array) < window:
        raise ValueError(f"Input array must have at least {window} elements for window size {window}.")

    # Use NumPy's cumulative sum for efficient moving average
    # This avoids redundant summations in the loop-based approach
    cumsum = np.cumsum(np.insert(counts_array, 0, 0))
    half_window = window // 2
    smooth_spec = np.empty_like(counts_array)

    # Edge handling: use available neighbors only
    for i in range(len(counts_array)):
        start = max(0, i - half_window)
        end = min(len(counts_array), i + half_window + 1)
        smooth_spec[i] = (cumsum[end] - cumsum[start]) / (end - start)

    return smooth_spec


def exponential_moving_average(
    counts: Union[Sequence[float], npt.NDArray[Any]],
    alpha: float = 0.3,
) -> npt.NDArray[Any]:
    """Exponential moving average (EMA) smoothing function.

    This function applies an exponential moving average where recent
    values have higher weight than older values. The smoothing factor
    alpha controls how quickly the weights decrease.

    EMA formula: S[i] = alpha * counts[i] + (1 - alpha) * S[i-1]

    Parameters
    ----------
    counts : array-like
        The data array to be smoothed
    alpha : float, optional
        Smoothing factor between 0 and 1. Default is 0.3.
        Higher alpha gives more weight to recent values (less smoothing).
        Lower alpha gives more weight to past values (more smoothing).

    Returns
    -------
    numpy.ndarray
        Smoothed array of the same length as input

    Raises
    ------
    ValueError
        If alpha is not between 0 and 1 (exclusive)
    """
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Alpha must be between 0 and 1 (exclusive).")

    # lfilter requires float input; float64 matches the output dtype of
    # the original np.zeros-based implementation.
    counts_array = np.asarray(counts, dtype=np.float64)

    if counts_array.size == 0:
        return counts_array.copy()

    # Represent EMA as a first-order IIR filter:
    #   y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
    # Transfer function: b = [alpha], a = [1, -(1-alpha)]
    # Initial conditions are set so the first output equals the first input,
    # matching the original loop behaviour (smooth_spec[0] = counts[0]).
    b = np.array([alpha])
    a = np.array([1.0, -(1.0 - alpha)])
    zi = lfilter_zi(b, a) * counts_array[0]
    smooth_spec, _ = lfilter(b, a, counts_array, zi=zi)
    # Preserve exact first-element equality (matches loop definition: S[0] = x[0]).
    smooth_spec[0] = counts_array[0]

    return smooth_spec


def find_energy_pos(ebins: npt.NDArray[Any], erg: float) -> Optional[int]:
    """Find the index of the energy bin that contains the given energy value.

    ebins is a NumPy array of energy bin boundaries.
    erg is an energy value in the same units as ebins (usually keV).
    Returns the index of the bin containing erg, or None if erg is outside the bins.
    Use NumPy's binary search for efficiency: find index i such that
    ebins[i] <= erg < ebins[i + 1]. Return None if out of range.
    """
    idx = int(np.searchsorted(ebins, erg, side="right") - 1)

    if idx < 0 or idx >= len(ebins) - 1:
        return None

    return idx


def calc_energy_efficiency(
    energy: float,
    eff_coeff: Sequence[float],
    eff_fit: EfficiencyFitType = EfficiencyFitType.LOG,
) -> float:
    """Detector efficiency calculation
    energy : Energy to calculate det eff
    eff_coeff : An array with the coefficients for the energy fit
        the length is not fixed, the length of the array determines the
        number of terms in the expansion
    eff_fit : Determines what type of fit to use (EfficiencyFitType enum)
    returns eff - Value of efficiency for the input
                  energy using the selected fitting eqn
    """
    # eff_fit used to choose between calibration fit eqns
    # energy to be in MeV

    if eff_fit not in set(EfficiencyFitType):
        raise ValueError("The selected eff_fit is not valid")

    log_eff = eff_coeff[0]

    for i in range(1, len(eff_coeff)):
        if eff_fit == EfficiencyFitType.LOG:
            log_eff += eff_coeff[i] * np.power(np.log(energy), i)
        elif eff_fit == EfficiencyFitType.INVERSE_ENERGY:
            log_eff += eff_coeff[i] * np.power(1 / energy, i)

    eff = np.exp(log_eff)

    return eff


def calc_bg(
    counts: npt.NDArray[Any],
    c1: int,
    c2: int,
    m: BackgroundMethod = BackgroundMethod.TRAPEZOID,
) -> float:
    """Returns background under a peak
    spec is an numpy array of the counts values
    c1 is channel number of the start of peak
    c2 is channel number of the peak end
    m is a BackgroundMethod enum selector for background calculation methods
    BackgroundMethod.TRAPEZOID is a simple trapezoid background from Maestro
    BackgroundMethod.LINEAR is a linear interpolation method
    BackgroundMethod.STEP is a step function method (average of edges)
    BackgroundMethod.SLIDING_AVERAGE is a sliding window average method
    """

    if check_channel_validity(c1, c2, counts):
        if m == BackgroundMethod.TRAPEZOID:
            bg = estimate_background_trapezoid(counts, c1, c2)
        elif m == BackgroundMethod.LINEAR:
            bg = estimate_background_linear(counts, c1, c2)
        elif m == BackgroundMethod.STEP:
            bg = estimate_background_step(counts, c1, c2)
        elif m == BackgroundMethod.SLIDING_AVERAGE:
            bg = estimate_background_sliding_average(counts, c1, c2)
        else:
            raise ValueError("m is not set to a valid method id")

    return bg


def estimate_background_trapezoid(counts: npt.NDArray[Any], c1: int, c2: int) -> float:
    """Estimate background under a peak using the Maestro trapezoid method.

    Uses up to two channels before `c1` and up to two channels after `c2`
    (adjusting at spectrum edges) to compute:
        bg = (low_sum + high_sum) * ((c2 - c1 + 1) / 6)

    """
    # Validate channel indices (will raise if invalid)
    check_channel_validity(c1, c2, counts)

    # Safe low window:
    low_start = max(0, c1 - 2)
    low_sum = float(np.sum(counts[low_start:c1])) if c1 > low_start else 0.0

    # Safe high window:
    high_end = min(len(counts), c2 + 2)
    high_sum = float(np.sum(counts[c2:high_end])) if high_end > c2 else 0.0

    width = c2 - c1 + 1
    bg = (low_sum + high_sum) * (width / 6.0)

    return float(bg)


def estimate_background_linear(counts: npt.NDArray[Any], c1: int, c2: int) -> float:
    """Estimate background using linear interpolation between edge points.

    This method uses a simple linear interpolation between the average of
    channels before c1 and after c2. The background under the peak is
    calculated by integrating the linear function across the peak region.

    Note: The peak region is defined by Python slicing convention [c1:c2),
    meaning c1 is inclusive and c2 is exclusive, giving width = c2 - c1.

    Parameters
    ----------
    counts : numpy array
        The spectrum counts data
    c1 : int
        Channel number of the start of peak (inclusive)
    c2 : int
        Channel number of the peak end (exclusive, as in Python slicing)

    Returns
    -------
    float
        Estimated background counts under the peak
    """
    check_channel_validity(c1, c2, counts)

    # Use two channels on each side for better statistics
    low_start = max(0, c1 - 2)
    low_count = len(counts[low_start:c1])
    low_avg = float(np.mean(counts[low_start:c1])) if low_count > 0 else 0.0

    high_end = min(len(counts), c2 + 2)
    high_count = len(counts[c2:high_end])
    high_avg = float(np.mean(counts[c2:high_end])) if high_count > 0 else 0.0

    # Linear interpolation: background is the trapezoidal area under the line
    # Width matches Python slicing: c2 - c1 (c2 is exclusive)
    width = c2 - c1
    bg = (low_avg + high_avg) * width / 2.0

    return float(bg)


def estimate_background_step(counts: npt.NDArray[Any], c1: int, c2: int) -> float:
    """Estimate background using a step function (average of edges).

    This method calculates the average of the background regions on both
    sides of the peak and uses this constant value as the background level
    under the peak.

    Note: The peak region is defined by Python slicing convention [c1:c2),
    meaning c1 is inclusive and c2 is exclusive, giving width = c2 - c1.

    Parameters
    ----------
    counts : numpy array
        The spectrum counts data
    c1 : int
        Channel number of the start of peak (inclusive)
    c2 : int
        Channel number of the peak end (exclusive, as in Python slicing)

    Returns
    -------
    float
        Estimated background counts under the peak
    """
    check_channel_validity(c1, c2, counts)

    # Use two channels on each side
    low_start = max(0, c1 - 2)
    low_count = len(counts[low_start:c1])
    low_avg = float(np.mean(counts[low_start:c1])) if low_count > 0 else 0.0

    high_end = min(len(counts), c2 + 2)
    high_count = len(counts[c2:high_end])
    high_avg = float(np.mean(counts[c2:high_end])) if high_count > 0 else 0.0

    # Step function: use the average of both sides
    avg_bg = (low_avg + high_avg) / 2.0
    # Width matches Python slicing: c2 - c1 (c2 is exclusive)
    width = c2 - c1
    bg = avg_bg * width

    return float(bg)


def estimate_background_sliding_average(
    counts: npt.NDArray[Any], c1: int, c2: int, window: int = 5
) -> float:
    """Estimate background using a sliding window average method.

    This method calculates the background by taking a moving average in the
    regions adjacent to the peak, then interpolating under the peak region.
    This method is more robust to local variations in the background.

    Note: The peak region is defined by Python slicing convention [c1:c2),
    meaning c1 is inclusive and c2 is exclusive, giving width = c2 - c1.

    Parameters
    ----------
    counts : numpy array
        The spectrum counts data
    c1 : int
        Channel number of the start of peak (inclusive)
    c2 : int
        Channel number of the peak end (exclusive, as in Python slicing)
    window : int, optional
        Size of the sliding window for averaging. Default is 5.

    Returns
    -------
    float
        Estimated background counts under the peak
    """
    check_channel_validity(c1, c2, counts)

    # Determine safe windows for averaging
    low_start = max(0, c1 - window)
    low_end = c1
    high_start = c2
    high_end = min(len(counts), c2 + window)

    # Calculate averages using available data
    if low_end > low_start:
        low_region = counts[low_start:low_end]
        low_avg = float(np.mean(low_region)) if len(low_region) > 0 else 0.0
    else:
        low_avg = 0.0

    if high_end > high_start:
        high_region = counts[high_start:high_end]
        high_avg = float(np.mean(high_region)) if len(high_region) > 0 else 0.0
    else:
        high_avg = 0.0

    # Linear interpolation between the two averaged regions
    # Width matches Python slicing: c2 - c1 (c2 is exclusive)
    width = c2 - c1
    bg = (low_avg + high_avg) * width / 2.0

    return float(bg)


def gross_count(counts: npt.NDArray[Any], c1: int, c2: int) -> int:
    """Returns total number of counts in a spectrum between two channels"""
    if check_channel_validity(c1, c2, counts):
        return int(np.sum(counts[c1:c2]))
    return 0


def check_channel_validity(c1: int, c2: int, counts: npt.NDArray[Any]) -> bool:
    """checks validity of the channel range"""
    # check channel bounds are valid
    if c1 > c2:
        raise ValueError("c1 must be less than c2")
    if c1 < 0:
        raise ValueError("c1 must be positive number above 0")
    if c2 > len(counts):
        raise ValueError("c2 must be less than max number of channels")
    return True


def net_counts(
    counts: npt.NDArray[Any],
    c1: int,
    c2: int,
    m: BackgroundMethod = BackgroundMethod.TRAPEZOID,
) -> float:
    """Calculates net counts between two channels"""
    bg = calc_bg(counts, c1, c2, m)
    gc = gross_count(counts, c1, c2)
    nc = gc - bg
    return float(nc)


def gaussian(
    x: npt.NDArray[Any], a: float, x0: float, sigma: float
) -> npt.NDArray[Any]:
    """gaussian used for curve fitting"""
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def double_gaussian(
    x: npt.NDArray[Any],
    a1: float, x01: float, sigma1: float,
    a2: float, x02: float, sigma2: float,
) -> npt.NDArray[Any]:
    """Sum of two Gaussians used for doublet curve fitting.

    Parameters
    ----------
    x : array-like
        Input x values
    a1 : float
        Amplitude of first Gaussian
    x01 : float
        Center of first Gaussian
    sigma1 : float
        Standard deviation of first Gaussian
    a2 : float
        Amplitude of second Gaussian
    x02 : float
        Center of second Gaussian
    sigma2 : float
        Standard deviation of second Gaussian

    Returns
    -------
    numpy.ndarray
        Sum of two Gaussian values at positions x
    """
    return gaussian(x, a1, x01, sigma1) + gaussian(x, a2, x02, sigma2)


def get_peak_roi(
    peak_pos: int, counts: npt.NDArray[Any], ebins: npt.NDArray[Any], offset: int = 10
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """extracts a region of the spectra around the peak_pos
    number of channels extracted is 2 x offset
    returns both the counts and energy bin values for that region
    """
    if (peak_pos - offset) < 0:
        raise ValueError("cannot extract channel below 0, reduce offset")
    if (peak_pos + offset) >= len(counts):
        raise ValueError("cannot extract channel beyond spec length")

    y = counts[peak_pos - offset: peak_pos + offset]
    x = ebins[peak_pos - offset: peak_pos + offset]

    return x, y


def fit_peak(
    x: npt.NDArray[Any], y: npt.NDArray[Any]
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Fit a single peak to a Gaussian.

    Parameters
    ----------
    x : array-like
        Energy bin positions for the region of interest.
    y : array-like
        Counts for the region of interest.

    Returns
    -------
    popt : numpy.ndarray
        Optimal fit parameters ``[amplitude, centroid, sigma]``.
    pcov : numpy.ndarray
        Estimated covariance matrix of *popt* (3×3).  The square root of
        the diagonal elements gives one-standard-deviation uncertainties
        for each parameter.
    """
    mean = np.sum(x * y) / np.sum(y)
    sigma = np.sqrt(np.sum(y * (x - mean) ** 2) / np.sum(y))

    popt, pcov = curve_fit(gaussian, x, y, p0=[max(y), mean, sigma], maxfev=10000)

    return popt, pcov


def fit_doublet(x: npt.NDArray[Any], y: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Fits a doublet (two partially overlapping peaks) to a double Gaussian.

    Initial estimates for each peak center are derived from the two largest
    local maxima in the ROI.  If fewer than two local maxima exist the ROI
    is split in half and the maximum of each half is used instead.

    Parameters
    ----------
    x : array-like
        Energy bin positions for the region of interest
    y : array-like
        Counts for the region of interest

    Returns
    -------
    popt : numpy.ndarray
        Array of 6 fitted parameters [a1, x01, sigma1, a2, x02, sigma2]
    pcov : numpy.ndarray
        Estimated covariance matrix of *popt* (6×6).

    Raises
    ------
    RuntimeError
        If the curve fit fails to converge
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Estimate a global sigma from the weighted spread of the data
    total = np.sum(y)
    if total != 0:
        mean_est = np.sum(x * y) / total
        sigma_est = np.sqrt(np.sum(y * (x - mean_est) ** 2) / total)
        if sigma_est == 0:
            sigma_est = (x[-1] - x[0]) / 6.0
    else:
        mean_est = (x[0] + x[-1]) / 2.0
        sigma_est = (x[-1] - x[0]) / 6.0

    # Find local maxima by comparing each point with its neighbours
    local_max_idx = [i for i in range(1, len(y) - 1) if y[i] > y[i - 1] and y[i] > y[i + 1]]

    if len(local_max_idx) >= 2:
        # Use the two highest local maxima, ordered by x position
        local_max_idx.sort(key=lambda i: y[i], reverse=True)
        i1, i2 = sorted(local_max_idx[:2])
        x01_est, a1_est = x[i1], float(y[i1])
        x02_est, a2_est = x[i2], float(y[i2])
    else:
        # Fall back: split the ROI in half, guarding against empty slices
        mid = max(1, min(len(x) // 2, len(x) - 1))
        i1 = int(np.argmax(y[:mid]))
        i2 = int(np.argmax(y[mid:])) + mid
        x01_est, a1_est = x[i1], float(y[i1])
        x02_est, a2_est = x[i2], float(y[i2])

    half_sigma = sigma_est / 2.0
    p0 = [a1_est, x01_est, half_sigma, a2_est, x02_est, half_sigma]

    popt, pcov = curve_fit(double_gaussian, x, y, p0=p0, maxfev=10000)

    return popt, pcov


# ---------------------------------------------------------------------------
# Gaussian area helpers (Option 2)
# ---------------------------------------------------------------------------

def gaussian_area(a: float, sigma: float) -> float:
    """Return the analytic area of a Gaussian peak.

    For a Gaussian of the form ``a * exp(-(x-x0)^2 / (2*sigma^2))`` the
    definite integral from −∞ to +∞ equals ``a * sigma * sqrt(2*pi)``.

    Parameters
    ----------
    a : float
        Peak amplitude.
    sigma : float
        Standard deviation (width parameter).

    Returns
    -------
    float
        Analytic peak area (counts).
    """
    return float(a * np.abs(sigma) * np.sqrt(2.0 * np.pi))


def gaussian_area_uncertainty(
    a: float, sigma: float, pcov: npt.NDArray[Any]
) -> float:
    """Return the one-sigma uncertainty on the analytic Gaussian peak area.

    Uses first-order error propagation through ``area = a * |sigma| * sqrt(2*pi)``:

    .. math::

        \\sigma_{\\text{area}} = \\sqrt{2\\pi}\\,
            \\sqrt{(\\sigma \\cdot \\sigma_a)^2
                  + (a \\cdot \\sigma_\\sigma)^2
                  + 2\\,a\\,\\sigma \\cdot \\mathrm{cov}(a,\\,\\sigma)}

    where ``sigma_a = sqrt(pcov[0,0])``, ``sigma_sigma = sqrt(pcov[2,2])``,
    and ``cov(a, sigma) = pcov[0,2]``.

    Parameters
    ----------
    a : float
        Fitted amplitude (``popt[0]``).
    sigma : float
        Fitted width (``popt[2]``).
    pcov : numpy.ndarray
        3×3 covariance matrix returned by :func:`fit_peak`.

    Returns
    -------
    float
        One-sigma uncertainty on the peak area (counts).
    """
    pcov = np.asarray(pcov, dtype=float)
    var_a = pcov[0, 0]
    var_sigma = pcov[2, 2]
    cov_a_sigma = pcov[0, 2]
    # Partial derivatives: dA/da = |sigma|*sqrt(2pi), dA/dsigma = a*sqrt(2pi)
    abs_sigma = np.abs(sigma)
    variance = (2.0 * np.pi) * (
        (abs_sigma * abs_sigma) * var_a
        + (a * a) * var_sigma
        + 2.0 * a * abs_sigma * cov_a_sigma
    )
    return float(np.sqrt(max(variance, 0.0)))


def fit_peak_area(
    x: npt.NDArray[Any], y: npt.NDArray[Any]
) -> Tuple[float, float]:
    """Fit a single Gaussian peak and return its area with uncertainty.

    Combines :func:`fit_peak` with :func:`gaussian_area` and
    :func:`gaussian_area_uncertainty` to give both the analytic peak area and
    its one-sigma uncertainty derived from the covariance matrix of the fit.

    Parameters
    ----------
    x : array-like
        Energy bin positions for the region of interest.
    y : array-like
        Counts for the region of interest.

    Returns
    -------
    area : float
        Analytic Gaussian peak area ``a * |sigma| * sqrt(2*pi)``.
    area_uncertainty : float
        One-sigma uncertainty on *area* propagated from the fit covariance.
    """
    popt, pcov = fit_peak(x, y)
    a, _x0, sigma = popt
    area = gaussian_area(a, sigma)
    unc = gaussian_area_uncertainty(a, sigma, pcov)
    return area, unc


def fit_doublet_areas(
    x: npt.NDArray[Any], y: npt.NDArray[Any]
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Fit a doublet and return the area and uncertainty for each sub-peak.

    Uses :func:`fit_doublet` and propagates uncertainties for each of the two
    Gaussian components using their respective sub-blocks of the covariance
    matrix.

    Parameters
    ----------
    x : array-like
        Energy bin positions for the region of interest.
    y : array-like
        Counts for the region of interest.

    Returns
    -------
    peak1 : tuple of (float, float)
        ``(area, area_uncertainty)`` for the first Gaussian component.
    peak2 : tuple of (float, float)
        ``(area, area_uncertainty)`` for the second Gaussian component.
    """
    popt, pcov = fit_doublet(x, y)
    a1, _x01, sigma1, a2, _x02, sigma2 = popt

    # Extract 3×3 sub-blocks for each peak's parameters
    # Peak 1: indices 0,1,2  →  a1, x01, sigma1
    idx1 = [0, 1, 2]
    pcov1 = pcov[np.ix_(idx1, idx1)]
    # Peak 2: indices 3,4,5  →  a2, x02, sigma2
    idx2 = [3, 4, 5]
    pcov2 = pcov[np.ix_(idx2, idx2)]

    area1 = gaussian_area(a1, sigma1)
    unc1 = gaussian_area_uncertainty(a1, sigma1, pcov1)
    area2 = gaussian_area(a2, sigma2)
    unc2 = gaussian_area_uncertainty(a2, sigma2, pcov2)

    return (area1, unc1), (area2, unc2)


# ---------------------------------------------------------------------------
# FWHM helpers (Option 4 from improvement plan)
# ---------------------------------------------------------------------------

#: Conversion factor from Gaussian sigma to FWHM: ``2 * sqrt(2 * ln(2))``.
_FWHM_FACTOR: float = 2.0 * np.sqrt(2.0 * np.log(2.0))


def peak_fwhm(sigma: float) -> float:
    """Return the Full Width at Half Maximum (FWHM) of a Gaussian peak.

    Uses the exact conversion ``FWHM = 2 * sqrt(2 * ln 2) * |sigma|``
    (approximately ``2.3548 * |sigma|``).

    Parameters
    ----------
    sigma : float
        Gaussian width parameter as returned in ``popt[2]`` by
        :func:`fit_peak`.  The sign is ignored.

    Returns
    -------
    float
        FWHM in the same units as *sigma* (channels or energy, depending on
        the coordinate axis used during fitting).
    """
    return _FWHM_FACTOR * abs(sigma)


def peak_fwhm_uncertainty(sigma: float, sigma_unc: float) -> float:
    """Return the one-sigma uncertainty on the FWHM.

    Since ``FWHM = k * |sigma|``, first-order propagation gives
    ``sigma_FWHM = k * sigma_sigma`` where ``k = 2*sqrt(2*ln2)``.

    Parameters
    ----------
    sigma : float
        Fitted Gaussian sigma (``popt[2]``).  Included for API symmetry with
        :func:`peak_fwhm` but not used in the calculation.
    sigma_unc : float
        One-sigma uncertainty on *sigma* (``sqrt(pcov[2, 2])``).

    Returns
    -------
    float
        One-sigma uncertainty on the FWHM.
    """
    return _FWHM_FACTOR * abs(sigma_unc)


def fit_peak_fwhm(
    x: npt.NDArray[Any], y: npt.NDArray[Any]
) -> Tuple[float, float]:
    """Fit a single Gaussian peak and return its FWHM with uncertainty.

    Convenience wrapper that calls :func:`fit_peak` and converts the fitted
    sigma to FWHM via :func:`peak_fwhm` and :func:`peak_fwhm_uncertainty`.

    Parameters
    ----------
    x : array-like
        Energy bin positions for the region of interest.
    y : array-like
        Counts for the region of interest.

    Returns
    -------
    fwhm : float
        Full Width at Half Maximum in the same units as *x*.
    fwhm_uncertainty : float
        One-sigma uncertainty on *fwhm* propagated from the fit covariance.
    """
    popt, pcov = fit_peak(x, y)
    sigma = popt[2]
    sigma_unc = float(np.sqrt(pcov[2, 2]))
    return peak_fwhm(sigma), peak_fwhm_uncertainty(sigma, sigma_unc)


# ---------------------------------------------------------------------------
# Goodness-of-fit statistics (Option 5 from improvement plan)
# ---------------------------------------------------------------------------

def fit_chi2(
    x: npt.NDArray[Any],
    y: npt.NDArray[Any],
    popt: npt.NDArray[Any],
    model_fn: Callable[..., npt.NDArray[Any]],
    n_params: Optional[int] = None,
) -> Tuple[float, float, int]:
    """Compute chi-squared goodness-of-fit statistics for any fitted model.

    Poisson statistics are assumed, so the variance in each bin is taken as
    ``max(y_i, 1)`` to avoid division by zero in empty bins.

    Parameters
    ----------
    x : array-like
        x-axis values (energy bins or channels) used during fitting.
    y : array-like
        Observed counts used during fitting.
    popt : array-like
        Optimal parameters returned by the fitting function (e.g.
        :func:`fit_peak` or :func:`fit_doublet`).
    model_fn : callable
        The model function with signature ``model_fn(x, *popt)``, e.g.
        :func:`gaussian` or :func:`double_gaussian`.
    n_params : int, optional
        Number of free parameters in the fit.  Defaults to ``len(popt)``.

    Returns
    -------
    chi2 : float
        Chi-squared statistic ``sum((y_i - y_fit_i)^2 / max(y_i, 1))``.
    reduced_chi2 : float
        Reduced chi-squared ``chi2 / ndof``.  Returns ``inf`` if
        ``ndof <= 0``.
    ndof : int
        Degrees of freedom ``len(y) - n_params``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    popt = np.asarray(popt, dtype=float)

    if n_params is None:
        n_params = len(popt)

    y_fit = model_fn(x, *popt)
    variance = np.maximum(y, 1.0)
    chi2 = float(np.sum((y - y_fit) ** 2 / variance))
    ndof = len(y) - n_params
    reduced_chi2 = chi2 / ndof if ndof > 0 else float("inf")
    return chi2, reduced_chi2, ndof


def fit_peak_chi2(
    x: npt.NDArray[Any],
    y: npt.NDArray[Any],
    popt: npt.NDArray[Any],
) -> Tuple[float, float, int]:
    """Goodness-of-fit statistics for a single-peak Gaussian fit.

    Convenience wrapper around :func:`fit_chi2` using :func:`gaussian` as
    the model and ``n_params = 3``.

    Parameters
    ----------
    x : array-like
        Energy bin positions for the ROI.
    y : array-like
        Counts for the ROI.
    popt : array-like
        Three-parameter Gaussian fit result ``[amplitude, centroid, sigma]``
        from :func:`fit_peak`.

    Returns
    -------
    chi2 : float
        Chi-squared statistic.
    reduced_chi2 : float
        Reduced chi-squared (chi2 per degree of freedom).
    ndof : int
        Degrees of freedom (``len(y) - 3``).
    """
    return fit_chi2(x, y, popt, gaussian, n_params=3)


def fit_doublet_chi2(
    x: npt.NDArray[Any],
    y: npt.NDArray[Any],
    popt: npt.NDArray[Any],
) -> Tuple[float, float, int]:
    """Goodness-of-fit statistics for a doublet fit.

    Convenience wrapper around :func:`fit_chi2` using :func:`double_gaussian`
    as the model and ``n_params = 6``.

    Parameters
    ----------
    x : array-like
        Energy bin positions for the ROI.
    y : array-like
        Counts for the ROI.
    popt : array-like
        Six-parameter doublet fit result
        ``[a1, x01, sigma1, a2, x02, sigma2]`` from :func:`fit_doublet`.

    Returns
    -------
    chi2 : float
        Chi-squared statistic.
    reduced_chi2 : float
        Reduced chi-squared (chi2 per degree of freedom).
    ndof : int
        Degrees of freedom (``len(y) - 6``).
    """
    return fit_chi2(x, y, popt, double_gaussian, n_params=6)


# ---------------------------------------------------------------------------
# Activity calculation (Option 3 from improvement plan)
# ---------------------------------------------------------------------------

def calc_activity(
    net_counts: float,
    live_time: float,
    emission_probability: float,
    efficiency: float,
) -> float:
    """Return the source activity in Becquerels (Bq).

    Converts a background-subtracted peak area to an absolute source
    activity using:

    .. math::

        A = \\frac{N_{\\text{net}}}{T_{\\text{live}} \\cdot I_{\\gamma}
                                    \\cdot \\varepsilon}

    where :math:`N_{\\text{net}}` is the net counts from
    :func:`net_counts`, :math:`T_{\\text{live}}` is the measurement
    live time in seconds, :math:`I_{\\gamma}` is the gamma-ray emission
    probability per decay (branching ratio), and :math:`\\varepsilon` is
    the detector efficiency at the photopeak energy.

    Parameters
    ----------
    net_counts : float
        Net peak counts after background subtraction (e.g. from
        :func:`net_counts` or :func:`fit_peak_area`).
    live_time : float
        Measurement live time in seconds (``PhSpectrum.live_time``).
    emission_probability : float
        Gamma-ray emission probability per decay in the range ``(0, 1]``
        (also called intensity or branching ratio).
    efficiency : float
        Absolute detector efficiency at the photopeak energy, in the range
        ``(0, 1]`` (e.g. from :func:`calc_energy_efficiency`).

    Returns
    -------
    float
        Source activity in Becquerels (Bq = disintegrations per second).

    Raises
    ------
    ValueError
        If *live_time* is not positive, or if *emission_probability* or
        *efficiency* are outside ``(0, 1]``.
    """
    if live_time <= 0.0:
        raise ValueError("live_time must be positive")
    if emission_probability <= 0.0 or emission_probability > 1.0:
        raise ValueError("emission_probability must be in the range (0, 1]")
    if efficiency <= 0.0 or efficiency > 1.0:
        raise ValueError("efficiency must be in the range (0, 1]")
    return float(net_counts) / (live_time * emission_probability * efficiency)


def calc_activity_uncertainty(
    net_counts_unc: float,
    live_time: float,
    emission_probability: float,
    efficiency: float,
) -> float:
    """Return the one-sigma uncertainty on the activity (Bq).

    Propagates the uncertainty on the net counts through the activity
    formula, treating live time, emission probability, and efficiency as
    exact:

    .. math::

        \\sigma_A = \\frac{\\sigma_{N_{\\text{net}}}}
                         {T_{\\text{live}} \\cdot I_{\\gamma}
                          \\cdot \\varepsilon}

    For the net-counts uncertainty use :func:`net_counts_uncertainty`
    (Poisson + background propagation) or :func:`gaussian_area_uncertainty`
    (fit covariance).

    Parameters
    ----------
    net_counts_unc : float
        One-sigma uncertainty on the net peak counts.
    live_time : float
        Measurement live time in seconds.
    emission_probability : float
        Gamma-ray emission probability per decay in ``(0, 1]``.
    efficiency : float
        Absolute detector efficiency at the photopeak energy in ``(0, 1]``.

    Returns
    -------
    float
        One-sigma uncertainty on the activity in Becquerels.

    Raises
    ------
    ValueError
        If *live_time* is not positive, or if *emission_probability* or
        *efficiency* are outside ``(0, 1]``.
    """
    if live_time <= 0.0:
        raise ValueError("live_time must be positive")
    if emission_probability <= 0.0 or emission_probability > 1.0:
        raise ValueError("emission_probability must be in the range (0, 1]")
    if efficiency <= 0.0 or efficiency > 1.0:
        raise ValueError("efficiency must be in the range (0, 1]")
    return float(net_counts_unc) / (live_time * emission_probability * efficiency)


# ---------------------------------------------------------------------------
# Per-method background variance (Option 4)
# ---------------------------------------------------------------------------

def estimate_background_trapezoid_uncertainty(
    counts: npt.NDArray[Any], c1: int, c2: int
) -> float:
    """Return the one-sigma uncertainty on the trapezoid background estimate.

    Assumes Poisson statistics so that ``var(N_i) = N_i`` for each channel.
    Propagating through ``bg = (low_sum + high_sum) * (width / 6)``:

    .. math::

        \\sigma_{bg} = \\frac{\\mathrm{width}}{6}
                      \\sqrt{\\sum_{i \\in \\text{low}} n_i
                           + \\sum_{i \\in \\text{high}} n_i}

    Parameters
    ----------
    counts : numpy.ndarray
        Spectrum counts array.
    c1, c2 : int
        ROI boundaries matching those passed to
        :func:`estimate_background_trapezoid`.

    Returns
    -------
    float
        One-sigma Poisson uncertainty on the background estimate.
    """
    check_channel_validity(c1, c2, counts)

    low_start = max(0, c1 - 2)
    low_window = counts[low_start:c1]
    high_end = min(len(counts), c2 + 2)
    high_window = counts[c2:high_end]

    width = c2 - c1 + 1
    scale = width / 6.0
    variance_bg = scale**2 * (
        float(np.sum(np.abs(low_window))) + float(np.sum(np.abs(high_window)))
    )
    return float(np.sqrt(variance_bg))


def estimate_background_linear_uncertainty(
    counts: npt.NDArray[Any], c1: int, c2: int
) -> float:
    """Return the one-sigma uncertainty on the linear-interpolation background.

    Propagates Poisson variance through
    ``bg = (low_avg + high_avg) * width / 2`` where the averages are taken
    over two-channel windows on each side:

    .. math::

        \\sigma_{bg} = \\frac{\\mathrm{width}}{2}
                      \\sqrt{\\frac{\\sum n_i}{n_{\\text{low}}^2}
                           + \\frac{\\sum n_j}{n_{\\text{high}}^2}}

    Parameters
    ----------
    counts : numpy.ndarray
        Spectrum counts array.
    c1, c2 : int
        ROI boundaries matching those passed to
        :func:`estimate_background_linear`.

    Returns
    -------
    float
        One-sigma Poisson uncertainty on the background estimate.
    """
    check_channel_validity(c1, c2, counts)

    low_start = max(0, c1 - 2)
    low_window = counts[low_start:c1]
    high_end = min(len(counts), c2 + 2)
    high_window = counts[c2:high_end]

    width = c2 - c1
    scale = width / 2.0

    # var(mean) = sum(N_i) / n^2 under Poisson
    n_low = len(low_window) if len(low_window) > 0 else 1
    n_high = len(high_window) if len(high_window) > 0 else 1
    var_low_avg = float(np.sum(np.abs(low_window))) / n_low**2
    var_high_avg = float(np.sum(np.abs(high_window))) / n_high**2

    variance_bg = scale**2 * (var_low_avg + var_high_avg)
    return float(np.sqrt(variance_bg))


def estimate_background_step_uncertainty(
    counts: npt.NDArray[Any], c1: int, c2: int
) -> float:
    """Return the one-sigma uncertainty on the step-function background.

    Propagates Poisson variance through
    ``bg = ((low_avg + high_avg) / 2) * width``:

    .. math::

        \\sigma_{bg} = \\frac{\\mathrm{width}}{2}
                      \\sqrt{\\frac{\\sum n_i}{4 n_{\\text{low}}^2}
                           + \\frac{\\sum n_j}{4 n_{\\text{high}}^2}}

    Parameters
    ----------
    counts : numpy.ndarray
        Spectrum counts array.
    c1, c2 : int
        ROI boundaries matching those passed to
        :func:`estimate_background_step`.

    Returns
    -------
    float
        One-sigma Poisson uncertainty on the background estimate.
    """
    check_channel_validity(c1, c2, counts)

    low_start = max(0, c1 - 2)
    low_window = counts[low_start:c1]
    high_end = min(len(counts), c2 + 2)
    high_window = counts[c2:high_end]

    width = c2 - c1

    n_low = len(low_window) if len(low_window) > 0 else 1
    n_high = len(high_window) if len(high_window) > 0 else 1
    # avg_bg = (low_avg + high_avg) / 2  →  var(avg_bg) = (var(low_avg) + var(high_avg)) / 4
    var_low_avg = float(np.sum(np.abs(low_window))) / n_low**2
    var_high_avg = float(np.sum(np.abs(high_window))) / n_high**2
    var_avg_bg = (var_low_avg + var_high_avg) / 4.0

    variance_bg = (width**2) * var_avg_bg
    return float(np.sqrt(variance_bg))


def estimate_background_sliding_average_uncertainty(
    counts: npt.NDArray[Any], c1: int, c2: int, window: int = 5
) -> float:
    """Return the one-sigma uncertainty on the sliding-average background.

    Propagates Poisson variance through
    ``bg = (low_avg + high_avg) * width / 2`` where the averages use a
    window of *window* channels:

    .. math::

        \\sigma_{bg} = \\frac{\\mathrm{width}}{2}
                      \\sqrt{\\frac{\\sum n_i}{n_{\\text{low}}^2}
                           + \\frac{\\sum n_j}{n_{\\text{high}}^2}}

    Parameters
    ----------
    counts : numpy.ndarray
        Spectrum counts array.
    c1, c2 : int
        ROI boundaries matching those passed to
        :func:`estimate_background_sliding_average`.
    window : int, optional
        Window size used in :func:`estimate_background_sliding_average`.
        Default is 5.

    Returns
    -------
    float
        One-sigma Poisson uncertainty on the background estimate.
    """
    check_channel_validity(c1, c2, counts)

    low_start = max(0, c1 - window)
    low_window = counts[low_start:c1]
    high_end = min(len(counts), c2 + window)
    high_window = counts[c2:high_end]

    width = c2 - c1

    n_low = len(low_window) if len(low_window) > 0 else 1
    n_high = len(high_window) if len(high_window) > 0 else 1
    var_low_avg = float(np.sum(np.abs(low_window))) / n_low**2
    var_high_avg = float(np.sum(np.abs(high_window))) / n_high**2

    scale = width / 2.0
    variance_bg = scale**2 * (var_low_avg + var_high_avg)
    return float(np.sqrt(variance_bg))


def calc_bg_uncertainty(
    counts: npt.NDArray[Any],
    c1: int,
    c2: int,
    m: BackgroundMethod = BackgroundMethod.TRAPEZOID,
) -> float:
    """Dispatcher: return the Poisson-propagated background uncertainty.

    Selects the appropriate analytical uncertainty function for the chosen
    background estimation method.

    Parameters
    ----------
    counts : numpy.ndarray
        Spectrum counts array.
    c1, c2 : int
        ROI channel boundaries.
    m : BackgroundMethod
        Background method selector (default: ``TRAPEZOID``).

    Returns
    -------
    float
        One-sigma uncertainty on the background estimate.
    """
    if m == BackgroundMethod.TRAPEZOID:
        return estimate_background_trapezoid_uncertainty(counts, c1, c2)
    elif m == BackgroundMethod.LINEAR:
        return estimate_background_linear_uncertainty(counts, c1, c2)
    elif m == BackgroundMethod.STEP:
        return estimate_background_step_uncertainty(counts, c1, c2)
    elif m == BackgroundMethod.SLIDING_AVERAGE:
        return estimate_background_sliding_average_uncertainty(counts, c1, c2)
    else:
        raise ValueError("m is not set to a valid method id")


def net_counts_uncertainty(
    counts: npt.NDArray[Any],
    c1: int,
    c2: int,
    m: BackgroundMethod = BackgroundMethod.TRAPEZOID,
) -> Tuple[float, float]:
    """Return net counts and their one-sigma Poisson uncertainty.

    Computes both the net peak area and the associated statistical
    uncertainty by propagating Poisson variance through:

    .. math::

        N_{\\text{net}} = N_{\\text{gross}} - B

    .. math::

        \\sigma_{\\text{net}} = \\sqrt{N_{\\text{gross}} + \\sigma_B^2}

    where :math:`\\sigma_B` is obtained from :func:`calc_bg_uncertainty` for
    the selected background method, and the Poisson variance on the gross
    counts is simply :math:`N_{\\text{gross}}`.

    Parameters
    ----------
    counts : numpy.ndarray
        Spectrum counts array.
    c1, c2 : int
        ROI channel boundaries.
    m : BackgroundMethod
        Background method selector (default: ``TRAPEZOID``).

    Returns
    -------
    net : float
        Net counts (peak area after background subtraction).
    uncertainty : float
        One-sigma statistical uncertainty on *net*.
    """
    nc = net_counts(counts, c1, c2, m)
    gc = gross_count(counts, c1, c2)
    sigma_bg = calc_bg_uncertainty(counts, c1, c2, m)
    uncertainty = float(np.sqrt(float(gc) + sigma_bg**2))
    return nc, uncertainty


# ---------------------------------------------------------------------------
# Background sensitivity analysis (Option 3)
# ---------------------------------------------------------------------------

def peak_area_with_background_sensitivity(
    counts: npt.NDArray[Any],
    c1: int,
    c2: int,
) -> Tuple[float, float, dict]:
    """Estimate peak area and systematic uncertainty across all background methods.

    Runs all four background-subtraction methods and reports the mean net
    count together with the standard deviation across methods as a measure of
    systematic sensitivity to the background choice.

    This is not a formal statistical uncertainty but a useful diagnostic: a
    large spread indicates that the result is strongly dependent on the chosen
    background model and should be treated with caution.

    Parameters
    ----------
    counts : numpy.ndarray
        Spectrum counts array.
    c1, c2 : int
        ROI channel boundaries.

    Returns
    -------
    mean_net : float
        Mean net counts averaged over the four background methods.
    std_net : float
        Standard deviation of net counts across the four methods.
    results : dict
        Per-method net counts keyed by :class:`BackgroundMethod` name,
        e.g. ``{'TRAPEZOID': 120.3, 'LINEAR': 118.7, ...}``.
    """
    check_channel_validity(c1, c2, counts)

    method_results: dict = {}
    for method in BackgroundMethod:
        nc = net_counts(counts, c1, c2, method)
        method_results[method.name] = nc

    values = np.array(list(method_results.values()), dtype=float)
    return float(np.mean(values)), float(np.std(values, ddof=0)), method_results


def identify_doublets(
    peaks: Sequence[int],
    ebins: npt.NDArray[Any],
    max_separation: float = 10.0,
) -> List[Tuple[int, int]]:
    """Identifies pairs of adjacent peaks that are close enough to form a doublet.

    Iterates over consecutive pairs of peaks and returns those whose energy
    separation is less than or equal to *max_separation*.

    Parameters
    ----------
    peaks : array-like
        Array of peak channel indices, as returned by :func:`peak_finder` or
        :func:`mariscotti_peak_finder`.  The array is assumed to be sorted in
        ascending channel order.
    ebins : array-like
        Energy bin array mapping channel indices to energy values.
    max_separation : float, optional
        Maximum energy separation between two adjacent peaks for the pair to
        be classified as a doublet, in the same units as *ebins*.
        Default is 10.0.

    Returns
    -------
    list of tuple of int
        List of ``(peak1, peak2)`` channel-index pairs where the two peaks
        are within *max_separation* of each other in energy.
    """
    peaks_array = np.asarray(peaks)
    ebins_array = np.asarray(ebins)

    doublets = []
    for i in range(len(peaks_array) - 1):
        energy1 = ebins_array[peaks_array[i]]
        energy2 = ebins_array[peaks_array[i + 1]]
        if abs(energy2 - energy1) <= max_separation:
            doublets.append((int(peaks_array[i]), int(peaks_array[i + 1])))

    return doublets


def peak_counts(
    peaks: Sequence[int],
    index: int,
    smooth_counts: npt.NDArray[Any],
    ebins: npt.NDArray[Any],
) -> Tuple[int, float]:
    """Index is the peak array index for the peak that counts is required for
    i.e [0], NOT the peak index itself i.e [3210]
    Returns the index of the peak and its calculated net count
    """
    x, y = get_peak_roi(peaks[index], smooth_counts, ebins, offset=10)

    length = len(x)
    start_pos = x[0]
    end_pos = x[length - 1]
    (start,) = np.where(ebins == start_pos)
    (end,) = np.where(ebins == end_pos)

    counts = net_counts(smooth_counts, start[0], end[0], m=1)

    return (peaks[index], counts)


def peak_finder(
    counts: npt.NDArray[Any], prominence: float, wlen: int
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Identifies the peaks and returns their index"""
    sf = five_point_smooth(counts)
    sf2 = five_point_smooth(sf)
    peaks, _ = find_peaks(sf2, prominence=prominence, wlen=wlen)

    return (sf2, peaks)


def mariscotti_peak_finder(
    counts: Union[Sequence[float], npt.NDArray[Any]],
    threshold: Optional[float] = None,
    smooth_iterations: int = 2
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Identifies peaks using the Mariscotti 2nd difference method.

    This method applies smoothing followed by second difference calculation
    to identify peaks. Peaks are identified where the second difference
    is significantly negative (below the threshold).

    Reference: M.A. Mariscotti, Nuclear Instruments and Methods 50 (1967) 309-320

    Parameters
    ----------
    counts : array-like
        The spectrum counts data
    threshold : float, optional
        Threshold value for peak identification. More negative values mean
        the second difference must be more negative to be considered a peak.
        If None (default), automatically set to mean - 1*std of negative
        second differences for better noise rejection.
    smooth_iterations : int, optional
        Number of smoothing iterations to apply. Default is 2.

    Returns
    -------
    tuple of (smoothed_counts, peaks)
        smoothed_counts : numpy array
            The smoothed spectrum after processing
        peaks : numpy array
            Array of indices where peaks were detected
    """
    if len(counts) < 5:
        raise ValueError("Input array must have at least 5 elements for Mariscotti peak finding.")

    counts_array = np.array(counts)

    # Apply smoothing iterations
    smoothed = counts_array.copy()
    for _ in range(smooth_iterations):
        smoothed = five_point_smooth(smoothed)

    # Calculate second difference using vectorized operations
    # Second difference: S''[i] = S[i+1] - 2*S[i] + S[i-1]
    second_diff = np.zeros(len(smoothed))
    second_diff[1:-1] = smoothed[2:] - 2 * smoothed[1:-1] + smoothed[:-2]

    # Auto-calculate threshold if not provided
    if threshold is None:
        # Use only negative second differences for statistics
        negative_diffs = second_diff[second_diff < 0]
        if len(negative_diffs) > 0:
            mean_neg = np.mean(negative_diffs)
            std_neg = np.std(negative_diffs)
            # Set threshold at mean - 1*std for balanced peak detection
            # Factor of 1.0 provides good balance between sensitivity and noise rejection
            AUTO_THRESHOLD_FACTOR = 1.0
            threshold = mean_neg - AUTO_THRESHOLD_FACTOR * std_neg
        else:
            threshold = 0.0

    # Find peaks using vectorized operations
    # A peak is where second_diff[i] < threshold and it's a local minimum
    # Note: Peaks at array boundaries (index 0 or len-1) are not detected
    # as the second difference and local minimum checks require neighbors
    is_below_threshold = second_diff < threshold
    is_local_min = np.zeros(len(second_diff), dtype=bool)
    is_local_min[1:-1] = ((second_diff[1:-1] < second_diff[:-2]) &
                          (second_diff[1:-1] < second_diff[2:]))

    # Peaks are where both conditions are met
    peak_mask = is_below_threshold & is_local_min
    peaks = np.where(peak_mask)[0]

    return (smoothed, peaks)


if __name__ == "__main__":
    pass
