# -*- coding: utf-8 -*-
"""
gamma spectrum analysis
"""

import numpy as np
from scipy.signal import find_peaks
from enum import IntEnum

import ph_spectrum

from typing import List, Optional, Sequence, Tuple, Union, Any
import numpy.typing as npt

# Re-export sub-module symbols for backward compatibility
from gs_smoothing import (  # noqa: F401
    five_point_smooth,
    three_point_smooth,
    moving_average,
    exponential_moving_average,
)
from gs_background import (  # noqa: F401
    BackgroundMethod,
    check_channel_validity,
    gross_count,
    estimate_background_trapezoid,
    estimate_background_linear,
    estimate_background_step,
    estimate_background_sliding_average,
    calc_bg,
    estimate_background_trapezoid_uncertainty,
    estimate_background_linear_uncertainty,
    estimate_background_step_uncertainty,
    estimate_background_sliding_average_uncertainty,
    calc_bg_uncertainty,
)
from gs_peak_fitting import (  # noqa: F401
    net_counts,
    gaussian,
    double_gaussian,
    get_peak_roi,
    fit_peak,
    fit_doublet,
    gaussian_area,
    gaussian_area_uncertainty,
    fit_peak_area,
    fit_doublet_areas,
    peak_fwhm,
    peak_fwhm_uncertainty,
    fit_peak_fwhm,
    fit_chi2,
    fit_peak_chi2,
    fit_doublet_chi2,
    net_counts_uncertainty,
    peak_area_with_background_sensitivity,
)


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
