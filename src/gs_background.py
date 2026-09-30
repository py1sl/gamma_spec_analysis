# -*- coding: utf-8 -*-
"""
Background estimation functions for gamma spectrum analysis.
"""

import numpy as np
from enum import IntEnum

from typing import Any
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


def gross_count(counts: npt.NDArray[Any], c1: int, c2: int) -> int:
    """Returns total number of counts in a spectrum between two channels"""
    if check_channel_validity(c1, c2, counts):
        return int(np.sum(counts[c1:c2]))
    return 0


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
