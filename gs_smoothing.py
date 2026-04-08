# -*- coding: utf-8 -*-
"""
Smoothing functions for gamma spectra.
"""

import numpy as np
from scipy.signal import lfilter, lfilter_zi

from typing import Sequence, Union
import numpy.typing as npt
from typing import Any


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
