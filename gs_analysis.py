# -*- coding: utf-8 -*-
"""
gamma spectrum analysis
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import gs_spe_reading
import ph_spectrum

from typing import Optional, Sequence, Tuple, Union, Any
import numpy.typing as npt


def get_spect(path: str) -> "ph_spectrum.PhSpectrum":
    """gets a spectrum
    returns the counts and the ebins
    """
    spec = gs_spe_reading.read_dollar_spe(path)
    return spec


def generate_ebins(spec: "ph_spectrum.PhSpectrum") -> npt.NDArray[Any]:
    """generate the energy bins boundaries from the energy fit co-efficients"""
    e_co_effs = spec.energy_fit_coefficients

    # ensure spec num_channels is set
    if spec.num_channels == 0:
        spec.num_channels = len(spec.counts)

    x = np.arange(spec.num_channels)

    # check validity of the co-efficients array
    if len(e_co_effs) == 2:
        ebins = e_co_effs[0] + x * e_co_effs[1]
    else:
        raise ValueError("The selected energy co-eff array is not valid")

    return ebins


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
    
    counts_array = np.array(counts)
    smooth_spec = np.zeros(len(counts_array))
    
    # Initialize with the first value
    smooth_spec[0] = counts_array[0]
    
    # Apply exponential moving average
    for i in range(1, len(counts_array)):
        smooth_spec[i] = alpha * counts_array[i] + (1 - alpha) * smooth_spec[i - 1]
    
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
    energy: float, eff_coeff: Sequence[float], eff_fit: int = 1
) -> float:
    """Detector efficiency calculation
    energy : Energy to calcuate det eff
    eff_coeff : An array with the coefficients for the energy fit
        the length is not fixed, the length of the array determines the
        number of terms in the expansion
    eff_fit : Determines what type of fit to use
    returns eff - Value of efficiency for the input
                  energy using the selected fitting eqn
    """
    # eff_fit used to choose between calibration fit eqns
    # energy to be in MeV

    if eff_fit not in {1, 2}:
        raise ValueError("The selected eff_fit is not valid")

    log_eff = eff_coeff[0]

    for i in range(1, len(eff_coeff)):
        if eff_fit == 1:
            log_eff += eff_coeff[i] * np.power(np.log(energy), i)
        elif eff_fit == 2:
            log_eff += eff_coeff[i] * np.power(1 / energy, i)

    eff = np.exp(log_eff)

    return eff


def calc_bg(counts: npt.NDArray[Any], c1: int, c2: int, m: int = 1) -> float:
    """Returns background under a peak
    spec is an numpy array of the counts values
    c1 is channel number of the start of peak
    c2 is channel number of the peak end
    m is a selector for different  background calculation methods
    m == 1 is a simple trapezoid background from Maestro
    m == 2 is a linear interpolation method
    m == 3 is a step function method (average of edges)
    m == 4 is a sliding window average method
    """

    if check_channel_validity(c1, c2, counts):
        if m == 1:
            bg = estimate_background_trapezoid(counts, c1, c2)
        elif m == 2:
            bg = estimate_background_linear(counts, c1, c2)
        elif m == 3:
            bg = estimate_background_step(counts, c1, c2)
        elif m == 4:
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
    low_sum = float(sum(counts[low_start:c1])) if c1 > low_start else 0.0

    # Safe high window:
    high_end = min(len(counts), c2 + 2)
    high_sum = float(sum(counts[c2:high_end])) if high_end > c2 else 0.0

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
        return int(sum(counts[c1:c2]))
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


def net_counts(counts: npt.NDArray[Any], c1: int, c2: int, m: int = 1) -> float:
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


def lognormal(
    x: npt.NDArray[Any], a: float, x0: float, sigma: float
) -> npt.NDArray[Any]:
    """Log-normal distribution used for curve fitting
    
    Parameters
    ----------
    x : array
        Input data points
    a : float
        Amplitude (peak height)
    x0 : float
        Location parameter (median of the distribution)
    sigma : float
        Shape parameter (related to standard deviation of log(x))
    
    Returns
    -------
    array
        Log-normal distribution values
    """
    # Avoid log(0) or negative values for x and x0
    x_safe = np.maximum(x, 1e-10)
    x0_safe = max(x0, 1e-10)
    return a * np.exp(-((np.log(x_safe / x0_safe)) ** 2) / (2 * sigma**2))


def weibull(
    x: npt.NDArray[Any], a: float, k: float, lambda_param: float
) -> npt.NDArray[Any]:
    """Weibull distribution used for curve fitting
    
    Parameters
    ----------
    x : array
        Input data points
    a : float
        Amplitude (scale factor for the distribution)
    k : float
        Shape parameter (must be > 0)
    lambda_param : float
        Scale parameter (must be > 0)
    
    Returns
    -------
    array
        Weibull distribution values
    """
    # Ensure x is non-negative for Weibull distribution
    x_safe = np.maximum(x, 0)
    # Avoid division by zero
    lambda_safe = max(lambda_param, 1e-10)
    k_safe = max(k, 1e-10)
    
    return a * (k_safe / lambda_safe) * ((x_safe / lambda_safe) ** (k_safe - 1)) * np.exp(-((x_safe / lambda_safe) ** k_safe))


def polynomial(
    x: npt.NDArray[Any], *coeffs: float
) -> npt.NDArray[Any]:
    """Polynomial function used for curve fitting
    
    Parameters
    ----------
    x : array
        Input data points
    *coeffs : float
        Polynomial coefficients in ascending order (c0 + c1*x + c2*x^2 + ...)
        Example: polynomial(x, 1, 2, 3) returns 1 + 2*x + 3*x^2
    
    Returns
    -------
    array
        Polynomial values
    """
    result = np.zeros_like(x, dtype=float)
    for i, coeff in enumerate(coeffs):
        result += coeff * (x ** i)
    return result


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

    y = counts[peak_pos - offset : peak_pos + offset]
    x = ebins[peak_pos - offset : peak_pos + offset]

    return x, y


def fit_peak(x: npt.NDArray[Any], y: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """fits a peak to a gaussian"""
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))

    popt, pcov = curve_fit(gaussian, x, y, p0=[max(y), mean, sigma], maxfev=10000)

    return popt


def fit_peak_lognormal(x: npt.NDArray[Any], y: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Fits a peak to a log-normal distribution
    
    Parameters
    ----------
    x : array
        X-axis data (energy bins or channels)
    y : array
        Y-axis data (counts)
    
    Returns
    -------
    array
        Optimal parameters [a, x0, sigma] for the log-normal fit
    """
    # Initial parameter estimates
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    
    # Ensure x values are positive for log-normal
    x_positive = np.maximum(x, 1e-10)
    
    # Initial guess: amplitude, location (median), shape parameter
    p0 = [max(y), mean, sigma / mean if mean > 0 else 1.0]
    
    popt, pcov = curve_fit(lognormal, x_positive, y, p0=p0, maxfev=10000)
    
    return popt


def fit_peak_weibull(x: npt.NDArray[Any], y: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Fits a peak to a Weibull distribution
    
    Parameters
    ----------
    x : array
        X-axis data (energy bins or channels)
    y : array
        Y-axis data (counts)
    
    Returns
    -------
    array
        Optimal parameters [a, k, lambda] for the Weibull fit
    """
    # Initial parameter estimates
    mean = sum(x * y) / sum(y)
    
    # Ensure x values are non-negative for Weibull
    x_nonneg = np.maximum(x, 0)
    
    # Initial guess: amplitude, shape parameter, scale parameter
    # k=2 gives Rayleigh-like distribution, lambda related to mean
    p0 = [max(y), 2.0, mean]
    
    popt, pcov = curve_fit(weibull, x_nonneg, y, p0=p0, maxfev=10000)
    
    return popt


def fit_peak_polynomial(
    x: npt.NDArray[Any], y: npt.NDArray[Any], degree: int = 2
) -> npt.NDArray[Any]:
    """Fits data to a polynomial of specified degree
    
    Parameters
    ----------
    x : array
        X-axis data (energy bins or channels)
    y : array
        Y-axis data (counts)
    degree : int, optional
        Degree of the polynomial (default: 2 for quadratic)
    
    Returns
    -------
    array
        Optimal polynomial coefficients in ascending order [c0, c1, c2, ...]
    """
    # Use numpy's polyfit which returns coefficients in descending order
    coeffs_desc = np.polyfit(x, y, degree)
    
    # Reverse to get ascending order for our polynomial function
    coeffs_asc = coeffs_desc[::-1]
    
    return coeffs_asc


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


def plot_spect_peaks(
    smooth_counts: npt.NDArray[Any],
    ebins: npt.NDArray[Any],
    peaks: Sequence[int],
    fname: Optional[str] = None,
) -> None:
    """Plots the spectra and highlights the peaks on the plot"""
    plt.clf()
    for peak in peaks:
        plt.plot(ebins[peak], smooth_counts[peak], "xr")
    plt.plot(ebins, smooth_counts)
    plt.xlabel("ebins")
    plt.ylabel("counts")
    plt.yscale("log")

    if fname:
        plt.savefig(fname)
    else:
        plt.show()


def plot_spec(
    counts: Union[Sequence[int], npt.NDArray[Any]],
    erg: Optional[npt.NDArray[Any]] = None,
    fname: Optional[str] = None,
) -> None:
    """simple plotting routine for spectra"""
    counts = np.array(counts).astype(int)
    plt.clf()

    if erg is None:
        x = np.arange(len(counts))
    else:
        x = erg

    plt.yscale("log")
    plt.step(x, counts)

    if fname:
        plt.savefig(fname)
    else:
        plt.show()


if __name__ == "__main__":
    get_spect()
