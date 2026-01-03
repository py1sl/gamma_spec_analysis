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

    smooth_spec = []

    # first 2 elements
    smooth_spec.extend(counts[:2])

    # smooth middle elements
    for i in range(2, len(counts) - 2):
        val = (1.0 / 9.0) * (
            counts[i - 2]
            + counts[i + 2]
            + (2 * counts[i + 1])
            + (2 * counts[i - 1])
            + (3 * counts[i])
        )
        smooth_spec.append(val)
    # last two elements
    smooth_spec.extend(counts[-2:])

    return np.array(smooth_spec)


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
    m == 1 is a simple trapesium background from Maestro
    """

    if check_channel_validity(c1, c2, counts):
        if m == 1:
            bg = estimate_background_trapezoid(counts, c1, c2)
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
        If None (default), automatically set to mean - 2*std of negative 
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
    
    # Calculate second difference
    # Second difference: S''[i] = S[i+1] - 2*S[i] + S[i-1]
    second_diff = np.zeros(len(smoothed))
    for i in range(1, len(smoothed) - 1):
        second_diff[i] = smoothed[i + 1] - 2 * smoothed[i] + smoothed[i - 1]
    
    # Auto-calculate threshold if not provided
    if threshold is None:
        # Use only negative second differences for statistics
        negative_diffs = second_diff[second_diff < 0]
        if len(negative_diffs) > 0:
            mean_neg = np.mean(negative_diffs)
            std_neg = np.std(negative_diffs)
            # Set threshold at mean - 1*std (less strict for better peak detection)
            threshold = mean_neg - 1 * std_neg
        else:
            threshold = 0.0
    
    # Find peaks where second difference is negative (below threshold)
    # A peak has a negative second difference
    peak_candidates = []
    for i in range(1, len(second_diff) - 1):
        # Peak at position i if second_diff[i] is below threshold
        # and it's a local minimum in the second difference
        if (second_diff[i] < threshold and 
            second_diff[i] < second_diff[i - 1] and 
            second_diff[i] < second_diff[i + 1]):
            peak_candidates.append(i)
    
    peaks = np.array(peak_candidates, dtype=int)
    
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
