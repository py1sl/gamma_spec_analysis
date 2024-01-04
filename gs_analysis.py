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


def plot_spec(counts, erg=None, fname=None):
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


def generate_ebins(spec):
    """makes the ebins from the energy fit co-efficients"""
    e_co_effs = spec.efit_co_eff
    if spec.num_channels == 0:
        spec.num_channels = len(spec.counts)
    x = np.arange(spec.num_channels)
    if len(e_co_effs) == 2:
        ebins = e_co_effs[0] + x * e_co_effs[1]
    else:
        raise ValueError("The selected energy co-eff array is not valid")

    return ebins


def five_point_smooth(counts):
    """5 point smoothing function.
    Recommended for use in low statistics in
    G.W. Phillips , Nucl. Instrum. Methods 153 (1978), 449
    Parameters
    ----------
    """
    smooth_spec = []
    smooth_spec.append(counts[0])
    smooth_spec.append(counts[1])

    spec_len = len(counts)
    i = 2
    while i < spec_len - 2:
        val = (1.0 / 9.0) * (
            counts[i - 2]
            + counts[i + 2]
            + (2 * counts[i + 1])
            + (2 * counts[i - 1])
            + (3 * counts[i])
        )
        smooth_spec.append(val)
        i = i + 1
    smooth_spec.append(counts[i])
    smooth_spec.append(counts[i + 1])

    return np.array(smooth_spec)


def find_energy_pos(ebins, erg):
    """given an energy , erg, finds the position of the ebin that falls
    into
    ebins is numpy array of energy bins
    erg is energy value in same units as ebins (usually keV)
    """
    for i, energy in enumerate(ebins[:-1]):
        if erg >= energy and erg < ebins[i + 1]:
            return i

    return False


def calc_e_eff(energy, eff_coeff, eff_fit=1):
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

    if eff_fit == 1:
        # eff_fit 1 uses series ao + a1(lnE)^1+ a2(lnE)^2+ ....
        log_eff = eff_coeff[0]
        i = 1
        while i < len(eff_coeff):
            log_eff = log_eff + (eff_coeff[i] * (np.power(np.log(energy), i)))
            i = i + 1
        eff = np.exp(log_eff)
    elif eff_fit == 2:
        # eff_fit 2 uses series a0 + a1(1/E)^1 + a2(1/E)^2+...
        log_eff = eff_coeff[0]
        i = 1
        while i < len(eff_coeff):
            log_eff = log_eff + (eff_coeff[i] * ((1 / energy) ** i))
            i = i + 1
        eff = np.exp(log_eff)
    else:
        raise ValueError("The selected eff_fit is not valid")

    return eff


def calc_bg(counts, c1, c2, m=1):
    """Returns background under a peak
    spec is an numpy array of the counts values
    c1 is channel number of the start of peak
    c2 is channel number of the peak end
    m is a selector for different  background calculation methods
    m == 1 is a simple trapesium background from Maestro
    """

    # check channels are appropraite
    if c1 > c2:
        raise ValueError("c1 must be less than c2")
    if c1 < 0:
        raise ValueError("c1 must be positive number above 0")
    if c2 > len(counts):
        raise ValueError("c2 must be less than max number of channels")

    if m == 1:
        low_sum = sum(counts[c1 - 2 : c1])
        high_sum = sum(counts[c2 : c2 + 2])
        bg = (low_sum + high_sum) * ((c2 - c1 + 1) / 6)
    else:
        raise ValueError("m is not set to a valid method id")

    return bg


def gross_count(counts, c1, c2):
    """Returns total number of counts in a spectrum between two channels"""

    # check channels are appropraite
    if c1 > c2:
        raise ValueError("c1 must be less than c2")
    if c1 < 0:
        raise ValueError("c1 must be positive number above 0")
    if c2 > len(counts):
        raise ValueError("c2 must be less than max number of channels")

    gc = sum(counts[c1:c2])
    return gc


def net_counts(counts, c1, c2, m=1):
    """Calculates net counts between two channels"""
    bg = calc_bg(counts, c1, c2, m)
    gc = gross_count(counts, c1, c2)
    nc = gc - bg
    return nc


def gaussian(x, a, x0, sigma):
    """gaussian used for curve fitting"""
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def get_peak_roi(peak_pos, counts, ebins, offset=10):
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


def fit_peak(x, y):
    """fits a peak to a gaussian"""
    mean = sum(x * y) / sum(y)
    sigma = sum(y * (x - mean) ** 2) / sum(y)
    popt, pcov = curve_fit(gaussian, x, y, p0=[1, mean, sigma], maxfev=10000)

    return popt


def get_spect(path):
    """gets  a spectrum
    returns the counts and the ebins
    """
    spec = gs_spe_reading.read_dollar_spe(path)

    return spec


def peak_finder(x, prominence, wlen):
    """Identifies the peaks and returns their index"""
    sf = five_point_smooth(x)
    sf2 = five_point_smooth(sf)
    peaks, _ = find_peaks(sf2, prominence=prominence, wlen=wlen)

    return (sf2, peaks)


def plot_spect_peaks(smooth_counts, ebins, peaks, fname=None):
    """Plots the spectra and highlights the peaks on the plot"""
    plt.clf()
    plt.plot(ebins[peaks], smooth_counts[peaks], "xr")
    plt.plot(ebins, smooth_counts)
    plt.xlabel("ebins")
    plt.ylabel("counts")
    plt.yscale("log")

    if fname:
        plt.savefig(fname)
    else:
        plt.show()


def peak_counts(peaks, index, smooth_counts, ebins):
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


if __name__ == "__main__":
    get_spect()
