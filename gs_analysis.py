# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 09:34:37 2020

@author: gai72996
 gamma spectrum analysis
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def read_file(path):
    """very boring utility function to read a file and create an
    list with each entry a single line from the file
    warning: do not use with very large files (Gb +)
    """
    with open(path) as f:
        lines = f.read().splitlines()
    f.close()

    return lines


def plot_spec(counts, erg=None, peaks=None):
    """simple plotting routine for spectra"""
    counts = np.array(counts).astype(int)
    plt.clf()

    if erg is None:
        x = np.arange(len(counts))
    else:
        x = erg

    """
    if peaks is not None:
        for xc in peaks:
            plt.axvline(x=xc, color='r')
    """

    plt.yscale("log")
    plt.step(x, counts)
    plt.show


def get_counts(line_data):
    """extracts the counts from the $ spe file"""
    counts = []
    for i, line in enumerate(line_data):
        if line == "$DATA:":
            startpoint = i + 2

            nchannels = line_data[i + 1]
            nchannels = nchannels.split()[-1]

            counts = line_data[startpoint : (startpoint + 1 + int(nchannels))]

    counts = np.array(counts).astype(int)
    return counts


def get_live_time(line_data):
    """extracts the live time from the $ spe file"""
    for i, line in enumerate(line_data):
        if line == "$MEAS_TIM:":
            live_time = line_data[i + 1]
            live_time = live_time.split()[0]
    return float(live_time)


def get_real_time(line_data):
    """extracts the real time from the $ spe file"""
    for i, line in enumerate(line_data):
        if line == "$MEAS_TIM:":
            real_time = line_data[i + 1]
            real_time = real_time.split()[-1]
    return float(real_time)


def get_e_fit(line_data):
    """extracts the energy fit co-efficients from the $ spe file"""
    for i, line in enumerate(line_data):
        if line == "$ENER_FIT:":
            efit = line_data[i + 1]
            efit = efit.split()
    return np.array(efit).astype(float)


def generate_ebins(lines, nbins):
    """makes the ebins from the energy fit co-efficients"""
    e_co_effs = get_e_fit(lines)

    x = np.arange(nbins)
    ebins = e_co_effs[0] + x * e_co_effs[1]

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

    return smooth_spec


def find_energy_pos(ebins, erg):
    """given an energy , erg, finds the position of the ebin that falls
    into
    ebins is numpy array of energy bins
    erg is energy value in same units as ebins (usually keV)
    """
    for i, energy in enumerate(ebins):
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
        eff = 0

    return eff


def calc_bg(spec, c1, c2, m=1):
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
    if c2 > len(spec):
        raise ValueError("c2 must be less than max number of channels")

    if m == 1:
        low_sum = sum(spec[c1 - 2 : c1])
        high_sum = sum(spec[c2 : c2 + 2])
        bg = (low_sum + high_sum) * ((c2 - c1 + 1) / 6)
    else:
        raise ValueError("m is not set to a valud method id")

    return bg


def gross_count(spec, c1, c2):
    """Returns total number of counts in a spectrum between two channels"""

    # check channels are appropraite
    if c1 > c2:
        raise ValueError("c1 must be less than c2")
    if c1 < 0:
        raise ValueError("c1 must be positive number above 0")
    if c2 > len(spec):
        raise ValueError("c2 must be less than max number of channels")

    gc = sum(spec[c1:c2])
    return gc


def net_counts(spec, c1, c2, m=1):
    """Calculates net counts between two channels"""
    bg = calc_bg(spec, c1, c2, m)
    gc = gross_count(spec, c1, c2)
    nc = gc - bg
    return nc


def gaussian(x, a, x0, sigma):
    """gaussian used for curve fitting"""
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))


def get_peak_roi(peak_pos, counts, ebins, offset=10):
    """extracts a region of the spectra around the peak_pos
    number of channels extracted is 2 x offset
    returns both the counts and energy bin values for that region
    """
    y = counts[peak_pos - offset : peak_pos + offset]
    x = ebins[peak_pos - offset : peak_pos + offset]

    return x, y


def fit_peak(x, y):
    """fits a peak to a gaussian"""
    mean = sum(x * y) / sum(y)
    sigma = sum(y * (x - mean) ** 2) / sum(y)
    popt, pcov = curve_fit(gaussian, x, y, p0=[1, mean, sigma])

    return popt


def get_spect(path):
    """gets  a spectrum
    returns the counts and the ebins
    """

    lines = read_file(path)
    counts = get_counts(lines)
    ebins = generate_ebins(lines, len(counts))

    return counts, ebins


if __name__ == "__main__":
    get_spect()
