"""
 gamma spectrum analysis - spe file reading
"""
from ph_spectrum import PhSpectrum
import numpy as np


def read_file(path):
    """very boring utility function to read a file and create an
    list with each entry a single line from the file
    warning: do not use with very large files (Gb +)
    """
    with open(path) as f:
        lines = f.read().splitlines()
    f.close()

    return lines


def check_file_type(path):
    """check which type of format file"""
    lines = read_file(path)


def read_dollar_spe(path):
    """read an ascii $spe format file"""
    lines = read_file(path)
    counts = get_counts(lines)
    live_time = get_live_time(lines)
    real_time = get_real_time(lines)
    e_fit_co_eff = get_e_fit(lines)

    spec = PhSpectrum(
        counts=counts,
        live_time=live_time,
        real_time=real_time,
        efit_co_eff=e_fit_co_eff,
        file_path=path,
    )

    return spec


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
        else:
            continue
    return None
