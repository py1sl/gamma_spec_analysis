"""
gamma spectrum analysis - spe file reading
"""

from typing import Dict, List, Optional, Sequence
from ph_spectrum import PhSpectrum
import numpy as np
import numpy.typing as npt
import re


def read_file(path: str) -> List[str]:
    """very boring utility function to read a file and create an
    list with each entry a single line from the file
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    return lines


def validate_dollar_spe_file(lines: List[str]) -> None:
    """check if this is a $ spe file"""
    if not any(line.strip().startswith("$SPEC_ID:") for line in lines):
        raise ValueError("This is not a valid $ spe file")


def read_dollar_spe(path: str) -> PhSpectrum:
    """read an ascii $spe format file"""
    lines = read_file(path)
    validate_dollar_spe_file(lines)
    counts = get_counts(lines)
    live_time = get_live_time(lines)
    real_time = get_real_time(lines)
    energy_fit_coeffs = get_energy_fit_coefficients(lines)
    date = get_start_date(lines)

    spec = PhSpectrum(
        counts=counts,
        live_time=live_time,
        real_time=real_time,
        energy_fit_coefficients=energy_fit_coeffs,
        file_path=path,
        start_time=date,
    )

    return spec


def get_counts(line_data: Sequence[str]) -> npt.NDArray[np.int64]:
    """extracts the counts from the $ spe file"""
    counts: List[str] = []
    for i, line in enumerate(line_data):
        if line.strip().startswith("$DATA:"):
            startpoint = i + 2
            nchannels_line = line_data[i + 1]
            nchannels = nchannels_line.split()[-1]

            counts = line_data[startpoint : (startpoint + 1 + int(nchannels))]
            break  # Exit loop once data is found

    return np.array(counts).astype(int)


def get_live_time(line_data: Sequence[str]) -> Optional[float]:
    """extracts the live time from the $ spe file"""
    for i, line in enumerate(line_data):
        if line.strip().startswith("$MEAS_TIM:"):
            live_time = line_data[i + 1]
            live_time = live_time.split()[0]
            return float(live_time)
    return None


def get_real_time(line_data: Sequence[str]) -> Optional[float]:
    """extracts the real time from the $ spe file"""
    for i, line in enumerate(line_data):
        if line.strip().startswith("$MEAS_TIM:"):
            real_time = line_data[i + 1]
            real_time = real_time.split()[-1]
            return float(real_time)
    return None


def get_start_date(line_data):
    """extract the measurement start date"""
    for i, line in enumerate(line_data):
        if line.strip().startswith("$DATE_MEA:"):
            # TODO convert to appropriate date format
            measurement_date = line_data[i + 1]
            return measurement_date
    return None


def get_energy_fit_coefficients(
    line_data: Sequence[str],
) -> Optional[npt.NDArray[np.float64]]:
    """extracts the energy fit co-efficients from the $ spe file"""
    for i, line in enumerate(line_data):
        if line.strip().startswith("$ENER_FIT:"):
            efit = line_data[i + 1]
            efit = efit.split()
            return np.array(efit).astype(float)
    return None


def get_dollar_keywords(line_data: Sequence[str]) -> Dict[str, list]:
    """Return a mapping of $-keywords to the list of line indices where they occur.

    A $-keyword is detected as a token at the start of a line like:
      $DATA:
      $MEAS_TIM:
      $ENER_FIT:

    Returns:
        dict mapping the keyword (including the leading '$' and without the trailing spaces,
        e.g. '$DATA') to a list of zero-based line indices where that keyword appears.
    """
    pattern = re.compile(r"^\s*(\$\w+)\s*:")
    keywords: Dict[str, list] = {}
    for idx, line in enumerate(line_data):
        m = pattern.match(line)
        if m:
            key = m.group(1)
            keywords.setdefault(key, []).append(idx)
    return keywords
