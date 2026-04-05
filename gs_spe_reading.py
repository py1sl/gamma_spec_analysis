"""
gamma spectrum analysis - spe file reading
"""

from typing import Any, Dict, List, Optional, Sequence
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

    # Get all keywords and their line indices in a single pass
    keywords_map = get_dollar_keywords(lines)

    counts = get_counts(lines, keywords_map)
    live_time = get_live_time(lines, keywords_map)
    real_time = get_real_time(lines, keywords_map)
    energy_fit_coeffs = get_energy_fit_coefficients(lines, keywords_map)
    date = get_start_date(lines, keywords_map)
    mca_cal = get_mca_cal(lines, keywords_map)
    shape_cal = get_shape_cal(lines, keywords_map)

    # Build keywords dictionary with optional calibration data
    keywords = {}
    if mca_cal is not None:
        keywords['mca_cal'] = mca_cal
    if shape_cal is not None:
        keywords['shape_cal'] = shape_cal

    spec = PhSpectrum(
        counts=counts,
        live_time=live_time,
        real_time=real_time,
        energy_fit_coefficients=energy_fit_coeffs,
        file_path=path,
        start_time=date,
        keywords=keywords,
    )

    return spec


def get_counts(line_data: Sequence[str], keywords_map: Dict[str, list]) -> npt.NDArray[np.int64]:
    """extracts the counts from the $ spe file"""
    counts: List[str] = []

    if "$DATA" in keywords_map and keywords_map["$DATA"]:
        i = keywords_map["$DATA"][0]  # Use first occurrence
        startpoint = i + 2
        nchannels_line = line_data[i + 1]
        nchannels = nchannels_line.split()[-1]
        counts = line_data[startpoint:(startpoint + 1 + int(nchannels))]

    return np.array(counts).astype(int)


def get_live_time(line_data: Sequence[str], keywords_map: Dict[str, list]) -> Optional[float]:
    """extracts the live time from the $ spe file"""
    if "$MEAS_TIM" in keywords_map and keywords_map["$MEAS_TIM"]:
        i = keywords_map["$MEAS_TIM"][0]  # Use first occurrence
        live_time = line_data[i + 1]
        live_time = live_time.split()[0]
        return float(live_time)
    return None


def get_real_time(line_data: Sequence[str], keywords_map: Dict[str, list]) -> Optional[float]:
    """extracts the real time from the $ spe file"""
    if "$MEAS_TIM" in keywords_map and keywords_map["$MEAS_TIM"]:
        i = keywords_map["$MEAS_TIM"][0]  # Use first occurrence
        real_time = line_data[i + 1]
        real_time = real_time.split()[-1]
        return float(real_time)
    return None


def get_start_date(line_data, keywords_map: Dict[str, list]):
    """extract the measurement start date"""
    if "$DATE_MEA" in keywords_map and keywords_map["$DATE_MEA"]:
        i = keywords_map["$DATE_MEA"][0]  # Use first occurrence
        # TODO convert to appropriate date format
        measurement_date = line_data[i + 1]
        return measurement_date
    return None


def get_energy_fit_coefficients(
    line_data: Sequence[str],
    keywords_map: Dict[str, list],
) -> Optional[npt.NDArray[np.float64]]:
    """extracts the energy fit co-efficients from the $ spe file"""
    if "$ENER_FIT" in keywords_map and keywords_map["$ENER_FIT"]:
        i = keywords_map["$ENER_FIT"][0]  # Use first occurrence
        efit = line_data[i + 1]
        efit = efit.split()
        return np.array(efit).astype(float)
    return None


def get_mca_cal(line_data: Sequence[str], keywords_map: Dict[str, list]) -> Optional[Dict[str, Any]]:
    """extracts the MCA calibration data from the $ spe file

    Returns a dictionary with the calibration data:
    - 'order': the order of the calibration polynomial (int)
    - 'coefficients': the calibration coefficients (list of floats)
    - 'unit': the unit string (e.g., 'keV')
    """
    if "$MCA_CAL" in keywords_map and keywords_map["$MCA_CAL"]:
        i = keywords_map["$MCA_CAL"][0]  # Use first occurrence
        try:
            order = int(line_data[i + 1].strip())
            coeff_line = line_data[i + 2].strip()
            parts = coeff_line.split()
            # Extract coefficients (all numeric values) and unit (any non-numeric value)
            # If multiple non-numeric values exist, the last one is kept as the unit
            coefficients = []
            unit = None
            for part in parts:
                try:
                    coefficients.append(float(part))
                except ValueError:
                    # This is a non-numeric value, capture it as the unit
                    unit = part

            return {
                'order': order,
                'coefficients': coefficients,
                'unit': unit
            }
        except (IndexError, ValueError):
            return None
    return None


def get_shape_cal(line_data: Sequence[str], keywords_map: Dict[str, list]) -> Optional[Dict[str, Any]]:
    """extracts the shape calibration data from the $ spe file

    Returns a dictionary with the calibration data:
    - 'order': the order of the calibration polynomial (int)
    - 'coefficients': the calibration coefficients (list of floats)
    """
    if "$SHAPE_CAL" in keywords_map and keywords_map["$SHAPE_CAL"]:
        i = keywords_map["$SHAPE_CAL"][0]  # Use first occurrence
        try:
            order = int(line_data[i + 1].strip())
            coeff_line = line_data[i + 2].strip()
            parts = coeff_line.split()
            # Extract only numeric values as coefficients
            coefficients = []
            for part in parts:
                try:
                    coefficients.append(float(part))
                except ValueError:
                    # Skip non-numeric values
                    pass

            return {
                'order': order,
                'coefficients': coefficients
            }
        except (IndexError, ValueError):
            return None
    return None


def validate_free_text_spe_file(lines: List[str]) -> None:
    """check if this is a free-text (colon-delimited) spe file"""
    has_spectrum = any(line.strip() == "SPECTRUM" for line in lines)
    has_real_time = any(line.strip().startswith("Real Time:") for line in lines)
    if not (has_spectrum and has_real_time):
        raise ValueError("This is not a valid free-text spe file")


def read_free_text_spe(path: str) -> PhSpectrum:
    """read a free-text colon-delimited spe format file (e.g. 93_test.spe)

    This format uses plain-text section headers and ``Key:  value`` pairs
    rather than the ``$KEYWORD:`` convention used by :func:`read_dollar_spe`.
    The SPECTRUM section contains lines of the form::

        <channel_number>:    <val0>    <val1>    <val2>    <val3>

    where each line holds four counts (in scientific notation) preceded by the
    starting channel index for that group.
    """
    lines = read_file(path)
    validate_free_text_spe_file(lines)

    counts = get_free_text_counts(lines)
    live_time = get_free_text_live_time(lines)
    real_time = get_free_text_real_time(lines)
    energy_fit_coeffs = get_free_text_energy_fit(lines)
    start_time = get_free_text_start_time(lines)
    spec_name = get_free_text_spec_name(lines)
    start_chan = get_free_text_start_channel(lines)

    spec = PhSpectrum(
        counts=counts,
        live_time=live_time,
        real_time=real_time,
        energy_fit_coefficients=energy_fit_coeffs,
        file_path=path,
        start_time=start_time,
        spec_name=spec_name if spec_name is not None else "",
        start_chan_num=start_chan if start_chan is not None else 0,
    )
    return spec


def _get_free_text_field(lines: List[str], key: str) -> Optional[str]:
    """Return the value part of the first ``key:  value`` line, stripped."""
    prefix = key + ":"
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped[len(prefix):].strip()
    return None


def get_free_text_live_time(lines: List[str]) -> Optional[float]:
    """extract live time from free-text spe"""
    val = _get_free_text_field(lines, "Live Time")
    return float(val) if val else None


def get_free_text_real_time(lines: List[str]) -> Optional[float]:
    """extract real time from free-text spe"""
    val = _get_free_text_field(lines, "Real Time")
    return float(val) if val else None


def get_free_text_start_time(lines: List[str]) -> Optional[str]:
    """extract acquisition start date/time from free-text spe"""
    date = _get_free_text_field(lines, "Acquisition start date")
    time = _get_free_text_field(lines, "Acquisition start time")
    if date and time:
        return f"{date} {time}"
    if date:
        return date
    return None


def get_free_text_spec_name(lines: List[str]) -> Optional[str]:
    """extract spectrum name from free-text spe"""
    return _get_free_text_field(lines, "Spectrum name")


def get_free_text_start_channel(lines: List[str]) -> Optional[int]:
    """extract starting channel number from free-text spe"""
    val = _get_free_text_field(lines, "Starting channel number")
    return int(val) if val is not None else None


def get_free_text_energy_fit(lines: List[str]) -> Optional[npt.NDArray[np.float64]]:
    """extract energy calibration coefficients from free-text spe

    The ``Energy Fit:`` line contains two or three space-separated floats.
    Only the first two (offset and slope) are returned to match the
    convention used by :func:`get_energy_fit_coefficients`.
    """
    val = _get_free_text_field(lines, "Energy Fit")
    if val is None:
        return None
    parts = val.split()
    coeffs = np.array(parts, dtype=np.float64)
    # Return only intercept and slope (first two coefficients)
    return coeffs[:2]


def get_free_text_counts(lines: List[str]) -> npt.NDArray[np.int64]:
    """extract spectrum counts from free-text spe

    Spectrum lines follow the pattern::

        <channel_num>:    <val0>    <val1>    <val2>    <val3>

    where values are in scientific notation.  The function locates the
    ``SPECTRUM`` header and reads all subsequent matching lines.
    """
    spectrum_pattern = re.compile(r"^\s*(\d+):\s+(.*)")
    in_spectrum = False
    # Values are in scientific notation so we accumulate as floats, then
    # convert to int64 (truncating any fractional part) before returning.
    raw_counts: List[float] = []
    for line in lines:
        if line.strip() == "SPECTRUM":
            in_spectrum = True
            continue
        if in_spectrum:
            m = spectrum_pattern.match(line)
            if m:
                values = m.group(2).split()
                raw_counts.extend(float(v) for v in values)
    return np.array(raw_counts, dtype=np.int64)


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
