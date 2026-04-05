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


def write_dollar_spe(spec: PhSpectrum, path: str) -> None:
    """Write a :class:`PhSpectrum` to an ASCII ``$`` SPE format file.

    The output follows the standard Ortec/Maestro ``$`` SPE layout:

    * ``$SPEC_ID:`` – :attr:`~PhSpectrum.spec_name` (blank line if empty).
    * ``$DATE_MEA:`` – :attr:`~PhSpectrum.start_time` (omitted when ``None``).
    * ``$MEAS_TIM:`` – ``<live_time> <real_time>`` (omitted when both are
      ``None``).
    * ``$DATA:`` – channel range followed by one count per line, right-justified
      in an 8-character field.
    * ``$ENER_FIT:`` – two energy calibration coefficients (omitted when
      :attr:`~PhSpectrum.energy_fit_coefficients` is ``None``).
    * ``$MCA_CAL:`` – polynomial calibration block from
      ``spec.keywords['mca_cal']`` (omitted when absent).
    * ``$SHAPE_CAL:`` – shape calibration block from
      ``spec.keywords['shape_cal']`` (omitted when absent).

    Parameters
    ----------
    spec:
        The spectrum to serialise.
    path:
        Destination file path.  The file is created or overwritten.
    """
    lines: List[str] = []

    # --- $SPEC_ID ---
    lines.append("$SPEC_ID:")
    lines.append(spec.spec_name if spec.spec_name else "")

    # --- $DATE_MEA ---
    if spec.start_time is not None:
        lines.append("$DATE_MEA:")
        lines.append(str(spec.start_time))

    # --- $MEAS_TIM ---
    if spec.live_time is not None or spec.real_time is not None:
        live = spec.live_time if spec.live_time is not None else 0
        real = spec.real_time if spec.real_time is not None else 0
        lines.append("$MEAS_TIM:")
        lines.append(f"{int(live)} {int(real)}")

    # --- $DATA ---
    counts = np.asarray(spec.counts, dtype=np.int64)
    num_channels = int(counts.size)
    start_chan = int(spec.start_chan_num) if spec.start_chan_num else 0
    end_chan = start_chan + num_channels - 1
    lines.append("$DATA:")
    lines.append(f"{start_chan} {end_chan}")
    for c in counts:
        lines.append(f"{int(c):8d}")

    # --- $ENER_FIT ---
    if spec.energy_fit_coefficients is not None:
        coeffs = spec.energy_fit_coefficients
        lines.append("$ENER_FIT:")
        lines.append(" ".join(str(c) for c in coeffs))

    # --- $MCA_CAL ---
    mca_cal = spec.keywords.get("mca_cal") if spec.keywords else None
    if mca_cal is not None:
        lines.append("$MCA_CAL:")
        lines.append(str(mca_cal["order"]))
        coeff_str = " ".join(f"{c:E}" for c in mca_cal["coefficients"])
        unit = mca_cal.get("unit")
        if unit:
            coeff_str += f" {unit}"
        lines.append(coeff_str)

    # --- $SHAPE_CAL ---
    shape_cal = spec.keywords.get("shape_cal") if spec.keywords else None
    if shape_cal is not None:
        lines.append("$SHAPE_CAL:")
        lines.append(str(shape_cal["order"]))
        lines.append(" ".join(f"{c:E}" for c in shape_cal["coefficients"]))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


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
