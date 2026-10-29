"""
gamma spectrum analysis - spe file writing
"""

from typing import List
from ph_spectrum import PhSpectrum
import numpy as np


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

    # --- $SPEC_REM ---
    spec_rem = spec.keywords.get("spec_rem") if spec.keywords else None
    if spec_rem is not None:
        lines.append("$SPEC_REM:")
        lines.extend(spec_rem)

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

    # --- $ROI ---
    roi = spec.keywords.get("roi") if spec.keywords else None
    if roi is not None:
        lines.append("$ROI:")
        lines.append(str(len(roi)))
        lines.extend(f"{int(start)} {int(end)}" for start, end in roi)

    # --- $PRESETS ---
    presets = spec.keywords.get("presets") if spec.keywords else None
    if presets is not None:
        lines.append("$PRESETS:")
        lines.extend(presets)

    # --- $ENER_FIT ---
    if spec.energy_fit_coefficients is not None:
        coeffs = spec.energy_fit_coefficients
        lines.append("$ENER_FIT:")
        lines.append(" ".join(f"{float(c):.6f}" for c in coeffs))

    # --- $MCA_CAL ---
    mca_cal = spec.keywords.get("mca_cal") if spec.keywords else None
    if mca_cal is not None:
        lines.append("$MCA_CAL:")
        lines.append(str(mca_cal["order"]))
        coeff_str = " ".join(_format_spe_float(c) for c in mca_cal["coefficients"])
        unit = mca_cal.get("unit")
        if unit:
            coeff_str += f" {unit}"
        lines.append(coeff_str)

    # --- $SHAPE_CAL ---
    shape_cal = spec.keywords.get("shape_cal") if spec.keywords else None
    if shape_cal is not None:
        lines.append("$SHAPE_CAL:")
        lines.append(str(shape_cal["order"]))
        lines.append(
            " ".join(_format_spe_float(c) for c in shape_cal["coefficients"])
        )

    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\r\n".join(lines) + "\r\n")


def _format_spe_float(value: float) -> str:
    """Format calibration coefficients using the SPE exponent convention."""
    mantissa, exponent = f"{value:.6E}".split("E")
    return f"{mantissa}E{int(exponent):+04d}"
