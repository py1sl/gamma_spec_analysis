# -*- coding: utf-8 -*-
"""
Multi-spectrum analysis utilities.

This module provides tools for working with a series of spectra acquired
one after another, enabling:

* **Spectrum co-addition** – sum a list of :class:`~ph_spectrum.PhSpectrum`
  objects to improve counting statistics.
* **Peak tracking** – extract the net count rate of a specific peak in each
  spectrum of a series, returning the rates together with elapsed times.
* **Half-life estimation** – fit an exponential decay model to tracked peak
  activities and return the half-life together with its uncertainty.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit

import gs_analysis
import ph_spectrum
from gs_analysis import BackgroundMethod

# Date format used in ORTEC .Spe files: "MM/DD/YYYY HH:MM:SS"
_SPE_DATE_FMT = "%m/%d/%Y %H:%M:%S"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_start_time(start_time: Optional[str]) -> Optional[datetime]:
    """Parse a start-time string from a .Spe file into a :class:`datetime`.

    Returns ``None`` when *start_time* is ``None`` or cannot be parsed.
    """
    if start_time is None:
        return None
    try:
        return datetime.strptime(start_time.strip(), _SPE_DATE_FMT)
    except ValueError:
        return None


def _exponential_decay(t: npt.NDArray[np.float64], a0: float, lam: float) -> npt.NDArray[np.float64]:
    """Exponential decay model: ``A(t) = a0 * exp(-lam * t)``."""
    return a0 * np.exp(-lam * t)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_spectra(spectra: List["ph_spectrum.PhSpectrum"]) -> "ph_spectrum.PhSpectrum":
    """Sum a list of spectra to produce a single co-added spectrum.

    All spectra must have the same number of channels.  The counts arrays are
    summed element-wise; ``live_time`` and ``real_time`` are summed when all
    spectra carry those values.  Energy calibration coefficients are taken
    from the first spectrum in the list.

    Parameters
    ----------
    spectra:
        A list of :class:`~ph_spectrum.PhSpectrum` objects to add together.
        Must contain at least one spectrum.

    Returns
    -------
    :class:`~ph_spectrum.PhSpectrum`
        A new spectrum whose counts equal the element-wise sum of the input
        spectra.

    Raises
    ------
    ValueError
        If *spectra* is empty, or if the spectra have different channel counts.
    """
    if len(spectra) == 0:
        raise ValueError("spectra list must not be empty")

    n_channels = spectra[0].num_channels
    for i, spec in enumerate(spectra[1:], start=1):
        if spec.num_channels != n_channels:
            raise ValueError(
                f"All spectra must have the same number of channels; "
                f"spectrum 0 has {n_channels}, spectrum {i} has {spec.num_channels}"
            )

    summed_counts = np.zeros(n_channels, dtype=np.int64)
    for spec in spectra:
        summed_counts += np.asarray(spec.counts, dtype=np.int64)

    # Sum timing information when all spectra carry it
    if all(s.live_time is not None for s in spectra):
        total_live: Optional[float] = float(sum(s.live_time for s in spectra))  # type: ignore[arg-type]
    else:
        total_live = None

    if all(s.real_time is not None for s in spectra):
        total_real: Optional[float] = float(sum(s.real_time for s in spectra))  # type: ignore[arg-type]
    else:
        total_real = None

    first = spectra[0]
    return ph_spectrum.PhSpectrum(
        spec_name=first.spec_name,
        counts=summed_counts,
        live_time=total_live,
        real_time=total_real,
        energy_fit_coefficients=first.energy_fit_coefficients,
        efficiency_fit_coefficients=first.efficiency_fit_coefficients,
        start_time=first.start_time,
    )


def get_elapsed_times(spectra: List["ph_spectrum.PhSpectrum"]) -> npt.NDArray[np.float64]:
    """Return elapsed times (in seconds) relative to the first spectrum.

    The function first attempts to derive elapsed times from the ``start_time``
    field of each spectrum (format ``"MM/DD/YYYY HH:MM:SS"``).  If any
    ``start_time`` is missing or unparseable the function falls back to
    accumulating ``real_time`` values: the elapsed time of the *k*-th spectrum
    is the sum of ``real_time`` for spectra 0 through *k*-1.

    Parameters
    ----------
    spectra:
        Ordered list of :class:`~ph_spectrum.PhSpectrum` objects.

    Returns
    -------
    numpy.ndarray of float64
        Array of elapsed-time values (seconds) with the same length as
        *spectra*.  The first element is always ``0.0``.

    Raises
    ------
    ValueError
        If *spectra* is empty.
    ValueError
        If ``start_time`` timestamps are not monotonically non-decreasing.
    ValueError
        If the fall-back ``real_time`` is ``None`` for any spectrum when
        ``start_time`` is unavailable.
    """
    if len(spectra) == 0:
        raise ValueError("spectra list must not be empty")

    # Attempt timestamp-based elapsed times
    datetimes = [_parse_start_time(s.start_time) for s in spectra]
    if all(dt is not None for dt in datetimes):
        t0 = datetimes[0]
        elapsed = np.array(
            [(dt - t0).total_seconds() for dt in datetimes],
            dtype=np.float64,
        )
        if np.any(np.diff(elapsed) < 0):
            raise ValueError(
                "start_time values are not monotonically non-decreasing; "
                "check the order of the spectra list"
            )
        return elapsed

    # Fall back: accumulate real_time
    if any(s.real_time is None for s in spectra):
        raise ValueError(
            "Cannot determine elapsed times: start_time is not available for "
            "all spectra and at least one spectrum is missing real_time"
        )

    elapsed_list: List[float] = [0.0]
    cumulative = 0.0
    for spec in spectra[:-1]:
        cumulative += float(spec.real_time)  # type: ignore[arg-type]
        elapsed_list.append(cumulative)

    return np.array(elapsed_list, dtype=np.float64)


def track_peak_activity(
    spectra: List["ph_spectrum.PhSpectrum"],
    energy: float,
    energy_tolerance: float = 5.0,
    bg_method: BackgroundMethod = BackgroundMethod.TRAPEZOID,
    roi_offset: int = 10,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Track a peak across a series of spectra and return count rates.

    For each spectrum the function:

    1. Converts *energy* to a channel index using the spectrum's energy
       calibration coefficients.
    2. Extracts a region-of-interest (ROI) around that channel.
    3. Computes the net count rate as ``net_counts / live_time``.

    Parameters
    ----------
    spectra:
        Ordered list of :class:`~ph_spectrum.PhSpectrum` objects.  All
        spectra must have ``energy_fit_coefficients`` and ``live_time`` set.
    energy:
        Nominal peak energy (keV) to track.
    energy_tolerance:
        Maximum allowed difference (keV) between *energy* and the nearest
        energy-bin value.  Used to guard against misidentified peaks.
        Default is ``5.0`` keV.
    bg_method:
        Background subtraction method to use when computing net counts.
        Default is :attr:`~gs_analysis.BackgroundMethod.TRAPEZOID`.
    roi_offset:
        Half-width (channels) of the ROI extracted around the peak.
        Default is ``10``.

    Returns
    -------
    elapsed_times : numpy.ndarray of float64
        Elapsed time (seconds) of each spectrum relative to the first.
    count_rates : numpy.ndarray of float64
        Net count rate (counts / second) for the peak in each spectrum.
        A ``NaN`` entry indicates that the peak could not be located or the
        count rate could not be computed for that spectrum.

    Raises
    ------
    ValueError
        If *spectra* is empty.
    ValueError
        If any spectrum is missing ``energy_fit_coefficients`` or ``live_time``.
    """
    if len(spectra) == 0:
        raise ValueError("spectra list must not be empty")

    for i, spec in enumerate(spectra):
        if spec.energy_fit_coefficients is None:
            raise ValueError(
                f"Spectrum {i} is missing energy_fit_coefficients; "
                "energy calibration is required for peak tracking"
            )
        if spec.live_time is None:
            raise ValueError(
                f"Spectrum {i} is missing live_time; "
                "live_time is required to compute a count rate"
            )

    elapsed_times = get_elapsed_times(spectra)
    count_rates = np.full(len(spectra), np.nan, dtype=np.float64)

    for idx, spec in enumerate(spectra):
        try:
            ebins = gs_analysis.generate_ebins(spec)
        except Exception:
            continue

        # Find the channel closest to the requested energy
        chan = gs_analysis.find_energy_pos(ebins, energy)
        if chan is None:
            continue

        # Guard against large energy mismatches
        if abs(ebins[chan] - energy) > energy_tolerance:
            continue

        smooth_counts = gs_analysis.five_point_smooth(spec.counts)

        try:
            x, y = gs_analysis.get_peak_roi(chan, smooth_counts, ebins, offset=roi_offset)
        except ValueError:
            continue

        # Determine ROI channel boundaries in the full spectrum
        c1 = chan - roi_offset
        c2 = chan + roi_offset
        if not gs_analysis.check_channel_validity(c1, c2, smooth_counts):
            continue

        try:
            nc = gs_analysis.net_counts(smooth_counts, c1, c2, m=bg_method)
        except Exception:
            continue

        count_rates[idx] = nc / float(spec.live_time)  # type: ignore[arg-type]

    return elapsed_times, count_rates


def estimate_half_life(
    elapsed_times: npt.NDArray[np.float64],
    count_rates: npt.NDArray[np.float64],
) -> Tuple[float, float]:
    """Estimate the half-life from a series of count rates.

    Fits the model ``A(t) = A0 * exp(-λ * t)`` to the supplied data and
    returns the half-life together with its uncertainty derived from the
    covariance matrix of the fit.

    Only data points where ``count_rates > 0`` and ``elapsed_times >= 0`` are
    used; ``NaN`` and non-positive values are silently dropped.

    Parameters
    ----------
    elapsed_times:
        Elapsed time (seconds) for each measurement.  Normally obtained from
        :func:`get_elapsed_times`.
    count_rates:
        Net count rate (counts / second) at each time point.  Normally
        obtained from :func:`track_peak_activity`.

    Returns
    -------
    half_life : float
        Estimated half-life in seconds.
    half_life_uncertainty : float
        One-sigma uncertainty on the half-life (seconds), propagated from the
        covariance of the decay-constant fit parameter.

    Raises
    ------
    ValueError
        If the arrays have different lengths.
    ValueError
        If fewer than two valid (positive, finite) data points remain after
        filtering.
    RuntimeError
        If the curve fit fails to converge.
    """
    elapsed_times = np.asarray(elapsed_times, dtype=np.float64)
    count_rates = np.asarray(count_rates, dtype=np.float64)

    if elapsed_times.shape != count_rates.shape:
        raise ValueError("elapsed_times and count_rates must have the same length")

    valid = (
        np.isfinite(count_rates)
        & (count_rates > 0)
        & np.isfinite(elapsed_times)
        & (elapsed_times >= 0)
    )
    t_fit = elapsed_times[valid]
    a_fit = count_rates[valid]

    if len(t_fit) < 2:
        raise ValueError(
            f"At least 2 valid data points are required for half-life fitting; "
            f"got {len(t_fit)}"
        )

    # Initial guess: a0 from the first data point, lambda from the ratio
    # of first and last activity (if they differ)
    a0_guess = float(a_fit[0])
    if a_fit[-1] > 0 and a_fit[0] > 0 and t_fit[-1] > t_fit[0]:
        lam_guess = -np.log(a_fit[-1] / a_fit[0]) / (t_fit[-1] - t_fit[0])
        lam_guess = max(lam_guess, 1e-15)
    else:
        lam_guess = 1e-5

    popt, pcov = curve_fit(
        _exponential_decay,
        t_fit,
        a_fit,
        p0=[a0_guess, lam_guess],
        bounds=([0, 0], [np.inf, np.inf]),
        maxfev=10000,
    )

    lam = popt[1]
    lam_std = float(np.sqrt(pcov[1, 1]))

    half_life = np.log(2) / lam
    # Propagate uncertainty: σ(t½) = (ln2 / λ²) * σ(λ)
    half_life_uncertainty = (np.log(2) / lam**2) * lam_std

    return float(half_life), float(half_life_uncertainty)
