# -*- coding: utf-8 -*-
"""
Peak fitting and characterisation functions for gamma spectrum analysis.
"""

import numpy as np
from scipy.optimize import curve_fit

from typing import Any, Callable, Optional, Sequence, Tuple, Union
import numpy.typing as npt

from gs_background import (
    BackgroundMethod,
    calc_bg,
    calc_bg_uncertainty,
    check_channel_validity,
    gross_count,
)


def net_counts(
    counts: npt.NDArray[Any],
    c1: int,
    c2: int,
    m: BackgroundMethod = BackgroundMethod.TRAPEZOID,
) -> float:
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


def double_gaussian(
    x: npt.NDArray[Any],
    a1: float, x01: float, sigma1: float,
    a2: float, x02: float, sigma2: float,
) -> npt.NDArray[Any]:
    """Sum of two Gaussians used for doublet curve fitting.

    Parameters
    ----------
    x : array-like
        Input x values
    a1 : float
        Amplitude of first Gaussian
    x01 : float
        Center of first Gaussian
    sigma1 : float
        Standard deviation of first Gaussian
    a2 : float
        Amplitude of second Gaussian
    x02 : float
        Center of second Gaussian
    sigma2 : float
        Standard deviation of second Gaussian

    Returns
    -------
    numpy.ndarray
        Sum of two Gaussian values at positions x
    """
    return gaussian(x, a1, x01, sigma1) + gaussian(x, a2, x02, sigma2)


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

    y = counts[peak_pos - offset: peak_pos + offset]
    x = ebins[peak_pos - offset: peak_pos + offset]

    return x, y


def fit_peak(
    x: npt.NDArray[Any], y: npt.NDArray[Any]
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Fit a single peak to a Gaussian.

    Parameters
    ----------
    x : array-like
        Energy bin positions for the region of interest.
    y : array-like
        Counts for the region of interest.

    Returns
    -------
    popt : numpy.ndarray
        Optimal fit parameters ``[amplitude, centroid, sigma]``.
    pcov : numpy.ndarray
        Estimated covariance matrix of *popt* (3×3).  The square root of
        the diagonal elements gives one-standard-deviation uncertainties
        for each parameter.
    """
    mean = np.sum(x * y) / np.sum(y)
    sigma = np.sqrt(np.sum(y * (x - mean) ** 2) / np.sum(y))

    popt, pcov = curve_fit(gaussian, x, y, p0=[max(y), mean, sigma], maxfev=10000)

    return popt, pcov


def fit_doublet(x: npt.NDArray[Any], y: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Fits a doublet (two partially overlapping peaks) to a double Gaussian.

    Initial estimates for each peak center are derived from the two largest
    local maxima in the ROI.  If fewer than two local maxima exist the ROI
    is split in half and the maximum of each half is used instead.

    Parameters
    ----------
    x : array-like
        Energy bin positions for the region of interest
    y : array-like
        Counts for the region of interest

    Returns
    -------
    popt : numpy.ndarray
        Array of 6 fitted parameters [a1, x01, sigma1, a2, x02, sigma2]
    pcov : numpy.ndarray
        Estimated covariance matrix of *popt* (6×6).

    Raises
    ------
    RuntimeError
        If the curve fit fails to converge
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Estimate a global sigma from the weighted spread of the data
    total = np.sum(y)
    if total != 0:
        mean_est = np.sum(x * y) / total
        sigma_est = np.sqrt(np.sum(y * (x - mean_est) ** 2) / total)
        if sigma_est == 0:
            sigma_est = (x[-1] - x[0]) / 6.0
    else:
        mean_est = (x[0] + x[-1]) / 2.0
        sigma_est = (x[-1] - x[0]) / 6.0

    # Find local maxima by comparing each point with its neighbours
    local_max_idx = [i for i in range(1, len(y) - 1) if y[i] > y[i - 1] and y[i] > y[i + 1]]

    if len(local_max_idx) >= 2:
        # Use the two highest local maxima, ordered by x position
        local_max_idx.sort(key=lambda i: y[i], reverse=True)
        i1, i2 = sorted(local_max_idx[:2])
        x01_est, a1_est = x[i1], float(y[i1])
        x02_est, a2_est = x[i2], float(y[i2])
    else:
        # Fall back: split the ROI in half, guarding against empty slices
        mid = max(1, min(len(x) // 2, len(x) - 1))
        i1 = int(np.argmax(y[:mid]))
        i2 = int(np.argmax(y[mid:])) + mid
        x01_est, a1_est = x[i1], float(y[i1])
        x02_est, a2_est = x[i2], float(y[i2])

    half_sigma = sigma_est / 2.0
    p0 = [a1_est, x01_est, half_sigma, a2_est, x02_est, half_sigma]

    popt, pcov = curve_fit(double_gaussian, x, y, p0=p0, maxfev=10000)

    return popt, pcov


# ---------------------------------------------------------------------------
# Gaussian area helpers
# ---------------------------------------------------------------------------

def gaussian_area(a: float, sigma: float) -> float:
    """Return the analytic area of a Gaussian peak.

    For a Gaussian of the form ``a * exp(-(x-x0)^2 / (2*sigma^2))`` the
    definite integral from −∞ to +∞ equals ``a * sigma * sqrt(2*pi)``.

    Parameters
    ----------
    a : float
        Peak amplitude.
    sigma : float
        Standard deviation (width parameter).

    Returns
    -------
    float
        Analytic peak area (counts).
    """
    return float(a * np.abs(sigma) * np.sqrt(2.0 * np.pi))


def gaussian_area_uncertainty(
    a: float, sigma: float, pcov: npt.NDArray[Any]
) -> float:
    """Return the one-sigma uncertainty on the analytic Gaussian peak area.

    Uses first-order error propagation through ``area = a * |sigma| * sqrt(2*pi)``:

    .. math::

        \\sigma_{\\text{area}} = \\sqrt{2\\pi}\\,
            \\sqrt{(\\sigma \\cdot \\sigma_a)^2
                  + (a \\cdot \\sigma_\\sigma)^2
                  + 2\\,a\\,\\sigma \\cdot \\mathrm{cov}(a,\\,\\sigma)}

    where ``sigma_a = sqrt(pcov[0,0])``, ``sigma_sigma = sqrt(pcov[2,2])``,
    and ``cov(a, sigma) = pcov[0,2]``.

    Parameters
    ----------
    a : float
        Fitted amplitude (``popt[0]``).
    sigma : float
        Fitted width (``popt[2]``).
    pcov : numpy.ndarray
        3×3 covariance matrix returned by :func:`fit_peak`.

    Returns
    -------
    float
        One-sigma uncertainty on the peak area (counts).
    """
    pcov = np.asarray(pcov, dtype=float)
    var_a = pcov[0, 0]
    var_sigma = pcov[2, 2]
    cov_a_sigma = pcov[0, 2]
    # Partial derivatives: dA/da = |sigma|*sqrt(2pi), dA/dsigma = a*sqrt(2pi)
    abs_sigma = np.abs(sigma)
    variance = (2.0 * np.pi) * (
        (abs_sigma * abs_sigma) * var_a
        + (a * a) * var_sigma
        + 2.0 * a * abs_sigma * cov_a_sigma
    )
    return float(np.sqrt(max(variance, 0.0)))


def fit_peak_area(
    x: npt.NDArray[Any], y: npt.NDArray[Any]
) -> Tuple[float, float]:
    """Fit a single Gaussian peak and return its area with uncertainty.

    Combines :func:`fit_peak` with :func:`gaussian_area` and
    :func:`gaussian_area_uncertainty` to give both the analytic peak area and
    its one-sigma uncertainty derived from the covariance matrix of the fit.

    Parameters
    ----------
    x : array-like
        Energy bin positions for the region of interest.
    y : array-like
        Counts for the region of interest.

    Returns
    -------
    area : float
        Analytic Gaussian peak area ``a * |sigma| * sqrt(2*pi)``.
    area_uncertainty : float
        One-sigma uncertainty on *area* propagated from the fit covariance.
    """
    popt, pcov = fit_peak(x, y)
    a, _x0, sigma = popt
    area = gaussian_area(a, sigma)
    unc = gaussian_area_uncertainty(a, sigma, pcov)
    return area, unc


def fit_doublet_areas(
    x: npt.NDArray[Any], y: npt.NDArray[Any]
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Fit a doublet and return the area and uncertainty for each sub-peak.

    Uses :func:`fit_doublet` and propagates uncertainties for each of the two
    Gaussian components using their respective sub-blocks of the covariance
    matrix.

    Parameters
    ----------
    x : array-like
        Energy bin positions for the region of interest.
    y : array-like
        Counts for the region of interest.

    Returns
    -------
    peak1 : tuple of (float, float)
        ``(area, area_uncertainty)`` for the first Gaussian component.
    peak2 : tuple of (float, float)
        ``(area, area_uncertainty)`` for the second Gaussian component.
    """
    popt, pcov = fit_doublet(x, y)
    a1, _x01, sigma1, a2, _x02, sigma2 = popt

    # Extract 3×3 sub-blocks for each peak's parameters
    # Peak 1: indices 0,1,2  →  a1, x01, sigma1
    idx1 = [0, 1, 2]
    pcov1 = pcov[np.ix_(idx1, idx1)]
    # Peak 2: indices 3,4,5  →  a2, x02, sigma2
    idx2 = [3, 4, 5]
    pcov2 = pcov[np.ix_(idx2, idx2)]

    area1 = gaussian_area(a1, sigma1)
    unc1 = gaussian_area_uncertainty(a1, sigma1, pcov1)
    area2 = gaussian_area(a2, sigma2)
    unc2 = gaussian_area_uncertainty(a2, sigma2, pcov2)

    return (area1, unc1), (area2, unc2)


# ---------------------------------------------------------------------------
# FWHM helpers
# ---------------------------------------------------------------------------

#: Conversion factor from Gaussian sigma to FWHM: ``2 * sqrt(2 * ln(2))``.
_FWHM_FACTOR: float = 2.0 * np.sqrt(2.0 * np.log(2.0))


def peak_fwhm(sigma: float) -> float:
    """Return the Full Width at Half Maximum (FWHM) of a Gaussian peak.

    Uses the exact conversion ``FWHM = 2 * sqrt(2 * ln 2) * |sigma|``
    (approximately ``2.3548 * |sigma|``).

    Parameters
    ----------
    sigma : float
        Gaussian width parameter as returned in ``popt[2]`` by
        :func:`fit_peak`.  The sign is ignored.

    Returns
    -------
    float
        FWHM in the same units as *sigma* (channels or energy, depending on
        the coordinate axis used during fitting).
    """
    return _FWHM_FACTOR * abs(sigma)


def peak_fwhm_uncertainty(sigma: float, sigma_unc: float) -> float:
    """Return the one-sigma uncertainty on the FWHM.

    Since ``FWHM = k * |sigma|``, first-order propagation gives
    ``sigma_FWHM = k * sigma_sigma`` where ``k = 2*sqrt(2*ln2)``.

    Parameters
    ----------
    sigma : float
        Fitted Gaussian sigma (``popt[2]``).  Included for API symmetry with
        :func:`peak_fwhm` but not used in the calculation.
    sigma_unc : float
        One-sigma uncertainty on *sigma* (``sqrt(pcov[2, 2])``).

    Returns
    -------
    float
        One-sigma uncertainty on the FWHM.
    """
    return _FWHM_FACTOR * abs(sigma_unc)


def fit_peak_fwhm(
    x: npt.NDArray[Any], y: npt.NDArray[Any]
) -> Tuple[float, float]:
    """Fit a single Gaussian peak and return its FWHM with uncertainty.

    Convenience wrapper that calls :func:`fit_peak` and converts the fitted
    sigma to FWHM via :func:`peak_fwhm` and :func:`peak_fwhm_uncertainty`.

    Parameters
    ----------
    x : array-like
        Energy bin positions for the region of interest.
    y : array-like
        Counts for the region of interest.

    Returns
    -------
    fwhm : float
        Full Width at Half Maximum in the same units as *x*.
    fwhm_uncertainty : float
        One-sigma uncertainty on *fwhm* propagated from the fit covariance.
    """
    popt, pcov = fit_peak(x, y)
    sigma = popt[2]
    sigma_unc = float(np.sqrt(pcov[2, 2]))
    return peak_fwhm(sigma), peak_fwhm_uncertainty(sigma, sigma_unc)


# ---------------------------------------------------------------------------
# Goodness-of-fit statistics
# ---------------------------------------------------------------------------

def fit_chi2(
    x: npt.NDArray[Any],
    y: npt.NDArray[Any],
    popt: npt.NDArray[Any],
    model_fn: Callable[..., npt.NDArray[Any]],
    n_params: Optional[int] = None,
) -> Tuple[float, float, int]:
    """Compute chi-squared goodness-of-fit statistics for any fitted model.

    Poisson statistics are assumed, so the variance in each bin is taken as
    ``max(y_i, 1)`` to avoid division by zero in empty bins.

    Parameters
    ----------
    x : array-like
        x-axis values (energy bins or channels) used during fitting.
    y : array-like
        Observed counts used during fitting.
    popt : array-like
        Optimal parameters returned by the fitting function (e.g.
        :func:`fit_peak` or :func:`fit_doublet`).
    model_fn : callable
        The model function with signature ``model_fn(x, *popt)``, e.g.
        :func:`gaussian` or :func:`double_gaussian`.
    n_params : int, optional
        Number of free parameters in the fit.  Defaults to ``len(popt)``.

    Returns
    -------
    chi2 : float
        Chi-squared statistic ``sum((y_i - y_fit_i)^2 / max(y_i, 1))``.
    reduced_chi2 : float
        Reduced chi-squared ``chi2 / ndof``.  Returns ``inf`` if
        ``ndof <= 0``.
    ndof : int
        Degrees of freedom ``len(y) - n_params``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    popt = np.asarray(popt, dtype=float)

    if n_params is None:
        n_params = len(popt)

    y_fit = model_fn(x, *popt)
    variance = np.maximum(y, 1.0)
    chi2 = float(np.sum((y - y_fit) ** 2 / variance))
    ndof = len(y) - n_params
    reduced_chi2 = chi2 / ndof if ndof > 0 else float("inf")
    return chi2, reduced_chi2, ndof


def fit_peak_chi2(
    x: npt.NDArray[Any],
    y: npt.NDArray[Any],
    popt: npt.NDArray[Any],
) -> Tuple[float, float, int]:
    """Goodness-of-fit statistics for a single-peak Gaussian fit.

    Convenience wrapper around :func:`fit_chi2` using :func:`gaussian` as
    the model and ``n_params = 3``.

    Parameters
    ----------
    x : array-like
        Energy bin positions for the ROI.
    y : array-like
        Counts for the ROI.
    popt : array-like
        Three-parameter Gaussian fit result ``[amplitude, centroid, sigma]``
        from :func:`fit_peak`.

    Returns
    -------
    chi2 : float
        Chi-squared statistic.
    reduced_chi2 : float
        Reduced chi-squared (chi2 per degree of freedom).
    ndof : int
        Degrees of freedom (``len(y) - 3``).
    """
    return fit_chi2(x, y, popt, gaussian, n_params=3)


def fit_doublet_chi2(
    x: npt.NDArray[Any],
    y: npt.NDArray[Any],
    popt: npt.NDArray[Any],
) -> Tuple[float, float, int]:
    """Goodness-of-fit statistics for a doublet fit.

    Convenience wrapper around :func:`fit_chi2` using :func:`double_gaussian`
    as the model and ``n_params = 6``.

    Parameters
    ----------
    x : array-like
        Energy bin positions for the ROI.
    y : array-like
        Counts for the ROI.
    popt : array-like
        Six-parameter doublet fit result
        ``[a1, x01, sigma1, a2, x02, sigma2]`` from :func:`fit_doublet`.

    Returns
    -------
    chi2 : float
        Chi-squared statistic.
    reduced_chi2 : float
        Reduced chi-squared (chi2 per degree of freedom).
    ndof : int
        Degrees of freedom (``len(y) - 6``).
    """
    return fit_chi2(x, y, popt, double_gaussian, n_params=6)


def net_counts_uncertainty(
    counts: npt.NDArray[Any],
    c1: int,
    c2: int,
    m: BackgroundMethod = BackgroundMethod.TRAPEZOID,
) -> Tuple[float, float]:
    """Return net counts and their one-sigma Poisson uncertainty.

    Computes both the net peak area and the associated statistical
    uncertainty by propagating Poisson variance through:

    .. math::

        N_{\\text{net}} = N_{\\text{gross}} - B

    .. math::

        \\sigma_{\\text{net}} = \\sqrt{N_{\\text{gross}} + \\sigma_B^2}

    where :math:`\\sigma_B` is obtained from :func:`calc_bg_uncertainty` for
    the selected background method, and the Poisson variance on the gross
    counts is simply :math:`N_{\\text{gross}}`.

    Parameters
    ----------
    counts : numpy.ndarray
        Spectrum counts array.
    c1, c2 : int
        ROI channel boundaries.
    m : BackgroundMethod
        Background method selector (default: ``TRAPEZOID``).

    Returns
    -------
    net : float
        Net counts (peak area after background subtraction).
    uncertainty : float
        One-sigma statistical uncertainty on *net*.
    """
    nc = net_counts(counts, c1, c2, m)
    gc = gross_count(counts, c1, c2)
    sigma_bg = calc_bg_uncertainty(counts, c1, c2, m)
    uncertainty = float(np.sqrt(float(gc) + sigma_bg**2))
    return nc, uncertainty


def peak_area_with_background_sensitivity(
    counts: npt.NDArray[Any],
    c1: int,
    c2: int,
) -> Tuple[float, float, dict]:
    """Estimate peak area and systematic uncertainty across all background methods.

    Runs all four background-subtraction methods and reports the mean net
    count together with the standard deviation across methods as a measure of
    systematic sensitivity to the background choice.

    This is not a formal statistical uncertainty but a useful diagnostic: a
    large spread indicates that the result is strongly dependent on the chosen
    background model and should be treated with caution.

    Parameters
    ----------
    counts : numpy.ndarray
        Spectrum counts array.
    c1, c2 : int
        ROI channel boundaries.

    Returns
    -------
    mean_net : float
        Mean net counts averaged over the four background methods.
    std_net : float
        Standard deviation of net counts across the four methods.
    results : dict
        Per-method net counts keyed by :class:`BackgroundMethod` name,
        e.g. ``{'TRAPEZOID': 120.3, 'LINEAR': 118.7, ...}``.
    """
    check_channel_validity(c1, c2, counts)

    method_results: dict = {}
    for method in BackgroundMethod:
        nc = net_counts(counts, c1, c2, method)
        method_results[method.name] = nc

    values = np.array(list(method_results.values()), dtype=float)
    return float(np.mean(values)), float(np.std(values, ddof=0)), method_results
