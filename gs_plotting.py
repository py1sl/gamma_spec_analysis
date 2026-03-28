# -*- coding: utf-8 -*-
"""
gamma spectrum plotting functions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from typing import Callable, List, Optional, Sequence, Tuple, Union, Any
import numpy.typing as npt

import gs_analysis
from gs_analysis import (
    BackgroundMethod,
    EfficiencyFitType,
    gaussian,
    double_gaussian,
    calc_bg,
    net_counts,
    calc_energy_efficiency,
    five_point_smooth,
    three_point_smooth,
    moving_average,
    exponential_moving_average,
)
import ph_spectrum as _ph_spectrum


# Human-readable names for each BackgroundMethod, used in plot titles.
_METHOD_NAMES = {
    BackgroundMethod.TRAPEZOID: "Trapezoid (Maestro)",
    BackgroundMethod.LINEAR: "Linear Interpolation",
    BackgroundMethod.STEP: "Step Function",
    BackgroundMethod.SLIDING_AVERAGE: "Sliding Window Average",
}


def _make_ax(ax: Optional[Axes]) -> Tuple[Axes, bool]:
    """Return *ax* unchanged, or create a new figure and return its axes.

    Returns ``(axes, created)`` where *created* is ``True`` when a new
    figure was created by this call.
    """
    if ax is None:
        _, new_ax = plt.subplots()
        return new_ax, True
    return ax, False


def _finish_plot(ax: Axes, created: bool, fname: Optional[str]) -> None:
    """Save or show the figure if we created it in this call.

    When *ax* was supplied by the caller no automatic save/show is
    performed — the caller owns the figure lifecycle.
    """
    if fname:
        plt.savefig(fname)
    elif created:
        plt.show()


def plot_spect_peaks(
    smooth_counts: npt.NDArray[Any],
    ebins: npt.NDArray[Any],
    peaks: Sequence[int],
    fname: Optional[str] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    xlabel: str = "Energy",
    ylabel: str = "Counts",
    yscale: str = "log",
) -> Axes:
    """Plot a smoothed spectrum and highlight detected peaks.

    Parameters
    ----------
    smooth_counts : array-like
        Smoothed spectrum counts.
    ebins : array-like
        Energy bin values corresponding to each channel.
    peaks : sequence of int
        Channel indices of detected peaks.
    fname : str, optional
        File path to save the figure.  If omitted the figure is shown
        interactively (only when *ax* is ``None``).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.  A new figure is created when ``None``.
    title : str, optional
        Plot title.
    xlabel : str, optional
        x-axis label.  Default ``"Energy"``.
    ylabel : str, optional
        y-axis label.  Default ``"Counts"``.
    yscale : str, optional
        y-axis scale (``"log"`` or ``"linear"``).  Default ``"log"``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax, created = _make_ax(ax)
    ax.plot(ebins, smooth_counts)
    for peak in peaks:
        ax.plot(ebins[peak], smooth_counts[peak], "xr")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    if title:
        ax.set_title(title)
    _finish_plot(ax, created, fname)
    return ax


def plot_spec(
    counts: Union[Sequence[int], npt.NDArray[Any]],
    erg: Optional[npt.NDArray[Any]] = None,
    fname: Optional[str] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    xlabel: str = "Channel",
    ylabel: str = "Counts",
    yscale: str = "log",
) -> Axes:
    """Step-plot a spectrum.

    Parameters
    ----------
    counts : array-like
        Spectrum counts.
    erg : array-like, optional
        Energy bin values.  Channel numbers are used on the x axis when
        omitted.
    fname : str, optional
        File path to save the figure.  If omitted the figure is shown
        interactively (only when *ax* is ``None``).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.  A new figure is created when ``None``.
    title : str, optional
        Plot title.
    xlabel : str, optional
        x-axis label.  Default ``"Channel"``.
    ylabel : str, optional
        y-axis label.  Default ``"Counts"``.
    yscale : str, optional
        y-axis scale (``"log"`` or ``"linear"``).  Default ``"log"``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    counts = np.array(counts).astype(int)
    ax, created = _make_ax(ax)
    x = np.arange(len(counts)) if erg is None else np.asarray(erg)
    ax.step(x, counts)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    if title:
        ax.set_title(title)
    _finish_plot(ax, created, fname)
    return ax


def plot_spectrum(
    spec: "_ph_spectrum.PhSpectrum",
    fname: Optional[str] = None,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> Axes:
    """Plot a :class:`~ph_spectrum.PhSpectrum` object.

    Convenience wrapper around :func:`plot_spec` that unpacks the energy
    bins and counts from the spectrum automatically.

    Parameters
    ----------
    spec : PhSpectrum
        Spectrum to plot.
    fname : str, optional
        File path to save the figure.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.  A new figure is created when ``None``.
    **kwargs
        Additional keyword arguments forwarded to :func:`plot_spec`
        (``title``, ``xlabel``, ``ylabel``, ``yscale``).

    Returns
    -------
    matplotlib.axes.Axes
    """
    erg = spec.ebin if spec.ebin.size > 0 else None
    return plot_spec(spec.counts, erg=erg, fname=fname, ax=ax, **kwargs)


def plot_spectra_overlay(
    spectra: Sequence[Any],
    labels: Optional[Sequence[str]] = None,
    ergs: Optional[Sequence[Optional[npt.NDArray[Any]]]] = None,
    fname: Optional[str] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    xlabel: str = "Channel",
    ylabel: str = "Counts",
    yscale: str = "log",
) -> Axes:
    """Overlay multiple spectra on the same axes.

    Each entry in *spectra* may be a :class:`~ph_spectrum.PhSpectrum`
    instance or a plain array-like of counts.

    Parameters
    ----------
    spectra : sequence
        Spectra to plot.  Each item is either a ``PhSpectrum`` or an
        array-like of counts.
    labels : sequence of str, optional
        Legend labels, one per spectrum.  Defaults to
        ``"Spectrum 0"``, ``"Spectrum 1"``, …
    ergs : sequence of array-like or None, optional
        Energy axes for raw count arrays, one per spectrum.  Use
        ``None`` as a placeholder to fall back to channel numbers for
        that spectrum.  Ignored for ``PhSpectrum`` items.
    fname : str, optional
        File path to save the figure.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.  A new figure is created when ``None``.
    title : str, optional
        Plot title.
    xlabel : str, optional
        x-axis label.  Default ``"Channel"``.
    ylabel : str, optional
        y-axis label.  Default ``"Counts"``.
    yscale : str, optional
        y-axis scale.  Default ``"log"``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax, created = _make_ax(ax)
    if labels is None:
        labels = [f"Spectrum {i}" for i in range(len(spectra))]
    for i, spec in enumerate(spectra):
        label = labels[i] if i < len(labels) else f"Spectrum {i}"
        if isinstance(spec, _ph_spectrum.PhSpectrum):
            counts = spec.counts
            x = spec.ebin if spec.ebin.size > 0 else np.arange(len(counts))
        else:
            counts = np.asarray(spec)
            if ergs is not None and i < len(ergs) and ergs[i] is not None:
                x = np.asarray(ergs[i])
            else:
                x = np.arange(len(counts))
        ax.step(x, counts, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    if title:
        ax.set_title(title)
    ax.legend()
    _finish_plot(ax, created, fname)
    return ax


def plot_peak_roi(
    x: npt.NDArray[Any],
    y: npt.NDArray[Any],
    fit_params: Optional[npt.NDArray[Any]] = None,
    fname: Optional[str] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    xlabel: str = "Energy",
    ylabel: str = "Counts",
) -> Axes:
    """Plot a peak region-of-interest with an optional Gaussian fit overlay.

    Accepts either single-peak (3-parameter) fit results from
    :func:`~gs_analysis.fit_peak` or doublet (6-parameter) results from
    :func:`~gs_analysis.fit_doublet`.

    Parameters
    ----------
    x : array-like
        Energy bin values for the ROI (as returned by
        :func:`~gs_analysis.get_peak_roi`).
    y : array-like
        Counts for the ROI.
    fit_params : array-like, optional
        Fit parameters.  *Length 3* → single Gaussian
        ``[amplitude, centre, sigma]``.  *Length 6* → doublet
        ``[a1, x01, sigma1, a2, x02, sigma2]``.
    fname : str, optional
        File path to save the figure.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.  A new figure is created when ``None``.
    title : str, optional
        Plot title.
    xlabel : str, optional
        x-axis label.  Default ``"Energy"``.
    ylabel : str, optional
        y-axis label.  Default ``"Counts"``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ax, created = _make_ax(ax)
    ax.step(x, y, label="Data", where="mid")
    if fit_params is not None:
        fit_params = np.asarray(fit_params, dtype=float)
        x_fine = np.linspace(x[0], x[-1], 500)
        if len(fit_params) == 3:
            ax.plot(x_fine, gaussian(x_fine, *fit_params), "r-", label="Gaussian fit")
        elif len(fit_params) == 6:
            ax.plot(x_fine, double_gaussian(x_fine, *fit_params), "r-", label="Doublet fit")
            a1, x01, s1, a2, x02, s2 = fit_params
            ax.plot(x_fine, gaussian(x_fine, a1, x01, s1), "g--", label="Peak 1")
            ax.plot(x_fine, gaussian(x_fine, a2, x02, s2), "b--", label="Peak 2")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    _finish_plot(ax, created, fname)
    return ax


def plot_peak_with_background(
    counts: npt.NDArray[Any],
    c1: int,
    c2: int,
    method: BackgroundMethod = BackgroundMethod.TRAPEZOID,
    fname: Optional[str] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    xlabel: str = "Channel",
    ylabel: str = "Counts",
    margin: int = 20,
) -> Axes:
    """Plot a peak region with the estimated background overlaid.

    Parameters
    ----------
    counts : array-like
        Full spectrum counts.
    c1, c2 : int
        Start and end channel of the peak region.
    method : BackgroundMethod, optional
        Background estimation method.  Default
        ``BackgroundMethod.TRAPEZOID``.
    fname : str, optional
        File path to save the figure.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.  A new figure is created when ``None``.
    title : str, optional
        Plot title.  Auto-generated from the method name and
        gross/net counts when omitted.
    xlabel : str, optional
        x-axis label.  Default ``"Channel"``.
    ylabel : str, optional
        y-axis label.  Default ``"Counts"``.
    margin : int, optional
        Channels to display on each side of the peak region.
        Default 20.

    Returns
    -------
    matplotlib.axes.Axes
    """
    counts = np.asarray(counts)
    ax, created = _make_ax(ax)
    x = np.arange(len(counts))
    ax.plot(x, counts, "b-", label="Spectrum", linewidth=1)
    ax.axvspan(c1, c2, alpha=0.2, color="yellow", label="Peak region")
    bg_total = calc_bg(counts, c1, c2, m=method)
    net = net_counts(counts, c1, c2, m=method)
    width = c2 - c1
    bg_per_channel = bg_total / width if width > 0 else 0
    ax.plot(
        [c1, c2], [bg_per_channel, bg_per_channel],
        "r--", linewidth=2, label="Est. background",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is None:
        method_name = _METHOD_NAMES.get(method, method.name)
        title = f"{method_name}\nBackground: {bg_total:.1f}, Net: {net:.1f}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    x_lo = max(0, c1 - margin)
    x_hi = min(len(counts), c2 + margin)
    ax.set_xlim(x_lo, x_hi)
    region = counts[x_lo:x_hi]
    if region.size > 0:
        ax.set_ylim(max(0, int(region.min()) - 50), int(region.max()) + 50)
    _finish_plot(ax, created, fname)
    return ax


def plot_background_methods_comparison(
    counts: npt.NDArray[Any],
    c1: int,
    c2: int,
    fname: Optional[str] = None,
    title: str = "Background Subtraction Methods",
    margin: int = 20,
) -> Figure:
    """Plot a 2×2 grid comparing all four background subtraction methods.

    Parameters
    ----------
    counts : array-like
        Full spectrum counts.
    c1, c2 : int
        Start and end channel of the peak region.
    fname : str, optional
        File path to save the figure.
    title : str, optional
        Figure super-title.
    margin : int, optional
        Channels to display on each side of the peak region.  Default 20.

    Returns
    -------
    matplotlib.figure.Figure
    """
    counts = np.asarray(counts)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16)
    method_axes = [
        (BackgroundMethod.TRAPEZOID, axes[0, 0]),
        (BackgroundMethod.LINEAR, axes[0, 1]),
        (BackgroundMethod.STEP, axes[1, 0]),
        (BackgroundMethod.SLIDING_AVERAGE, axes[1, 1]),
    ]
    for method, ax in method_axes:
        plot_peak_with_background(counts, c1, c2, method=method, ax=ax, margin=margin)
    plt.tight_layout()
    if fname:
        plt.savefig(fname)
    else:
        plt.show()
    return fig


def plot_efficiency_curve(
    erg_range: Tuple[float, float],
    eff_coeff: Sequence[float],
    fit_type: EfficiencyFitType = EfficiencyFitType.LOG,
    n_points: int = 200,
    erg_points: Optional[npt.NDArray[Any]] = None,
    eff_points: Optional[npt.NDArray[Any]] = None,
    fname: Optional[str] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    xlabel: str = "Energy (MeV)",
    ylabel: str = "Efficiency",
) -> Axes:
    """Plot the detector efficiency curve.

    Parameters
    ----------
    erg_range : tuple of float
        ``(min_energy, max_energy)`` in MeV.
    eff_coeff : sequence of float
        Efficiency fit coefficients.
    fit_type : EfficiencyFitType, optional
        Fit equation type.  Default ``EfficiencyFitType.LOG``.
    n_points : int, optional
        Number of evaluation points along the curve.  Default 200.
    erg_points : array-like, optional
        Measured energy values (MeV) to overlay as scatter points.
    eff_points : array-like, optional
        Measured efficiency values corresponding to *erg_points*.
    fname : str, optional
        File path to save the figure.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.  A new figure is created when ``None``.
    title : str, optional
        Plot title.
    xlabel : str, optional
        x-axis label.  Default ``"Energy (MeV)"``.
    ylabel : str, optional
        y-axis label.  Default ``"Efficiency"``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax, created = _make_ax(ax)
    energies = np.linspace(erg_range[0], erg_range[1], n_points)
    efficiencies = np.array(
        [calc_energy_efficiency(e, eff_coeff, fit_type) for e in energies]
    )
    ax.plot(energies, efficiencies, "b-", label="Efficiency curve")
    if erg_points is not None and eff_points is not None:
        ax.scatter(
            np.asarray(erg_points), np.asarray(eff_points),
            color="red", zorder=5, label="Measured data",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _finish_plot(ax, created, fname)
    return ax


def plot_smoothing_comparison(
    counts: Union[Sequence[float], npt.NDArray[Any]],
    erg: Optional[npt.NDArray[Any]] = None,
    smoothers: Optional[List[Tuple[str, Callable]]] = None,
    fname: Optional[str] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    xlabel: str = "Channel",
    ylabel: str = "Counts",
    yscale: str = "log",
) -> Axes:
    """Overlay raw counts with one or more smoothed versions.

    Parameters
    ----------
    counts : array-like
        Raw spectrum counts (must have at least 5 elements for the
        default smoothers).
    erg : array-like, optional
        Energy values for the x axis.  Channel numbers used when omitted.
    smoothers : list of (str, callable), optional
        ``(label, function)`` pairs.  Each function must accept a 1-D
        counts array and return a smoothed array of the same length.
        Defaults to all four smoothing functions in :mod:`gs_analysis`.
    fname : str, optional
        File path to save the figure.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.  A new figure is created when ``None``.
    title : str, optional
        Plot title.
    xlabel : str, optional
        x-axis label.  Default ``"Channel"``.
    ylabel : str, optional
        y-axis label.  Default ``"Counts"``.
    yscale : str, optional
        y-axis scale.  Default ``"log"``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if smoothers is None:
        smoothers = [
            ("5-point smooth", five_point_smooth),
            ("3-point smooth", three_point_smooth),
            ("Moving average (5)", moving_average),
            ("EMA (α=0.3)", exponential_moving_average),
        ]
    counts_arr = np.asarray(counts, dtype=float)
    x = np.arange(len(counts_arr)) if erg is None else np.asarray(erg)
    ax, created = _make_ax(ax)
    ax.step(x, counts_arr, color="gray", alpha=0.5, label="Raw", linewidth=1)
    for label, fn in smoothers:
        ax.plot(x, fn(counts_arr), label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    if title:
        ax.set_title(title)
    ax.legend()
    _finish_plot(ax, created, fname)
    return ax


def plot_doublet_fit(
    x: npt.NDArray[Any],
    y: npt.NDArray[Any],
    popt: npt.NDArray[Any],
    fname: Optional[str] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    xlabel: str = "Energy",
    ylabel: str = "Counts",
) -> Axes:
    """Plot a doublet decomposition with the total fit and individual components.

    Parameters
    ----------
    x : array-like
        Energy bin values for the ROI.
    y : array-like
        Counts for the ROI.
    popt : array-like
        Six-parameter doublet fit result
        ``[a1, x01, sigma1, a2, x02, sigma2]`` from
        :func:`~gs_analysis.fit_doublet`.
    fname : str, optional
        File path to save the figure.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.  A new figure is created when ``None``.
    title : str, optional
        Plot title.
    xlabel : str, optional
        x-axis label.  Default ``"Energy"``.
    ylabel : str, optional
        y-axis label.  Default ``"Counts"``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    popt = np.asarray(popt, dtype=float)
    ax, created = _make_ax(ax)
    x_fine = np.linspace(x[0], x[-1], 500)
    a1, x01, s1, a2, x02, s2 = popt
    ax.step(x, y, color="gray", label="Data", where="mid")
    ax.plot(x_fine, double_gaussian(x_fine, *popt), "r-", linewidth=2, label="Total fit")
    ax.plot(x_fine, gaussian(x_fine, a1, x01, s1), "g--", label=f"Peak 1 (x\u2080={x01:.2f})")
    ax.plot(x_fine, gaussian(x_fine, a2, x02, s2), "b--", label=f"Peak 2 (x\u2080={x02:.2f})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    _finish_plot(ax, created, fname)
    return ax


if __name__ == "__main__":
    pass
