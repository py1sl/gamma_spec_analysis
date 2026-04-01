#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script demonstrating the different background subtraction methods
in gs_analysis, with visualisation via gs_plotting.

This script shows how to use the four available background subtraction methods:
1. BackgroundMethod.TRAPEZOID - Maestro-style trapezoid background
2. BackgroundMethod.LINEAR    - Linear interpolation between edge averages
3. BackgroundMethod.STEP      - Constant background from average of edges
4. BackgroundMethod.SLIDING_AVERAGE - Moving average in adjacent regions

It also demonstrates several of the other plotting helpers added to
gs_plotting: plot_spectrum, plot_spectra_overlay, plot_smoothing_comparison,
and plot_peak_roi.
"""

import sys
import os

# Add parent directory to path so the package modules can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import gs_analysis as gs
from gs_analysis import BackgroundMethod
import gs_plotting as gsp
import tempfile


def create_synthetic_spectrum():
    """Create a synthetic gamma spectrum with multiple peaks."""
    x = np.arange(200)

    # Create a realistic background (slowly varying)
    background = 100 + 20 * np.sin(x / 30) + 10 * np.exp(-x / 100)

    # Add some peaks at different positions
    peak1 = 500 * np.exp(-((x - 50) ** 2) / (2 * 4 ** 2))
    peak2 = 800 * np.exp(-((x - 100) ** 2) / (2 * 6 ** 2))
    peak3 = 300 * np.exp(-((x - 150) ** 2) / (2 * 3 ** 2))

    # Add Poisson noise
    counts = background + peak1 + peak2 + peak3
    counts = np.random.poisson(counts)

    return counts


def compare_background_methods(counts, c1, c2):
    """
    Compare all background subtraction methods for a given peak region.

    Parameters
    ----------
    counts : numpy array
        The spectrum counts data
    c1 : int
        Start channel of peak region
    c2 : int
        End channel of peak region

    Returns
    -------
    dict
        Dictionary containing results for each method
    """
    methods = {
        BackgroundMethod.TRAPEZOID: "Trapezoid (Maestro)",
        BackgroundMethod.LINEAR: "Linear Interpolation",
        BackgroundMethod.STEP: "Step Function",
        BackgroundMethod.SLIDING_AVERAGE: "Sliding Window Average"
    }

    results = {}

    for method, method_name in methods.items():
        bg = gs.calc_bg(counts, c1, c2, m=method)
        net = gs.net_counts(counts, c1, c2, m=method)
        gross = gs.gross_count(counts, c1, c2)

        results[method] = {
            'name': method_name,
            'background': bg,
            'net_counts': net,
            'gross_counts': gross
        }

    return results


def main():
    """Main demonstration function."""
    print("=" * 70)
    print("Background Subtraction Methods Demonstration")
    print("=" * 70)
    print()

    # Create synthetic spectrum
    print("Creating synthetic gamma spectrum with multiple peaks...")
    counts = create_synthetic_spectrum()
    print(f"Spectrum created with {len(counts)} channels")
    print()

    # Define peak regions to analyze
    peak_regions = [
        (45, 55, "Peak 1 (channel ~50)"),
        (90, 110, "Peak 2 (channel ~100)"),
        (145, 155, "Peak 3 (channel ~150)")
    ]

    # Analyze each peak with all methods
    for c1, c2, peak_name in peak_regions:
        print(f"\n{peak_name}")
        print("-" * 70)
        print(f"Peak region: channels {c1} to {c2}")
        print()

        results = compare_background_methods(counts, c1, c2)

        print(f"{'Method':<30} {'Background':<15} {'Net Counts':<15}")
        print("-" * 70)

        for method in sorted(results.keys(), key=lambda m: m.value):
            res = results[method]
            print(f"{res['name']:<30} {res['background']:<15.2f} {res['net_counts']:<15.2f}")

        print()

    print("=" * 70)
    print("Method Descriptions:")
    print("=" * 70)
    print()
    print(f"{BackgroundMethod.TRAPEZOID.value}. Trapezoid (Maestro):")
    print("   Uses up to 2 channels before and after the peak region.")
    print("   Implements the Maestro software's trapezoid background method.")
    print()
    print(f"{BackgroundMethod.LINEAR.value}. Linear Interpolation:")
    print("   Averages 2 channels on each side of the peak and linearly")
    print("   interpolates the background under the peak region.")
    print()
    print(f"{BackgroundMethod.STEP.value}. Step Function:")
    print("   Uses the average of background regions on both sides as a")
    print("   constant background level under the peak.")
    print()
    print(f"{BackgroundMethod.SLIDING_AVERAGE.value}. Sliding Window Average:")
    print("   Uses a moving average window (default 5 channels) in regions")
    print("   adjacent to the peak for more robust background estimation.")
    print()

    temp_dir = tempfile.gettempdir()

    # --- 1. Background methods comparison (2×2 grid) -----------------
    print("Generating background-methods comparison for Peak 2 ...")
    c1, c2 = 90, 110
    output_file = os.path.join(temp_dir, "background_methods_comparison.png")
    gsp.plot_background_methods_comparison(
        counts, c1, c2,
        title="Background Subtraction Methods Comparison",
        fname=output_file,
    )
    print(f"Visualization saved to: {output_file}")
    print()

    # --- 2. Full spectrum plot ----------------------------------------
    print("Generating full-spectrum plot ...")
    output_file = os.path.join(temp_dir, "full_spectrum.png")
    gsp.plot_spec(
        counts,
        title="Synthetic Gamma Spectrum",
        xlabel="Channel",
        ylabel="Counts",
        fname=output_file,
    )
    print(f"Spectrum plot saved to: {output_file}")
    print()

    # --- 3. Spectra overlay (raw vs. 5-point smoothed) ---------------
    print("Generating spectra overlay (raw vs. smoothed) ...")
    output_file = os.path.join(temp_dir, "spectra_overlay.png")
    smoothed = gs.five_point_smooth(gs.five_point_smooth(counts.astype(float)))
    gsp.plot_spectra_overlay(
        [counts, smoothed.astype(int)],
        labels=["Raw", "5-point smooth (×2)"],
        title="Raw vs. Smoothed Spectrum",
        fname=output_file,
    )
    print(f"Overlay plot saved to: {output_file}")
    print()

    # --- 4. Smoothing comparison for the full spectrum ----------------
    print("Generating smoothing comparison ...")
    output_file = os.path.join(temp_dir, "smoothing_comparison.png")
    gsp.plot_smoothing_comparison(
        counts,
        title="Smoothing Methods Comparison",
        fname=output_file,
    )
    print(f"Smoothing comparison saved to: {output_file}")
    print()

    # --- 5. Peak ROI plot for Peak 2 with a Gaussian fit --------------
    print("Generating peak ROI plot ...")
    c1, c2 = 90, 110
    x_roi = np.arange(c1, c2, dtype=float)
    y_roi = counts[c1:c2].astype(float)
    try:
        popt, _pcov = gs.fit_peak(x_roi, y_roi)
        output_file = os.path.join(temp_dir, "peak_roi.png")
        gsp.plot_peak_roi(
            x_roi, y_roi,
            fit_params=popt,
            title="Peak 2 ROI with Gaussian Fit",
            xlabel="Channel",
            fname=output_file,
        )
        print(f"Peak ROI plot saved to: {output_file}")
    except Exception as exc:
        print(f"  (Peak ROI fit skipped: {exc})")
    print()

    print("=" * 70)
    print("Demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
