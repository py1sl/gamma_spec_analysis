#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script demonstrating the different background subtraction methods
in gs_analysis.

This script shows how to use the four available background subtraction methods:
1. Trapezoid method (m=1) - Maestro-style trapezoid background
2. Linear interpolation (m=2) - Linear interpolation between edge averages
3. Step function (m=3) - Constant background from average of edges
4. Sliding window average (m=4) - Moving average in adjacent regions

Author: Generated for gamma_spec_analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import gs_analysis as gs


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
        1: "Trapezoid (Maestro)",
        2: "Linear Interpolation",
        3: "Step Function",
        4: "Sliding Window Average"
    }
    
    results = {}
    
    for method_id, method_name in methods.items():
        bg = gs.calc_bg(counts, c1, c2, m=method_id)
        net = gs.net_counts(counts, c1, c2, m=method_id)
        gross = gs.gross_count(counts, c1, c2)
        
        results[method_id] = {
            'name': method_name,
            'background': bg,
            'net_counts': net,
            'gross_counts': gross
        }
    
    return results


def visualize_background_methods(counts, c1, c2, title="Background Subtraction Methods"):
    """
    Visualize the different background subtraction methods.
    
    Parameters
    ----------
    counts : numpy array
        The spectrum counts data
    c1 : int
        Start channel of peak region
    c2 : int
        End channel of peak region
    title : str, optional
        Plot title
    """
    x = np.arange(len(counts))
    
    # Create subplots for each method
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16)
    
    methods = {
        1: ("Trapezoid (Maestro)", axes[0, 0]),
        2: ("Linear Interpolation", axes[0, 1]),
        3: ("Step Function", axes[1, 0]),
        4: ("Sliding Window Average", axes[1, 1])
    }
    
    for method_id, (method_name, ax) in methods.items():
        # Plot the spectrum
        ax.plot(x, counts, 'b-', label='Spectrum', linewidth=1)
        
        # Highlight the peak region
        ax.axvspan(c1, c2, alpha=0.2, color='yellow', label='Peak region')
        
        # Calculate and show background
        bg_total = gs.calc_bg(counts, c1, c2, m=method_id)
        net = gs.net_counts(counts, c1, c2, m=method_id)
        
        # Estimate background per channel for visualization
        width = c2 - c1
        bg_per_channel = bg_total / width if width > 0 else 0
        
        # Draw background line
        ax.plot([c1, c2], [bg_per_channel, bg_per_channel], 
                'r--', linewidth=2, label=f'Est. background')
        
        ax.set_xlabel('Channel')
        ax.set_ylabel('Counts')
        ax.set_title(f'{method_name}\nBackground: {bg_total:.1f}, Net: {net:.1f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set reasonable y-axis limits
        y_min = max(0, counts[c1:c2].min() - 50)
        y_max = counts[c1:c2].max() + 50
        ax.set_ylim(y_min, y_max)
        
        # Set x-axis to show region around peak
        margin = 20
        ax.set_xlim(max(0, c1 - margin), min(len(counts), c2 + margin))
    
    plt.tight_layout()
    return fig


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
        
        for method_id in sorted(results.keys()):
            res = results[method_id]
            print(f"{res['name']:<30} {res['background']:<15.2f} {res['net_counts']:<15.2f}")
        
        print()
    
    print("=" * 70)
    print("Method Descriptions:")
    print("=" * 70)
    print()
    print("1. Trapezoid (Maestro):")
    print("   Uses up to 2 channels before and after the peak region.")
    print("   Implements the Maestro software's trapezoid background method.")
    print()
    print("2. Linear Interpolation:")
    print("   Averages 2 channels on each side of the peak and linearly")
    print("   interpolates the background under the peak region.")
    print()
    print("3. Step Function:")
    print("   Uses the average of background regions on both sides as a")
    print("   constant background level under the peak.")
    print()
    print("4. Sliding Window Average:")
    print("   Uses a moving average window (default 5 channels) in regions")
    print("   adjacent to the peak for more robust background estimation.")
    print()
    
    # Create visualizations for one peak
    print("Generating visualization for Peak 2...")
    c1, c2 = 90, 110
    fig = visualize_background_methods(counts, c1, c2, 
                                      "Background Subtraction Methods Comparison")
    
    # Save the figure
    output_file = "/tmp/background_methods_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    print()
    
    print("=" * 70)
    print("Demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
