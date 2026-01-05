"""
gamma spectrum creator - functions to create synthetic ph_spectrum objects
"""

from typing import List, Optional, Tuple
import numpy as np
import numpy.typing as npt
from ph_spectrum import PhSpectrum


def create_gaussian_peak(
    energy: float,
    emission_rate: float,
    num_bins: int,
    energy_per_bin: float,
    fwhm_factor: float = 0.02,
) -> npt.NDArray[np.int64]:
    """Create a Gaussian peak at a specific energy.
    
    Parameters
    ----------
    energy : float
        Peak energy in keV
    emission_rate : float
        Peak emission rate (intensity/counts)
    num_bins : int
        Total number of bins in the spectrum
    energy_per_bin : float
        Energy per bin in keV
    fwhm_factor : float, optional
        Factor to calculate FWHM from energy (default 0.02 means 2% of energy)
        
    Returns
    -------
    np.ndarray
        Array of counts for the Gaussian peak
    """
    # Create energy bins
    ebins = np.arange(num_bins) * energy_per_bin
    
    # Calculate FWHM and sigma
    fwhm = fwhm_factor * energy
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    
    # Generate Gaussian peak
    peak_counts = emission_rate * np.exp(-((ebins - energy) ** 2) / (2 * sigma ** 2))
    
    return peak_counts.astype(np.int64)


def create_spectrum_from_peaks(
    peak_energies: List[float],
    emission_rates: List[float],
    num_bins: int,
    energy_range: Optional[Tuple[float, float]] = None,
    background_spectrum: Optional[PhSpectrum] = None,
    fwhm_factor: float = 0.02,
    spec_name: str = "synthetic_spectrum",
) -> PhSpectrum:
    """Create a ph_spectrum object from a list of gamma peak energies and emission rates.
    
    Parameters
    ----------
    peak_energies : List[float]
        List of gamma peak energies in keV
    emission_rates : List[float]
        List of emission rates (intensities) for each peak
    num_bins : int
        Number of bins in the spectrum
    energy_range : Tuple[float, float], optional
        Energy range as (min_energy, max_energy) in keV.
        If None, defaults to (0, max(peak_energies) * 1.5)
    background_spectrum : PhSpectrum, optional
        Optional background ph_spectrum object to add to the created spectrum
    fwhm_factor : float, optional
        Factor to calculate FWHM from energy (default 0.02 means 2% of energy)
    spec_name : str, optional
        Name for the created spectrum (default "synthetic_spectrum")
        
    Returns
    -------
    PhSpectrum
        A new PhSpectrum object with the synthetic gamma peaks
        
    Raises
    ------
    ValueError
        If peak_energies and emission_rates have different lengths
        If num_bins is less than 1
        If background_spectrum has different number of bins
    """
    # Validate inputs
    if len(peak_energies) != len(emission_rates):
        raise ValueError("peak_energies and emission_rates must have the same length")
    
    if num_bins < 1:
        raise ValueError("num_bins must be at least 1")
    
    if background_spectrum is not None and len(background_spectrum.counts) != num_bins:
        raise ValueError(
            f"background_spectrum has {len(background_spectrum.counts)} bins, "
            f"but num_bins is {num_bins}"
        )
    
    # Determine energy range
    if energy_range is None:
        if len(peak_energies) > 0:
            max_peak = max(peak_energies)
            energy_range = (0.0, max_peak * 1.5)
        else:
            energy_range = (0.0, 3000.0)  # Default to 3 MeV
    
    min_energy, max_energy = energy_range
    energy_per_bin = (max_energy - min_energy) / num_bins
    
    # Initialize counts array
    counts = np.zeros(num_bins, dtype=np.int64)
    
    # Add each peak
    for energy, rate in zip(peak_energies, emission_rates):
        if min_energy <= energy <= max_energy:
            peak_counts = create_gaussian_peak(
                energy, rate, num_bins, energy_per_bin, fwhm_factor
            )
            counts += peak_counts
    
    # Add background if provided
    if background_spectrum is not None:
        counts += background_spectrum.counts.astype(np.int64)
    
    # Calculate energy fit coefficients (linear: E = a + b*channel)
    # E = min_energy + channel * energy_per_bin
    energy_fit_coefficients = [min_energy, energy_per_bin]
    
    # Create and return PhSpectrum object
    spectrum = PhSpectrum(
        spec_name=spec_name,
        start_chan_num=0,
        num_channels=num_bins,
        channels=list(range(num_bins)),
        counts=counts,
        energy_fit_coefficients=energy_fit_coefficients,
    )
    
    return spectrum


def create_flat_background(
    num_bins: int,
    background_level: float,
    energy_range: Optional[Tuple[float, float]] = None,
    spec_name: str = "flat_background",
) -> PhSpectrum:
    """Create a flat background spectrum.
    
    Parameters
    ----------
    num_bins : int
        Number of bins in the spectrum
    background_level : float
        Constant background count level per bin
    energy_range : Tuple[float, float], optional
        Energy range as (min_energy, max_energy) in keV.
        If None, defaults to (0, 3000)
    spec_name : str, optional
        Name for the background spectrum (default "flat_background")
        
    Returns
    -------
    PhSpectrum
        A PhSpectrum object with flat background
        
    Raises
    ------
    ValueError
        If num_bins is less than 1
        If background_level is negative
    """
    if num_bins < 1:
        raise ValueError("num_bins must be at least 1")
    
    if background_level < 0:
        raise ValueError("background_level must be non-negative")
    
    # Determine energy range
    if energy_range is None:
        energy_range = (0.0, 3000.0)
    
    min_energy, max_energy = energy_range
    energy_per_bin = (max_energy - min_energy) / num_bins
    
    # Create flat background counts
    counts = np.full(num_bins, int(background_level), dtype=np.int64)
    
    # Calculate energy fit coefficients
    energy_fit_coefficients = [min_energy, energy_per_bin]
    
    # Create and return PhSpectrum object
    spectrum = PhSpectrum(
        spec_name=spec_name,
        start_chan_num=0,
        num_channels=num_bins,
        channels=list(range(num_bins)),
        counts=counts,
        energy_fit_coefficients=energy_fit_coefficients,
    )
    
    return spectrum
