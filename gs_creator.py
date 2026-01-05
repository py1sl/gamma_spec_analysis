"""
gamma spectrum creator - functions to create synthetic ph_spectrum objects
"""

from typing import List, Optional, Tuple, Sequence
import numpy as np
import numpy.typing as npt
from ph_spectrum import PhSpectrum
import gs_analysis


def create_gaussian_peak(
    energy: float,
    emission_rate: float,
    num_bins: int,
    energy_per_bin: float,
    min_energy: float = 0.0,
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
    min_energy : float, optional
        Minimum energy (offset) for the spectrum in keV (default 0.0)
    fwhm_factor : float, optional
        Factor to calculate FWHM from energy (default 0.02 means 2% of energy)
        
    Returns
    -------
    np.ndarray
        Array of counts for the Gaussian peak
    """
    # Create energy bins accounting for offset
    ebins = np.arange(num_bins) * energy_per_bin + min_energy
    
    # Calculate FWHM and sigma
    fwhm = fwhm_factor * energy
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    
    # Generate Gaussian peak
    peak_counts = emission_rate * np.exp(-((ebins - energy) ** 2) / (2 * sigma ** 2))
    
    return peak_counts.astype(np.int64)


def create_compton_continuum(
    energy: float,
    emission_rate: float,
    num_bins: int,
    energy_per_bin: float,
    min_energy: float = 0.0,
    compton_fraction: float = 0.5,
) -> npt.NDArray[np.float64]:
    """Create Compton continuum for a gamma peak.
    
    When gamma rays undergo Compton scattering in the detector, they can deposit
    partial energy, creating a continuum from low energies up to the Compton edge.
    
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
    min_energy : float, optional
        Minimum energy (offset) for the spectrum in keV (default 0.0)
    compton_fraction : float, optional
        Fraction of peak intensity that goes into Compton continuum (default 0.5)
        
    Returns
    -------
    np.ndarray
        Array of counts for the Compton continuum
    """
    # Create energy bins
    ebins = np.arange(num_bins) * energy_per_bin + min_energy
    
    # Calculate Compton edge energy
    # E_compton = E_gamma / (1 + m_e*c^2 / (2*E_gamma))
    # where m_e*c^2 = 511 keV (electron rest mass energy)
    m_e_c2 = 511.0  # keV
    compton_edge = energy / (1 + m_e_c2 / (2 * energy))
    
    # Initialize continuum array
    continuum = np.zeros(num_bins, dtype=np.float64)
    
    # Only add continuum below the Compton edge
    mask = (ebins > 0) & (ebins < compton_edge)
    
    if np.any(mask):
        # Simplified continuum shape: decreasing from Compton edge to lower energies
        # Using a power law shape that's common in gamma spectroscopy
        # Intensity ~ (E / E_compton)^2 for a simple approximation
        continuum[mask] = compton_fraction * emission_rate * (ebins[mask] / compton_edge) ** 2
    
    return continuum


def create_spectrum_from_peaks(
    peak_energies: List[float],
    emission_rates: List[float],
    num_bins: int,
    energy_range: Optional[Tuple[float, float]] = None,
    background_spectrum: Optional[PhSpectrum] = None,
    fwhm_factor: float = 0.02,
    spec_name: str = "synthetic_spectrum",
    efficiency_coefficients: Optional[Sequence[float]] = None,
    efficiency_fit_type: int = 1,
    include_compton: bool = False,
    compton_fraction: float = 0.5,
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
    efficiency_coefficients : Sequence[float], optional
        Efficiency fit coefficients to adjust peak heights based on energy.
        If provided, peak heights will be scaled by the detector efficiency.
    efficiency_fit_type : int, optional
        Type of efficiency fit to use (1 or 2, default 1).
        1: logarithmic fit, 2: inverse energy fit.
        See gs_analysis.calc_energy_efficiency for details.
    include_compton : bool, optional
        If True, generate Compton continuum for each peak (default False)
    compton_fraction : float, optional
        Fraction of peak intensity that goes into Compton continuum (default 0.5)
        
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
    
    # Initialize counts array (use float64 for accumulation, convert to int64 at end)
    counts = np.zeros(num_bins, dtype=np.float64)
    
    # Add each peak
    for energy, rate in zip(peak_energies, emission_rates):
        if min_energy <= energy <= max_energy:
            # Apply efficiency correction if coefficients provided
            effective_rate = rate
            if efficiency_coefficients is not None:
                # Convert keV to MeV for efficiency calculation
                energy_mev = energy / 1000.0
                efficiency = gs_analysis.calc_energy_efficiency(
                    energy_mev, efficiency_coefficients, efficiency_fit_type
                )
                effective_rate = rate * efficiency
            
            # Add photopeak
            peak_counts = create_gaussian_peak(
                energy, effective_rate, num_bins, energy_per_bin, min_energy, fwhm_factor
            )
            counts += peak_counts
            
            # Add Compton continuum if requested
            if include_compton:
                compton_counts = create_compton_continuum(
                    energy, effective_rate, num_bins, energy_per_bin, 
                    min_energy, compton_fraction
                )
                counts += compton_counts
    
    # Add background if provided
    if background_spectrum is not None:
        counts += background_spectrum.counts.astype(np.float64)
    
    # Convert to integer counts
    counts = counts.astype(np.int64)
    
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
        efficiency_fit_coefficients=efficiency_coefficients,
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
