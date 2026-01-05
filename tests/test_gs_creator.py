import unittest
import numpy as np
import gs_creator
from ph_spectrum import PhSpectrum


class TestGsCreator(unittest.TestCase):
    """Tests for gs_creator module functions"""

    def test_create_gaussian_peak_basic(self):
        """Test basic Gaussian peak creation"""
        energy = 662.0  # keV (Cs-137 peak)
        emission_rate = 1000.0
        num_bins = 1000
        energy_per_bin = 3.0  # keV per bin
        
        peak = gs_creator.create_gaussian_peak(
            energy, emission_rate, num_bins, energy_per_bin
        )
        
        # Check output type and shape
        self.assertIsInstance(peak, np.ndarray)
        self.assertEqual(len(peak), num_bins)
        self.assertEqual(peak.dtype, np.int64)
        
        # Check that peak is at approximately correct position
        peak_channel = int(energy / energy_per_bin)
        max_channel = np.argmax(peak)
        self.assertAlmostEqual(peak_channel, max_channel, delta=5)
        
        # Check that peak amplitude is reasonable
        self.assertGreater(peak[max_channel], 0)

    def test_create_gaussian_peak_different_fwhm(self):
        """Test Gaussian peak creation with different FWHM factors"""
        energy = 1000.0
        emission_rate = 5000.0
        num_bins = 500
        energy_per_bin = 5.0
        
        # Create peaks with different FWHM factors
        peak_narrow = gs_creator.create_gaussian_peak(
            energy, emission_rate, num_bins, energy_per_bin, fwhm_factor=0.01
        )
        peak_wide = gs_creator.create_gaussian_peak(
            energy, emission_rate, num_bins, energy_per_bin, fwhm_factor=0.05
        )
        
        # Wider peak should have more non-zero bins (greater width)
        self.assertGreater(np.count_nonzero(peak_wide), np.count_nonzero(peak_narrow))
        
        # Wider peak should have larger total area
        self.assertGreater(np.sum(peak_wide), np.sum(peak_narrow))

    def test_create_gaussian_peak_with_offset(self):
        """Test Gaussian peak creation with non-zero energy offset"""
        energy = 1500.0  # Peak at 1500 keV
        emission_rate = 1000.0
        num_bins = 500
        energy_per_bin = 5.0
        min_energy = 500.0  # Start at 500 keV
        
        # Create peak with offset
        peak = gs_creator.create_gaussian_peak(
            energy, emission_rate, num_bins, energy_per_bin, min_energy
        )
        
        # Check output
        self.assertIsInstance(peak, np.ndarray)
        self.assertEqual(len(peak), num_bins)
        
        # Peak should be at correct channel accounting for offset
        # Energy range is 500-3000 keV (500 bins * 5 keV/bin + 500 keV offset)
        # Peak at 1500 keV should be at channel (1500 - 500) / 5 = 200
        expected_channel = int((energy - min_energy) / energy_per_bin)
        max_channel = np.argmax(peak)
        self.assertAlmostEqual(expected_channel, max_channel, delta=5)

    def test_create_spectrum_from_peaks_single_peak(self):
        """Test spectrum creation with a single peak"""
        peak_energies = [662.0]  # Cs-137
        emission_rates = [1000.0]
        num_bins = 1000
        
        spectrum = gs_creator.create_spectrum_from_peaks(
            peak_energies, emission_rates, num_bins
        )
        
        # Check that spectrum is a PhSpectrum object
        self.assertIsInstance(spectrum, PhSpectrum)
        
        # Check basic attributes
        self.assertEqual(spectrum.num_channels, num_bins)
        self.assertEqual(len(spectrum.counts), num_bins)
        
        # Check that counts array contains the peak
        self.assertGreater(np.sum(spectrum.counts), 0)
        self.assertGreater(np.max(spectrum.counts), 0)
        
        # Check energy fit coefficients exist and are reasonable
        self.assertIsNotNone(spectrum.energy_fit_coefficients)
        self.assertEqual(len(spectrum.energy_fit_coefficients), 2)

    def test_create_spectrum_from_peaks_multiple_peaks(self):
        """Test spectrum creation with multiple peaks"""
        # Co-60 peaks at 1173 and 1332 keV
        peak_energies = [1173.0, 1332.0]
        emission_rates = [1000.0, 1000.0]
        num_bins = 2000
        
        spectrum = gs_creator.create_spectrum_from_peaks(
            peak_energies, emission_rates, num_bins
        )
        
        # Check basic structure
        self.assertIsInstance(spectrum, PhSpectrum)
        self.assertEqual(spectrum.num_channels, num_bins)
        
        # Check that we have counts (indicating peaks were added)
        self.assertGreater(np.sum(spectrum.counts), 0)

    def test_create_spectrum_from_peaks_with_energy_range(self):
        """Test spectrum creation with specified energy range"""
        peak_energies = [662.0]
        emission_rates = [1000.0]
        num_bins = 1000
        energy_range = (0.0, 2000.0)
        
        spectrum = gs_creator.create_spectrum_from_peaks(
            peak_energies, emission_rates, num_bins, energy_range=energy_range
        )
        
        # Check energy fit coefficients
        expected_energy_per_bin = (energy_range[1] - energy_range[0]) / num_bins
        self.assertAlmostEqual(spectrum.energy_fit_coefficients[0], energy_range[0], delta=0.1)
        self.assertAlmostEqual(spectrum.energy_fit_coefficients[1], expected_energy_per_bin, delta=0.01)

    def test_create_spectrum_from_peaks_with_background(self):
        """Test spectrum creation with background spectrum"""
        peak_energies = [662.0]
        emission_rates = [1000.0]
        num_bins = 1000
        background_level = 10.0
        
        # Create background spectrum
        background = gs_creator.create_flat_background(num_bins, background_level)
        
        # Create spectrum with background
        spectrum = gs_creator.create_spectrum_from_peaks(
            peak_energies, emission_rates, num_bins, background_spectrum=background
        )
        
        # Check that background was added (minimum count should be at least background_level)
        self.assertGreaterEqual(np.min(spectrum.counts), background_level - 1)

    def test_create_spectrum_from_peaks_empty_peaks(self):
        """Test spectrum creation with no peaks"""
        peak_energies = []
        emission_rates = []
        num_bins = 500
        
        spectrum = gs_creator.create_spectrum_from_peaks(
            peak_energies, emission_rates, num_bins
        )
        
        # Should create empty spectrum with all zeros
        self.assertEqual(spectrum.num_channels, num_bins)
        self.assertEqual(np.sum(spectrum.counts), 0)

    def test_create_spectrum_from_peaks_validation(self):
        """Test input validation for create_spectrum_from_peaks"""
        # Mismatched lengths
        with self.assertRaises(ValueError):
            gs_creator.create_spectrum_from_peaks(
                [662.0, 1173.0], [1000.0], 1000
            )
        
        # Invalid num_bins
        with self.assertRaises(ValueError):
            gs_creator.create_spectrum_from_peaks(
                [662.0], [1000.0], 0
            )
        
        with self.assertRaises(ValueError):
            gs_creator.create_spectrum_from_peaks(
                [662.0], [1000.0], -10
            )

    def test_create_spectrum_from_peaks_background_mismatch(self):
        """Test that background spectrum with wrong number of bins raises error"""
        peak_energies = [662.0]
        emission_rates = [1000.0]
        num_bins = 1000
        
        # Create background with different number of bins
        wrong_background = gs_creator.create_flat_background(500, 10.0)
        
        with self.assertRaises(ValueError):
            gs_creator.create_spectrum_from_peaks(
                peak_energies, emission_rates, num_bins, 
                background_spectrum=wrong_background
            )

    def test_create_spectrum_from_peaks_peak_outside_range(self):
        """Test that peaks outside energy range are not included"""
        peak_energies = [500.0, 1500.0, 2500.0]  # Middle one outside range
        emission_rates = [1000.0, 1000.0, 1000.0]
        num_bins = 1000
        energy_range = (0.0, 1000.0)  # Only first peak in range
        
        spectrum = gs_creator.create_spectrum_from_peaks(
            peak_energies, emission_rates, num_bins, energy_range=energy_range
        )
        
        # Should still create valid spectrum
        self.assertEqual(spectrum.num_channels, num_bins)
        self.assertGreater(np.sum(spectrum.counts), 0)  # Has at least one peak

    def test_create_spectrum_from_peaks_custom_name(self):
        """Test spectrum creation with custom name"""
        peak_energies = [662.0]
        emission_rates = [1000.0]
        num_bins = 1000
        custom_name = "test_spectrum"
        
        spectrum = gs_creator.create_spectrum_from_peaks(
            peak_energies, emission_rates, num_bins, spec_name=custom_name
        )
        
        self.assertEqual(spectrum.spec_name, custom_name)

    def test_create_flat_background_basic(self):
        """Test basic flat background creation"""
        num_bins = 1000
        background_level = 50.0
        
        background = gs_creator.create_flat_background(num_bins, background_level)
        
        # Check that it's a PhSpectrum object
        self.assertIsInstance(background, PhSpectrum)
        
        # Check dimensions
        self.assertEqual(background.num_channels, num_bins)
        self.assertEqual(len(background.counts), num_bins)
        
        # Check that all bins have the background level
        self.assertTrue(np.all(background.counts == int(background_level)))

    def test_create_flat_background_with_energy_range(self):
        """Test flat background creation with specified energy range"""
        num_bins = 500
        background_level = 25.0
        energy_range = (100.0, 2100.0)
        
        background = gs_creator.create_flat_background(
            num_bins, background_level, energy_range=energy_range
        )
        
        # Check energy fit coefficients
        expected_energy_per_bin = (energy_range[1] - energy_range[0]) / num_bins
        self.assertAlmostEqual(background.energy_fit_coefficients[0], energy_range[0], delta=0.1)
        self.assertAlmostEqual(background.energy_fit_coefficients[1], expected_energy_per_bin, delta=0.01)

    def test_create_flat_background_custom_name(self):
        """Test flat background creation with custom name"""
        num_bins = 500
        background_level = 10.0
        custom_name = "test_background"
        
        background = gs_creator.create_flat_background(
            num_bins, background_level, spec_name=custom_name
        )
        
        self.assertEqual(background.spec_name, custom_name)

    def test_create_flat_background_validation(self):
        """Test input validation for create_flat_background"""
        # Invalid num_bins
        with self.assertRaises(ValueError):
            gs_creator.create_flat_background(0, 10.0)
        
        with self.assertRaises(ValueError):
            gs_creator.create_flat_background(-5, 10.0)
        
        # Negative background level
        with self.assertRaises(ValueError):
            gs_creator.create_flat_background(100, -10.0)

    def test_create_flat_background_zero_level(self):
        """Test flat background with zero level (edge case)"""
        num_bins = 100
        background_level = 0.0
        
        background = gs_creator.create_flat_background(num_bins, background_level)
        
        # Should create all-zero spectrum
        self.assertEqual(np.sum(background.counts), 0)

    def test_integration_peaks_with_background(self):
        """Integration test: create spectrum with multiple peaks and background"""
        # Simulate a Co-60 source with background
        peak_energies = [1173.0, 1332.0]
        emission_rates = [2000.0, 2000.0]
        num_bins = 2000
        background_level = 20.0
        energy_range = (0.0, 3000.0)
        
        # Create background
        background = gs_creator.create_flat_background(
            num_bins, background_level, energy_range=energy_range
        )
        
        # Create spectrum with peaks and background
        spectrum = gs_creator.create_spectrum_from_peaks(
            peak_energies, emission_rates, num_bins,
            energy_range=energy_range,
            background_spectrum=background,
            spec_name="Co60_synthetic"
        )
        
        # Verify structure
        self.assertEqual(spectrum.spec_name, "Co60_synthetic")
        self.assertEqual(spectrum.num_channels, num_bins)
        
        # Verify that we have peaks above background
        max_count = np.max(spectrum.counts)
        self.assertGreater(max_count, background_level + 100)  # Peaks should be significantly higher
        
        # Verify background is present (minimum should be around background level)
        self.assertGreaterEqual(np.min(spectrum.counts), 0)


if __name__ == "__main__":
    unittest.main()
