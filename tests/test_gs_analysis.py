import unittest
from unittest.mock import patch, mock_open, call
import numpy as np
import matplotlib.pyplot as plt
import gs_analysis as gs
import ph_spectrum


class analysis_test_case(unittest.TestCase):
    """tests for analysis functions"""

    def test_counts(self):
        """tests related to the counts"""
        # gross counts
        counts = [1, 1, 1, 1, 1]
        gc = gs.gross_count(counts, 1, 4)
        self.assertEqual(gc, 3)
        self.assertRaises(ValueError, gs.gross_count, counts, -1, 4)
        self.assertRaises(ValueError, gs.gross_count, counts, 1, 10)
        self.assertRaises(ValueError, gs.gross_count, counts, 10, 4)

        # background
        self.assertRaises(ValueError, gs.calc_bg, counts, -1, 4)
        self.assertRaises(ValueError, gs.calc_bg, counts, 1, 10)
        self.assertRaises(ValueError, gs.calc_bg, counts, 10, 4)
        self.assertRaises(ValueError, gs.calc_bg, counts, 1, 4, 2)

        # net
        self.assertRaises(ValueError, gs.net_counts, counts, -1, 4)
        self.assertRaises(ValueError, gs.net_counts, counts, 1, 10)
        self.assertRaises(ValueError, gs.net_counts, counts, 10, 4)

    def test_ebins(self):
        """tests rlated to energy bins"""
        # testing find e pos
        ebins = [1, 2, 3, 4, 5]
        self.assertEqual(gs.find_energy_pos(ebins, 1.5), 0)
        self.assertEqual(gs.find_energy_pos(ebins, 1), 0)
        self.assertEqual(gs.find_energy_pos(ebins, 4.9), 3)
        self.assertFalse(gs.find_energy_pos(ebins, -1))
        self.assertFalse(gs.find_energy_pos(ebins, 5))
        self.assertFalse(gs.find_energy_pos(ebins, 10))
        self.assertFalse(gs.find_energy_pos(ebins, 0))

        # generating ebins
        spec = gs.get_spect("../test_data/Ba_133_raised_1.Spe")
        ebins = gs.generate_ebins(spec)
        self.assertEqual(len(ebins), len(spec.counts))
        
        # Test generate_ebins with invalid coefficients (not length 2)
        spec_invalid = ph_spectrum.PhSpectrum()
        spec_invalid.energy_fit_coefficients = [1.0, 2.0, 3.0]  # Length 3, should fail
        spec_invalid.num_channels = 10
        spec_invalid.counts = np.array([1] * 10)
        self.assertRaises(ValueError, gs.generate_ebins, spec_invalid)
        
        # Test generate_ebins with num_channels == 0
        spec_zero = ph_spectrum.PhSpectrum()
        spec_zero.energy_fit_coefficients = [1.0, 2.0]
        spec_zero.num_channels = 0
        spec_zero.counts = np.array([1, 2, 3, 4, 5])
        ebins_zero = gs.generate_ebins(spec_zero)
        self.assertEqual(len(ebins_zero), 5)
        self.assertEqual(spec_zero.num_channels, 5)

    def test_roi(self):
        """tests for extracting a region of interest"""
        spec = gs.get_spect("../test_data/Ba_133_raised_1.Spe")
        ebins = gs.generate_ebins(spec)
        peak_ebin, data = gs.get_peak_roi(230, spec.counts, ebins)
        self.assertEqual(len(data), 20)
        self.assertEqual(len(data), len(peak_ebin))
        self.assertRaises(ValueError, gs.get_peak_roi, 2, spec.counts, ebins)
        self.assertRaises(ValueError, gs.get_peak_roi, 10000, spec.counts, ebins)

    def test_eff_fit(self):
        """tests for efficency function fitting"""
        self.assertRaises(ValueError, gs.calc_energy_efficiency, 1.3, [1, 1, 1, 1], 5)
        
        # Test eff_fit=1 (logarithmic fit)
        eff_coeff = [1.0, 0.5, 0.1]
        energy = 1.0  # MeV
        eff = gs.calc_energy_efficiency(energy, eff_coeff, eff_fit=1)
        self.assertIsInstance(eff, float)
        self.assertGreater(eff, 0)
        
        # Test eff_fit=2 (inverse energy fit)
        eff = gs.calc_energy_efficiency(energy, eff_coeff, eff_fit=2)
        self.assertIsInstance(eff, float)
        self.assertGreater(eff, 0)

    def test_smoothing(self):
        """tests related to smoothing functions"""
        # testing 5 point smooth
        spec = gs.get_spect("../test_data/Ba_133_raised_1.Spe")
        smoothed = gs.five_point_smooth(spec.counts)
        self.assertEqual(len(smoothed), len(spec.counts))
        self.assertEqual(smoothed[0], spec.counts[0])
        self.assertEqual(smoothed[-1], spec.counts[-1])
        
        # Test with array that's too short
        short_array = [1, 2, 3, 4]
        self.assertRaises(ValueError, gs.five_point_smooth, short_array)

    def test_getting_data(self):
        """tests for getting data"""
        spec = gs.get_spect("../test_data/Ba_133_raised_1.Spe")
        self.assertTrue(len(spec.counts) > 0)

    def test_background_calculation(self):
        """tests for background calculation functions"""
        counts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Test calc_bg with valid parameters
        bg = gs.calc_bg(counts, 2, 7, m=1)
        self.assertIsInstance(bg, float)
        self.assertGreaterEqual(bg, 0)
        
        # Test estimate_background_trapezoid directly
        bg_trap = gs.estimate_background_trapezoid(counts, 2, 7)
        self.assertIsInstance(bg_trap, float)
        self.assertGreaterEqual(bg_trap, 0)
        
        # Test edge case: channels at the start
        bg_start = gs.estimate_background_trapezoid(counts, 0, 3)
        self.assertIsInstance(bg_start, float)
        
        # Test edge case: channels at the end
        bg_end = gs.estimate_background_trapezoid(counts, 7, 9)
        self.assertIsInstance(bg_end, float)
    
    def test_net_counts(self):
        """tests for net counts calculation"""
        counts = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        
        # Test net_counts function
        nc = gs.net_counts(counts, 2, 7, m=1)
        self.assertIsInstance(nc, float)
    
    def test_gaussian(self):
        """tests for gaussian function"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        a = 10.0
        x0 = 3.0
        sigma = 1.0
        
        result = gs.gaussian(x, a, x0, sigma)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), len(x))
        # Peak should be at x0
        max_idx = np.argmax(result)
        self.assertEqual(x[max_idx], x0)
    
    def test_fit_peak(self):
        """tests for peak fitting function"""
        # Create synthetic peak data
        x = np.linspace(0, 10, 50)
        # Create a Gaussian peak
        a = 100.0
        x0 = 5.0
        sigma = 1.0
        y = gs.gaussian(x, a, x0, sigma)
        # Add some noise
        y = y + np.random.normal(0, 1, len(y))
        
        # Fit the peak
        popt = gs.fit_peak(x, y)
        self.assertEqual(len(popt), 3)
        # Check that fitted parameters are reasonable
        self.assertAlmostEqual(popt[1], x0, delta=0.5)  # x0
        self.assertAlmostEqual(popt[2], sigma, delta=0.5)  # sigma
    
    def test_peak_finder(self):
        """tests for peak finding function"""
        spec = gs.get_spect("../test_data/Ba_133_raised_1.Spe")
        
        # Run peak finder with reasonable parameters
        smoothed, peaks = gs.peak_finder(spec.counts, prominence=100, wlen=50)
        
        self.assertIsInstance(smoothed, np.ndarray)
        self.assertIsInstance(peaks, np.ndarray)
        self.assertEqual(len(smoothed), len(spec.counts))
        self.assertGreater(len(peaks), 0)  # Should find some peaks
    
    def test_peak_counts(self):
        """tests for peak counts calculation"""
        spec = gs.get_spect("../test_data/Ba_133_raised_1.Spe")
        ebins = gs.generate_ebins(spec)
        smoothed, peaks = gs.peak_finder(spec.counts, prominence=100, wlen=50)
        
        if len(peaks) > 0:
            # Test peak_counts for first peak
            peak_idx, counts = gs.peak_counts(peaks, 0, smoothed, ebins)
            self.assertEqual(peak_idx, peaks[0])
            self.assertIsInstance(counts, float)


class TestPlotting(unittest.TestCase):
    """tests relating to plotting functions"""

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.show")
    def test_plot_spec(self, mock_show, mock_savefig):
        counts = [1, 10, 100, 1000]
        erg = [1, 2, 3, 4]
        fname = "test_plot.png"

        # called with just counts
        gs.plot_spec(counts)
        # Assert that show was called
        mock_show.assert_called_once()

        # called with counts and energy
        gs.plot_spec(counts, erg=erg)
        # Assert that show was called
        mock_show.assert_called()

        # called with a file name
        mock_savefig.reset_mock()
        gs.plot_spec(counts, fname=fname)
        # Assert that savefig was called with the specified filename
        mock_savefig.assert_called_once_with(fname)

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.show")
    def test_plot_spect_peaks(self, mock_show, mock_savefig):
        counts = [1, 10, 100, 1000]
        erg = [1, 2, 3, 4]
        peaks = [3]
        fname = "test_plot.png"

        # called with data
        gs.plot_spect_peaks(counts, erg, peaks)
        # Assert that show was called
        mock_show.assert_called_once()

        # called with data and fname
        # Assert that savefig was called with the specified filename
        # Assert that show was called
        gs.plot_spect_peaks(counts, erg, peaks, fname)
        mock_savefig.assert_called_once_with(fname)


if __name__ == "__main__":
    unittest.main()
