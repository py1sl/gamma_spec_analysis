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
        self.assertRaises(ValueError, gs.calc_bg, counts, 1, 4, 5)  # Invalid method

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
    
    def test_three_point_smooth(self):
        """tests for 3 point smoothing function"""
        # Test with simple data
        counts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        smoothed = gs.three_point_smooth(counts)
        
        # Check length is preserved
        self.assertEqual(len(smoothed), len(counts))
        
        # Check first and last elements are unchanged
        self.assertEqual(smoothed[0], counts[0])
        self.assertEqual(smoothed[-1], counts[-1])
        
        # Check middle element is average of 3 points
        # For index 1: (1 + 2 + 3) / 3 = 2.0
        self.assertAlmostEqual(smoothed[1], 2.0)
        # For index 5: (5 + 6 + 7) / 3 = 6.0
        self.assertAlmostEqual(smoothed[5], 6.0)
        
        # Test with array that's too short
        short_array = [1, 2]
        self.assertRaises(ValueError, gs.three_point_smooth, short_array)
    
    def test_moving_average(self):
        """tests for moving average smoothing function"""
        # Test with simple data and default window
        counts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        smoothed = gs.moving_average(counts, window=5)
        
        # Check length is preserved
        self.assertEqual(len(smoothed), len(counts))
        
        # Check that smoothing reduces variance
        self.assertLessEqual(np.std(smoothed), np.std(counts))
        
        # Test with window=3
        smoothed_3 = gs.moving_average(counts, window=3)
        self.assertEqual(len(smoothed_3), len(counts))
        
        # Middle element should be average of 3 points
        # For index 5: (5 + 6 + 7) / 3 = 6.0
        self.assertAlmostEqual(smoothed_3[5], 6.0)
        
        # Test with invalid window size (even number)
        self.assertRaises(ValueError, gs.moving_average, counts, window=4)
        
        # Test with invalid window size (negative)
        self.assertRaises(ValueError, gs.moving_average, counts, window=-1)
        
        # Test with array shorter than window
        short_array = [1, 2, 3]
        self.assertRaises(ValueError, gs.moving_average, short_array, window=5)
    
    def test_exponential_moving_average(self):
        """tests for exponential moving average smoothing function"""
        # Test with simple data
        counts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        smoothed = gs.exponential_moving_average(counts, alpha=0.3)
        
        # Check length is preserved
        self.assertEqual(len(smoothed), len(counts))
        
        # First element should be unchanged
        self.assertEqual(smoothed[0], counts[0])
        
        # Second element: alpha * counts[1] + (1 - alpha) * smoothed[0]
        # = 0.3 * 2 + 0.7 * 1 = 0.6 + 0.7 = 1.3
        self.assertAlmostEqual(smoothed[1], 1.3)
        
        # Test with different alpha values
        smoothed_low = gs.exponential_moving_average(counts, alpha=0.1)
        smoothed_high = gs.exponential_moving_average(counts, alpha=0.9)
        
        # Lower alpha should result in smoother output
        self.assertLessEqual(np.std(smoothed_low), np.std(smoothed_high))
        
        # Test with invalid alpha (too low)
        self.assertRaises(ValueError, gs.exponential_moving_average, counts, alpha=0.0)
        
        # Test with invalid alpha (too high)
        self.assertRaises(ValueError, gs.exponential_moving_average, counts, alpha=1.0)
        
        # Test with invalid alpha (negative)
        self.assertRaises(ValueError, gs.exponential_moving_average, counts, alpha=-0.1)
        
        # Test with invalid alpha (greater than 1)
        self.assertRaises(ValueError, gs.exponential_moving_average, counts, alpha=1.5)

    def test_five_point_smooth_correctness(self):
        """Test that five_point_smooth produces correct results"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = gs.five_point_smooth(data)
        
        # First two and last two elements should be unchanged
        self.assertEqual(result[0], data[0])
        self.assertEqual(result[1], data[1])
        self.assertEqual(result[-2], data[-2])
        self.assertEqual(result[-1], data[-1])
        
        # Check middle element calculation using actual test data
        # For index 2: (1/9) * (data[0] + data[4] + 2*data[3] + 2*data[1] + 3*data[2])
        expected_idx2 = (1.0 / 9.0) * (data[0] + data[4] + 2 * data[3] + 2 * data[1] + 3 * data[2])
        np.testing.assert_almost_equal(result[2], expected_idx2)

    def test_three_point_smooth_correctness(self):
        """Test that three_point_smooth produces correct results"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = gs.three_point_smooth(data)
        
        # First and last elements should be unchanged
        self.assertEqual(result[0], data[0])
        self.assertEqual(result[-1], data[-1])
        
        # Check middle element calculation
        # For index 1: (1 + 2 + 3) / 3 = 2
        expected_idx1 = (data[0] + data[1] + data[2]) / 3.0
        np.testing.assert_almost_equal(result[1], expected_idx1)

    def test_moving_average_correctness(self):
        """Test that moving_average produces correct results"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = gs.moving_average(data, window=5)
        
        # Check middle element calculation
        # For index 5 with window=5: mean of [4, 5, 6, 7, 8] = 6
        expected_idx5 = np.mean(data[3:8])
        np.testing.assert_almost_equal(result[5], expected_idx5)

    def test_smoothing_maintains_signal_integrity(self):
        """Test that smoothing preserves important signal properties"""
        # Create a signal with known characteristics
        data = np.concatenate([
            np.zeros(100),
            np.full(50, 100),  # Peak
            np.zeros(100)
        ])
        
        # Apply smoothing
        result_5pt = gs.five_point_smooth(data)
        result_3pt = gs.three_point_smooth(data)
        result_ma = gs.moving_average(data, window=5)
        
        # All should maintain similar total counts (conservation)
        np.testing.assert_allclose(np.sum(result_5pt), np.sum(data), rtol=0.1)
        np.testing.assert_allclose(np.sum(result_3pt), np.sum(data), rtol=0.1)
        np.testing.assert_allclose(np.sum(result_ma), np.sum(data), rtol=0.1)
        
        # Peak location should be preserved (within a few bins)
        peak_orig = np.argmax(data)
        peak_5pt = np.argmax(result_5pt)
        peak_3pt = np.argmax(result_3pt)
        peak_ma = np.argmax(result_ma)
        
        self.assertLess(abs(peak_5pt - peak_orig), 5)
        self.assertLess(abs(peak_3pt - peak_orig), 5)
        self.assertLess(abs(peak_ma - peak_orig), 5)

    def test_getting_data(self):
        """tests for getting data"""
        spec = gs.get_spect("../test_data/Ba_133_raised_1.Spe")
        self.assertTrue(len(spec.counts) > 0)

    def test_background_calculation(self):
        """tests for background calculation functions"""
        counts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Test calc_bg with valid parameters (method 1 - trapezoid)
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
        
        # Test method 2 - linear interpolation
        bg_linear = gs.calc_bg(counts, 2, 7, m=2)
        self.assertIsInstance(bg_linear, float)
        self.assertGreaterEqual(bg_linear, 0)
        
        # Test estimate_background_linear directly
        bg_linear_direct = gs.estimate_background_linear(counts, 2, 7)
        self.assertIsInstance(bg_linear_direct, float)
        self.assertGreaterEqual(bg_linear_direct, 0)
        
        # Test method 3 - step function
        bg_step = gs.calc_bg(counts, 2, 7, m=3)
        self.assertIsInstance(bg_step, float)
        self.assertGreaterEqual(bg_step, 0)
        
        # Test estimate_background_step directly
        bg_step_direct = gs.estimate_background_step(counts, 2, 7)
        self.assertIsInstance(bg_step_direct, float)
        self.assertGreaterEqual(bg_step_direct, 0)
        
        # Test method 4 - sliding window average
        bg_sliding = gs.calc_bg(counts, 2, 7, m=4)
        self.assertIsInstance(bg_sliding, float)
        self.assertGreaterEqual(bg_sliding, 0)
        
        # Test estimate_background_sliding_average directly
        bg_sliding_direct = gs.estimate_background_sliding_average(counts, 2, 7)
        self.assertIsInstance(bg_sliding_direct, float)
        self.assertGreaterEqual(bg_sliding_direct, 0)
        
        # Test invalid method number
        with self.assertRaises(ValueError):
            gs.calc_bg(counts, 2, 7, m=5)
    
    def test_background_methods_comparison(self):
        """Compare different background subtraction methods"""
        # Create a synthetic peak with known background
        x = np.arange(100)
        background_level = 50.0
        peak = 200 * np.exp(-((x - 50) ** 2) / (2 * 5 ** 2))
        counts = background_level + peak
        
        # Define peak region (channels 40-60)
        c1, c2 = 40, 60
        
        # All methods should give reasonable backgrounds
        bg_trap = gs.calc_bg(counts, c1, c2, m=1)
        bg_linear = gs.calc_bg(counts, c1, c2, m=2)
        bg_step = gs.calc_bg(counts, c1, c2, m=3)
        bg_sliding = gs.calc_bg(counts, c1, c2, m=4)
        
        # All should be positive
        self.assertGreater(bg_trap, 0)
        self.assertGreater(bg_linear, 0)
        self.assertGreater(bg_step, 0)
        self.assertGreater(bg_sliding, 0)
        
        # For a flat background, all methods should give similar results
        # (within reasonable tolerance given the peak interference)
        expected_bg = background_level * (c2 - c1)
        
        # Check that all methods are within reasonable range of expected
        # (allowing for variation due to peak edges)
        for bg_value in [bg_trap, bg_linear, bg_step, bg_sliding]:
            self.assertGreater(bg_value, expected_bg * 0.5)
            self.assertLess(bg_value, expected_bg * 2.0)
    
    def test_background_edge_cases_all_methods(self):
        """Test edge cases for all background methods"""
        counts = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
        
        # Test at spectrum edges for all methods
        for method in [1, 2, 3, 4]:
            # At start of spectrum
            bg_start = gs.calc_bg(counts, 0, 3, m=method)
            self.assertIsInstance(bg_start, float)
            self.assertGreaterEqual(bg_start, 0)
            
            # At end of spectrum
            bg_end = gs.calc_bg(counts, 7, 9, m=method)
            self.assertIsInstance(bg_end, float)
            self.assertGreaterEqual(bg_end, 0)
            
            # Single channel peak
            bg_single = gs.calc_bg(counts, 5, 6, m=method)
            self.assertIsInstance(bg_single, float)
            self.assertGreaterEqual(bg_single, 0)
    
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
    
    def test_mariscotti_peak_finder_basic(self):
        """tests for Mariscotti peak finding function - basic functionality"""
        # Create synthetic data with a clear peak
        x = np.arange(100)
        # Create a Gaussian peak at position 50
        counts = 10 + 100 * np.exp(-((x - 50) ** 2) / (2 * 5 ** 2))
        
        # Run Mariscotti peak finder with auto threshold
        smoothed, peaks = gs.mariscotti_peak_finder(counts)
        
        # Verify output types and shapes
        self.assertIsInstance(smoothed, np.ndarray)
        self.assertIsInstance(peaks, np.ndarray)
        self.assertEqual(len(smoothed), len(counts))
        
        # Should find at least one peak
        self.assertGreater(len(peaks), 0)
        
        # Peak should be near position 50
        peak_found_near_50 = any(abs(p - 50) < 10 for p in peaks)
        self.assertTrue(peak_found_near_50, "Expected to find a peak near position 50")
    
    def test_mariscotti_peak_finder_multiple_peaks(self):
        """tests for Mariscotti peak finding with multiple peaks"""
        # Create synthetic data with two peaks
        x = np.arange(200)
        counts = (10 + 
                  80 * np.exp(-((x - 50) ** 2) / (2 * 5 ** 2)) + 
                  60 * np.exp(-((x - 150) ** 2) / (2 * 5 ** 2)))
        
        # Run Mariscotti peak finder with auto threshold
        smoothed, peaks = gs.mariscotti_peak_finder(counts)
        
        # Should find multiple peaks
        self.assertGreaterEqual(len(peaks), 1)
    
    def test_mariscotti_peak_finder_no_peaks(self):
        """tests for Mariscotti peak finding with flat data"""
        # Create flat data - should find no peaks
        counts = np.ones(100) * 50
        
        smoothed, peaks = gs.mariscotti_peak_finder(counts, threshold=-0.1)
        
        # Should find no peaks (or very few)
        self.assertEqual(len(peaks), 0)
    
    def test_mariscotti_peak_finder_edge_cases(self):
        """tests for Mariscotti peak finding edge cases"""
        # Test with minimum size array
        counts_min = np.array([1, 2, 3, 4, 5])
        smoothed, peaks = gs.mariscotti_peak_finder(counts_min)
        self.assertEqual(len(smoothed), 5)
        self.assertIsInstance(peaks, np.ndarray)
        
        # Test with array that's too short
        short_array = [1, 2, 3, 4]
        self.assertRaises(ValueError, gs.mariscotti_peak_finder, short_array)
    
    def test_mariscotti_peak_finder_parameters(self):
        """tests for Mariscotti peak finding with different parameters"""
        # Create synthetic data
        x = np.arange(100)
        counts = 10 + 100 * np.exp(-((x - 50) ** 2) / (2 * 5 ** 2))
        
        # Test with different smoothing iterations
        smoothed1, peaks1 = gs.mariscotti_peak_finder(counts, smooth_iterations=1)
        smoothed2, peaks2 = gs.mariscotti_peak_finder(counts, smooth_iterations=3)
        
        self.assertEqual(len(smoothed1), len(counts))
        self.assertEqual(len(smoothed2), len(counts))
        
        # Test with explicit threshold values
        smoothed_low, peaks_low = gs.mariscotti_peak_finder(counts, threshold=-0.1)
        smoothed_high, peaks_high = gs.mariscotti_peak_finder(counts, threshold=-10.0)
        
        # Higher (less negative) threshold should find more or equal peaks
        self.assertGreaterEqual(len(peaks_low), len(peaks_high))
        
        # Test with auto threshold (None)
        smoothed_auto, peaks_auto = gs.mariscotti_peak_finder(counts, threshold=None)
        self.assertIsInstance(peaks_auto, np.ndarray)
        self.assertEqual(len(smoothed_auto), len(counts))


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
