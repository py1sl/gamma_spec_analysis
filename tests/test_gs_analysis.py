import unittest
import numpy as np
import gs_analysis as gs
from gs_analysis import EfficiencyFitType
import gs_spe_reading
import ph_spectrum


class analysis_test_case(unittest.TestCase):
    """tests for functions defined in gs_analysis (energy bins, efficiency,
    activity, peak finding, and doublet identification)"""

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
        spec = gs_spe_reading.read_dollar_spe("../test_data/Ba_133_raised_1.Spe")
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

    def test_eff_fit(self):
        """tests for efficency function fitting"""
        self.assertRaises(ValueError, gs.calc_energy_efficiency, 1.3, [1, 1, 1, 1], 5)

        # Test EfficiencyFitType.LOG (logarithmic fit)
        eff_coeff = [1.0, 0.5, 0.1]
        energy = 1.0  # MeV
        eff = gs.calc_energy_efficiency(energy, eff_coeff, eff_fit=EfficiencyFitType.LOG)
        self.assertIsInstance(eff, float)
        self.assertGreater(eff, 0)

        # Test EfficiencyFitType.INVERSE_ENERGY (inverse energy fit)
        eff = gs.calc_energy_efficiency(energy, eff_coeff, eff_fit=EfficiencyFitType.INVERSE_ENERGY)
        self.assertIsInstance(eff, float)
        self.assertGreater(eff, 0)

    def test_identify_doublets(self):
        """tests for identify_doublets function"""
        # Build a simple energy bin array: channel i -> i keV
        ebins = np.arange(0, 500, dtype=float)

        # Three peaks: channels 100 (100 keV), 107 (107 keV), 300 (300 keV)
        peaks = np.array([100, 107, 300])

        # With max_separation=10 keV, (100,107) is a doublet; (107,300) is not
        doublets = gs.identify_doublets(peaks, ebins, max_separation=10.0)
        self.assertEqual(len(doublets), 1)
        self.assertEqual(doublets[0], (100, 107))

        # With max_separation=5 keV, no doublets
        doublets_none = gs.identify_doublets(peaks, ebins, max_separation=5.0)
        self.assertEqual(len(doublets_none), 0)

        # With max_separation=200 keV, both adjacent pairs qualify
        doublets_all = gs.identify_doublets(peaks, ebins, max_separation=200.0)
        self.assertEqual(len(doublets_all), 2)

    def test_identify_doublets_empty(self):
        """tests identify_doublets with empty or single-peak input"""
        ebins = np.arange(0, 500, dtype=float)

        # Empty peaks list
        doublets = gs.identify_doublets([], ebins)
        self.assertEqual(doublets, [])

        # Single peak – no pairs to compare
        doublets = gs.identify_doublets([200], ebins)
        self.assertEqual(doublets, [])

    def test_peak_finder(self):
        """tests for peak finding function"""
        spec = gs_spe_reading.read_dollar_spe("../test_data/Ba_133_raised_1.Spe")

        # Run peak finder with reasonable parameters
        smoothed, peaks = gs.peak_finder(spec.counts, prominence=100, wlen=50)

        self.assertIsInstance(smoothed, np.ndarray)
        self.assertIsInstance(peaks, np.ndarray)
        self.assertEqual(len(smoothed), len(spec.counts))
        self.assertGreater(len(peaks), 0)  # Should find some peaks

    def test_peak_counts(self):
        """tests for peak counts calculation"""
        spec = gs_spe_reading.read_dollar_spe("../test_data/Ba_133_raised_1.Spe")
        ebins = gs.generate_ebins(spec)
        smoothed, peaks = gs.peak_finder(spec.counts, prominence=100, wlen=50)

        if len(peaks) > 0:
            # Test peak_counts for first peak
            peak_idx, counts = gs.peak_counts(peaks, 0, smoothed, ebins)
            self.assertEqual(peak_idx, peaks[0])
            self.assertIsInstance(counts, float)

    def test_mariscotti_peak_finder_basic(self):
        """tests for Mariscotti peak finding function - basic functionality"""
        x = np.arange(100)
        counts = 10 + 100 * np.exp(-((x - 50) ** 2) / (2 * 5 ** 2))

        smoothed, peaks = gs.mariscotti_peak_finder(counts)

        self.assertIsInstance(smoothed, np.ndarray)
        self.assertIsInstance(peaks, np.ndarray)
        self.assertEqual(len(smoothed), len(counts))
        self.assertGreater(len(peaks), 0)
        peak_found_near_50 = any(abs(p - 50) < 10 for p in peaks)
        self.assertTrue(peak_found_near_50, "Expected to find a peak near position 50")

    def test_mariscotti_peak_finder_multiple_peaks(self):
        """tests for Mariscotti peak finding with multiple peaks"""
        x = np.arange(200)
        counts = (10 +
                  80 * np.exp(-((x - 50) ** 2) / (2 * 5 ** 2)) +
                  60 * np.exp(-((x - 150) ** 2) / (2 * 5 ** 2)))

        smoothed, peaks = gs.mariscotti_peak_finder(counts)
        self.assertGreaterEqual(len(peaks), 1)

    def test_mariscotti_peak_finder_no_peaks(self):
        """tests for Mariscotti peak finding with flat data"""
        counts = np.ones(100) * 50
        smoothed, peaks = gs.mariscotti_peak_finder(counts, threshold=-0.1)
        self.assertEqual(len(peaks), 0)

    def test_mariscotti_peak_finder_edge_cases(self):
        """tests for Mariscotti peak finding edge cases"""
        counts_min = np.array([1, 2, 3, 4, 5])
        smoothed, peaks = gs.mariscotti_peak_finder(counts_min)
        self.assertEqual(len(smoothed), 5)
        self.assertIsInstance(peaks, np.ndarray)

        short_array = [1, 2, 3, 4]
        self.assertRaises(ValueError, gs.mariscotti_peak_finder, short_array)

    def test_mariscotti_peak_finder_parameters(self):
        """tests for Mariscotti peak finding with different parameters"""
        x = np.arange(100)
        counts = 10 + 100 * np.exp(-((x - 50) ** 2) / (2 * 5 ** 2))

        smoothed1, peaks1 = gs.mariscotti_peak_finder(counts, smooth_iterations=1)
        smoothed2, peaks2 = gs.mariscotti_peak_finder(counts, smooth_iterations=3)
        self.assertEqual(len(smoothed1), len(counts))
        self.assertEqual(len(smoothed2), len(counts))

        smoothed_low, peaks_low = gs.mariscotti_peak_finder(counts, threshold=-0.1)
        smoothed_high, peaks_high = gs.mariscotti_peak_finder(counts, threshold=-10.0)
        self.assertGreaterEqual(len(peaks_low), len(peaks_high))

        smoothed_auto, peaks_auto = gs.mariscotti_peak_finder(counts, threshold=None)
        self.assertIsInstance(peaks_auto, np.ndarray)
        self.assertEqual(len(smoothed_auto), len(counts))

    # ------------------------------------------------------------------
    # Activity calculation
    # ------------------------------------------------------------------

    def test_calc_activity_known_value(self):
        """calc_activity must equal N / (T * I * eps)."""
        net = 1000.0
        T = 100.0
        I = 0.85
        eps = 0.05
        expected = net / (T * I * eps)
        self.assertAlmostEqual(gs.calc_activity(net, T, I, eps), expected)

    def test_calc_activity_returns_float(self):
        act = gs.calc_activity(500.0, 60.0, 0.90, 0.10)
        self.assertIsInstance(act, float)
        self.assertGreater(act, 0.0)

    def test_calc_activity_invalid_live_time(self):
        self.assertRaises(ValueError, gs.calc_activity, 100.0, 0.0, 0.85, 0.05)
        self.assertRaises(ValueError, gs.calc_activity, 100.0, -1.0, 0.85, 0.05)

    def test_calc_activity_invalid_emission_probability(self):
        self.assertRaises(ValueError, gs.calc_activity, 100.0, 60.0, 0.0, 0.05)
        self.assertRaises(ValueError, gs.calc_activity, 100.0, 60.0, 1.5, 0.05)
        self.assertRaises(ValueError, gs.calc_activity, 100.0, 60.0, -0.1, 0.05)

    def test_calc_activity_invalid_efficiency(self):
        self.assertRaises(ValueError, gs.calc_activity, 100.0, 60.0, 0.85, 0.0)
        self.assertRaises(ValueError, gs.calc_activity, 100.0, 60.0, 0.85, 1.5)
        self.assertRaises(ValueError, gs.calc_activity, 100.0, 60.0, 0.85, -0.1)

    def test_calc_activity_uncertainty_known_value(self):
        """calc_activity_uncertainty = sigma_N / (T * I * eps)."""
        sigma_N = 50.0
        T, I, eps = 100.0, 0.85, 0.05
        expected = sigma_N / (T * I * eps)
        self.assertAlmostEqual(gs.calc_activity_uncertainty(sigma_N, T, I, eps), expected)

    def test_calc_activity_uncertainty_returns_float(self):
        unc = gs.calc_activity_uncertainty(30.0, 60.0, 0.90, 0.10)
        self.assertIsInstance(unc, float)
        self.assertGreaterEqual(unc, 0.0)

    def test_calc_activity_uncertainty_invalid_args(self):
        """Same input validation as calc_activity."""
        self.assertRaises(ValueError, gs.calc_activity_uncertainty, 10.0, 0.0, 0.85, 0.05)
        self.assertRaises(ValueError, gs.calc_activity_uncertainty, 10.0, 60.0, 0.0, 0.05)
        self.assertRaises(ValueError, gs.calc_activity_uncertainty, 10.0, 60.0, 0.85, 0.0)

    def test_calc_activity_roundtrip(self):
        """Activity / uncertainty ratio should equal net / net_unc (same denominator)."""
        net, net_unc = 800.0, 40.0
        T, I, eps = 120.0, 0.80, 0.06
        act = gs.calc_activity(net, T, I, eps)
        act_unc = gs.calc_activity_uncertainty(net_unc, T, I, eps)
        self.assertAlmostEqual(act / act_unc, net / net_unc, places=10)


class backward_compat_test_case(unittest.TestCase):
    """Smoke tests: gs_analysis must still re-export all sub-module symbols."""

    def test_smoothing_re_exports(self):
        self.assertTrue(callable(gs.five_point_smooth))
        self.assertTrue(callable(gs.three_point_smooth))
        self.assertTrue(callable(gs.moving_average))
        self.assertTrue(callable(gs.exponential_moving_average))

    def test_background_re_exports(self):
        self.assertTrue(callable(gs.gross_count))
        self.assertTrue(callable(gs.calc_bg))
        self.assertTrue(callable(gs.calc_bg_uncertainty))
        self.assertTrue(hasattr(gs, "BackgroundMethod"))

    def test_peak_fitting_re_exports(self):
        self.assertTrue(callable(gs.gaussian))
        self.assertTrue(callable(gs.double_gaussian))
        self.assertTrue(callable(gs.get_peak_roi))
        self.assertTrue(callable(gs.fit_peak))
        self.assertTrue(callable(gs.fit_doublet))
        self.assertTrue(callable(gs.net_counts))
        self.assertTrue(callable(gs.net_counts_uncertainty))
        self.assertTrue(callable(gs.peak_fwhm))
        self.assertTrue(callable(gs.fit_chi2))


if __name__ == "__main__":
    unittest.main()
