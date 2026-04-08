import unittest
import numpy as np
import gs_peak_fitting as pf
import gs_analysis
import gs_spe_reading
from gs_background import BackgroundMethod


class peak_fitting_test_case(unittest.TestCase):
    """tests for peak fitting and characterisation functions in gs_peak_fitting"""

    def test_net_counts(self):
        """tests for net_counts function including invalid channel handling"""
        counts = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        nc = pf.net_counts(counts, 2, 7, m=BackgroundMethod.TRAPEZOID)
        self.assertIsInstance(nc, float)

        # Invalid channel ranges
        short = [1, 1, 1, 1, 1]
        self.assertRaises(ValueError, pf.net_counts, short, -1, 4)
        self.assertRaises(ValueError, pf.net_counts, short, 1, 10)
        self.assertRaises(ValueError, pf.net_counts, short, 10, 4)

    def test_roi(self):
        """tests for extracting a region of interest"""
        spec = gs_spe_reading.read_dollar_spe("../test_data/Ba_133_raised_1.Spe")
        ebins = gs_analysis.generate_ebins(spec)
        peak_ebin, data = pf.get_peak_roi(230, spec.counts, ebins)
        self.assertEqual(len(data), 20)
        self.assertEqual(len(data), len(peak_ebin))
        self.assertRaises(ValueError, pf.get_peak_roi, 2, spec.counts, ebins)
        self.assertRaises(ValueError, pf.get_peak_roi, 10000, spec.counts, ebins)

    def test_gaussian(self):
        """tests for gaussian function"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        a = 10.0
        x0 = 3.0
        sigma = 1.0

        result = pf.gaussian(x, a, x0, sigma)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), len(x))
        # Peak should be at x0
        max_idx = np.argmax(result)
        self.assertEqual(x[max_idx], x0)

    def test_fit_peak(self):
        """tests for peak fitting function"""
        x = np.linspace(0, 10, 50)
        a = 100.0
        x0 = 5.0
        sigma = 1.0
        y = pf.gaussian(x, a, x0, sigma)
        y = y + np.random.normal(0, 1, len(y))

        popt, pcov = pf.fit_peak(x, y)
        self.assertEqual(len(popt), 3)
        self.assertAlmostEqual(popt[1], x0, delta=0.5)  # x0
        self.assertAlmostEqual(popt[2], sigma, delta=0.5)  # sigma

    def test_double_gaussian(self):
        """tests for double_gaussian function"""
        x = np.linspace(0, 20, 100)
        a1, x01, sigma1 = 100.0, 5.0, 1.0
        a2, x02, sigma2 = 80.0, 15.0, 1.2
        y = pf.double_gaussian(x, a1, x01, sigma1, a2, x02, sigma2)

        # Output array has the correct length
        self.assertEqual(len(y), len(x))
        # Values are non-negative
        self.assertTrue(np.all(y >= 0))
        # Maximum is near one of the two peak centres
        peak_x = x[np.argmax(y)]
        self.assertTrue(abs(peak_x - x01) < 1.5 or abs(peak_x - x02) < 1.5)
        # Equals sum of individual Gaussians
        y1 = pf.gaussian(x, a1, x01, sigma1)
        y2 = pf.gaussian(x, a2, x02, sigma2)
        np.testing.assert_array_almost_equal(y, y1 + y2)

    def test_fit_doublet(self):
        """tests for doublet fitting function"""
        np.random.seed(42)
        x = np.linspace(0, 20, 200)
        a1, x01, sigma1 = 100.0, 7.0, 0.8
        a2, x02, sigma2 = 90.0, 10.0, 0.9
        y = pf.double_gaussian(x, a1, x01, sigma1, a2, x02, sigma2)
        y = y + np.random.normal(0, 1, len(y))

        popt, pcov = pf.fit_doublet(x, y)

        # Returns 6 parameters
        self.assertEqual(len(popt), 6)
        # Recovered centres should be close to the true values (within 1 unit)
        centres = sorted([popt[1], popt[4]])
        self.assertAlmostEqual(centres[0], min(x01, x02), delta=1.0)
        self.assertAlmostEqual(centres[1], max(x01, x02), delta=1.0)

    def test_fit_doublet_fallback(self):
        """tests fit_doublet when no two distinct local maxima exist"""
        np.random.seed(7)
        x = np.linspace(0, 10, 100)
        a1, x01, sigma1 = 100.0, 4.5, 1.5
        a2, x02, sigma2 = 100.0, 5.5, 1.5
        y = pf.double_gaussian(x, a1, x01, sigma1, a2, x02, sigma2)

        popt, pcov = pf.fit_doublet(x, y)
        self.assertEqual(len(popt), 6)
        # Amplitudes should be positive
        self.assertGreater(popt[0], 0)
        self.assertGreater(popt[3], 0)

    def test_fit_peak_returns_pcov(self):
        """fit_peak must return (popt, pcov) and pcov must be a 3x3 matrix."""
        np.random.seed(0)
        x = np.linspace(0, 10, 50)
        y = pf.gaussian(x, 100.0, 5.0, 1.0) + np.random.normal(0, 1, 50)
        popt, pcov = pf.fit_peak(x, y)
        self.assertEqual(len(popt), 3)
        self.assertEqual(pcov.shape, (3, 3))
        # Variances must be non-negative
        self.assertTrue(np.all(np.diag(pcov) >= 0))

    def test_fit_doublet_returns_pcov(self):
        """fit_doublet must return (popt, pcov) and pcov must be a 6x6 matrix."""
        np.random.seed(42)
        x = np.linspace(0, 20, 200)
        y = pf.double_gaussian(x, 100.0, 7.0, 0.8, 90.0, 10.0, 0.9)
        y = y + np.random.normal(0, 2, 200)
        popt, pcov = pf.fit_doublet(x, y)
        self.assertEqual(len(popt), 6)
        self.assertEqual(pcov.shape, (6, 6))

    def test_gaussian_area(self):
        """gaussian_area must equal a * |sigma| * sqrt(2*pi)."""
        a, sigma = 50.0, 2.0
        expected = a * sigma * np.sqrt(2.0 * np.pi)
        self.assertAlmostEqual(pf.gaussian_area(a, sigma), expected)
        # Negative sigma (allowed: absolute value is used)
        self.assertAlmostEqual(pf.gaussian_area(a, -sigma), expected)

    def test_gaussian_area_uncertainty(self):
        """gaussian_area_uncertainty must return a non-negative float."""
        np.random.seed(2)
        x = np.linspace(0, 10, 60)
        y = pf.gaussian(x, 80.0, 5.0, 1.0) + np.random.normal(0, 0.5, 60)
        popt, pcov = pf.fit_peak(x, y)
        a, _x0, sigma = popt
        unc = pf.gaussian_area_uncertainty(a, sigma, pcov)
        self.assertIsInstance(unc, float)
        self.assertGreaterEqual(unc, 0.0)

    def test_fit_peak_area(self):
        """fit_peak_area must return a positive area with non-negative uncertainty."""
        np.random.seed(3)
        x = np.linspace(0, 10, 60)
        a, x0, sigma = 100.0, 5.0, 1.0
        y = pf.gaussian(x, a, x0, sigma) + np.random.normal(0, 1, 60)
        area, unc = pf.fit_peak_area(x, y)
        expected_area = pf.gaussian_area(a, sigma)
        # Area should be close to analytic expectation
        self.assertAlmostEqual(area, expected_area, delta=expected_area * 0.1)
        self.assertGreaterEqual(unc, 0.0)

    def test_fit_doublet_areas(self):
        """fit_doublet_areas must return sensible areas and non-negative uncertainties."""
        np.random.seed(4)
        x = np.linspace(0, 20, 200)
        a1, x01, s1 = 100.0, 7.0, 0.8
        a2, x02, s2 = 90.0, 10.0, 0.9
        y = pf.double_gaussian(x, a1, x01, s1, a2, x02, s2)
        y = y + np.random.normal(0, 1, 200)
        (area1, unc1), (area2, unc2) = pf.fit_doublet_areas(x, y)
        self.assertGreater(area1, 0.0)
        self.assertGreater(area2, 0.0)
        self.assertGreaterEqual(unc1, 0.0)
        self.assertGreaterEqual(unc2, 0.0)

    def test_net_counts_uncertainty(self):
        """net_counts_uncertainty must return (net, uncertainty) with uncertainty >= 0."""
        counts = np.array([5, 4, 6, 50, 100, 80, 50, 5, 4, 6], dtype=float)
        for method in BackgroundMethod:
            net, unc = pf.net_counts_uncertainty(counts, 3, 7, method)
            self.assertIsInstance(net, float)
            self.assertGreaterEqual(unc, 0.0)

    def test_net_counts_uncertainty_poisson_scaling(self):
        """Uncertainty grows with signal strength (Poisson: sigma ~ sqrt(N))."""
        low_counts = np.array([1, 1, 1, 5, 10, 8, 5, 1, 1, 1], dtype=float)
        high_counts = low_counts * 100.0
        _, unc_low = pf.net_counts_uncertainty(low_counts, 3, 7)
        _, unc_high = pf.net_counts_uncertainty(high_counts, 3, 7)
        self.assertGreater(unc_high, unc_low)
        # Uncertainty should scale roughly as sqrt(N): sqrt(100) = 10x
        self.assertAlmostEqual(unc_high / unc_low, 10.0, delta=2.0)

    def test_peak_area_with_background_sensitivity(self):
        """peak_area_with_background_sensitivity must return mean, std, and per-method dict."""
        counts = np.array([5, 4, 6, 50, 100, 80, 50, 5, 4, 6], dtype=float)
        mean_net, std_net, results = pf.peak_area_with_background_sensitivity(counts, 3, 7)
        self.assertIsInstance(mean_net, float)
        self.assertGreaterEqual(std_net, 0.0)
        # Should have one entry per BackgroundMethod
        self.assertEqual(len(results), len(BackgroundMethod))
        for name in results:
            self.assertIn(name, [m.name for m in BackgroundMethod])

    def test_peak_area_sensitivity_flat_background(self):
        """When background is truly flat all methods should agree closely."""
        counts = np.array([10, 10, 10, 110, 200, 110, 10, 10, 10, 10], dtype=float)
        _mean, std_net, _results = pf.peak_area_with_background_sensitivity(counts, 3, 7)
        # All methods should give similar background → small std
        self.assertLess(std_net, 50.0)

    # ------------------------------------------------------------------
    # FWHM helpers
    # ------------------------------------------------------------------

    def test_peak_fwhm_known_value(self):
        """FWHM = 2*sqrt(2*ln2)*sigma ≈ 2.3548*sigma."""
        sigma = 2.0
        expected = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
        self.assertAlmostEqual(pf.peak_fwhm(sigma), expected)

    def test_peak_fwhm_negative_sigma(self):
        """peak_fwhm uses absolute value – result equals fwhm for |sigma|."""
        self.assertAlmostEqual(pf.peak_fwhm(-1.5), pf.peak_fwhm(1.5))

    def test_peak_fwhm_uncertainty_scaling(self):
        """FWHM uncertainty scales linearly with sigma_unc."""
        self.assertAlmostEqual(
            pf.peak_fwhm_uncertainty(1.0, 0.1),
            pf.peak_fwhm_uncertainty(1.0, 0.2) / 2.0,
        )

    def test_fit_peak_fwhm_returns_positive(self):
        """fit_peak_fwhm must return a positive FWHM and non-negative uncertainty."""
        np.random.seed(10)
        x = np.linspace(0, 10, 60)
        y = pf.gaussian(x, 100.0, 5.0, 1.0) + np.random.normal(0, 0.5, 60)
        fwhm, fwhm_unc = pf.fit_peak_fwhm(x, y)
        self.assertGreater(fwhm, 0.0)
        self.assertGreaterEqual(fwhm_unc, 0.0)

    def test_fit_peak_fwhm_matches_peak_fwhm(self):
        """fit_peak_fwhm(x, y) must equal peak_fwhm(sigma) from fit_peak."""
        np.random.seed(11)
        x = np.linspace(0, 10, 60)
        y = pf.gaussian(x, 100.0, 5.0, 1.2) + np.random.normal(0, 0.5, 60)
        popt, pcov = pf.fit_peak(x, y)
        expected_fwhm = pf.peak_fwhm(popt[2])
        expected_unc = pf.peak_fwhm_uncertainty(popt[2], np.sqrt(pcov[2, 2]))
        # fit_peak_fwhm must produce exactly the same values
        np.random.seed(11)
        x2 = np.linspace(0, 10, 60)
        y2 = pf.gaussian(x2, 100.0, 5.0, 1.2) + np.random.normal(0, 0.5, 60)
        fwhm, fwhm_unc = pf.fit_peak_fwhm(x2, y2)
        self.assertAlmostEqual(fwhm, expected_fwhm)
        self.assertAlmostEqual(fwhm_unc, expected_unc)

    def test_fit_peak_fwhm_known_sigma(self):
        """For a noise-free Gaussian the FWHM should be close to the analytic value."""
        x = np.linspace(0, 10, 200)
        sigma = 1.0
        y = pf.gaussian(x, 200.0, 5.0, sigma)
        fwhm, _ = pf.fit_peak_fwhm(x, y)
        expected = pf.peak_fwhm(sigma)
        self.assertAlmostEqual(fwhm, expected, delta=0.05)

    # ------------------------------------------------------------------
    # Goodness-of-fit statistics
    # ------------------------------------------------------------------

    def test_fit_peak_chi2_perfect_fit(self):
        """For a noise-free Gaussian fit, chi2 should be near zero."""
        x = np.linspace(0, 10, 100)
        y = pf.gaussian(x, 200.0, 5.0, 1.0)
        popt, _ = pf.fit_peak(x, y)
        chi2, reduced_chi2, ndof = pf.fit_peak_chi2(x, y, popt)
        self.assertAlmostEqual(chi2, 0.0, delta=1e-6)
        self.assertEqual(ndof, len(x) - 3)
        self.assertGreater(reduced_chi2, 0.0)

    def test_fit_doublet_chi2_perfect_fit(self):
        """For a noise-free doublet fit, chi2 should be near zero."""
        x = np.linspace(0, 20, 150)
        y = pf.double_gaussian(x, 100.0, 7.0, 0.8, 90.0, 13.0, 0.9)
        popt, _ = pf.fit_doublet(x, y)
        chi2, reduced_chi2, ndof = pf.fit_doublet_chi2(x, y, popt)
        self.assertAlmostEqual(chi2, 0.0, delta=1e-4)
        self.assertEqual(ndof, len(x) - 6)

    def test_fit_chi2_ndof(self):
        """fit_chi2 ndof = len(y) - n_params."""
        x = np.linspace(0, 10, 50)
        y = pf.gaussian(x, 100.0, 5.0, 1.0)
        popt, _ = pf.fit_peak(x, y)
        _, _, ndof = pf.fit_chi2(x, y, popt, pf.gaussian, n_params=3)
        self.assertEqual(ndof, 47)

    def test_fit_chi2_poor_fit_higher_value(self):
        """A deliberately bad fit should give a larger chi2 than the true fit."""
        np.random.seed(20)
        x = np.linspace(0, 10, 80)
        y = pf.gaussian(x, 100.0, 5.0, 1.0) + np.random.normal(0, 1, 80)
        y = np.maximum(y, 0.0)
        popt_good, _ = pf.fit_peak(x, y)
        chi2_good, _, _ = pf.fit_peak_chi2(x, y, popt_good)
        # A wrong centroid gives a worse fit
        popt_bad = np.array([popt_good[0], popt_good[1] + 2.0, popt_good[2]])
        chi2_bad, _, _ = pf.fit_peak_chi2(x, y, popt_bad)
        self.assertGreater(chi2_bad, chi2_good)

    def test_fit_chi2_zero_bin_handling(self):
        """Bins with zero counts must not cause division by zero."""
        x = np.linspace(0, 10, 50)
        y = pf.gaussian(x, 100.0, 5.0, 1.0)
        # Force some bins to zero
        y_with_zeros = y.copy()
        y_with_zeros[:5] = 0.0
        popt, _ = pf.fit_peak(x, y)
        chi2, reduced_chi2, _ = pf.fit_peak_chi2(x, y_with_zeros, popt)
        self.assertIsInstance(chi2, float)
        self.assertFalse(np.isnan(chi2))

    def test_fit_chi2_inf_reduced_chi2_when_no_dof(self):
        """If ndof <= 0, reduced_chi2 should be inf."""
        x = np.array([1.0, 2.0, 3.0])
        y = pf.gaussian(x, 10.0, 2.0, 0.5)
        popt, _ = pf.fit_peak(x, y)
        # n_params == len(y) means ndof == 0
        _, reduced_chi2, ndof = pf.fit_chi2(x, y, popt, pf.gaussian, n_params=len(y))
        self.assertEqual(ndof, 0)
        self.assertEqual(reduced_chi2, float("inf"))


if __name__ == "__main__":
    unittest.main()
