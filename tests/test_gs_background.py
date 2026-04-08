import unittest
import numpy as np
import gs_background as bg
from gs_background import BackgroundMethod


class background_test_case(unittest.TestCase):
    """tests for background estimation functions in gs_background"""

    def test_gross_count(self):
        """tests for gross_count function"""
        counts = [1, 1, 1, 1, 1]
        gc = bg.gross_count(counts, 1, 4)
        self.assertEqual(gc, 3)
        self.assertRaises(ValueError, bg.gross_count, counts, -1, 4)
        self.assertRaises(ValueError, bg.gross_count, counts, 1, 10)
        self.assertRaises(ValueError, bg.gross_count, counts, 10, 4)

    def test_calc_bg_invalid_channels(self):
        """tests that calc_bg raises ValueError for invalid channel ranges"""
        counts = [1, 1, 1, 1, 1]
        self.assertRaises(ValueError, bg.calc_bg, counts, -1, 4)
        self.assertRaises(ValueError, bg.calc_bg, counts, 1, 10)
        self.assertRaises(ValueError, bg.calc_bg, counts, 10, 4)
        self.assertRaises(ValueError, bg.calc_bg, counts, 1, 4, 5)  # Invalid method

    def test_background_calculation(self):
        """tests for background calculation functions"""
        counts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # Test calc_bg with valid parameters (BackgroundMethod.TRAPEZOID)
        bg_val = bg.calc_bg(counts, 2, 7, m=BackgroundMethod.TRAPEZOID)
        self.assertIsInstance(bg_val, float)
        self.assertGreaterEqual(bg_val, 0)

        # Test estimate_background_trapezoid directly
        bg_trap = bg.estimate_background_trapezoid(counts, 2, 7)
        self.assertIsInstance(bg_trap, float)
        self.assertGreaterEqual(bg_trap, 0)

        # Test edge case: channels at the start
        bg_start = bg.estimate_background_trapezoid(counts, 0, 3)
        self.assertIsInstance(bg_start, float)

        # Test edge case: channels at the end
        bg_end = bg.estimate_background_trapezoid(counts, 7, 9)
        self.assertIsInstance(bg_end, float)

        # Test BackgroundMethod.LINEAR - linear interpolation
        bg_linear = bg.calc_bg(counts, 2, 7, m=BackgroundMethod.LINEAR)
        self.assertIsInstance(bg_linear, float)
        self.assertGreaterEqual(bg_linear, 0)

        # Test estimate_background_linear directly
        bg_linear_direct = bg.estimate_background_linear(counts, 2, 7)
        self.assertIsInstance(bg_linear_direct, float)
        self.assertGreaterEqual(bg_linear_direct, 0)

        # Test BackgroundMethod.STEP - step function
        bg_step = bg.calc_bg(counts, 2, 7, m=BackgroundMethod.STEP)
        self.assertIsInstance(bg_step, float)
        self.assertGreaterEqual(bg_step, 0)

        # Test estimate_background_step directly
        bg_step_direct = bg.estimate_background_step(counts, 2, 7)
        self.assertIsInstance(bg_step_direct, float)
        self.assertGreaterEqual(bg_step_direct, 0)

        # Test BackgroundMethod.SLIDING_AVERAGE - sliding window average
        bg_sliding = bg.calc_bg(counts, 2, 7, m=BackgroundMethod.SLIDING_AVERAGE)
        self.assertIsInstance(bg_sliding, float)
        self.assertGreaterEqual(bg_sliding, 0)

        # Test estimate_background_sliding_average directly
        bg_sliding_direct = bg.estimate_background_sliding_average(counts, 2, 7)
        self.assertIsInstance(bg_sliding_direct, float)
        self.assertGreaterEqual(bg_sliding_direct, 0)

        # Test invalid method number
        with self.assertRaises(ValueError):
            bg.calc_bg(counts, 2, 7, m=5)

    def test_background_methods_comparison(self):
        """Compare different background subtraction methods"""
        x = np.arange(100)
        background_level = 50.0
        peak = 200 * np.exp(-((x - 50) ** 2) / (2 * 5 ** 2))
        counts = background_level + peak

        c1, c2 = 40, 60

        bg_trap = bg.calc_bg(counts, c1, c2, m=BackgroundMethod.TRAPEZOID)
        bg_linear = bg.calc_bg(counts, c1, c2, m=BackgroundMethod.LINEAR)
        bg_step = bg.calc_bg(counts, c1, c2, m=BackgroundMethod.STEP)
        bg_sliding = bg.calc_bg(counts, c1, c2, m=BackgroundMethod.SLIDING_AVERAGE)

        # All should be positive
        self.assertGreater(bg_trap, 0)
        self.assertGreater(bg_linear, 0)
        self.assertGreater(bg_step, 0)
        self.assertGreater(bg_sliding, 0)

        # For a flat background, all methods should give similar results
        # (within reasonable tolerance given the peak interference)
        expected_bg = background_level * (c2 - c1)

        for bg_value in [bg_trap, bg_linear, bg_step, bg_sliding]:
            self.assertGreater(bg_value, expected_bg * 0.5)
            self.assertLess(bg_value, expected_bg * 2.0)

    def test_background_edge_cases_all_methods(self):
        """Test edge cases for all background methods"""
        counts = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

        for method in BackgroundMethod:
            # At start of spectrum
            bg_start = bg.calc_bg(counts, 0, 3, m=method)
            self.assertIsInstance(bg_start, float)
            self.assertGreaterEqual(bg_start, 0)

            # At end of spectrum
            bg_end = bg.calc_bg(counts, 7, 9, m=method)
            self.assertIsInstance(bg_end, float)
            self.assertGreaterEqual(bg_end, 0)

            # Single channel peak
            bg_single = bg.calc_bg(counts, 5, 6, m=method)
            self.assertIsInstance(bg_single, float)
            self.assertGreaterEqual(bg_single, 0)

    def test_estimate_background_trapezoid_uncertainty(self):
        """Trapezoid background uncertainty must be non-negative."""
        counts = np.array([5, 4, 6, 50, 100, 80, 50, 5, 4, 6], dtype=float)
        unc = bg.estimate_background_trapezoid_uncertainty(counts, 3, 7)
        self.assertGreaterEqual(unc, 0.0)

    def test_estimate_background_linear_uncertainty(self):
        """Linear background uncertainty must be non-negative."""
        counts = np.array([5, 4, 6, 50, 100, 80, 50, 5, 4, 6], dtype=float)
        unc = bg.estimate_background_linear_uncertainty(counts, 3, 7)
        self.assertGreaterEqual(unc, 0.0)

    def test_estimate_background_step_uncertainty(self):
        """Step background uncertainty must be non-negative."""
        counts = np.array([5, 4, 6, 50, 100, 80, 50, 5, 4, 6], dtype=float)
        unc = bg.estimate_background_step_uncertainty(counts, 3, 7)
        self.assertGreaterEqual(unc, 0.0)

    def test_estimate_background_sliding_average_uncertainty(self):
        """Sliding-average background uncertainty must be non-negative."""
        counts = np.array([5, 4, 6, 50, 100, 80, 50, 5, 4, 6], dtype=float)
        unc = bg.estimate_background_sliding_average_uncertainty(counts, 3, 7)
        self.assertGreaterEqual(unc, 0.0)

    def test_calc_bg_uncertainty_dispatches(self):
        """calc_bg_uncertainty must dispatch to all four methods without error."""
        counts = np.array([5, 4, 6, 50, 100, 80, 50, 5, 4, 6], dtype=float)
        for method in BackgroundMethod:
            unc = bg.calc_bg_uncertainty(counts, 3, 7, method)
            self.assertGreaterEqual(unc, 0.0)
        # Invalid method raises ValueError
        self.assertRaises(ValueError, bg.calc_bg_uncertainty, counts, 3, 7, 99)


if __name__ == "__main__":
    unittest.main()
