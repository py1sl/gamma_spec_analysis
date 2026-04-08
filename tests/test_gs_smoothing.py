import unittest
import numpy as np
import gs_smoothing as sm
import gs_spe_reading


class smoothing_test_case(unittest.TestCase):
    """tests for smoothing functions in gs_smoothing"""

    def test_smoothing(self):
        """tests related to five_point_smooth"""
        spec = gs_spe_reading.read_dollar_spe("../test_data/Ba_133_raised_1.Spe")
        smoothed = sm.five_point_smooth(spec.counts)
        self.assertEqual(len(smoothed), len(spec.counts))
        self.assertEqual(smoothed[0], spec.counts[0])
        self.assertEqual(smoothed[-1], spec.counts[-1])

        # Test with array that's too short
        short_array = [1, 2, 3, 4]
        self.assertRaises(ValueError, sm.five_point_smooth, short_array)

    def test_three_point_smooth(self):
        """tests for 3 point smoothing function"""
        counts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        smoothed = sm.three_point_smooth(counts)

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
        self.assertRaises(ValueError, sm.three_point_smooth, short_array)

    def test_moving_average(self):
        """tests for moving average smoothing function"""
        counts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        smoothed = sm.moving_average(counts, window=5)

        # Check length is preserved
        self.assertEqual(len(smoothed), len(counts))

        # Check that smoothing reduces variance
        self.assertLessEqual(np.std(smoothed), np.std(counts))

        # Test with window=3
        smoothed_3 = sm.moving_average(counts, window=3)
        self.assertEqual(len(smoothed_3), len(counts))

        # Middle element should be average of 3 points
        # For index 5: (5 + 6 + 7) / 3 = 6.0
        self.assertAlmostEqual(smoothed_3[5], 6.0)

        # Test with invalid window size (even number)
        self.assertRaises(ValueError, sm.moving_average, counts, window=4)

        # Test with invalid window size (negative)
        self.assertRaises(ValueError, sm.moving_average, counts, window=-1)

        # Test with array shorter than window
        short_array = [1, 2, 3]
        self.assertRaises(ValueError, sm.moving_average, short_array, window=5)

    def test_exponential_moving_average(self):
        """tests for exponential moving average smoothing function"""
        counts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        smoothed = sm.exponential_moving_average(counts, alpha=0.3)

        # Check length is preserved
        self.assertEqual(len(smoothed), len(counts))

        # First element should be unchanged
        self.assertEqual(smoothed[0], counts[0])

        # Second element: alpha * counts[1] + (1 - alpha) * smoothed[0]
        # = 0.3 * 2 + 0.7 * 1 = 0.6 + 0.7 = 1.3
        self.assertAlmostEqual(smoothed[1], 1.3)

        # Test with different alpha values
        smoothed_low = sm.exponential_moving_average(counts, alpha=0.1)
        smoothed_high = sm.exponential_moving_average(counts, alpha=0.9)

        # Lower alpha should result in smoother output
        self.assertLessEqual(np.std(smoothed_low), np.std(smoothed_high))

        # Test with invalid alpha (too low)
        self.assertRaises(ValueError, sm.exponential_moving_average, counts, alpha=0.0)

        # Test with invalid alpha (too high)
        self.assertRaises(ValueError, sm.exponential_moving_average, counts, alpha=1.0)

        # Test with invalid alpha (negative)
        self.assertRaises(ValueError, sm.exponential_moving_average, counts, alpha=-0.1)

        # Test with invalid alpha (greater than 1)
        self.assertRaises(ValueError, sm.exponential_moving_average, counts, alpha=1.5)

    def test_five_point_smooth_correctness(self):
        """Test that five_point_smooth produces correct results"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = sm.five_point_smooth(data)

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
        result = sm.three_point_smooth(data)

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
        result = sm.moving_average(data, window=5)

        # Check middle element calculation
        # For index 5 with window=5: mean of [4, 5, 6, 7, 8] = 6
        expected_idx5 = np.mean(data[3:8])
        np.testing.assert_almost_equal(result[5], expected_idx5)

    def test_smoothing_maintains_signal_integrity(self):
        """Test that smoothing preserves important signal properties"""
        data = np.concatenate([
            np.zeros(100),
            np.full(50, 100),  # Peak
            np.zeros(100)
        ])

        result_5pt = sm.five_point_smooth(data)
        result_3pt = sm.three_point_smooth(data)
        result_ma = sm.moving_average(data, window=5)

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


if __name__ == "__main__":
    unittest.main()
