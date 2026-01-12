"""
Performance tests to verify the optimization improvements
"""
import unittest
import time
import numpy as np
import gs_analysis as gs


class PerformanceTest(unittest.TestCase):
    """Tests to verify performance improvements in smoothing functions"""

    def setUp(self):
        """Create test data for performance benchmarks"""
        # Create large arrays for performance testing
        np.random.seed(42)
        self.small_data = np.random.randint(0, 1000, 1000)
        self.large_data = np.random.randint(0, 1000, 10000)
        
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
        expected_idx2 = (1.0 / 9.0) * (data[0] + data[4] + 2*data[3] + 2*data[1] + 3*data[2])
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
        
    def test_five_point_smooth_performance(self):
        """Benchmark five_point_smooth function"""
        # Run multiple iterations to get reliable timing
        iterations = 10
        start = time.time()
        for _ in range(iterations):
            result = gs.five_point_smooth(self.large_data)
        elapsed = time.time() - start
        
        # Verify result is correct type and length
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), len(self.large_data))
        
        # Verify performance is reasonable (should be < 1ms per call for 10k elements)
        avg_time = elapsed / iterations
        self.assertLess(avg_time, 0.001, 
                       f"five_point_smooth took {avg_time*1000:.3f}ms, expected < 1ms")
        
    def test_three_point_smooth_performance(self):
        """Benchmark three_point_smooth function"""
        iterations = 10
        start = time.time()
        for _ in range(iterations):
            result = gs.three_point_smooth(self.large_data)
        elapsed = time.time() - start
        
        # Verify result is correct type and length
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), len(self.large_data))
        
        # Verify performance is reasonable (should be < 1ms per call for 10k elements)
        avg_time = elapsed / iterations
        self.assertLess(avg_time, 0.001, 
                       f"three_point_smooth took {avg_time*1000:.3f}ms, expected < 1ms")
        
    def test_moving_average_performance(self):
        """Benchmark moving_average function"""
        iterations = 10
        start = time.time()
        for _ in range(iterations):
            result = gs.moving_average(self.large_data, window=5)
        elapsed = time.time() - start
        
        # Verify result is correct type and length
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), len(self.large_data))
        
        # Verify performance is reasonable (should be < 10ms per call for 10k elements)
        avg_time = elapsed / iterations
        self.assertLess(avg_time, 0.010, 
                       f"moving_average took {avg_time*1000:.3f}ms, expected < 10ms")
        
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


if __name__ == '__main__':
    unittest.main(verbosity=2)
