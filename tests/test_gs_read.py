import unittest
import gs_spe_reading as gsr


class read_ascii_dollar_spe_test_case(unittest.TestCase):
    """tests for file reading functions"""

    def test_read_file(self):
        """tests related to the initial file reading"""
        data = gsr.read_file("../test_data/Ba_133_raised_1.Spe")
        self.assertEqual(len(data), 8261)
        self.assertEqual(data[0], "$SPEC_ID:")
        self.assertEqual(data[-2], "3")

    def test_read_times(self):
        """tests related to measurement times and  dates"""
        data = gsr.read_file("../test_data/Ba_133_raised_1.Spe")
        self.assertEqual(gsr.get_live_time(data), 326)
        self.assertEqual(gsr.get_real_time(data), 431)
        self.assertEqual(gsr.get_start_date(data), "02/25/2020 14:24:52")

    def test_get_fits(self):
        """tests related to the energy, efficiency and shape calibration"""
        data = gsr.read_file("../test_data/Ba_133_raised_1.Spe")
        e_data = gsr.get_energy_fit_coefficients(data)
        self.assertEqual(len(e_data), 2)
        self.assertEqual(e_data[0], 0.323476)
        self.assertEqual(e_data[1], 0.365473)

    def test_get_counts(self):
        """tests about reading to count data"""
        data = gsr.read_file("../test_data/Ba_133_raised_1.Spe")
        counts = gsr.get_counts(data)
        self.assertEqual(len(counts), 8192)
        self.assertEqual(counts[0], 0)
        self.assertEqual(counts[-1], 0)

    def test_get_dollar_keywords_presence(self):
        """Ensure expected keywords exist in the real test .Spe file"""
        data = gsr.read_file("../test_data/Ba_133_raised_1.Spe")
        kws = gsr.get_dollar_keywords(data)

        # expected keywords (adjust if your file uses different tags)
        self.assertIn("$SPEC_ID", kws)
        self.assertIn("$DATA", kws)
        self.assertIn("$MEAS_TIM", kws)
        self.assertIn("$DATE_MEA", kws)
        self.assertIn("$ENER_FIT", kws)

        # check some known positions from existing tests
        self.assertEqual(kws["$SPEC_ID"][0], 0)
        self.assertGreater(len(kws["$DATA"]), 0)

    def test_read_dollar_spe(self):
        """testing the read $ spe function"""

        spec = gsr.read_dollar_spe("../test_data/Ba_133_raised_1.Spe")
        self.assertEqual(len(spec.counts), 8192)
        e_data = spec.energy_fit_coefficients
        self.assertEqual(len(e_data), 2)
        self.assertEqual(e_data[0], 0.323476)
        self.assertEqual(e_data[1], 0.365473)
        self.assertEqual(spec.peaks, [])


if __name__ == "__main__":
    unittest.main()
