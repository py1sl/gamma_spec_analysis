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

    def test_get_mca_cal(self):
        """tests related to reading MCA calibration data"""
        data = gsr.read_file("../test_data/Co_60_raised_1.Spe")
        mca_cal = gsr.get_mca_cal(data)
        self.assertIsNotNone(mca_cal)
        self.assertEqual(mca_cal['order'], 3)
        self.assertEqual(len(mca_cal['coefficients']), 3)
        self.assertAlmostEqual(mca_cal['coefficients'][0], 0.323476, places=5)
        self.assertAlmostEqual(mca_cal['coefficients'][1], 0.365473, places=5)
        self.assertAlmostEqual(mca_cal['coefficients'][2], 3.057753e-8, places=13)
        self.assertEqual(mca_cal['unit'], 'keV')

    def test_get_shape_cal(self):
        """tests related to reading shape calibration data"""
        data = gsr.read_file("../test_data/Co_60_raised_1.Spe")
        shape_cal = gsr.get_shape_cal(data)
        self.assertIsNotNone(shape_cal)
        self.assertEqual(shape_cal['order'], 3)
        self.assertEqual(len(shape_cal['coefficients']), 3)
        self.assertAlmostEqual(shape_cal['coefficients'][0], 1.604150, places=5)
        self.assertAlmostEqual(shape_cal['coefficients'][1], 2.041100e-3, places=8)
        self.assertAlmostEqual(shape_cal['coefficients'][2], -3.766970e-7, places=12)

    def test_read_dollar_spe_with_keywords(self):
        """testing that keywords are populated in the PhSpectrum object"""
        spec = gsr.read_dollar_spe("../test_data/Co_60_raised_1.Spe")
        self.assertIn('mca_cal', spec.keywords)
        self.assertIn('shape_cal', spec.keywords)
        
        mca_cal = spec.keywords['mca_cal']
        self.assertEqual(mca_cal['order'], 3)
        self.assertEqual(mca_cal['unit'], 'keV')
        self.assertEqual(len(mca_cal['coefficients']), 3)
        
        shape_cal = spec.keywords['shape_cal']
        self.assertEqual(shape_cal['order'], 3)
        self.assertEqual(len(shape_cal['coefficients']), 3)


if __name__ == "__main__":
    unittest.main()
