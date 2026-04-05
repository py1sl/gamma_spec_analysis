import os
import tempfile
import unittest
import gs_spe_reading as gsr
from ph_spectrum import PhSpectrum


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
        keywords_map = gsr.get_dollar_keywords(data)
        self.assertEqual(gsr.get_live_time(data, keywords_map), 326)
        self.assertEqual(gsr.get_real_time(data, keywords_map), 431)
        self.assertEqual(gsr.get_start_date(data, keywords_map), "02/25/2020 14:24:52")

    def test_get_fits(self):
        """tests related to the energy, efficiency and shape calibration"""
        data = gsr.read_file("../test_data/Ba_133_raised_1.Spe")
        keywords_map = gsr.get_dollar_keywords(data)
        e_data = gsr.get_energy_fit_coefficients(data, keywords_map)
        self.assertEqual(len(e_data), 2)
        self.assertEqual(e_data[0], 0.323476)
        self.assertEqual(e_data[1], 0.365473)

    def test_get_counts(self):
        """tests about reading to count data"""
        data = gsr.read_file("../test_data/Ba_133_raised_1.Spe")
        keywords_map = gsr.get_dollar_keywords(data)
        counts = gsr.get_counts(data, keywords_map)
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
        keywords_map = gsr.get_dollar_keywords(data)
        mca_cal = gsr.get_mca_cal(data, keywords_map)
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
        keywords_map = gsr.get_dollar_keywords(data)
        shape_cal = gsr.get_shape_cal(data, keywords_map)
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


class write_dollar_spe_test_case(unittest.TestCase):
    """tests for write_dollar_spe"""

    def _write_and_read(self, spec):
        """Write spec to a temp file and read it back."""
        with tempfile.NamedTemporaryFile(suffix=".Spe", delete=False) as f:
            tmp_path = f.name
        try:
            gsr.write_dollar_spe(spec, tmp_path)
            return gsr.read_dollar_spe(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_roundtrip_counts(self):
        """counts survive a write-then-read roundtrip"""
        spec = PhSpectrum(
            counts=[0, 5, 12, 3, 0],
            live_time=100.0,
            real_time=120.0,
        )
        result = self._write_and_read(spec)
        self.assertEqual(list(result.counts), [0, 5, 12, 3, 0])

    def test_roundtrip_times(self):
        """live_time and real_time survive a roundtrip"""
        spec = PhSpectrum(
            counts=[1, 2, 3],
            live_time=300.0,
            real_time=400.0,
        )
        result = self._write_and_read(spec)
        self.assertEqual(result.live_time, 300.0)
        self.assertEqual(result.real_time, 400.0)

    def test_roundtrip_start_time(self):
        """start_time survives a roundtrip"""
        spec = PhSpectrum(
            counts=[1, 2, 3],
            start_time="02/25/2020 14:24:52",
        )
        result = self._write_and_read(spec)
        self.assertEqual(result.start_time, "02/25/2020 14:24:52")

    def test_roundtrip_energy_fit_coefficients(self):
        """energy_fit_coefficients survive a roundtrip"""
        spec = PhSpectrum(
            counts=[1, 2, 3],
            energy_fit_coefficients=[0.323476, 0.365473],
        )
        result = self._write_and_read(spec)
        self.assertAlmostEqual(result.energy_fit_coefficients[0], 0.323476, places=5)
        self.assertAlmostEqual(result.energy_fit_coefficients[1], 0.365473, places=5)

    def test_roundtrip_mca_and_shape_cal(self):
        """keywords (mca_cal, shape_cal) survive a roundtrip"""
        spec = PhSpectrum(
            counts=[0, 1, 2],
            keywords={
                'mca_cal': {
                    'order': 3,
                    'coefficients': [0.323476, 0.365473, 3.057753e-8],
                    'unit': 'keV',
                },
                'shape_cal': {
                    'order': 3,
                    'coefficients': [1.604150, 2.041100e-3, -3.766970e-7],
                },
            },
        )
        result = self._write_and_read(spec)
        mca = result.keywords['mca_cal']
        self.assertEqual(mca['order'], 3)
        self.assertEqual(mca['unit'], 'keV')
        self.assertAlmostEqual(mca['coefficients'][0], 0.323476, places=5)
        shape = result.keywords['shape_cal']
        self.assertEqual(shape['order'], 3)
        self.assertAlmostEqual(shape['coefficients'][0], 1.604150, places=5)

    def test_roundtrip_full_file(self):
        """reading and rewriting a real SPE file produces equivalent spectrum"""
        original = gsr.read_dollar_spe("../test_data/Ba_133_raised_1.Spe")
        roundtripped = self._write_and_read(original)
        self.assertEqual(len(roundtripped.counts), len(original.counts))
        self.assertTrue((roundtripped.counts == original.counts).all())
        self.assertEqual(roundtripped.live_time, original.live_time)
        self.assertEqual(roundtripped.real_time, original.real_time)
        self.assertEqual(roundtripped.start_time, original.start_time)

    def test_spec_name_in_file(self):
        """spec_name is written to $SPEC_ID section"""
        spec = PhSpectrum(spec_name="my_spectrum", counts=[1, 2, 3])
        with tempfile.NamedTemporaryFile(suffix=".Spe", delete=False) as f:
            tmp_path = f.name
        try:
            gsr.write_dollar_spe(spec, tmp_path)
            lines = gsr.read_file(tmp_path)
            spec_id_idx = next(i for i, l in enumerate(lines) if l.strip() == "$SPEC_ID:")
            self.assertEqual(lines[spec_id_idx + 1], "my_spectrum")
        finally:
            os.unlink(tmp_path)

    def test_omits_date_when_none(self):
        """$DATE_MEA section is omitted when start_time is None"""
        spec = PhSpectrum(counts=[1, 2, 3], start_time=None)
        with tempfile.NamedTemporaryFile(suffix=".Spe", delete=False) as f:
            tmp_path = f.name
        try:
            gsr.write_dollar_spe(spec, tmp_path)
            lines = gsr.read_file(tmp_path)
            self.assertFalse(any("$DATE_MEA" in l for l in lines))
        finally:
            os.unlink(tmp_path)

    def test_omits_times_when_none(self):
        """$MEAS_TIM section is omitted when both live_time and real_time are None"""
        spec = PhSpectrum(counts=[1, 2, 3], live_time=None, real_time=None)
        with tempfile.NamedTemporaryFile(suffix=".Spe", delete=False) as f:
            tmp_path = f.name
        try:
            gsr.write_dollar_spe(spec, tmp_path)
            lines = gsr.read_file(tmp_path)
            self.assertFalse(any("$MEAS_TIM" in l for l in lines))
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
