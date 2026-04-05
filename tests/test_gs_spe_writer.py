import os
import tempfile
import unittest
import gs_spe_reading as gsr
import gs_spe_writer as gsw
from ph_spectrum import PhSpectrum


class write_dollar_spe_test_case(unittest.TestCase):
    """tests for write_dollar_spe"""

    def _write_and_read(self, spec):
        """Write spec to a temp file and read it back."""
        with tempfile.NamedTemporaryFile(suffix=".Spe", delete=False) as f:
            tmp_path = f.name
        try:
            gsw.write_dollar_spe(spec, tmp_path)
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
            gsw.write_dollar_spe(spec, tmp_path)
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
            gsw.write_dollar_spe(spec, tmp_path)
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
            gsw.write_dollar_spe(spec, tmp_path)
            lines = gsr.read_file(tmp_path)
            self.assertFalse(any("$MEAS_TIM" in l for l in lines))
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
