"""Tests for the PhSpectrum data model (ph_spectrum.py)."""

import unittest
import numpy as np
import gs_analysis as gs
from ph_spectrum import PhSpectrum


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spectrum(
    n: int = 512,
    count_val: int = 100,
    live_time: float = 100.0,
    real_time: float = 105.0,
    start_chan_num: int = 0,
    energy_coeffs=(0.0, 1.0),
    spec_name: str = "test",
) -> PhSpectrum:
    """Return a flat PhSpectrum for testing."""
    return PhSpectrum(
        spec_name=spec_name,
        start_chan_num=start_chan_num,
        counts=np.full(n, count_val, dtype=np.int64),
        live_time=live_time,
        real_time=real_time,
        energy_fit_coefficients=list(energy_coeffs),
    )


# ---------------------------------------------------------------------------
# Tests: generate_ebins (instance method)
# ---------------------------------------------------------------------------

class TestGenerateEbins(unittest.TestCase):

    def test_basic_linear_calibration(self):
        """E = a0 + a1 * channel should be computed correctly."""
        spec = _make_spectrum(n=5, energy_coeffs=(10.0, 2.0))
        ebins = spec.generate_ebins()
        np.testing.assert_array_almost_equal(ebins, [10.0, 12.0, 14.0, 16.0, 18.0])

    def test_returns_float64(self):
        spec = _make_spectrum(n=4, energy_coeffs=(0.0, 1.5))
        self.assertEqual(spec.generate_ebins().dtype, np.float64)

    def test_length_matches_num_channels(self):
        spec = _make_spectrum(n=100)
        self.assertEqual(len(spec.generate_ebins()), 100)

    def test_zero_intercept_coefficients(self):
        """Energy = 0 + 1 * channel (identity mapping)."""
        spec = _make_spectrum(n=10, energy_coeffs=(0.0, 1.0))
        ebins = spec.generate_ebins()
        np.testing.assert_array_equal(ebins, np.arange(10, dtype=np.float64))

    def test_raises_when_coefficients_none(self):
        spec = PhSpectrum(counts=np.ones(10, dtype=np.int64))
        with self.assertRaises(ValueError):
            spec.generate_ebins()

    def test_raises_when_coefficients_wrong_length(self):
        """Only two-element coefficient arrays are valid."""
        spec = _make_spectrum(n=10)
        spec.energy_fit_coefficients = [1.0, 2.0, 3.0]  # length 3
        with self.assertRaises(ValueError):
            spec.generate_ebins()

    def test_raises_when_coefficients_length_one(self):
        spec = _make_spectrum(n=10)
        spec.energy_fit_coefficients = [1.0]
        with self.assertRaises(ValueError):
            spec.generate_ebins()

    def test_derives_num_channels_when_zero(self):
        """When num_channels is 0 it should be derived from counts."""
        spec = PhSpectrum()
        spec.energy_fit_coefficients = [1.0, 2.0]
        spec.num_channels = 0
        spec.counts = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        ebins = spec.generate_ebins()
        self.assertEqual(len(ebins), 5)
        self.assertEqual(spec.num_channels, 5)

    def test_consistent_with_gs_analysis_wrapper(self):
        """The instance method and gs_analysis.generate_ebins must agree."""
        spec = _make_spectrum(n=512, energy_coeffs=(0.5, 0.3))
        np.testing.assert_array_almost_equal(
            spec.generate_ebins(), gs.generate_ebins(spec)
        )


# ---------------------------------------------------------------------------
# Tests: __add__ fixes (channels and start_chan_num)
# ---------------------------------------------------------------------------

class TestAddFix(unittest.TestCase):

    def test_result_channels_populated(self):
        """The result of __add__ should have channels list = range(n)."""
        s1 = _make_spectrum(n=512)
        s2 = _make_spectrum(n=512)
        result = s1 + s2
        self.assertEqual(result.channels, list(range(512)))

    def test_result_start_chan_num_from_left(self):
        """start_chan_num should be taken from the left operand."""
        s1 = _make_spectrum(n=64, start_chan_num=7)
        s2 = _make_spectrum(n=64, start_chan_num=0)
        result = s1 + s2
        self.assertEqual(result.start_chan_num, 7)

    def test_result_channels_length_matches_counts(self):
        s1 = _make_spectrum(n=128)
        s2 = _make_spectrum(n=128)
        result = s1 + s2
        self.assertEqual(len(result.channels), len(result.counts))


# ---------------------------------------------------------------------------
# Tests: __sub__
# ---------------------------------------------------------------------------

class TestSub(unittest.TestCase):

    def test_subtracts_counts_elementwise(self):
        s1 = _make_spectrum(n=8, count_val=10)
        s2 = _make_spectrum(n=8, count_val=3)
        result = s1 - s2
        np.testing.assert_array_equal(result.counts, np.full(8, 7, dtype=np.int64))

    def test_negative_counts_preserved(self):
        """Subtraction that yields negative bins must not clamp to zero."""
        s1 = _make_spectrum(n=4, count_val=5)
        s2 = _make_spectrum(n=4, count_val=10)
        result = s1 - s2
        np.testing.assert_array_equal(result.counts, np.full(4, -5, dtype=np.int64))

    def test_result_counts_dtype_int64(self):
        s1 = _make_spectrum(n=16, count_val=50)
        s2 = _make_spectrum(n=16, count_val=20)
        self.assertEqual((s1 - s2).counts.dtype, np.int64)

    def test_live_time_from_left_operand(self):
        s1 = _make_spectrum(n=8, live_time=300.0)
        s2 = _make_spectrum(n=8, live_time=600.0)
        self.assertAlmostEqual((s1 - s2).live_time, 300.0)

    def test_real_time_from_left_operand(self):
        s1 = _make_spectrum(n=8, real_time=310.0)
        s2 = _make_spectrum(n=8, real_time=620.0)
        self.assertAlmostEqual((s1 - s2).real_time, 310.0)

    def test_spec_name_from_left_operand(self):
        s1 = _make_spectrum(n=8, spec_name="sample")
        s2 = _make_spectrum(n=8, spec_name="background")
        self.assertEqual((s1 - s2).spec_name, "sample")

    def test_energy_coefficients_from_left_operand(self):
        s1 = _make_spectrum(n=8, energy_coeffs=(0.0, 2.0))
        s2 = _make_spectrum(n=8, energy_coeffs=(1.0, 3.0))
        self.assertEqual(list((s1 - s2).energy_fit_coefficients), [0.0, 2.0])

    def test_start_chan_num_from_left_operand(self):
        s1 = _make_spectrum(n=8, start_chan_num=3)
        s2 = _make_spectrum(n=8, start_chan_num=0)
        self.assertEqual((s1 - s2).start_chan_num, 3)

    def test_result_channels_populated(self):
        s1 = _make_spectrum(n=64)
        s2 = _make_spectrum(n=64)
        result = s1 - s2
        self.assertEqual(result.channels, list(range(64)))

    def test_raises_on_channel_mismatch(self):
        s1 = _make_spectrum(n=512)
        s2 = _make_spectrum(n=256)
        with self.assertRaises(ValueError):
            _ = s1 - s2

    def test_returns_not_implemented_for_non_spectrum(self):
        s = _make_spectrum()
        result = s.__sub__(42)
        self.assertIs(result, NotImplemented)

    def test_self_minus_self_is_zero(self):
        s = _make_spectrum(n=16, count_val=50)
        result = s - s
        np.testing.assert_array_equal(result.counts, np.zeros(16, dtype=np.int64))

    def test_live_time_none_when_left_missing(self):
        s1 = PhSpectrum(counts=np.ones(8, dtype=np.int64))
        s2 = _make_spectrum(n=8, live_time=100.0)
        self.assertIsNone((s1 - s2).live_time)

    def test_num_channels_set_in_result(self):
        s1 = _make_spectrum(n=32)
        s2 = _make_spectrum(n=32)
        self.assertEqual((s1 - s2).num_channels, 32)


# ---------------------------------------------------------------------------
# Tests: normalise_to_livetime
# ---------------------------------------------------------------------------

class TestNormaliseToLivetime(unittest.TestCase):

    def test_basic_normalisation(self):
        """counts / live_time should give the count rate."""
        spec = _make_spectrum(n=4, count_val=200, live_time=100.0)
        rates = spec.normalise_to_livetime()
        np.testing.assert_array_almost_equal(rates, np.full(4, 2.0))

    def test_returns_float64(self):
        spec = _make_spectrum(n=8, live_time=50.0)
        self.assertEqual(spec.normalise_to_livetime().dtype, np.float64)

    def test_length_matches_counts(self):
        spec = _make_spectrum(n=256, live_time=60.0)
        self.assertEqual(len(spec.normalise_to_livetime()), 256)

    def test_raises_when_live_time_none(self):
        spec = PhSpectrum(counts=np.ones(8, dtype=np.int64))
        with self.assertRaises(ValueError):
            spec.normalise_to_livetime()

    def test_raises_when_live_time_zero(self):
        spec = _make_spectrum(n=8)
        spec.live_time = 0.0
        with self.assertRaises(ValueError):
            spec.normalise_to_livetime()

    def test_fractional_live_time(self):
        """Non-integer live_time should produce correct float rates."""
        spec = _make_spectrum(n=3, count_val=10, live_time=0.5)
        rates = spec.normalise_to_livetime()
        np.testing.assert_array_almost_equal(rates, [20.0, 20.0, 20.0])

    def test_does_not_modify_counts_in_place(self):
        """Calling normalise_to_livetime must not mutate the spectrum."""
        spec = _make_spectrum(n=4, count_val=100, live_time=50.0)
        _ = spec.normalise_to_livetime()
        np.testing.assert_array_equal(spec.counts, np.full(4, 100, dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
