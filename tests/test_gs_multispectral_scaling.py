"""Tests for gs_multispectral_scaling module."""

import math
import unittest
from datetime import datetime, timedelta

import numpy as np

import gs_multispectral_scaling as gms
from gs_analysis import BackgroundMethod
from ph_spectrum import PhSpectrum


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_flat_spectrum(n_channels: int = 512,
                        counts_per_channel: int = 100,
                        live_time: float = 100.0,
                        real_time: float = 105.0,
                        start_time: str | None = None,
                        energy_coeffs=(0.0, 1.0)) -> PhSpectrum:
    """Return a PhSpectrum filled with *counts_per_channel* in every bin."""
    counts = np.full(n_channels, counts_per_channel, dtype=np.int64)
    spec = PhSpectrum(
        counts=counts,
        live_time=live_time,
        real_time=real_time,
        start_time=start_time,
        energy_fit_coefficients=list(energy_coeffs),
    )
    return spec


def _make_peak_spectrum(n_channels: int = 512,
                        peak_channel: int = 256,
                        peak_height: int = 1000,
                        bg_level: int = 10,
                        live_time: float = 100.0,
                        real_time: float = 105.0,
                        start_time: str | None = None,
                        energy_coeffs=(0.0, 1.0)) -> PhSpectrum:
    """Return a PhSpectrum with a simple triangular peak at *peak_channel*."""
    counts = np.full(n_channels, bg_level, dtype=np.int64)
    # Add a narrow Gaussian-like peak (±5 channels)
    half_width = 5
    for offset in range(-half_width, half_width + 1):
        ch = peak_channel + offset
        if 0 <= ch < n_channels:
            counts[ch] += int(peak_height * np.exp(-0.5 * (offset / 2.0) ** 2))
    spec = PhSpectrum(
        counts=counts,
        live_time=live_time,
        real_time=real_time,
        start_time=start_time,
        energy_fit_coefficients=list(energy_coeffs),
    )
    return spec


def _spe_time(dt: datetime) -> str:
    """Format a datetime the same way ORTEC .Spe files do."""
    return dt.strftime("%m/%d/%Y %H:%M:%S")


# ---------------------------------------------------------------------------
# Tests: add_spectra
# ---------------------------------------------------------------------------

class TestAddSpectra(unittest.TestCase):

    def test_add_two_flat_spectra(self):
        """Summed counts should equal the element-wise sum."""
        s1 = _make_flat_spectrum(counts_per_channel=100)
        s2 = _make_flat_spectrum(counts_per_channel=200)
        result = gms.add_spectra([s1, s2])
        np.testing.assert_array_equal(result.counts, np.full(512, 300, dtype=np.int64))

    def test_add_single_spectrum_unchanged(self):
        """Adding a single spectrum should return an equal spectrum."""
        s = _make_flat_spectrum(counts_per_channel=50)
        result = gms.add_spectra([s])
        np.testing.assert_array_equal(result.counts, s.counts)

    def test_live_time_summed(self):
        """live_time must be summed across all spectra."""
        specs = [_make_flat_spectrum(live_time=100.0) for _ in range(3)]
        result = gms.add_spectra(specs)
        self.assertAlmostEqual(result.live_time, 300.0)

    def test_real_time_summed(self):
        """real_time must be summed across all spectra."""
        specs = [_make_flat_spectrum(real_time=110.0) for _ in range(4)]
        result = gms.add_spectra(specs)
        self.assertAlmostEqual(result.real_time, 440.0)

    def test_live_time_none_when_any_missing(self):
        """live_time must be None when any input is missing it."""
        s1 = _make_flat_spectrum(live_time=100.0)
        s2 = PhSpectrum(counts=np.ones(512, dtype=np.int64))  # live_time=None
        result = gms.add_spectra([s1, s2])
        self.assertIsNone(result.live_time)

    def test_real_time_none_when_any_missing(self):
        """real_time must be None when any input is missing it."""
        s1 = _make_flat_spectrum(real_time=110.0)
        s2 = PhSpectrum(counts=np.ones(512, dtype=np.int64))
        result = gms.add_spectra([s1, s2])
        self.assertIsNone(result.real_time)

    def test_raises_on_empty_list(self):
        with self.assertRaises(ValueError):
            gms.add_spectra([])

    def test_raises_on_mismatched_channels(self):
        s1 = _make_flat_spectrum(n_channels=512)
        s2 = _make_flat_spectrum(n_channels=1024)
        with self.assertRaises(ValueError):
            gms.add_spectra([s1, s2])

    def test_energy_coefficients_from_first_spectrum(self):
        """Energy calibration must be taken from the first spectrum."""
        s1 = _make_flat_spectrum(energy_coeffs=(0.0, 2.0))
        s2 = _make_flat_spectrum(energy_coeffs=(1.0, 3.0))
        result = gms.add_spectra([s1, s2])
        self.assertEqual(list(result.energy_fit_coefficients), [0.0, 2.0])

    def test_add_many_spectra(self):
        """Sum of 10 identical spectra should equal 10× original counts."""
        specs = [_make_flat_spectrum(counts_per_channel=5) for _ in range(10)]
        result = gms.add_spectra(specs)
        np.testing.assert_array_equal(result.counts, np.full(512, 50, dtype=np.int64))

    def test_counts_dtype_is_int64(self):
        """Summed counts array should be int64."""
        specs = [_make_flat_spectrum(counts_per_channel=10) for _ in range(2)]
        result = gms.add_spectra(specs)
        self.assertEqual(result.counts.dtype, np.int64)


# ---------------------------------------------------------------------------
# Tests: get_elapsed_times
# ---------------------------------------------------------------------------

class TestGetElapsedTimes(unittest.TestCase):

    def test_from_start_time_strings(self):
        """Elapsed times from timestamps should be correct."""
        t0 = datetime(2024, 1, 1, 12, 0, 0)
        times = [t0 + timedelta(seconds=s) for s in [0, 60, 180, 300]]
        specs = [_make_flat_spectrum(start_time=_spe_time(t)) for t in times]
        elapsed = gms.get_elapsed_times(specs)
        np.testing.assert_array_almost_equal(elapsed, [0.0, 60.0, 180.0, 300.0])

    def test_first_elapsed_is_zero(self):
        """The first elapsed time must always be 0."""
        t0 = datetime(2023, 6, 15, 9, 0, 0)
        specs = [_make_flat_spectrum(start_time=_spe_time(t0 + timedelta(hours=h)))
                 for h in range(5)]
        elapsed = gms.get_elapsed_times(specs)
        self.assertEqual(elapsed[0], 0.0)

    def test_fallback_to_real_time(self):
        """When start_time is absent, real_time should be accumulated."""
        specs = [_make_flat_spectrum(real_time=100.0, start_time=None) for _ in range(4)]
        elapsed = gms.get_elapsed_times(specs)
        np.testing.assert_array_almost_equal(elapsed, [0.0, 100.0, 200.0, 300.0])

    def test_single_spectrum_returns_zero(self):
        """A single-element list should return [0.0]."""
        s = _make_flat_spectrum(real_time=50.0)
        elapsed = gms.get_elapsed_times([s])
        np.testing.assert_array_equal(elapsed, [0.0])

    def test_raises_on_empty_list(self):
        with self.assertRaises(ValueError):
            gms.get_elapsed_times([])

    def test_raises_when_no_times_available(self):
        """Missing both start_time and real_time must raise ValueError."""
        specs = [PhSpectrum(counts=np.ones(128, dtype=np.int64)) for _ in range(3)]
        with self.assertRaises(ValueError):
            gms.get_elapsed_times(specs)

    def test_raises_on_non_monotonic_timestamps(self):
        """Out-of-order timestamps must raise ValueError."""
        t0 = datetime(2024, 3, 1, 10, 0, 0)
        times = [t0, t0 + timedelta(minutes=5), t0 + timedelta(minutes=2)]
        specs = [_make_flat_spectrum(start_time=_spe_time(t)) for t in times]
        with self.assertRaises(ValueError):
            gms.get_elapsed_times(specs)


# ---------------------------------------------------------------------------
# Tests: track_peak_activity
# ---------------------------------------------------------------------------

class TestTrackPeakActivity(unittest.TestCase):

    def _make_decaying_series(self, n: int = 5, half_life_s: float = 200.0):
        """Create a series of spectra whose peak height decays exponentially."""
        t0 = datetime(2024, 1, 1, 0, 0, 0)
        dt_s = 50.0  # 50 s between measurements
        lam = math.log(2) / half_life_s
        specs = []
        for k in range(n):
            elapsed = k * dt_s
            activity = 1000.0 * math.exp(-lam * elapsed)
            ts = _spe_time(t0 + timedelta(seconds=elapsed))
            specs.append(_make_peak_spectrum(
                peak_channel=256,
                peak_height=int(activity),
                bg_level=5,
                live_time=dt_s,
                real_time=dt_s,
                start_time=ts,
            ))
        return specs

    def test_returns_correct_lengths(self):
        specs = self._make_decaying_series(n=4)
        times, rates = gms.track_peak_activity(specs, energy=256.0)
        self.assertEqual(len(times), 4)
        self.assertEqual(len(rates), 4)

    def test_rates_are_positive(self):
        specs = self._make_decaying_series(n=5)
        _, rates = gms.track_peak_activity(specs, energy=256.0)
        for r in rates:
            if not np.isnan(r):
                self.assertGreater(r, 0.0)

    def test_rates_decrease_over_time(self):
        """Count rates from a decaying source must decrease monotonically."""
        specs = self._make_decaying_series(n=6, half_life_s=300.0)
        _, rates = gms.track_peak_activity(specs, energy=256.0)
        valid = rates[~np.isnan(rates)]
        for i in range(len(valid) - 1):
            self.assertGreater(valid[i], valid[i + 1])

    def test_raises_on_empty_list(self):
        with self.assertRaises(ValueError):
            gms.track_peak_activity([], energy=256.0)

    def test_raises_when_missing_calibration(self):
        spec = PhSpectrum(counts=np.ones(512, dtype=np.int64), live_time=100.0)
        with self.assertRaises(ValueError):
            gms.track_peak_activity([spec], energy=256.0)

    def test_raises_when_missing_live_time(self):
        spec = _make_flat_spectrum()
        spec.live_time = None
        with self.assertRaises(ValueError):
            gms.track_peak_activity([spec], energy=256.0)

    def test_nan_for_out_of_range_energy(self):
        """Energy outside the spectrum range must produce NaN count rate."""
        specs = [_make_peak_spectrum(n_channels=512, peak_channel=256)]
        _, rates = gms.track_peak_activity(specs, energy=9999.0)
        self.assertTrue(np.isnan(rates[0]))

    def test_custom_bg_method_accepted(self):
        specs = self._make_decaying_series(n=3)
        times, rates = gms.track_peak_activity(
            specs, energy=256.0, bg_method=BackgroundMethod.LINEAR
        )
        self.assertEqual(len(times), 3)


# ---------------------------------------------------------------------------
# Tests: estimate_half_life
# ---------------------------------------------------------------------------

class TestEstimateHalfLife(unittest.TestCase):

    def _synthetic_data(self, half_life_s: float = 300.0, n: int = 10,
                        dt: float = 60.0, a0: float = 1000.0):
        """Generate perfect exponential decay data."""
        lam = math.log(2) / half_life_s
        times = np.arange(n, dtype=np.float64) * dt
        rates = a0 * np.exp(-lam * times)
        return times, rates

    def test_recovers_known_half_life(self):
        """Estimated half-life should be close to the true value."""
        true_hl = 300.0  # seconds
        times, rates = self._synthetic_data(half_life_s=true_hl)
        hl, hl_unc = gms.estimate_half_life(times, rates)
        self.assertAlmostEqual(hl, true_hl, delta=1.0)

    def test_uncertainty_is_non_negative(self):
        times, rates = self._synthetic_data(half_life_s=600.0)
        _, hl_unc = gms.estimate_half_life(times, rates)
        self.assertGreaterEqual(hl_unc, 0.0)

    def test_half_life_positive(self):
        times, rates = self._synthetic_data(half_life_s=120.0)
        hl, _ = gms.estimate_half_life(times, rates)
        self.assertGreater(hl, 0.0)

    def test_raises_on_mismatched_lengths(self):
        times = np.array([0.0, 10.0, 20.0])
        rates = np.array([100.0, 80.0])
        with self.assertRaises(ValueError):
            gms.estimate_half_life(times, rates)

    def test_raises_on_too_few_points(self):
        with self.assertRaises(ValueError):
            gms.estimate_half_life(np.array([0.0]), np.array([100.0]))

    def test_nan_values_are_ignored(self):
        """NaN count rates should be silently dropped before fitting."""
        true_hl = 300.0
        times, rates = self._synthetic_data(half_life_s=true_hl, n=8)
        rates[2] = np.nan
        rates[5] = np.nan
        hl, _ = gms.estimate_half_life(times, rates)
        self.assertAlmostEqual(hl, true_hl, delta=5.0)

    def test_raises_on_too_few_after_nan_removal(self):
        """Fewer than 2 valid points after NaN removal must raise."""
        times = np.array([0.0, 60.0, 120.0])
        rates = np.array([100.0, np.nan, np.nan])
        with self.assertRaises(ValueError):
            gms.estimate_half_life(times, rates)

    def test_integration_with_track_peak_activity(self):
        """track_peak_activity output can be fed directly into estimate_half_life."""
        half_life_s = 300.0
        lam = math.log(2) / half_life_s
        t0 = datetime(2024, 1, 1, 0, 0, 0)
        dt_s = 60.0
        specs = []
        for k in range(8):
            elapsed = k * dt_s
            activity = 2000.0 * math.exp(-lam * elapsed)
            ts = _spe_time(t0 + timedelta(seconds=elapsed))
            specs.append(_make_peak_spectrum(
                peak_channel=200,
                peak_height=int(activity),
                bg_level=5,
                live_time=dt_s,
                real_time=dt_s,
                start_time=ts,
            ))
        elapsed_times, count_rates = gms.track_peak_activity(specs, energy=200.0)
        hl, hl_unc = gms.estimate_half_life(elapsed_times, count_rates)
        # Allow generous tolerance because net counts include background
        self.assertAlmostEqual(hl, half_life_s, delta=half_life_s * 0.3)
        self.assertGreaterEqual(hl_unc, 0.0)


# ---------------------------------------------------------------------------
# Tests: _parse_start_time (internal helper – tested via get_elapsed_times)
# ---------------------------------------------------------------------------

class TestParseStartTime(unittest.TestCase):

    def test_valid_ortec_format(self):
        dt = gms._parse_start_time("02/25/2020 14:24:52")
        self.assertEqual(dt, datetime(2020, 2, 25, 14, 24, 52))

    def test_none_input(self):
        self.assertIsNone(gms._parse_start_time(None))

    def test_invalid_string(self):
        self.assertIsNone(gms._parse_start_time("not a date"))

    def test_whitespace_stripped(self):
        dt = gms._parse_start_time("  03/01/2021 08:00:00  ")
        self.assertEqual(dt, datetime(2021, 3, 1, 8, 0, 0))


if __name__ == "__main__":
    unittest.main()
