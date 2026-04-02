import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for testing
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import gs_plotting as gsp
from ph_spectrum import PhSpectrum
from gs_analysis import BackgroundMethod, EfficiencyFitType


class TestPlotting(unittest.TestCase):
    """tests relating to plotting functions"""

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.show")
    def test_plot_spec(self, mock_show, mock_savefig):
        counts = [1, 10, 100, 1000]
        erg = [1, 2, 3, 4]
        fname = "test_plot.png"

        # called with just counts
        gsp.plot_spec(counts)
        # Assert that show was called
        mock_show.assert_called_once()

        # called with counts and energy
        gsp.plot_spec(counts, erg=erg)
        # Assert that show was called
        mock_show.assert_called()

        # called with a file name
        mock_savefig.reset_mock()
        gsp.plot_spec(counts, fname=fname)
        # Assert that savefig was called with the specified filename
        mock_savefig.assert_called_once_with(fname)

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.show")
    def test_plot_spect_peaks(self, mock_show, mock_savefig):
        counts = [1, 10, 100, 1000]
        erg = [1, 2, 3, 4]
        peaks = [3]
        fname = "test_plot.png"

        # called with data
        gsp.plot_spect_peaks(counts, erg, peaks)
        # Assert that show was called
        mock_show.assert_called_once()

        # called with data and fname
        # Assert that savefig was called with the specified filename
        # Assert that show was not called
        gsp.plot_spect_peaks(counts, erg, peaks, fname)
        mock_savefig.assert_called_once_with(fname)
        mock_show.assert_called_once()


class TestNewPlottingFunctions(unittest.TestCase):
    """Tests for the new plotting capabilities added in the improvement pass."""

    def setUp(self):
        """Common test data shared across all new-function tests."""
        # A 100-channel spectrum with a simple peak around channel 40-60
        x = np.arange(100)
        bg = np.ones(100) * 10
        peak = 200 * np.exp(-((x - 50) ** 2) / (2 * 5 ** 2))
        self.counts = (bg + peak).astype(int) + 1  # ensure all positive
        self.erg = np.linspace(0.0, 1.0, 100)
        self.c1 = 40
        self.c2 = 60
        plt.close("all")

    def tearDown(self):
        plt.close("all")

    # ------------------------------------------------------------------
    # plot_spec / plot_spect_peaks – return type and new parameters
    # ------------------------------------------------------------------

    def test_plot_spec_returns_axes(self):
        with patch("matplotlib.pyplot.show"):
            ax = gsp.plot_spec(self.counts)
        self.assertIsInstance(ax, Axes)

    def test_plot_spec_with_ax(self):
        fig, ax_in = plt.subplots()
        ax_out = gsp.plot_spec(self.counts, ax=ax_in)
        self.assertIs(ax_out, ax_in)

    def test_plot_spec_no_show_when_ax_provided(self):
        fig, ax_in = plt.subplots()
        with patch("matplotlib.pyplot.show") as mock_show:
            gsp.plot_spec(self.counts, ax=ax_in)
        mock_show.assert_not_called()

    def test_plot_spec_title_and_labels(self):
        with patch("matplotlib.pyplot.show"):
            ax = gsp.plot_spec(
                self.counts,
                title="My spectrum",
                xlabel="keV",
                ylabel="N",
                yscale="linear",
            )
        self.assertEqual(ax.get_title(), "My spectrum")
        self.assertEqual(ax.get_xlabel(), "keV")
        self.assertEqual(ax.get_ylabel(), "N")

    def test_plot_spect_peaks_returns_axes(self):
        with patch("matplotlib.pyplot.show"):
            ax = gsp.plot_spect_peaks(self.counts, self.erg, [50])
        self.assertIsInstance(ax, Axes)

    def test_plot_spect_peaks_with_ax(self):
        fig, ax_in = plt.subplots()
        ax_out = gsp.plot_spect_peaks(self.counts, self.erg, [50], ax=ax_in)
        self.assertIs(ax_out, ax_in)

    # ------------------------------------------------------------------
    # plot_spectrum
    # ------------------------------------------------------------------

    def test_plot_spectrum_phspectrum(self):
        spec = PhSpectrum(counts=self.counts, ebin=self.erg)
        with patch("matplotlib.pyplot.show"):
            ax = gsp.plot_spectrum(spec)
        self.assertIsInstance(ax, Axes)

    def test_plot_spectrum_no_ebin(self):
        spec = PhSpectrum(counts=self.counts)
        with patch("matplotlib.pyplot.show"):
            ax = gsp.plot_spectrum(spec)
        self.assertIsInstance(ax, Axes)

    def test_plot_spectrum_with_ax(self):
        spec = PhSpectrum(counts=self.counts, ebin=self.erg)
        fig, ax_in = plt.subplots()
        ax_out = gsp.plot_spectrum(spec, ax=ax_in)
        self.assertIs(ax_out, ax_in)

    def test_plot_spectrum_saves_file(self):
        spec = PhSpectrum(counts=self.counts, ebin=self.erg)
        with patch("matplotlib.pyplot.savefig") as mock_save:
            gsp.plot_spectrum(spec, fname="out.png")
        mock_save.assert_called_once_with("out.png")

    # ------------------------------------------------------------------
    # plot_spectra_overlay
    # ------------------------------------------------------------------

    def test_plot_spectra_overlay_list_of_arrays(self):
        spectra = [self.counts, self.counts * 2]
        with patch("matplotlib.pyplot.show"):
            ax = gsp.plot_spectra_overlay(spectra, labels=["A", "B"])
        self.assertIsInstance(ax, Axes)

    def test_plot_spectra_overlay_phspectrum_objects(self):
        spec1 = PhSpectrum(counts=self.counts, ebin=self.erg)
        spec2 = PhSpectrum(counts=self.counts * 2, ebin=self.erg)
        with patch("matplotlib.pyplot.show"):
            ax = gsp.plot_spectra_overlay([spec1, spec2])
        self.assertIsInstance(ax, Axes)

    def test_plot_spectra_overlay_with_ergs(self):
        spectra = [self.counts, self.counts * 2]
        ergs = [self.erg, self.erg]
        with patch("matplotlib.pyplot.show"):
            ax = gsp.plot_spectra_overlay(spectra, ergs=ergs)
        self.assertIsInstance(ax, Axes)

    def test_plot_spectra_overlay_default_labels(self):
        with patch("matplotlib.pyplot.show"):
            ax = gsp.plot_spectra_overlay([self.counts, self.counts])
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertIn("Spectrum 0", legend_texts)
        self.assertIn("Spectrum 1", legend_texts)

    def test_plot_spectra_overlay_saves_file(self):
        with patch("matplotlib.pyplot.savefig") as mock_save:
            gsp.plot_spectra_overlay([self.counts], fname="overlay.png")
        mock_save.assert_called_once_with("overlay.png")

    # ------------------------------------------------------------------
    # plot_peak_roi
    # ------------------------------------------------------------------

    def test_plot_peak_roi_no_fit(self):
        x = self.erg[40:60]
        y = self.counts[40:60]
        with patch("matplotlib.pyplot.show"):
            ax = gsp.plot_peak_roi(x, y)
        self.assertIsInstance(ax, Axes)

    def test_plot_peak_roi_single_gaussian(self):
        x = np.linspace(0, 1, 50)
        y = 100 * np.exp(-((x - 0.5) ** 2) / (2 * 0.05 ** 2)) + 1
        fit_params = [100.0, 0.5, 0.05]
        with patch("matplotlib.pyplot.show"):
            ax = gsp.plot_peak_roi(x, y, fit_params=fit_params)
        self.assertIsInstance(ax, Axes)

    def test_plot_peak_roi_doublet(self):
        x = np.linspace(0, 1, 50)
        y = (80 * np.exp(-((x - 0.35) ** 2) / (2 * 0.05 ** 2))
             + 60 * np.exp(-((x - 0.65) ** 2) / (2 * 0.05 ** 2)) + 1)
        fit_params = [80.0, 0.35, 0.05, 60.0, 0.65, 0.05]
        with patch("matplotlib.pyplot.show"):
            ax = gsp.plot_peak_roi(x, y, fit_params=fit_params)
        self.assertIsInstance(ax, Axes)

    def test_plot_peak_roi_with_ax(self):
        fig, ax_in = plt.subplots()
        ax_out = gsp.plot_peak_roi(self.erg[40:60], self.counts[40:60], ax=ax_in)
        self.assertIs(ax_out, ax_in)

    # ------------------------------------------------------------------
    # plot_peak_with_background
    # ------------------------------------------------------------------

    def test_plot_peak_with_background_default(self):
        with patch("matplotlib.pyplot.show"):
            ax = gsp.plot_peak_with_background(self.counts, self.c1, self.c2)
        self.assertIsInstance(ax, Axes)

    def test_plot_peak_with_background_all_methods(self):
        for method in BackgroundMethod:
            plt.close("all")
            with patch("matplotlib.pyplot.show"):
                ax = gsp.plot_peak_with_background(
                    self.counts, self.c1, self.c2, method=method
                )
            self.assertIsInstance(ax, Axes)

    def test_plot_peak_with_background_custom_title(self):
        with patch("matplotlib.pyplot.show"):
            ax = gsp.plot_peak_with_background(
                self.counts, self.c1, self.c2, title="Custom"
            )
        self.assertEqual(ax.get_title(), "Custom")

    def test_plot_peak_with_background_with_ax(self):
        fig, ax_in = plt.subplots()
        ax_out = gsp.plot_peak_with_background(
            self.counts, self.c1, self.c2, ax=ax_in
        )
        self.assertIs(ax_out, ax_in)

    def test_plot_peak_with_background_saves_file(self):
        with patch("matplotlib.pyplot.savefig") as mock_save:
            gsp.plot_peak_with_background(
                self.counts, self.c1, self.c2, fname="bg.png"
            )
        mock_save.assert_called_once_with("bg.png")

    # ------------------------------------------------------------------
    # plot_background_methods_comparison
    # ------------------------------------------------------------------

    @patch("matplotlib.pyplot.show")
    def test_plot_background_methods_comparison_returns_figure(self, _):
        fig = gsp.plot_background_methods_comparison(self.counts, self.c1, self.c2)
        self.assertIsInstance(fig, Figure)

    def test_plot_background_methods_comparison_saves_file(self):
        with patch("matplotlib.pyplot.savefig") as mock_save:
            gsp.plot_background_methods_comparison(
                self.counts, self.c1, self.c2, fname="comp.png"
            )
        mock_save.assert_called_once_with("comp.png")

    @patch("matplotlib.pyplot.show")
    def test_plot_background_methods_comparison_custom_title(self, _):
        fig = gsp.plot_background_methods_comparison(
            self.counts, self.c1, self.c2, title="My Title"
        )
        self.assertIsInstance(fig, Figure)

    # ------------------------------------------------------------------
    # plot_efficiency_curve
    # ------------------------------------------------------------------

    def test_plot_efficiency_curve_log_fit(self):
        eff_coeff = [-2.0, 1.0]
        with patch("matplotlib.pyplot.show"):
            ax = gsp.plot_efficiency_curve((0.1, 2.0), eff_coeff)
        self.assertIsInstance(ax, Axes)

    def test_plot_efficiency_curve_inverse_energy(self):
        eff_coeff = [-2.0, 0.5]
        with patch("matplotlib.pyplot.show"):
            ax = gsp.plot_efficiency_curve(
                (0.5, 3.0), eff_coeff, fit_type=EfficiencyFitType.INVERSE_ENERGY
            )
        self.assertIsInstance(ax, Axes)

    def test_plot_efficiency_curve_with_measured_points(self):
        eff_coeff = [-2.0, 1.0]
        erg_pts = np.array([0.5, 1.0, 1.5])
        eff_pts = np.array([0.08, 0.12, 0.09])
        with patch("matplotlib.pyplot.show"):
            ax = gsp.plot_efficiency_curve(
                (0.1, 2.0), eff_coeff,
                erg_points=erg_pts, eff_points=eff_pts,
            )
        self.assertIsInstance(ax, Axes)

    def test_plot_efficiency_curve_with_ax(self):
        fig, ax_in = plt.subplots()
        ax_out = gsp.plot_efficiency_curve((0.1, 2.0), [-2.0], ax=ax_in)
        self.assertIs(ax_out, ax_in)

    def test_plot_efficiency_curve_saves_file(self):
        with patch("matplotlib.pyplot.savefig") as mock_save:
            gsp.plot_efficiency_curve((0.1, 2.0), [-2.0], fname="eff.png")
        mock_save.assert_called_once_with("eff.png")

    # ------------------------------------------------------------------
    # plot_smoothing_comparison
    # ------------------------------------------------------------------

    def test_plot_smoothing_comparison_defaults(self):
        with patch("matplotlib.pyplot.show"):
            ax = gsp.plot_smoothing_comparison(self.counts)
        self.assertIsInstance(ax, Axes)

    def test_plot_smoothing_comparison_custom_smoothers(self):
        from gs_analysis import three_point_smooth
        with patch("matplotlib.pyplot.show"):
            ax = gsp.plot_smoothing_comparison(
                self.counts,
                smoothers=[("3-pt", three_point_smooth)],
            )
        self.assertIsInstance(ax, Axes)

    def test_plot_smoothing_comparison_with_erg(self):
        with patch("matplotlib.pyplot.show"):
            ax = gsp.plot_smoothing_comparison(self.counts, erg=self.erg)
        self.assertIsInstance(ax, Axes)

    def test_plot_smoothing_comparison_with_ax(self):
        fig, ax_in = plt.subplots()
        ax_out = gsp.plot_smoothing_comparison(self.counts, ax=ax_in)
        self.assertIs(ax_out, ax_in)

    def test_plot_smoothing_comparison_saves_file(self):
        with patch("matplotlib.pyplot.savefig") as mock_save:
            gsp.plot_smoothing_comparison(self.counts, fname="smooth.png")
        mock_save.assert_called_once_with("smooth.png")

    # ------------------------------------------------------------------
    # plot_doublet_fit
    # ------------------------------------------------------------------

    def test_plot_doublet_fit(self):
        x = np.linspace(0, 1, 50)
        y = (80 * np.exp(-((x - 0.35) ** 2) / (2 * 0.05 ** 2))
             + 60 * np.exp(-((x - 0.65) ** 2) / (2 * 0.05 ** 2)) + 1)
        popt = [80.0, 0.35, 0.05, 60.0, 0.65, 0.05]
        with patch("matplotlib.pyplot.show"):
            ax = gsp.plot_doublet_fit(x, y, popt)
        self.assertIsInstance(ax, Axes)

    def test_plot_doublet_fit_with_ax(self):
        x = np.linspace(0, 1, 50)
        y = np.ones(50)
        popt = [1.0, 0.3, 0.05, 1.0, 0.7, 0.05]
        fig, ax_in = plt.subplots()
        ax_out = gsp.plot_doublet_fit(x, y, popt, ax=ax_in)
        self.assertIs(ax_out, ax_in)

    def test_plot_doublet_fit_saves_file(self):
        x = np.linspace(0, 1, 50)
        y = np.ones(50)
        popt = [1.0, 0.3, 0.05, 1.0, 0.7, 0.05]
        with patch("matplotlib.pyplot.savefig") as mock_save:
            gsp.plot_doublet_fit(x, y, popt, fname="doublet.png")
        mock_save.assert_called_once_with("doublet.png")

    def test_plot_doublet_fit_title(self):
        x = np.linspace(0, 1, 50)
        y = np.ones(50)
        popt = [1.0, 0.3, 0.05, 1.0, 0.7, 0.05]
        with patch("matplotlib.pyplot.show"):
            ax = gsp.plot_doublet_fit(x, y, popt, title="Doublet")
        self.assertEqual(ax.get_title(), "Doublet")

    # ------------------------------------------------------------------
    # Option 9 – normalise by live time
    # ------------------------------------------------------------------

    def test_plot_spectrum_normalise_ylabel(self):
        """plot_spectrum with normalise=True should use count-rate y-label."""
        spec = PhSpectrum(counts=self.counts, ebin=self.erg, live_time=100.0)
        with patch("matplotlib.pyplot.show"):
            ax = gsp.plot_spectrum(spec, normalise=True)
        self.assertEqual(ax.get_ylabel(), "Count Rate (counts/s)")

    def test_plot_spectrum_normalise_data(self):
        """plot_spectrum with normalise=True should scale y-data by live_time."""
        live_time = 50.0
        spec = PhSpectrum(counts=self.counts, ebin=self.erg, live_time=live_time)
        fig, ax_in = plt.subplots()
        gsp.plot_spectrum(spec, normalise=True, ax=ax_in)
        # Retrieve the plotted y-data (step plot produces two arrays)
        lines = ax_in.lines
        # The step plot stores data; verify maximum y is near max(counts)/live_time
        plotted_max = max(line.get_ydata().max() for line in lines)
        self.assertAlmostEqual(plotted_max, self.counts.max() / live_time, delta=1.0)

    def test_plot_spectrum_normalise_ylabel_override(self):
        """Caller can override the default count-rate ylabel."""
        spec = PhSpectrum(counts=self.counts, ebin=self.erg, live_time=100.0)
        with patch("matplotlib.pyplot.show"):
            ax = gsp.plot_spectrum(spec, normalise=True, ylabel="Custom")
        self.assertEqual(ax.get_ylabel(), "Custom")

    def test_plot_spectrum_no_normalise_ylabel(self):
        """Without normalise, ylabel should remain the default 'Counts'."""
        spec = PhSpectrum(counts=self.counts, ebin=self.erg)
        with patch("matplotlib.pyplot.show"):
            ax = gsp.plot_spectrum(spec)
        self.assertEqual(ax.get_ylabel(), "Counts")

    def test_plot_spectra_overlay_normalise_ylabel(self):
        """plot_spectra_overlay normalise=True should use count-rate y-label."""
        spec1 = PhSpectrum(counts=self.counts, ebin=self.erg, live_time=60.0)
        spec2 = PhSpectrum(counts=self.counts * 2, ebin=self.erg, live_time=120.0)
        with patch("matplotlib.pyplot.show"):
            ax = gsp.plot_spectra_overlay([spec1, spec2], normalise=True)
        self.assertEqual(ax.get_ylabel(), "Count Rate (counts/s)")

    def test_plot_spectra_overlay_normalise_explicit_ylabel(self):
        """Explicit ylabel should take precedence even when normalise=True."""
        spec = PhSpectrum(counts=self.counts, ebin=self.erg, live_time=60.0)
        with patch("matplotlib.pyplot.show"):
            ax = gsp.plot_spectra_overlay([spec], normalise=True, ylabel="My label")
        self.assertEqual(ax.get_ylabel(), "My label")

    def test_plot_spectra_overlay_normalise_plain_arrays_unchanged(self):
        """Plain count arrays should not be normalised (no live_time available)."""
        raw = self.counts
        with patch("matplotlib.pyplot.show"):
            ax_norm = gsp.plot_spectra_overlay([raw], normalise=True)
            ax_plain = gsp.plot_spectra_overlay([raw], normalise=False)
        # Both should plot the same data
        y_norm = ax_norm.lines[0].get_ydata()
        y_plain = ax_plain.lines[0].get_ydata()
        np.testing.assert_array_equal(y_norm, y_plain)


if __name__ == "__main__":
    unittest.main()
