import unittest
from unittest.mock import patch, mock_open, call
import numpy as np
import matplotlib.pyplot as plt
import gs_analysis as gs


class analysis_test_case(unittest.TestCase):
    """tests for analysis functions"""

    def test_counts(self):
        # gross counts
        counts = [1, 1, 1, 1, 1]
        gc = gs.gross_count(counts, 1, 4)
        self.assertEqual(gc, 3)
        self.assertRaises(ValueError, gs.gross_count, counts, -1, 4)
        self.assertRaises(ValueError, gs.gross_count, counts, 1, 10)
        self.assertRaises(ValueError, gs.gross_count, counts, 10, 4)

        # background
        self.assertRaises(ValueError, gs.calc_bg, counts, -1, 4)
        self.assertRaises(ValueError, gs.calc_bg, counts, 1, 10)
        self.assertRaises(ValueError, gs.calc_bg, counts, 10, 4)
        self.assertRaises(ValueError, gs.calc_bg, counts, 1, 4, 2)

        # net
        self.assertRaises(ValueError, gs.net_counts, counts, -1, 4)
        self.assertRaises(ValueError, gs.net_counts, counts, 1, 10)
        self.assertRaises(ValueError, gs.net_counts, counts, 10, 4)

    def test_ebins(self):
        # testing find e pos
        ebins = [1, 2, 3, 4, 5]
        self.assertEqual(gs.find_energy_pos(ebins, 1.5), 0)
        self.assertEqual(gs.find_energy_pos(ebins, 1), 0)
        self.assertEqual(gs.find_energy_pos(ebins, 4.9), 3)
        self.assertFalse(gs.find_energy_pos(ebins, -1))
        self.assertFalse(gs.find_energy_pos(ebins, 5))
        self.assertFalse(gs.find_energy_pos(ebins, 10))
        self.assertFalse(gs.find_energy_pos(ebins, 0))

        # generating ebins
        spec = gs.get_spect("../test_data/Ba_133_raised_1.Spe")
        ebins = gs.generate_ebins(spec)
        self.assertEqual(len(ebins), len(spec.counts))

    def test_roi(self):
        spec = gs.get_spect("../test_data/Ba_133_raised_1.Spe")
        ebins = gs.generate_ebins(spec)
        peak_ebin, data = gs.get_peak_roi(230, spec.counts, ebins)
        self.assertEqual(len(data), 20)
        self.assertEqual(len(data), len(peak_ebin))
        self.assertRaises(ValueError, gs.get_peak_roi, 2, spec.counts, ebins)
        self.assertRaises(ValueError, gs.get_peak_roi, 10000, spec.counts, ebins)

    def test_eff_fit(self):
        self.assertRaises(ValueError, gs.calc_e_eff, 1.3, [1, 1, 1, 1], 5)

    def test_smoothing(self):
        # testing 5 point smooth
        spec = gs.get_spect("../test_data/Ba_133_raised_1.Spe")
        smoothed = gs.five_point_smooth(spec.counts)
        self.assertEqual(len(smoothed), len(spec.counts))
        self.assertEqual(smoothed[0], spec.counts[0])
        self.assertEqual(smoothed[-1], spec.counts[-1])


class TestPlotting(unittest.TestCase):
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.show")
    def test_plot_spec(self, mock_show, mock_savefig):
        counts = [1, 10, 100, 1000]
        erg = [1, 2, 3, 4]
        fname = "test_plot.png"

        # called with just counts
        gs.plot_spec(counts)
        # Assert that savefig was called
        mock_savefig.assert_called_once_with(None)
        # Assert that show was called
        mock_show.assert_called_once()

        # called with counts and energy
        gs.plot_spec(counts, erg=erg)
        # Assert that savefig was called
        mock_savefig.assert_called_once_with(None)
        # Assert that show was called
        mock_show.assert_called_once()

        # called with a file name
        gs.plot_spec(counts, fname=fname)
        # Assert that savefig was called with the specified filename
        mock_savefig.assert_called_once_with(fname)


if __name__ == "__main__":
    unittest.main()
