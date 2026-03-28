import unittest
from unittest.mock import patch
import gs_plotting as gsp


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


if __name__ == "__main__":
    unittest.main()
