import unittest
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
        data, ebins = gs.get_spect("../test_data/Ba_133_raised_1.Spe")
        self.assertEqual(len(ebins), 8192)

        # testing find e pos
        ebins = [1, 2, 3, 4, 5]
        self.assertEqual(gs.find_energy_pos(ebins, 1.5), 0)
        self.assertEqual(gs.find_energy_pos(ebins, 1), 0)
        self.assertEqual(gs.find_energy_pos(ebins, 4.9), 3)
        self.assertFalse(gs.find_energy_pos(ebins, -1))
        self.assertFalse(gs.find_energy_pos(ebins, 5))
        self.assertFalse(gs.find_energy_pos(ebins, 10))
        self.assertFalse(gs.find_energy_pos(ebins, 0))

    def test_roi(self):
        counts, ebins = gs.get_spect("../test_data/Ba_133_raised_1.Spe")
        peak_ebin, data = gs.get_peak_roi(230, counts, ebins)
        self.assertEqual(len(data), 20)
        self.assertEqual(len(data), len(peak_ebin))
        self.assertRaises(ValueError, gs.get_peak_roi, 2, counts, ebins)
        self.assertRaises(ValueError, gs.get_peak_roi, 10000, counts, ebins)

    def test_eff_fit(self):
        self.assertRaises(ValueError, gs.calc_e_eff, 1.3, [1, 1, 1, 1], 5)


if __name__ == "__main__":
    unittest.main()
