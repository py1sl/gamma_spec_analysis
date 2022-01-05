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
        
        # net
        self.assertRaises(ValueError, gs.net_counts, counts, -1, 4)
        self.assertRaises(ValueError, gs.net_counts, counts, 1, 10)
        self.assertRaises(ValueError, gs.net_counts, counts, 10, 4)
        
    def test_ebins(self):
        data = gs.read_file("../test_data/Ba_133_raised_1.Spe")
        self.assertEqual(len(data), 8261)


if __name__ == "__main__":
    unittest.main()
