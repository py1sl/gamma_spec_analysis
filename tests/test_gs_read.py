import unittest
import gs_analysis as gs

class read_test_case(unittest.TestCase):
    """ tests for file reading functions"""

    def test_read_file(self):
        data = gs.read_file("../test_data/Ba_133_raised_1.Spe")
        self.assertEqual(len(data), 8261)
        self.assertEqual(data[0], "$SPEC_ID:")
        self.assertEqual(data[-2], "3")
        
    def test_read_times(self):
        data = gs.read_file("../test_data/Ba_133_raised_1.Spe")
        self.assertEqual(gs.get_live_time(data), 326)
        self.assertEqual(gs.get_real_time(data), 431)
        
    def test_get_fits(self):
        data = gs.read_file("../test_data/Ba_133_raised_1.Spe")
        e_data = gs.get_e_fit(data)
        self.assertEqual(len(e_data), 2)
        self.assertEqual(e_data[0], 0.323476)
        self.assertEqual(e_data[1], 0.365473)

        
if __name__ == '__main__':
    unittest.main()
