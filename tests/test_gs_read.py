import unittest
import gs_analysis as gs

class read_test_case(unittest.TestCase):
    """ tests for perimeter functions"""

    def test_read_file(self):
        data = gs.read_file("test_data/Ba_133_raised_1.Spe"
        self.assertEqual(len(data), 8261)
