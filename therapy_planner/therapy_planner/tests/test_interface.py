from ..interface import PlannerInterface
import numpy as np
from numpy.testing import assert_array_equal
import os


# Read file relative to current directory (when using pytest)
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


class TestInterface:
    def test_read_maps(self):
        plan = PlannerInterface(os.path.join(__location__, 'test.map'))
        maps = plan.get_maps()
        target_map = np.array([[1, 2, 3],
                               [4, 5, 6],
                               [7, 8, 9]])
        assert_array_equal(maps["target"], target_map)
        max_map = np.array([[5, 5, 5],
                            [5, 5, 5],
                            [5, 5, 5]])
        assert_array_equal(maps['max'], max_map)
        min_map = np.array([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]])
        assert_array_equal(maps["min"], min_map)

    def test_read_float_maps(self):
        plan = PlannerInterface(os.path.join(__location__, 'test_float.map'))
        maps = plan.get_maps()
        target_map = np.array([[1.1, 2, 3],
                               [4, 5, 6],
                               [7, 8, 9.0]])
        assert_array_equal(maps["target"], target_map)
        max_map = np.array([[5, 5, 5],
                            [5, 5.5, 5],
                            [5.2, 5, 5]])
        assert_array_equal(maps['max'], max_map)
        min_map = np.array([[0, 0, 0],
                            [0.1, 0, 0.5],
                            [0, 0, 0]])
        assert_array_equal(maps["min"], min_map)
