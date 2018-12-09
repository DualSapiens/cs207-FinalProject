from ..interface import BeamDirection, Beam, PlannerInterface
import numpy as np
import pytest
from numpy.testing import assert_array_equal
import os


# Read file relative to current directory (when using pytest)
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


class TestBeam:
    def test_beam_init(self):
        with pytest.raises(Exception):
            beam = Beam("diagonal")
        beam = Beam(BeamDirection.Horizontal)
        assert beam.direction == BeamDirection.Horizontal
        assert beam.name == "Horizontal beam"

    def test_beam_solve(self):
        # adapted from example in [Boye02] in documentation.
        beam = Beam(BeamDirection.Horizontal)
        beamlets = np.array([1, 7, 9, 8, 8, 7, 2, 10, 3])
        intensity = 1.
        beam.solve(beamlets, intensity)
        assert beam.collimator["left"] == [0, 1, 1, 1, 1, 1, 1, 2, 2, 7, 7, 7, 7, 7, 7, 7, 7]
        assert beam.collimator["right"] == [3, 5, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9]


class TestInterface:
    def test_read_maps(self):
        plan = PlannerInterface(os.path.join(__location__, 'test.map'))
        maps = plan.get_maps()
        target_map = np.array([[1, 2, 3],
                               [4, 5, 6],
                               [7, 8, 9]])
        assert_array_equal(maps["target"], target_map)
        max_map = np.array([[10, 10, 10],
                            [10, 10, 10],
                            [10, 10, 10]])
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
        max_map = np.array([[50, 50, 50],
                            [50, 55.5, 50],
                            [52.2, 50, 50]])
        assert_array_equal(maps['max'], max_map)
        min_map = np.array([[0, 0, 0],
                            [0.1, 0, 0.5],
                            [0, 0, 0]])
        assert_array_equal(maps["min"], min_map)

    def test_missing_values_map(self):
        missing_value = np.NaN
        plan = PlannerInterface(os.path.join(__location__, 'test_missing_values.map'))
        maps = plan.get_maps()
        target_map = np.array([[1, missing_value, 3],
                               [4, 5, 6.3],
                               [missing_value, 8, 9]])
        assert_array_equal(maps["target"], target_map)
        max_map = np.array([[50, 50, missing_value],
                            [50.1, 50, 50],
                            [missing_value, 50.2, 50]])
        assert_array_equal(maps['max'], max_map)
        min_map = np.array([[0, 0, missing_value],
                            [0.2, 0.5, missing_value],
                            [missing_value, 0, 0]])
        assert_array_equal(maps["min"], min_map)

    def test_optimize(self):
        plan = PlannerInterface(os.path.join(__location__, 'test_optimize.map'))
        intensity = 0.2
        plan.optimize(intensity, smoothness=0.012)
        maps = plan.get_maps()
        assert np.allclose(maps["optimized"], [[3.8, 2.8512664],
                                               [1.69183011, 0.93239382]])
        horiz_beam = plan._beams[0]
        vert_beam = plan._beams[1]
        assert np.allclose(horiz_beam.beamlets, [2.2, 0.2])
        assert np.allclose(vert_beam.beamlets, [1.6, 0.8])
        assert np.allclose(horiz_beam.collimator["left"], [0]*11)
        assert np.allclose(horiz_beam.collimator["right"], [1]*10 + [2])
        assert np.allclose(vert_beam.collimator["left"], [0]*8)
        assert np.allclose(vert_beam.collimator["right"], [1]*4 + [2]*4)
        assert horiz_beam.exposure_time == 11
        assert vert_beam.exposure_time == 8

    def test_rotate(self):
        plan = PlannerInterface(os.path.join(__location__, 'test_optimize.map'))
        intensity = 0.2
        plan.optimize(intensity, smoothness=0.02, allow_rotation=True)
        assert plan.rotate_angle == 90
        maps = plan.get_maps()
        assert np.allclose(maps["optimized"], [[2.8, 0.77295753],
                                               [3.83774517, 1.86478764]])
        horiz_beam = plan._beams[0]
        vert_beam = plan._beams[1]
        assert np.allclose(horiz_beam.beamlets, [0.4, 1.6])
        assert np.allclose(vert_beam.beamlets, [2.4, 0.4])
        assert np.allclose(horiz_beam.collimator["left"], [0]*2 + [1]*6)
        assert np.allclose(horiz_beam.collimator["right"], [2]*8)
        assert np.allclose(vert_beam.collimator["left"], [0]*12)
        assert np.allclose(vert_beam.collimator["right"], [1]*10 + [2]*2)
        assert horiz_beam.exposure_time == 8
        assert vert_beam.exposure_time == 12


