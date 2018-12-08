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
        assert beam.collimator["left"] == [0, 1, 1, 1, 1, 1, 1, 2, 2, 7, 7, 7 ,7 ,7 ,7 ,7, 7]
        assert beam.collimator["right"] == [3, 5, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9]

class TestInterface:
    def test_read_maps(self):
        plan = PlannerInterface(os.path.join(__location__, 'test.map'))
        maps = plan.get_maps()
        target_map = np.array([[1, 2, 3],
                               [4, 5, 6],
                               [7, 8, 9]])
        assert_array_equal(maps["target"], target_map)

        
