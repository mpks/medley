import numpy as np
from numpy.testing import assert_allclose

from medley.projection import project_position
from medley.synthetic_experiment import make_beam, make_detector


def test_synthetic_detector_is_downstream_of_the_sample():
    det = make_detector(578.3, 512, 512, 0.075, 0.075)
    # electrons travel along -z, so the panel must be at negative z
    assert det.origin[2] < 0


def test_synthetic_direct_beam_hits_beam_centre():
    beam = make_beam(0.02851)
    det = make_detector(578.3, 512, 512, 0.075, 0.075)
    px, py = project_position(0.0, 0.0, 0.0, beam, det)
    assert_allclose([px, py], det.beam_centre_px, atol=1e-6)
