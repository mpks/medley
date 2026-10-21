import numpy as np
import pytest
from numpy.testing import assert_allclose

from medley.experiment import _beam_from_dict, _detector_from_dict
from medley.projection import (
    panel_matrix,
    project_position,
    project_ray,
    abtem_to_lab,
)

# A DIALS-style setup: detector 500 mm downstream at z < 0,
# beam travelling along -z.  In dxtbx, beam "direction" points
# from the sample to the source, so it is +z here.
WAVELENGTH = 0.0251          # Å, ~200 keV electrons
PX = 0.075                   # mm
DIST = 500.0                 # mm
BEAM_CENTRE = (1024.0, 1024.0)

PANEL = {
    "fast_axis": [1.0, 0.0, 0.0],
    "slow_axis": [0.0, -1.0, 0.0],
    # origin = position of pixel (0, 0), chosen so the beam
    # hits pixel BEAM_CENTRE
    "origin": [-BEAM_CENTRE[0] * PX, BEAM_CENTRE[1] * PX, -DIST],
    "pixel_size": [PX, PX],
    "image_size": [2048, 2048],
}


@pytest.fixture
def beam():
    return _beam_from_dict({"wavelength": WAVELENGTH,
                            "direction": [0.0, 0.0, 1.0]})


@pytest.fixture
def detector(beam):
    return _detector_from_dict({"panels": [PANEL]}, beam.direction)


# --- panel geometry ---------------------------------------------

def test_detector_reader_finds_beam_centre_and_distance(detector):
    assert_allclose(detector.beam_centre_px, BEAM_CENTRE)
    assert_allclose(detector.distance_mm, DIST)


def test_ray_towards_panel_origin_hits_pixel_zero(detector):
    D = panel_matrix(detector)
    px, py = project_ray(detector.origin, D, PX, PX)
    assert_allclose([px, py], [0.0, 0.0], atol=1e-9)


def test_ray_to_known_panel_point_hits_known_pixel(detector):
    # Point 10 px along fast and 20 px along slow from the origin.
    point = (np.asarray(detector.origin)
             + 10 * PX * np.asarray(detector.fast_axis)
             + 20 * PX * np.asarray(detector.slow_axis))
    D = panel_matrix(detector)
    px, py = project_ray(point, D, PX, PX)
    assert_allclose([px, py], [10.0, 20.0], atol=1e-9)


def test_ray_length_does_not_matter(detector):
    D = panel_matrix(detector)
    ray = np.array([0.01, 0.02, -1.0])
    assert_allclose(project_ray(ray, D, PX, PX),
                    project_ray(7.3 * ray, D, PX, PX))


# --- abTEM frame -> lab frame ------------------------------------

def test_abtem_beam_axis_maps_to_direction_of_travel(beam):
    # dxtbx direction (0, 0, 1) = sample -> source, so the
    # electrons travel along -z.
    R = abtem_to_lab(beam)
    assert_allclose(R @ [0, 0, 1], [0, 0, -1], atol=1e-12)


def test_abtem_to_lab_handles_tilted_beam():
    d = np.array([0.004, -0.003, 1.0])
    tilted = _beam_from_dict({"wavelength": WAVELENGTH,
                              "direction": d.tolist()})
    R = abtem_to_lab(tilted)
    assert_allclose(R @ [0, 0, 1], -d / np.linalg.norm(d), atol=1e-12)


def test_abtem_to_lab_is_a_proper_rotation(beam):
    R = abtem_to_lab(beam)
    assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_ray_away_from_panel_is_rejected(detector):
    D = panel_matrix(detector)
    assert project_ray([0.01, 0.02, +1.0], D, PX, PX) is None


# --- direct beam ------------------------------------------------

def test_direct_beam_lands_on_beam_centre(beam, detector):
    px, py = project_position(0.0, 0.0, 0.0, beam, detector)
    assert_allclose([px, py], BEAM_CENTRE, atol=1e-6)


# --- beam direction convention ----------------------------------

def test_reflection_lands_where_dials_puts_it(beam, detector):
    # In DIALS, s0 = -direction / wavelength and a reflection with
    # lab-frame reciprocal vector g scatters along s1 = s0 + g.
    k0 = 1.0 / WAVELENGTH
    s0 = -k0 * np.array([0.0, 0.0, 1.0])

    # Pick a g that is exactly on the Ewald sphere: choose the
    # scattered direction first, then g = s1 - s0.
    s1_dir = np.array([0.012, 0.005, -1.0])
    s1 = k0 * s1_dir / np.linalg.norm(s1_dir)
    g = s1 - s0

    # Medley hands abTEM the crystal rotated by R.T, so abTEM
    # reports this reflection as R.T @ g.
    R = abtem_to_lab(beam)
    g_abtem = R.T @ g

    D = panel_matrix(detector)
    dials_px = project_ray(s1, D, PX, PX)
    medley_px = project_position(*g_abtem, beam, detector)

    assert_allclose(medley_px, dials_px, atol=1e-3)


def test_abtem_to_lab_for_beam_travelling_along_plus_z():
    b = _beam_from_dict({"wavelength": WAVELENGTH,
                         "direction": [0.0, 0.0, -1.0]})
    R = abtem_to_lab(b)
    assert_allclose(R @ [0, 0, 1], [0, 0, 1], atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)
