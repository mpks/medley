import numpy as np
from numpy.testing import assert_allclose

from medley.experiment import (
    axis_angle_rotation_matrix,
    get_B_matrix,
    goniometer_matrices,
    polar_rotation,
)


def test_rotation_90_deg_about_z_sends_x_to_y():
    R = axis_angle_rotation_matrix([0, 0, 1], 90)
    assert_allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_rotation_axis_need_not_be_normalised():
    R1 = axis_angle_rotation_matrix([0, 0, 1], 30)
    R2 = axis_angle_rotation_matrix([0, 0, 5], 30)
    assert_allclose(R1, R2)


def test_rotation_is_orthogonal_with_det_one():
    R = axis_angle_rotation_matrix([1, 2, 3], 37)
    assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)


# --- get_B_matrix -----------------------------------------------

def test_B_for_cubic_cell_is_diagonal_one_over_a():
    ort = 10.0 * np.eye(3)            # rows are a, b, c
    assert_allclose(get_B_matrix(ort), 0.1 * np.eye(3), atol=1e-12)


def test_B_reproduces_reciprocal_metric_of_triclinic_cell():
    # B^T B must equal the reciprocal metric tensor inv(G),
    # where G = ort @ ort.T is the real-space metric.  This
    # holds in any Cartesian frame, so it checks B without
    # depending on which way the axes point.
    ort = np.array([
        [5.0, 0.0, 0.0],
        [1.2, 6.0, 0.0],
        [0.7, -0.9, 7.5],
    ])
    B = get_B_matrix(ort)
    G = ort @ ort.T
    assert_allclose(B.T @ B, np.linalg.inv(G), rtol=1e-10)


def test_inverse_B_rows_have_the_cell_lengths():
    ort = np.array([
        [5.0, 0.0, 0.0],
        [1.2, 6.0, 0.0],
        [0.7, -0.9, 7.5],
    ])
    cell_rows = np.linalg.inv(get_B_matrix(ort))
    assert_allclose(np.linalg.norm(cell_rows, axis=1),
                    np.linalg.norm(ort, axis=1), rtol=1e-10)


# --- polar_rotation ---------------------------------------------

def test_polar_rotation_of_a_rotation_is_itself():
    R = axis_angle_rotation_matrix([1, 1, 0], 25)
    assert_allclose(polar_rotation(R), R, atol=1e-12)


def test_polar_rotation_returns_proper_rotation():
    rng = np.random.default_rng(0)
    M = np.eye(3) + 0.1 * rng.normal(size=(3, 3))
    Q = polar_rotation(M)
    assert_allclose(Q @ Q.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(Q), 1.0)


# --- goniometer_matrices ----------------------------------------

def test_single_axis_goniometer_is_read_as_given():
    gonio = {
        "setting_rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        "fixed_rotation":   [1, 0, 0, 0, 1, 0, 0, 0, 1],
        "rotation_axis":    [1.0, 0.0, 0.0],
    }
    S, F, axis = goniometer_matrices(gonio)
    assert_allclose(S, np.eye(3))
    assert_allclose(F, np.eye(3))
    assert_allclose(axis, [1, 0, 0])


def test_multi_axis_goniometer_at_zero_angles_is_identity():
    gonio = {
        "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "angles": [0.0, 0.0, 0.0],
        "scan_axis": 1,
    }
    S, F, axis = goniometer_matrices(gonio)
    assert_allclose(S, np.eye(3), atol=1e-12)
    assert_allclose(F, np.eye(3), atol=1e-12)
    assert_allclose(axis, [0, 1, 0])
