"""The crystal handed to abTEM must see the beam as the real experiment does.

These tests check the reflection table itself (which hkl are
excited, and where), without any detector projection.
"""
import numpy as np
from numpy.testing import assert_allclose
from ase import Atoms

from medley.engine import orient_atoms
from medley.experiment import Geometry, _beam_from_dict, axis_angle_rotation_matrix, get_B_matrix
from medley.projection import abtem_to_lab

WAVELENGTH = 0.0251
K0 = 1.0 / WAVELENGTH


def make_geometry():
    ort = np.array([[5.0, 0.0, 0.0],
                    [1.2, 6.0, 0.0],
                    [0.7, -0.9, 7.5]])
    B = get_B_matrix(ort)
    U = axis_angle_rotation_matrix([0.3, -0.5, 0.8], 41)
    return Geometry(B_mats=[B, B], U_mats=[U, U],
                    scan_point_angles=np.array([0.0, 1.0]),
                    F=np.eye(3), S=np.eye(3),
                    rotation_axis=np.array([0.0, 1.0, 0.0]))


def base_atoms(geometry):
    cell = np.linalg.inv(geometry.base_B())
    return Atoms("C", scaled_positions=[[0.1, 0.2, 0.3]],
                 cell=cell, pbc=True)


def excitation_abtem(g):
    # abTEM's condition: beam along +z, s_g = 0 when |k0 z + g| = k0
    return np.linalg.norm(np.array([0, 0, K0]) + g, axis=1) - K0


def excitation_real(g, beam):
    s0 = -K0 * np.asarray(beam.direction) / np.linalg.norm(beam.direction)
    return np.linalg.norm(s0 + g, axis=1) - K0


def check(direction):
    beam = _beam_from_dict({"wavelength": WAVELENGTH,
                            "direction": list(direction)})
    geom = make_geometry()
    phi = 0.6
    hkl = np.array([[h, k, l] for h in range(-3, 4)
                    for k in range(-3, 4) for l in range(-3, 4)])

    atoms = orient_atoms(base_atoms(geom), geom, phi, beam)
    g_abtem = hkl @ np.linalg.pinv(atoms.cell[:]).T      # what abTEM computes
    g_lab = hkl @ np.linalg.pinv(geom.oriented_cell(phi)).T

    # Every reflection is exactly as close to the Ewald sphere in
    # abTEM's calculation as in the real experiment ...
    assert_allclose(excitation_abtem(g_abtem),
                    excitation_real(g_lab, beam), atol=1e-9)
    # ... and rotating abTEM's vectors back gives the lab vectors.
    assert_allclose(g_abtem @ abtem_to_lab(beam).T, g_lab, atol=1e-12)


def test_beam_along_minus_z():
    check([0.0, 0.0, 1.0])


def test_tilted_beam():
    check([0.004, -0.003, 1.0])


def test_crystal_keeps_its_hand():
    beam = _beam_from_dict({"wavelength": WAVELENGTH,
                            "direction": [0.0, 0.0, 1.0]})
    geom = make_geometry()
    atoms = orient_atoms(base_atoms(geom), geom, 0.6, beam)
    assert np.linalg.det(atoms.cell[:]) > 0
