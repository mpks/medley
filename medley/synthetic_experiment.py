"""
synthetic_experiment.py — build a synthetic electron diffraction
experiment from a CIF file and a random crystal orientation.

Produces the same domain objects as experiment.load_experiment
(Detector, Beam, Geometry, scan angles) so the simulation engine
can consume either an experiment-derived or a synthetic setup
without knowing which it is.

For the synthetic case:
  - U is a uniformly random rotation seeded by orientation_seed
  - S and F are identity (no goniometer offsets)
  - the rotation axis is +y
  - B is built from the CIF unit cell
  - the scan angles are supplied by the caller

The sample-to-detector distance can either be given directly or
computed from a desired corner resolution via the calibration
module.  Calibration is kept as a separate, explicit step:
build_synthetic always takes a distance in mm, and the
build_synthetic_calibrated wrapper resolves the distance first
and then calls build_synthetic.
"""

from __future__ import annotations

import numpy as np
import ase.io
from scipy.spatial.transform import Rotation

from dynamic.simulation.experiment import (
    Detector,
    Beam,
    Geometry,
    get_B_matrix,
    _wavelength_to_energy_eV,
)
from dynamic.simulation.calibration import (
    distance_from_resolution,
)


# ----------------------------------------------------------------
# Builders for the individual pieces
# ----------------------------------------------------------------

def random_U(orientation_seed):
    """
    Uniform random rotation matrix over SO(3), seeded for
    reproducibility.
    """
    rot = Rotation.random(random_state=orientation_seed)
    return rot.as_matrix()


def B_from_cif(cif_file):
    """
    Build the reciprocal metric B from the unit cell of a CIF.

    Reads the cell with ASE, forms the real-space cell (rows
    a, b, c) and passes it to get_B_matrix.
    """
    atoms = ase.io.read(cif_file)
    ort = np.array(atoms.get_cell())   # rows = a, b, c
    return get_B_matrix(ort)


def make_beam(wavelength_A):
    """Build a Beam propagating along +z from a wavelength."""
    energy_eV = _wavelength_to_energy_eV(wavelength_A)
    return Beam(
        wavelength_A=wavelength_A,
        energy_eV=energy_eV,
        direction=np.array([0.0, 0.0, 1.0]),
    )


def make_detector(distance_mm, npx, npy,
                  px_x_mm, px_y_mm,
                  beam_centre_px=None):
    """
    Build a Detector with axes aligned to the lab frame.

    Pixel size is given separately for x and y to allow
    non-square pixels.  If beam_centre_px is None it defaults
    to the detector centre (npx/2, npy/2).
    """
    if beam_centre_px is None:
        beam_centre_px = (npx / 2.0, npy / 2.0)
    if abs(px_x_mm - px_y_mm) > 1e-12:
        print(
            "WARNING: non-square pixels "
            f"({px_x_mm} != {px_y_mm}); the Detector stores a "
            "single pixel_size_mm (x). Downstream projection "
            "assumes square pixels."
        )
    fast = np.array([1.0, 0.0, 0.0])
    slow = np.array([0.0, 1.0, 0.0])
    # Origin places pixel (0,0) so the beam (+z from the
    # sample) pierces the panel at beam_centre_px, distance_mm
    # along +z:
    #   origin + cx*px_x*fast + cy*px_y*slow = (0, 0, distance)
    cx, cy = beam_centre_px
    origin = (
        np.array([0.0, 0.0, distance_mm])
        - cx * px_x_mm * fast
        - cy * px_y_mm * slow
    )
    return Detector(
        distance_mm=distance_mm,
        npx=npx,
        npy=npy,
        pixel_size_mm=px_x_mm,
        beam_centre_px=beam_centre_px,
        fast_axis=fast,
        slow_axis=slow,
        origin=origin,
    )


def make_scan_angles(start_deg, delta_deg, n_images):
    """
    Per-image start angles for a scan, following the DIALS
    convention of omitting the final scan point.

    For start=-5, delta=0.5, n_images=3 this returns
    [-5.0, -4.5, -4.0].
    """
    return start_deg + delta_deg * np.arange(n_images)


# ----------------------------------------------------------------
# Top-level builder (distance supplied directly)
# ----------------------------------------------------------------

def build_synthetic(cif_file, wavelength_A, distance_mm,
                    npx, npy, px_x_mm, px_y_mm,
                    start_deg, delta_deg, n_images,
                    orientation_seed,
                    beam_centre_px=None):
    """
    Build a complete synthetic experiment from an explicit
    sample-to-detector distance.

    Returns
    -------
    (detector, beam, geometry, scan_angles_deg)
        Matching the signature of experiment.load_experiment.
    """
    beam = make_beam(wavelength_A)
    detector = make_detector(distance_mm, npx, npy,
                             px_x_mm, px_y_mm,
                             beam_centre_px)

    B = B_from_cif(cif_file)
    U = random_U(orientation_seed)
    identity = np.eye(3)

    # Static synthetic model: N images -> N+1 scan points, the
    # same U and B copied to each (arrays for a uniform code
    # path with the scan-varying experiment loader).
    n_scan_points = n_images + 1
    scan_point_angles = start_deg + delta_deg * np.arange(
        n_scan_points
    )
    U_mats = [U.copy() for _ in range(n_scan_points)]
    B_mats = [B.copy() for _ in range(n_scan_points)]

    geometry = Geometry(
        B_mats=B_mats,
        U_mats=U_mats,
        scan_point_angles=scan_point_angles,
        F=identity,
        S=identity,
        rotation_axis=np.array([0.0, 1.0, 0.0]),
        orientation_seed=orientation_seed,
    )

    scan_angles = make_scan_angles(start_deg, delta_deg,
                                   n_images)
    return detector, beam, geometry, scan_angles


# ----------------------------------------------------------------
# Convenience wrapper (distance from corner resolution)
# ----------------------------------------------------------------

def build_synthetic_calibrated(cif_file, wavelength_A,
                               npx, npy, px_x_mm, px_y_mm,
                               start_deg, delta_deg, n_images,
                               orientation_seed,
                               g_max=None, d_min=None,
                               distance_mm=None,
                               beam_centre_px=None,
                               report=True):
    """
    Build a synthetic experiment, resolving the detector
    distance from a desired corner resolution.

    Calibration is an explicit, separate step: this wrapper
    calls calibration.distance_from_resolution to obtain the
    distance (from exactly one of g_max, d_min or
    distance_mm), then delegates to build_synthetic.

    Returns
    -------
    (detector, beam, geometry, scan_angles_deg)
    """
    distance = distance_from_resolution(
        wavelength_A=wavelength_A,
        npx=npx, npy=npy,
        px_x_mm=px_x_mm, px_y_mm=px_y_mm,
        g_max=g_max, d_min=d_min,
        distance_mm=distance_mm,
        report=report,
    )
    return build_synthetic(
        cif_file=cif_file,
        wavelength_A=wavelength_A,
        distance_mm=distance,
        npx=npx, npy=npy,
        px_x_mm=px_x_mm, px_y_mm=px_y_mm,
        start_deg=start_deg, delta_deg=delta_deg,
        n_images=n_images,
        orientation_seed=orientation_seed,
        beam_centre_px=beam_centre_px,
    )
