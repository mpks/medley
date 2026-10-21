"""
projection.py — project reciprocal-space diffraction spots onto
the detector panel, in pixel coordinates.

The abTEM simulation is taken as physical truth and is left
untouched: the beam propagates along +z in abTEM's frame and
each reflection comes out as a reciprocal-space vector
g = (kx, ky, kz) with no detector baked in.  sg_max controls
how much of the Ewald sphere is sampled; that is abTEM's
concern, not the projection's.

abTEM always propagates its beam along +z.  In the lab the
electrons travel along p = -beam.direction (dxtbx 'direction'
points from the sample towards the source).  The crystal is
therefore handed to abTEM rotated by R.T, where R = abtem_to_lab
is the proper rotation with R @ z = p (see engine.orient_atoms),
and abTEM's results are rotated back with the same R:

    s1 = R @ ( (0, 0, k0) + (kx, ky, kz) )

with k0 = 1 / wavelength and (kx, ky, kz) in abTEM's frame.

s1 is then projected onto the real, possibly tilted, detector
panel using the panel basis (fast axis, slow axis, origin) via
the dxtbx D-matrix projection:

    d = [ fast | slow | origin ]   (columns)
    D = d^{-1}

D maps the lab ray s1 to panel coordinates; the perspective
divide by the third component gives millimetres along fast and
slow, divided by the pixel sizes to get pixels.
"""

from __future__ import annotations

import numpy as np


def panel_matrix(detector):
    """
    Return D = inv([fast | slow | origin]) for the detector
    panel, the matrix that projects a lab-frame ray onto the
    panel.
    """
    d = np.column_stack([
        np.asarray(detector.fast_axis, dtype=float),
        np.asarray(detector.slow_axis, dtype=float),
        np.asarray(detector.origin, dtype=float),
    ])
    return np.linalg.inv(d)


def beam_s0(beam):
    """
    Incident wavevector s0 in the lab frame, as in DIALS:
    s0 = -direction / wavelength, since dxtbx 'direction' points
    from the sample towards the source.
    """
    d = np.asarray(beam.direction, dtype=float)
    return -d / (np.linalg.norm(d) * beam.wavelength_A)


def _rotate_onto(a, b):
    """
    Rotation matrix that turns unit vector a onto unit vector b
    (Rodrigues formula).  Used only for small angles here, where
    it is well conditioned.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    c = float(np.dot(a, b))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        raise ValueError("_rotate_onto: vectors are antiparallel.")
    axis = axis / s
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def abtem_to_lab(beam):
    """
    Rotation R from abTEM's frame to the laboratory frame.

    abTEM always propagates its beam along +z.  In the lab, the
    electrons travel along p = -beam.direction, because dxtbx
    stores 'direction' pointing from the sample towards the
    source.  R is the proper rotation with R @ (0, 0, 1) = p:

        lab vector   = R   @ abTEM vector
        abTEM vector = R.T @ lab vector

    The crystal is rotated into abTEM's frame with R.T before
    the simulation, and abTEM's results are rotated back with R,
    so abTEM's +z beam meets the crystal exactly as the real
    beam does, including any small tilt.

    R is built as a 180 deg turn about x (+z -> -z) followed by
    the small rotation -z -> p.  Any rotation about the beam axis
    could be added without changing the physics; this choice is
    simply well conditioned for p close to -z.
    """
    p = -np.asarray(beam.direction, dtype=float)
    p = p / np.linalg.norm(p)
    flip = np.diag([1.0, -1.0, -1.0])
    tilt = _rotate_onto([0.0, 0.0, -1.0], p)
    return tilt @ flip


def project_ray(s1, D, px_fast_mm, px_slow_mm):
    """
    Project a scattered ray s1 onto the panel using the panel
    projection matrix D.

    Returns (px, py) in pixels, or None if the ray does not reach
    the panel.  With d = [fast | slow | origin], s1 = v[0] fast +
    v[1] slow + v[2] origin, so the ray meets the panel plane at
    s1 / v[2].  That point lies in front of the sample only if
    v[2] > 0; a ray with v[2] <= 0 travels away from the panel
    (or parallel to it) and never hits it.
    """
    v = D @ np.asarray(s1, dtype=float)
    if v[2] <= 0:
        return None
    x_mm = v[0] / v[2]
    y_mm = v[1] / v[2]
    return x_mm / px_fast_mm, y_mm / px_slow_mm


def project_position(kx, ky, kz, beam, detector,
                     D=None, R=None, k0=None):
    """
    Project a single abTEM reciprocal-space vector onto the
    detector.

    The abTEM scattered ray in the +z frame is
    s1_abtem = (0, 0, k0) + (kx, ky, kz).  The whole cone is
    rotated back to the lab by R = abtem_to_lab(beam) (which maps
    +z onto the direction the electrons travel) before
    projection:  s1 = R @ s1_abtem.

    Parameters
    ----------
    kx, ky, kz : float
        Scattering vector in abTEM's frame (A^-1).
    beam : Beam
        Supplies the wavelength and the experimental beam
        direction (used to build R and k0).
    detector : Detector
        Supplies the panel basis (fast/slow/origin) and pixel
        size.
    D : ndarray, optional
        Precomputed panel matrix; pass in a loop.
    R : ndarray, optional
        Precomputed abtem_to_lab(beam); pass in a loop.
    k0 : float, optional
        Precomputed 1 / wavelength; pass in a loop.

    Returns
    -------
    (px, py) : tuple of float, or None if the ray does not
        reach the panel.
    """
    if D is None:
        D = panel_matrix(detector)
    if R is None:
        R = abtem_to_lab(beam)
    if k0 is None:
        k0 = 1.0 / beam.wavelength_A
    s1_abtem = np.array([kx, ky, k0 + kz])
    s1 = R @ s1_abtem
    px_mm = detector.pixel_size_mm
    return project_ray(s1, D, px_mm, px_mm)


def project_spots(spots, detector, beam):
    """
    Project all spots onto the detector.

    The abTEM scattered cone is rotated back to the lab by
    R = abtem_to_lab(beam) and then projected onto the real
    detector panel (fast/slow/origin).  The positions
    (kx, ky, kz) are in abTEM's frame.

    Parameters
    ----------
    spots : Spots
        Reciprocal-space positions (N, 3), millers, intensities.
    detector : Detector
    beam : Beam

    Returns
    -------
    list of dict, one per spot that projects onto the panel
    (rays that do not reach the panel are omitted), each
    with keys:
        'hkl'       : tuple (h, k, l)
        'px', 'py'  : float pixel coordinates
        'intensity' : float

    Spots are NOT filtered by detector bounds here; that is the
    caller's choice.
    """
    D = panel_matrix(detector)
    R = abtem_to_lab(beam)
    k0 = 1.0 / beam.wavelength_A
    px_mm = detector.pixel_size_mm

    out = []
    positions = spots.positions
    millers = spots.millers
    intensities = spots.intensities

    for i in range(len(intensities)):
        kx = positions[i][0]
        ky = positions[i][1]
        kz = positions[i][2]
        s1_abtem = np.array([kx, ky, k0 + kz])
        s1 = R @ s1_abtem
        proj = project_ray(s1, D, px_mm, px_mm)
        if proj is None:
            continue
        px, py = proj
        hkl = (
            int(millers[i][0]),
            int(millers[i][1]),
            int(millers[i][2]),
        )
        out.append({
            "hkl": hkl,
            "px": px,
            "py": py,
            "intensity": float(intensities[i]),
        })
    return out


def in_detector_bounds(px, py, npx, npy):
    """True if (px, py) lies within the detector pixel grid."""
    return (0 <= px < npx) and (0 <= py < npy)
