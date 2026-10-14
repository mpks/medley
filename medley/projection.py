"""
projection.py — project reciprocal-space diffraction spots onto
the detector panel, in pixel coordinates.

The abTEM simulation is taken as physical truth and is left
untouched: the beam propagates along +z in abTEM's frame and
each reflection comes out as a reciprocal-space vector
g = (kx, ky, kz) with no detector baked in.  sg_max controls
how much of the Ewald sphere is sampled; that is abTEM's
concern, not the projection's.

The projection is the only place the real detector geometry
enters.  abTEM produces the whole scattered cone with the beam
along +z; the real beam is tilted slightly.  We rotate the
entire cone by R_tilt, the rotation that maps +z onto the
experimental beam direction, so the direct beam and every
diffracted vector align with the real beam:

    s1 = R_tilt @ ( (0, 0, k0) + (kx, ky, kz) )

with k0 = 1 / wavelength.  g = (kx, ky, kz) comes from abTEM
unchanged.

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
    Incident wavevector s0 = k0 * (unit beam propagation
    direction), taken from the experiment beam.  k0 = 1 /
    wavelength.  The direction is normalised so only its
    orientation (the slight experimental tilt away from +z)
    matters, not its stored magnitude.
    """
    k0 = 1.0 / beam.wavelength_A
    d = np.asarray(beam.direction, dtype=float)
    d = d / np.linalg.norm(d)
    return k0 * d


def tilt_rotation(beam):
    """
    Rotation R_tilt that maps the abTEM optical axis +z onto the
    experimental incident beam direction.

    abTEM computes the whole scattered cone with the beam along
    +z.  The real beam is tilted slightly.  R_tilt rotates the
    entire cone (every scattered wavevector, including the
    direct beam) so the direct beam aligns with the experimental
    direction, then the cone is projected onto the detector.

    The rotation is about the axis z x d through the angle
    between z and d; for d == +z it is the identity.
    """
    z = np.array([0.0, 0.0, 1.0])
    d = np.asarray(beam.direction, dtype=float)
    d = d / np.linalg.norm(d)

    axis = np.cross(z, d)
    s = np.linalg.norm(axis)
    c = float(np.dot(z, d))
    if s < 1e-12:
        # Parallel (or antiparallel) to +z.
        if c > 0:
            return np.eye(3)
        # 180 deg: rotate about x.
        return np.diag([1.0, -1.0, -1.0])
    axis = axis / s
    angle = np.arctan2(s, c)
    # Rodrigues rotation matrix.
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return (np.eye(3) + np.sin(angle) * K
            + (1.0 - np.cos(angle)) * (K @ K))


def project_ray(s1, D, px_fast_mm, px_slow_mm):
    """
    Project a scattered ray s1 onto the panel using the panel
    projection matrix D.

    Returns (px, py) in pixels, or None only if the ray is
    parallel to the panel (no intersection).  The sign of the
    perspective divisor depends on the panel's origin/normal
    convention, so it is not used to reject spots.
    """
    v = D @ np.asarray(s1, dtype=float)
    if v[2] == 0:
        return None
    x_mm = v[0] / v[2]
    y_mm = v[1] / v[2]
    return x_mm / px_fast_mm, y_mm / px_slow_mm


def project_position(kx, ky, kz, beam, detector,
                     D=None, R_tilt=None, k0=None):
    """
    Project a single abTEM reciprocal-space vector onto the
    detector.

    The abTEM scattered ray in the +z frame is
    s1_abtem = (0, 0, k0) + (kx, ky, kz).  The whole cone is
    rotated by R_tilt (which maps +z onto the experimental beam
    direction) before projection, so the direct beam aligns with
    the experimental beam and every diffracted vector rotates
    rigidly with it:  s1 = R_tilt @ s1_abtem.

    Parameters
    ----------
    kx, ky, kz : float
        abTEM reciprocal-space components (A^-1), passed through
        unchanged.
    beam : Beam
        Supplies the wavelength and the experimental beam
        direction (used to build R_tilt and k0).
    detector : Detector
        Supplies the panel basis (fast/slow/origin) and pixel
        size.
    D : ndarray, optional
        Precomputed panel matrix; pass in a loop.
    R_tilt : ndarray, optional
        Precomputed tilt_rotation(beam); pass in a loop.
    k0 : float, optional
        Precomputed 1 / wavelength; pass in a loop.

    Returns
    -------
    (px, py) : tuple of float, or None only if the ray is
        parallel to the panel.
    """
    if D is None:
        D = panel_matrix(detector)
    if R_tilt is None:
        R_tilt = tilt_rotation(beam)
    if k0 is None:
        k0 = 1.0 / beam.wavelength_A
    s1_abtem = np.array([kx, ky, k0 + kz])
    s1 = R_tilt @ s1_abtem
    px_mm = detector.pixel_size_mm
    return project_ray(s1, D, px_mm, px_mm)


def project_spots(spots, detector, beam):
    """
    Project all spots onto the detector.

    The whole abTEM scattered cone is rotated by R_tilt (mapping
    +z onto the experimental beam direction) and then projected
    onto the real detector panel (fast/slow/origin).  The abTEM
    positions (kx, ky, kz) are used as given.

    Parameters
    ----------
    spots : Spots
        Reciprocal-space positions (N, 3), millers, intensities.
    detector : Detector
    beam : Beam

    Returns
    -------
    list of dict, one per spot that projects onto the panel
    (only rays exactly parallel to the panel are omitted), each
    with keys:
        'hkl'       : tuple (h, k, l)
        'px', 'py'  : float pixel coordinates
        'intensity' : float

    Spots are NOT filtered by detector bounds here; that is the
    caller's choice.
    """
    D = panel_matrix(detector)
    R_tilt = tilt_rotation(beam)
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
        s1 = R_tilt @ s1_abtem
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
