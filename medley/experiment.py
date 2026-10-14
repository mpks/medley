"""
experiment.py — domain model for an electron diffraction
experiment and loading of geometry from a DIALS .expt file.

Defines the immutable data classes that describe what is being
measured (Detector, Beam, Geometry, Scan) and the Spots produced
by a simulation.  Also provides the cell-orientation geometry
(Geometry.oriented_cell) shared by every simulator backend, and
a reader that populates these objects from a DIALS experiment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation


# ----------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------

@dataclass(frozen=True)
class Detector:
    """Flat-panel detector geometry."""
    distance_mm: float
    npx: int
    npy: int
    pixel_size_mm: float
    beam_centre_px: tuple          # (cx, cy)
    fast_axis: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0])
    )
    slow_axis: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.0, 0.0])
    )
    origin: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )


@dataclass(frozen=True)
class Beam:
    """Incident electron beam, propagating along +z."""
    wavelength_A: float
    energy_eV: float
    direction: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0])
    )


@dataclass(frozen=True)
class Geometry:
    """
    Crystal orientation and metric, sampled at scan points.

    A rotation scan of N images has N+1 scan points (the image
    boundaries).  The crystal setting (orientation U and metric
    B) is stored at each scan point so a scan-varying model can
    be represented; for a static model every entry is identical.

    B_mats : list of 3x3   reciprocal metric at each scan point
    U_mats : list of 3x3   orientation (pure rotation) at each
        scan point
    scan_point_angles : ndarray (N+1,)
        Absolute angle (deg) of each scan point; used to locate
        and interpolate within an interval.
    F : 3x3  goniometer fixed rotation
    S : 3x3  goniometer setting rotation
    rotation_axis : 3-vector  scan rotation axis
    orientation_seed : int or None
        Seed used to generate U (synthetic only); None for
        experiment-derived geometry.
    """
    B_mats: list
    U_mats: list
    scan_point_angles: np.ndarray
    F: np.ndarray
    S: np.ndarray
    rotation_axis: np.ndarray
    orientation_seed: int | None = None

    def base_B(self) -> np.ndarray:
        """The metric used to set up the base atoms (the first
        scan point; all entries are equal in the static case)."""
        return self.B_mats[0]

    def _interpolate_setting(self, angle_deg):
        """
        Interpolate the orientation U and metric B at an
        absolute scan angle.

        U is interpolated by spherical linear interpolation
        (SLERP) of the bracketing scan-point rotations; B is
        linearly interpolated.  Angles outside the scan-point
        range clamp to the nearest interval.

        Returns (U, B) at the angle.
        """
        angles = self.scan_point_angles
        n = len(angles)
        if n == 1:
            return self.U_mats[0], self.B_mats[0]

        # Locate the interval [i, i+1] containing angle_deg.
        i = int(np.searchsorted(angles, angle_deg) - 1)
        i = max(0, min(i, n - 2))
        a0 = angles[i]
        a1 = angles[i + 1]
        span = a1 - a0
        t = 0.0 if span == 0 else (angle_deg - a0) / span
        t = max(0.0, min(1.0, t))

        U = _slerp_rotation(self.U_mats[i],
                            self.U_mats[i + 1], t)
        B = (1.0 - t) * self.B_mats[i] + t * self.B_mats[i + 1]
        return U, B

    def oriented_cell(self, angle_deg: float) -> np.ndarray:
        """
        Return the real-space cell (ASE row convention) at the
        given scan angle.

          step 1: crystal-frame cell = B(angle)^{-1}  (rows)
          step 2: rotate into the lab frame by
                  S @ R(angle) @ F @ U(angle)

        U and B are interpolated from the scan-point arrays, so
        a scan-varying crystal model is honoured; the explicit
        scan rotation R(angle) is still applied on top (it is
        not contained in U).

        Returns
        -------
        cell_rows : ndarray (3, 3)
            Rows are the real-space vectors a, b, c.
        """
        U, B = self._interpolate_setting(angle_deg)
        crystal_cell = np.linalg.inv(B)        # rows = a, b, c
        R = axis_angle_rotation_matrix(self.rotation_axis,
                                       angle_deg)
        full = self.S @ R @ self.F @ U
        cell_rows = (full @ crystal_cell.T).T
        return cell_rows


@dataclass(frozen=True)
class Scan:
    """
    Rotation scan description.

    angles_deg : ndarray
        Per-image start angles (the final scan point is implied,
        following the DIALS convention of omitting it).
    n_substeps : int
        Number of sub-angles integrated within each image.
    """
    angles_deg: np.ndarray
    n_substeps: int

    @property
    def delta_deg(self) -> float:
        """Oscillation width per image."""
        if len(self.angles_deg) < 2:
            raise ValueError(
                "Scan needs at least 2 angles to define a "
                "delta; got "
                f"{len(self.angles_deg)}."
            )
        return float(self.angles_deg[1]
                     - self.angles_deg[0])

    @property
    def n_images(self) -> int:
        return len(self.angles_deg)

    def substep_angles(self, image_index: int) -> np.ndarray:
        """
        Sub-angles for one image, tiling [start, start+delta)
        with endpoint excluded so that concatenating adjacent
        images produces a uniform non-overlapping series.
        """
        start = self.angles_deg[image_index]
        return np.linspace(
            start,
            start + self.delta_deg,
            self.n_substeps,
            endpoint=False,
        )


@dataclass(frozen=True)
class Spots:
    """
    Diffraction spots from one orientation.

    positions : ndarray (N, 3)  kx, ky, kz in Å^-1
    millers   : list of [h, k, l]
    intensities : ndarray (N,)
    """
    positions: np.ndarray
    millers: list
    intensities: np.ndarray


# ----------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------

def axis_angle_rotation_matrix(axis, phi_deg):
    """
    Rotation matrix for a rotation of phi_deg degrees about
    the given axis (need not be normalized).
    """
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    rotvec = axis * np.deg2rad(phi_deg)
    return Rotation.from_rotvec(rotvec).as_matrix()


def polar_rotation(M):
    """
    Nearest pure rotation to a 3x3 matrix, via the polar
    decomposition M = Q P (Q orthogonal, P symmetric positive
    semi-definite).  Returns Q with det +1.

    Used to extract a clean orientation U from A @ inv(B), which
    may carry a small non-rotational part if A was refined with
    a slightly different metric than the cell-derived B.
    """
    from scipy.linalg import polar
    Q, _P = polar(np.asarray(M, dtype=float))
    if np.linalg.det(Q) < 0:
        # Reflect the least-significant axis to enforce det +1.
        u, _s, vt = np.linalg.svd(Q)
        d = np.ones(3)
        d[-1] = -1.0
        Q = u @ np.diag(d) @ vt
    return Q


def _slerp_rotation(U1, U2, t):
    """
    Spherical linear interpolation between two rotation
    matrices, returning the rotation a fraction t (0..1) of the
    way from U1 to U2.

    Implemented as U(t) = dR(t) @ U1, where dR(t) is the
    relative rotation U2 @ U1^T scaled to fraction t via its
    axis-angle form (matching the scan-point interpolation used
    by DIALS).
    """
    if t <= 0.0:
        return np.asarray(U1, dtype=float)
    if t >= 1.0:
        return np.asarray(U2, dtype=float)
    U1 = np.asarray(U1, dtype=float)
    U2 = np.asarray(U2, dtype=float)
    M = U2 @ U1.T
    rotvec = Rotation.from_matrix(M).as_rotvec()
    dR = Rotation.from_rotvec(rotvec * t).as_matrix()
    return dR @ U1


def get_B_matrix(ort):
    """
    Build the reciprocal metric B from a real-space cell.

    ort : 3x3, rows are real-space cell vectors a, b, c (Å).
    Returns B : 3x3, columns are a*, b*, c* (Å^-1), in the
    Busing-Levy convention (a* along x, b* in the xy plane).
    """
    a_vec, b_vec, c_vec = ort[0], ort[1], ort[2]
    a = np.linalg.norm(a_vec)
    b = np.linalg.norm(b_vec)
    c = np.linalg.norm(c_vec)
    alpha = np.arccos(b_vec.dot(c_vec) / (b * c))
    beta = np.arccos(a_vec.dot(c_vec) / (a * c))
    gamma = np.arccos(a_vec.dot(b_vec) / (a * b))

    cos_a, cos_b, cos_g = (
        np.cos(alpha), np.cos(beta), np.cos(gamma)
    )
    sin_g = np.sin(gamma)
    V = a * b * c * np.sqrt(
        1 - cos_a**2 - cos_b**2 - cos_g**2
        + 2 * cos_a * cos_b * cos_g
    )
    B = np.array([
        [1 / a, 0, 0],
        [-cos_g / (a * sin_g), 1 / (b * sin_g), 0],
        [
            b * c * (
                cos_g * (cos_a - cos_b * cos_g) / sin_g
                - cos_b * sin_g
            ) / V,
            -(a * c * (cos_a - cos_b * cos_g)) / (V * sin_g),
            (a * b * sin_g) / V,
        ],
    ])
    return B


# ----------------------------------------------------------------
# DIALS experiment reader
# ----------------------------------------------------------------

def load_experiment(expt_file):
    """
    Read a DIALS .expt JSON file and return
    (detector, beam, geometry, scan_angles_deg).

    Assumes fixed U, B, F, S throughout the scan (no
    scan-varying crystal model).
    """
    with open(expt_file) as fh:
        data = json.load(fh)

    experiments = data["experiment"]
    if len(experiments) == 0:
        raise ValueError("No experiments found in expt file.")
    if len(experiments) > 1:
        print(
            f"WARNING: {len(experiments)} experiments found, "
            "using the first one."
        )

    exp = experiments[0]
    crystal = data["crystal"][exp["crystal"]]
    gonio = data["goniometer"][exp["goniometer"]]
    scan = data["scan"][exp["scan"]]
    beam_d = data["beam"][exp["beam"]]
    detector_d = data["detector"][exp["detector"]]

    # --- geometry (crystal orientation) -------------------
    # S = np.array(gonio["setting_rotation"]).reshape(3, 3)
    # F = np.array(gonio["fixed_rotation"]).reshape(3, 3)
    # rotation_axis = np.array(gonio["rotation_axis"])
    S, F, rotation_axis = goniometer_matrices(gonio)

    angles = np.array(scan["properties"]["oscillation"])

    ar = crystal["real_space_a"]
    br = crystal["real_space_b"]
    cr = crystal["real_space_c"]
    ort = np.array([ar, br, cr])

    # B always comes from the cell vectors (the cell-derived
    # metric).  This is reused for every scan point.
    B = get_B_matrix(ort)

    # Scan points sit at the image boundaries: N images give
    # N+1 scan points.  Their absolute angles run from the
    # first oscillation start in steps of delta.
    n_images = len(angles)
    if n_images >= 2:
        delta = float(angles[1] - angles[0])
    else:
        delta = float(scan["properties"].get(
            "oscillation_width", [1.0])[0]
        ) if isinstance(
            scan["properties"].get("oscillation_width"), list
        ) else 1.0
    start = float(angles[0]) if n_images else 0.0
    n_scan_points = n_images + 1
    scan_point_angles = start + delta * np.arange(
        n_scan_points
    )

    U_mats, B_mats = _build_setting_arrays(
        crystal, ort, B, n_scan_points
    )

    geometry = Geometry(
        B_mats=B_mats,
        U_mats=U_mats,
        scan_point_angles=scan_point_angles,
        F=F, S=S,
        rotation_axis=rotation_axis,
        orientation_seed=None,
    )

    beam = _beam_from_dict(beam_d)
    detector = _detector_from_dict(detector_d, beam.direction)

    return detector, beam, geometry, angles


def _build_setting_arrays(crystal, ort, B, n_scan_points):
    """
    Build per-scan-point orientation (U) and metric (B) arrays.

    For every scan point we have a setting matrix A (from
    'A_at_scan_points' when present, otherwise the single static
    A = inv(ort) copied to each point).  For each A:

      U = polar_rotation(A @ inv(B_cell))   nearest pure
          rotation to the setting matrix divided by the
          cell-derived metric (as before);
      B = U^{-1} @ A                          the metric that
          makes A = U B exact.

    Because U is a pure rotation, |B h| = |A h| for every
    reflection, so the simulated reciprocal-lattice magnitudes
    match the experiment's setting matrix exactly.  The
    cell-derived B_cell is used only as the reference for the
    polar decomposition; the stored B is recomputed from A and
    U and is interpolated per scan point.
    """
    B_cell_inv = np.linalg.inv(B)
    A_sp = crystal.get("A_at_scan_points", None)

    if A_sp is not None and len(A_sp) > 0:
        A_list = [
            np.array(a, dtype=float).reshape(3, 3)
            for a in A_sp
        ]
        if len(A_list) != n_scan_points:
            print(
                f"WARNING: A_at_scan_points has {len(A_list)} "
                f"entries, expected {n_scan_points}; using "
                "the values as given."
            )
    else:
        # Static model: a single setting matrix A = inv(ort),
        # copied to every scan point.
        A_static = np.linalg.inv(ort)
        A_list = [
            A_static.copy() for _ in range(n_scan_points)
        ]

    U_mats = []
    B_mats = []
    for A in A_list:
        U = polar_rotation(A @ B_cell_inv)
        U_mats.append(U)
        # B = U^{-1} A = U^T A (U is a rotation); A = U B exact,
        # and |B h| = |A h|.
        B_mats.append(U.T @ A)

    return U_mats, B_mats


def _beam_from_dict(beam_d):
    """Build a Beam from the DIALS beam dictionary."""
    wavelength_A = float(beam_d["wavelength"])
    # DIALS stores the beam direction in "direction" (and the
    # equivalent "sample_to_source_direction"). The beam centre
    # projection is invariant under its sign, so we store it as
    # given without flipping.
    direction = beam_d.get("direction", None)
    if direction is None:
        direction = beam_d.get(
            "sample_to_source_direction", [0.0, 0.0, 1.0]
        )
    direction = np.array(direction, dtype=float)
    energy_eV = _wavelength_to_energy_eV(wavelength_A)
    return Beam(
        wavelength_A=wavelength_A,
        energy_eV=energy_eV,
        direction=direction,
    )


def _detector_from_dict(detector_d, beam_direction):
    """
    Build a Detector from the DIALS detector dictionary.

    DIALS detectors are a list of panels; we use panel 0.
    The panel gives fast_axis, slow_axis, origin (mm),
    pixel_size (mm), image_size (px).  The sample-to-detector
    distance is the projection of origin onto the panel
    normal, and the beam centre is where the incident beam
    (beam_direction) pierces the panel, matching the DIALS
    Panel.get_beam_centre_px(s0) calculation.
    """
    panels = detector_d["panels"]
    panel = panels[0]

    fast = np.array(panel["fast_axis"], dtype=float)
    slow = np.array(panel["slow_axis"], dtype=float)
    origin = np.array(panel["origin"], dtype=float)
    pixel_size = panel["pixel_size"]          # (fast, slow) mm
    image_size = panel["image_size"]          # (npx, npy)

    npx = int(image_size[0])
    npy = int(image_size[1])
    px_fast = float(pixel_size[0])
    px_slow = float(pixel_size[1])

    # The incident beam direction (its sign does not affect the
    # projection onto the panel).
    beam_dir = np.asarray(beam_direction, dtype=float)

    # Distance: perpendicular distance from the sample (lab
    # origin) to the panel plane, = |origin . normal|.
    normal = np.cross(fast, slow)
    normal = normal / np.linalg.norm(normal)
    distance_mm = abs(float(origin.dot(normal)))

    # Beam centre: where the incident beam pierces the panel,
    # via the dxtbx D-matrix projection (matches DIALS).
    cx_px, cy_px = _ray_intersection_px(beam_dir, fast, slow,
                                        origin, px_fast,
                                        px_slow)

    if abs(px_fast - px_slow) > 1e-9:
        print(
            "WARNING: non-square pixels "
            f"({px_fast} != {px_slow}); using fast size."
        )

    return Detector(
        distance_mm=distance_mm,
        npx=npx,
        npy=npy,
        pixel_size_mm=px_fast,
        beam_centre_px=(cx_px, cy_px),
        fast_axis=fast,
        slow_axis=slow,
        origin=origin,
    )


def _ray_intersection_px(s1, fast, slow, origin,
                         px_fast, px_slow):
    """
    Project a scattered ray onto the panel, returning the
    intersection in pixels — the dxtbx Panel approach.

    The panel basis matrix d has the fast axis, slow axis and
    origin as its columns:

        d = [ fast | slow | origin ]

    Its inverse D = d^{-1} projects a lab-frame ray s1 onto
    the panel.  The first two components of D @ s1, divided by
    the third (the perspective divide), give the position on
    the panel in millimetres along fast and slow; dividing by
    the pixel sizes converts to pixels.  No trigonometry is
    needed.

    The lab origin is at the beam-sample intersection, so the
    ray s1 is taken as a direction from that origin.
    """
    d = np.column_stack([fast, slow, origin])
    D = np.linalg.inv(d)
    v = D @ np.asarray(s1, dtype=float)
    if v[2] == 0:
        raise ValueError(
            "Ray is parallel to the panel; no intersection."
        )
    x_mm = v[0] / v[2]
    y_mm = v[1] / v[2]
    cx_px = x_mm / px_fast
    cy_px = y_mm / px_slow
    return float(cx_px), float(cy_px)


def _wavelength_to_energy_eV(wavelength_A):
    """
    Relativistic electron energy (eV) for a given de Broglie
    wavelength in Angstrom.  Inverts
        lambda = hc / sqrt(E^2 + 2 m0c2 E).
    """
    wl_m = wavelength_A * 1e-10
    m0c2 = 510998.9461                       # eV
    hc = 4.135667696e-15 * 299792458.0       # eV·m
    lhs = (hc / wl_m) ** 2
    # E^2 + 2 m0c2 E - lhs = 0
    energy_eV = -m0c2 + np.sqrt(m0c2 ** 2 + lhs)
    return float(energy_eV)


def goniometer_matrices(gonio):
    """
    Return (S, F, rotation_axis) for a goniometer dict.

    Two dxtbx forms are supported:

      * explicit single-axis: the dict has 'setting_rotation',
        'fixed_rotation' and 'rotation_axis' directly.

      * multi-axis: the dict has
        'axes' (ordered crystal-to-goniometer),
        'angles' (fixed motor settings, degrees) and
        'scan_axis' (the index of the scanned axis).
        dxtbx composes these into

            F = product of the axis rotations BEFORE the scan
                axis (crystal side), at their fixed angles;
            S = product of the axis rotations AFTER the scan
                axis (base side), at their fixed angles;

        and the scan rotation axis is axes[scan_axis] (the datum
        direction; the effective lab axis is S @ axis, which the
        rotation R(angle) about `rotation_axis` combined with S
        in oriented_cell reproduces).

    The full goniostat rotation is S @ R(scan_axis, angle) @ F.
    """

    if "setting_rotation" in gonio and "fixed_rotation" in gonio:
        S = np.array(gonio["setting_rotation"]).reshape(3, 3)
        F = np.array(gonio["fixed_rotation"]).reshape(3, 3)
        rotation_axis = np.array(gonio["rotation_axis"])
        return S, F, rotation_axis

    axes = [np.array(a, dtype=float) for a in gonio["axes"]]
    angles = [float(a) for a in gonio["angles"]]
    scan_axis = int(gonio["scan_axis"])

    # Axes before the scan axis (crystal side) -> F.
    # Composition order: the axis nearest the scan axis is
    # applied first (innermost), matching S R F acting on the
    # crystal.  F = R(k-1) @ ... @ R(0).
    F = np.eye(3)
    for i in range(scan_axis):
        F = axis_angle_rotation_matrix(axes[i], angles[i]) @ F
    print("F", F)

    # Axes after the scan axis (base side) -> S.
    # S = R(n-1) @ ... @ R(scan_axis+1).
    S = np.eye(3)
    for i in range(scan_axis + 1, len(axes)):
        S = axis_angle_rotation_matrix(axes[i], angles[i]) @ S
    print("S", S)

    rotation_axis = axes[scan_axis]
    return S, F, rotation_axis
