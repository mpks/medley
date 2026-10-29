"""
engine.py — integrate a single diffraction image.

The engine integrates one image of a rotation scan: it loops the
substep angles of that image, runs the Simulator at each, projects
the spots to the detector, and accumulates per Miller index the
total intensity and the intensity-weighted centroid (px, py).  It
returns one ImageResult plus that image's per-substep rocking
intensities.

The engine is backend-agnostic (it depends only on the abstract
Simulator interface and the domain data classes) and deals with a
single image only.  Looping over images, image selection, output
writing and scan-wide rocking-curve assembly all live in the
driver layer above.

Parallelism is over the substeps of the image: the substeps are
split into n_workers batches, each handled in a separate process;
the engine merges the partial accumulators.

Several crystal thicknesses can be simulated in one run.  Each
substep then makes one Bloch-wave calculation for all of them
(abTEM diagonalises once and evaluates every thickness from the
same eigenvectors), and one accumulator per thickness collects
the results.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import ase.io

from medley.projection import abtem_to_lab, project_spots
from medley.integration import (
    make_integrator,
    merge_states,
    spot_table,
)


# ----------------------------------------------------------------
# Engine parameters and results
# ----------------------------------------------------------------

@dataclass(frozen=True)
class EngineParams:
    """
    Engine-level (backend-independent) configuration.

    base_thickness_A : float or sequence of float
        Crystal thickness(es) along the beam at zero tilt (A);
        the per-substep thickness is base_thickness_A /
        cos(angle).  Stored as a tuple; several values simulate
        several thicknesses in one run.
    intensity_cut : float
        Reflections whose integrated intensity is at or below
        this value are dropped from an image after integration.
    n_workers : int
        Number of worker processes for substep parallelism.
    rocking_hkl : tuple
        Miller indices (as tuples) to record rocking curves for.
        Empty means no rocking curves.
    """
    base_thickness_A: tuple
    intensity_cut: float = 0.0
    n_workers: int = 1
    rocking_hkl: tuple = ()

    def __post_init__(self):
        t = tuple(float(x) for x in np.atleast_1d(self.base_thickness_A))
        if not t:
            raise ValueError("base_thickness_A needs at least one value")
        if len(set(t)) != len(t):
            raise ValueError(f"base_thickness_A has repeated values: {t}")
        object.__setattr__(self, "base_thickness_A", t)


@dataclass(frozen=True)
class ImageResult:
    """
    Integrated diffraction for one image, at every thickness.

    T is the number of thicknesses and N the number of spots; the
    spots are the same at every thickness.

    millers : list of (h, k, l), length N
    thicknesses_A : tuple of float, length T
        Base thicknesses (at zero tilt), in run order.
    px, py : ndarray (T, N)
        Intensity-weighted detector centroids (pixels).
    intensities : ndarray (T, N)
        Integrated intensities (angular integral over substeps).
    signal : ndarray (T, npy, npx)
        Integrated sharp signal image (no blur).  The noise halo
        is produced from this at render time.
    method : str
        Integration method used ("vector" or "raster").
    angle_centre_deg : float
        Centre angle of the image (start + delta/2).
    image_index : int
    """
    millers: list
    thicknesses_A: tuple
    px: np.ndarray
    py: np.ndarray
    intensities: np.ndarray
    signal: np.ndarray
    method: str
    angle_centre_deg: float
    image_index: int

    def at(self, t_index=0):
        """The result at one thickness, as an ImageAtThickness."""
        return ImageAtThickness(
            millers=self.millers,
            thickness_A=self.thicknesses_A[t_index],
            px=self.px[t_index],
            py=self.py[t_index],
            intensities=self.intensities[t_index],
            signal=self.signal[t_index],
            method=self.method,
            angle_centre_deg=self.angle_centre_deg,
            image_index=self.image_index,
        )


@dataclass(frozen=True)
class ImageAtThickness:
    """
    One thickness of an ImageResult: px, py, intensities (N,)
    and signal (npy, npx), as the CBF writer and plots expect.
    """
    millers: list
    thickness_A: float
    px: np.ndarray
    py: np.ndarray
    intensities: np.ndarray
    signal: np.ndarray
    method: str
    angle_centre_deg: float
    image_index: int


# ----------------------------------------------------------------
# Atom orientation
# ----------------------------------------------------------------

def make_base_atoms(cif_file, geometry):
    """
    Read the CIF and place the atoms into the B^{-1} crystal
    cell, preserving the CIF fractional coordinates.

    Uses scale_atoms=True so the fractional coordinates are
    kept (convention-independent) while the cell becomes
    B^{-1}.  A volume check (15% tolerance) guards against a
    CIF cell whose metric grossly mismatches the experiment B;
    small differences (e.g. CIF and ED data at different
    temperatures) are expected and allowed.
    """
    atoms = ase.io.read(cif_file)
    v_before = atoms.get_volume()
    crystal_cell = np.linalg.inv(geometry.base_B())
    atoms.set_cell(crystal_cell, scale_atoms=True)
    v_after = atoms.get_volume()
    if not np.isclose(v_before, v_after, rtol=0.15):
        raise ValueError(
            "Cell volume changed on alignment: "
            f"{v_before:.2f} -> {v_after:.2f} A^3. "
            "CIF cell and experiment B metric differ by "
            "more than 15%."
        )
    return atoms


def orient_atoms(base_atoms, geometry, angle_deg, beam):
    """
    Set the base atoms into abTEM's frame for a scan angle.

    The lab-frame cell at this angle (interpolated from the
    scan-point setting arrays, including the scan rotation
    R(angle)) comes from geometry.oriented_cell.  abTEM always
    sends its beam along +z, whereas the real electrons travel
    along -beam.direction, so the cell is rotated into abTEM's
    frame with R.T, where R = abtem_to_lab(beam):

        cell rows (a, b, c)  ->  (R.T a, R.T b, R.T c)
                              =  oriented_cell @ R

    abTEM's +z beam then meets the crystal exactly as the real
    beam does.  The atoms are placed into this cell with
    scale_atoms=True, so the CIF fractional coordinates are
    preserved, then wrapped.  All vectors that come back from
    abTEM are in this frame and must be rotated back with R
    (projection.project_spots does this).
    """
    R = abtem_to_lab(beam)
    atoms = base_atoms.copy()
    new_cell = geometry.oriented_cell(angle_deg) @ R
    atoms.set_cell(new_cell, scale_atoms=True)
    atoms.wrap()
    return atoms


# ----------------------------------------------------------------
# Worker: process a batch of substeps for one image
# ----------------------------------------------------------------

def _process_substep_batch(args):
    """
    Process a batch of substep angles for a single image.

    Runs in a worker process.  Builds a partial integrator
    state (signal/noise arrays and per-hkl sums) and the partial
    rocking-curve data for this batch.

    args is a tuple (picklable):
      (cif_file, geometry, simulator, detector, beam,
       base_thicknesses_A, angle_batch, rocking_hkl_set,
       integrator, weight)

    weight is the angular width delta_deg / n_substeps applied
    to every substep so the integrated intensity is invariant
    to the number of substeps.  The rocking curves store the
    raw (unweighted) intensity.

    Returns (states, rocking)
      states  : list of IntegratorState, one per thickness
      rocking : dict (h,k,l) -> dict(angle -> (T,) raw intensity)
    """
    (cif_file, geometry, simulator, detector, beam,
     base_thicknesses_A, angle_batch, rocking_hkl_set,
     integrator, weight) = args

    base_atoms = make_base_atoms(cif_file, geometry)
    base = np.asarray(base_thicknesses_A, dtype=float)

    states = [integrator.new_state() for _ in base]
    rocking = {}

    for angle in angle_batch:
        atoms = orient_atoms(base_atoms, geometry, angle, beam)
        tilt_rad = np.deg2rad(angle)
        thicknesses_A = base / np.cos(tilt_rad)

        spots_per_t = simulator.simulate(atoms, thicknesses_A)
        projected = [project_spots(spots, detector, beam)
                     for spots in spots_per_t]

        for state, proj in zip(states, projected):
            integrator.add_substep(state, proj, weight)

        if rocking_hkl_set:
            _record_rocking(
                rocking, projected, rocking_hkl_set, angle
            )

    return states, rocking


def _record_rocking(rocking, projected, rocking_set, angle):
    """
    Store the raw (unweighted) intensity of each tracked
    reflection at this substep angle, one value per thickness.
    projected is a list (one per thickness) of projected spots.
    """
    n_t = len(projected)
    for t, proj in enumerate(projected):
        for s in proj:
            hkl = s["hkl"]
            if hkl in rocking_set:
                per_angle = rocking.setdefault(hkl, {})
                values = per_angle.setdefault(float(angle),
                                              np.zeros(n_t))
                values[t] = s["intensity"]


# ----------------------------------------------------------------
# Merging partial results
# ----------------------------------------------------------------

def _merge_rocking(partials):
    """Merge the rocking dicts from the partial results."""
    rocking = {}
    for _state, p_rock in partials:
        for hkl, ang_map in p_rock.items():
            if hkl not in rocking:
                rocking[hkl] = {}
            rocking[hkl].update(ang_map)
    return rocking


# ----------------------------------------------------------------
# Per-image integration
# ----------------------------------------------------------------

def _integrate_image(integrator, states, thicknesses_A,
                     intensity_cut, method, angle_centre_deg,
                     image_index):
    """
    Finalise the merged per-thickness states into an ImageResult
    with one shared spot list.
    """
    millers, px, py, ints = spot_table(states, intensity_cut)
    signal = np.stack([
        integrator.build_signal(state, px[t], py[t], ints[t])
        for t, state in enumerate(states)
    ])
    return ImageResult(
        millers=millers,
        thicknesses_A=tuple(thicknesses_A),
        px=px,
        py=py,
        intensities=ints,
        signal=signal,
        method=method,
        angle_centre_deg=angle_centre_deg,
        image_index=image_index,
    )


# ----------------------------------------------------------------
# Top-level driver
# ----------------------------------------------------------------

def run_image(cif_file, detector, beam, geometry, scan,
              image_index, simulator, params,
              integration_params):
    """
    Integrate a single image of the scan.

    Parameters
    ----------
    cif_file : str
    detector : Detector
    beam : Beam
    geometry : Geometry
    scan : Scan
    image_index : int
        Which image of the scan to integrate.
    simulator : Simulator
    params : EngineParams
    integration_params : IntegrationParams
        Selects the integrator (vector or raster) and its
        psf_sigma / spot_percent.

    Returns
    -------
    (image_result, image_rocking)
      image_result : ImageResult
          Integrated spots and signal for this image, at every
          thickness.
      image_rocking : dict (h, k, l) -> dict(angle -> (T,) intensity)
          Per-substep raw intensities for the tracked
          reflections of this image.  The driver places these
          on the full scan axis.
    """
    rocking_set = set(params.rocking_hkl)
    sub_angles = scan.substep_angles(image_index)
    start = scan.angles_deg[image_index]
    centre = start + 0.5 * scan.delta_deg

    # Angular weight per substep: invariant integrated intensity.
    weight = scan.delta_deg / scan.n_substeps

    integrator = make_integrator(detector, integration_params)

    partials = _run_image_substeps(
        cif_file, detector, beam, geometry, simulator,
        params, sub_angles, rocking_set, integrator, weight,
    )

    n_t = len(params.base_thickness_A)
    merged = [
        merge_states([states[t] for states, _rock in partials],
                     detector.npx, detector.npy)
        for t in range(n_t)
    ]
    image_rocking = _merge_rocking(partials)

    image_result = _integrate_image(
        integrator, merged, params.base_thickness_A,
        params.intensity_cut, integration_params.method, centre,
        image_index,
    )
    return image_result, image_rocking


def _run_image_substeps(cif_file, detector, beam, geometry,
                        simulator, params, sub_angles,
                        rocking_set, integrator, weight):
    """
    Dispatch one image's substeps across worker processes and
    return the list of (state, rocking) partial results.
    """
    batches = _split_batches(sub_angles, params.n_workers)
    work = [
        (cif_file, geometry, simulator, detector, beam,
         params.base_thickness_A, batch, rocking_set,
         integrator, weight)
        for batch in batches
    ]

    if params.n_workers == 1:
        return [_process_substep_batch(w) for w in work]

    with ProcessPoolExecutor(
        max_workers=params.n_workers
    ) as executor:
        results = list(
            executor.map(_process_substep_batch, work)
        )
    return results


def _split_batches(angles, n_workers):
    """Split an array of angles into n_workers contiguous
    batches (some may be empty if there are fewer angles
    than workers)."""
    return [
        batch for batch in np.array_split(angles, n_workers)
        if len(batch) > 0
    ]
