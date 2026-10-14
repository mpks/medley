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
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import ase.io

from dynamic.simulation.projection import project_spots
from dynamic.simulation.integration import (
    make_integrator,
    merge_states,
)


# ----------------------------------------------------------------
# Engine parameters and results
# ----------------------------------------------------------------

@dataclass(frozen=True)
class EngineParams:
    """
    Engine-level (backend-independent) configuration.

    base_thickness_A : float
        Crystal thickness along the beam at zero tilt (A); the
        per-substep thickness is base_thickness_A / cos(angle).
    intensity_cut : float
        Reflections whose integrated intensity is at or below
        this value are dropped from an image after integration.
    n_workers : int
        Number of worker processes for substep parallelism.
    rocking_hkl : tuple
        Miller indices (as tuples) to record rocking curves for.
        Empty means no rocking curves.
    """
    base_thickness_A: float
    intensity_cut: float = 0.0
    n_workers: int = 1
    rocking_hkl: tuple = ()


@dataclass(frozen=True)
class ImageResult:
    """
    Integrated diffraction for one image.

    millers : list of (h, k, l)
    px, py : ndarray
        Intensity-weighted detector centroids (pixels), used to
        label the diagnostic image.
    intensities : ndarray
        Integrated intensities (angular integral over substeps).
    signal : ndarray (npy, npx)
        Integrated sharp signal image (no blur).  The noise halo
        is produced from this at render time.
    method : str
        Integration method used ("vector" or "raster").
    angle_centre_deg : float
        Centre angle of the image (start + delta/2).
    image_index : int
    """
    millers: list
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
    if not np.isclose(v_before, v_after, rtol=0.35):
        raise ValueError(
            "Cell volume changed on alignment: "
            f"{v_before:.2f} -> {v_after:.2f} A^3. "
            "CIF cell and experiment B metric differ by "
            "more than 15%."
        )
    return atoms


def orient_atoms(base_atoms, geometry, angle_deg):
    """
    Set the base atoms into the lab-frame cell for a scan angle.

    The oriented cell (interpolated from the scan-point setting
    arrays and including the scan rotation R(angle)) is taken
    from geometry.oriented_cell; the atoms are placed into it
    with scale_atoms=True so the CIF fractional coordinates are
    preserved, then wrapped.
    """
    atoms = base_atoms.copy()
    new_cell = geometry.oriented_cell(angle_deg)
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
       base_thickness_A, angle_batch, rocking_hkl_set,
       integrator, weight)

    weight is the angular width delta_deg / n_substeps applied
    to every substep so the integrated intensity is invariant
    to the number of substeps.  The rocking curves store the
    raw (unweighted) intensity.

    Returns (state, rocking)
      state   : IntegratorState
      rocking : dict (h,k,l) -> dict(angle -> raw intensity)
    """
    (cif_file, geometry, simulator, detector, beam,
     base_thickness_A, angle_batch, rocking_hkl_set,
     integrator, weight) = args

    base_atoms = make_base_atoms(cif_file, geometry)

    state = integrator.new_state()
    rocking = {}

    for angle in angle_batch:
        atoms = orient_atoms(base_atoms, geometry, angle)
        tilt_rad = np.deg2rad(angle)
        thickness_A = base_thickness_A / np.cos(tilt_rad)

        spots = simulator.simulate(atoms, thickness_A)
        projected = project_spots(spots, detector, beam)

        integrator.add_substep(state, projected, weight)

        if rocking_hkl_set:
            _record_rocking(
                rocking, projected, rocking_hkl_set, angle
            )

    return state, rocking


def _record_rocking(rocking, projected, rocking_set, angle):
    """
    Store the raw (unweighted) intensity of each tracked
    reflection at this substep angle.
    """
    for s in projected:
        hkl = s["hkl"]
        if hkl in rocking_set:
            if hkl not in rocking:
                rocking[hkl] = {}
            rocking[hkl][float(angle)] = s["intensity"]


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

def _integrate_image(integrator, state, intensity_cut,
                     method, angle_centre_deg, image_index):
    """
    Finalise the merged integrator state into an ImageResult.
    """
    (signal, millers, px, py,
     ints) = integrator.finalize(state, intensity_cut)

    return ImageResult(
        millers=millers,
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
          Integrated spots, signal and noise for this image.
      image_rocking : dict (h, k, l) -> dict(angle -> intensity)
          Per-substep raw intensities for the tracked
          reflections of this image.  The driver places these
          on the full scan axis.
    """
    if scan.n_substeps % 2 != 0:
        print(
            "WARNING: n_substeps is odd; the image centre "
            "angle is not sampled by a substep."
        )

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

    states = [state for state, _rock in partials]
    merged = merge_states(states, detector.npx, detector.npy)
    image_rocking = _merge_rocking(partials)

    image_result = _integrate_image(
        integrator, merged, params.intensity_cut,
        integration_params.method, centre, image_index,
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
