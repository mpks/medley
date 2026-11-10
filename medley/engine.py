"""
engine.py — integrate diffraction images.

The engine integrates images of a rotation scan: for each image it
runs the Simulator at the image's substep angles, projects the
spots to the detector, and accumulates per Miller index the total
intensity and the intensity-weighted centroid (px, py).  Each image
gives one ImageResult plus its per-substep rocking intensities.

The engine is backend-agnostic (it depends only on the abstract
Simulator interface and the domain data classes).  Image selection,
output writing and scan-wide rocking-curve assembly live in the
driver layer above.

Parallelism
-----------
run_images keeps ONE pool of n_workers processes for all the images
it is given.  Every image's substeps are cut into chunks of
substeps_per_task, and every (image, chunk) pair is a task in one
queue, so a worker that finishes takes the next chunk of whichever
image: no core waits for the slowest batch of an image, and
n_substeps need not divide by n_workers.  The queue is kept a few
tasks ahead of the workers, so they stay busy while the parent
merges a finished image (in chunk order, so the result does not
depend on n_workers or on which chunk finished first) and the
driver writes its outputs.  An optional per-image follow-up task
(the driver's excitation errors) runs in the same pool.

Each worker limits numpy's linear algebra (via threadpoolctl, if
installed) and dask (which abTEM uses) to threads_per_worker
threads, so n_workers processes on n_workers cores do not each
start a thread per core.  With n_workers = 1 everything runs in
the calling process with no limits, so abTEM and numpy may use
threads themselves.

Several crystal thicknesses can be simulated in one run.  Each
substep then makes one Bloch-wave calculation for all of them
(abTEM diagonalises once and evaluates every thickness from the
same eigenvectors), and one accumulator per thickness collects
the results.
"""

from __future__ import annotations

import math
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    wait,
)
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
        Number of worker processes, shared by all images of a
        run (1: everything in the calling process).
    rocking_hkl : tuple
        Miller indices (as tuples) to record rocking curves for.
        Empty means no rocking curves.
    substeps_per_task : int or None
        Substeps per task.  None chooses: 1 for vector
        integration (best balance; each task returns only small
        per-spot sums), and n_substeps / n_workers (rounded up)
        for raster, whose tasks each return full detector-sized
        arrays.
    threads_per_worker : int
        Threads each worker process may use for numpy's linear
        algebra and for dask.
    """
    base_thickness_A: tuple
    intensity_cut: float = 0.0
    n_workers: int = 1
    rocking_hkl: tuple = ()
    substeps_per_task: int | None = None
    threads_per_worker: int = 1

    def __post_init__(self):
        t = tuple(float(x) for x in np.atleast_1d(self.base_thickness_A))
        if not t:
            raise ValueError("base_thickness_A needs at least one value")
        if len(set(t)) != len(t):
            raise ValueError(f"base_thickness_A has repeated values: {t}")
        object.__setattr__(self, "base_thickness_A", t)
        if self.n_workers < 1:
            raise ValueError("n_workers must be at least 1")
        if self.substeps_per_task is not None and self.substeps_per_task < 1:
            raise ValueError("substeps_per_task must be at least 1")
        if self.threads_per_worker < 1:
            raise ValueError("threads_per_worker must be at least 1")


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
# Worker processes
# ----------------------------------------------------------------

# Per-process cache of the CIF atoms placed in the B^-1 cell, so a
# worker reads the CIF once per run rather than once per task.
_BASE_ATOMS = {}

# Kept alive so the thread limits stay in force in the worker.
_THREAD_LIMITS = None


def _base_atoms(cif_file, geometry):
    """make_base_atoms, cached per process by CIF and B matrix."""
    try:
        key = (cif_file, np.asarray(geometry.base_B()).tobytes())
    except AttributeError:          # no B matrix to key on
        return make_base_atoms(cif_file, geometry)
    if key not in _BASE_ATOMS:
        _BASE_ATOMS[key] = make_base_atoms(cif_file, geometry)
    return _BASE_ATOMS[key]


def _worker_init(threads):
    """
    Runs once in each worker process: limit numpy's linear algebra
    and dask to `threads` threads.
    """
    global _THREAD_LIMITS
    try:
        from threadpoolctl import threadpool_limits
        _THREAD_LIMITS = threadpool_limits(limits=threads)
    except ImportError:
        pass
    try:
        import dask
        if threads == 1:
            dask.config.set(scheduler="synchronous")
        else:
            dask.config.set(scheduler="threads", num_workers=threads)
    except ImportError:
        pass


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

    base_atoms = _base_atoms(cif_file, geometry)
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
# Running images
# ----------------------------------------------------------------

def substeps_per_task(params, method, n_substeps):
    """The chunk size: params.substeps_per_task, or the default."""
    if params.substeps_per_task is not None:
        return params.substeps_per_task
    if method == "raster":
        return max(1, math.ceil(n_substeps / params.n_workers))
    return 1


def run_images(cif_file, detector, beam, geometry, scan,
               image_indices, simulator, params,
               integration_params, extra_task=None,
               mp_context=None):
    """
    Integrate several images with one shared pool of workers.

    A generator: yields (image_index, image_result, image_rocking,
    extra) for each image as soon as it is finished, which need
    not be in index order.

      image_result  : ImageResult, every thickness
      image_rocking : dict (h, k, l) -> dict(angle -> (T,) intensity)
      extra         : the result of extra_task for this image, or
                      None

    extra_task, if given, is called in the parent with the
    finished ImageResult and returns (function, args); function
    (a picklable, module-level function) then runs in the pool,
    and the image is yielded once it is done.  The driver uses it
    for the excitation errors.

    mp_context : multiprocessing context for the pool (default:
    the platform's).
    """
    image_indices = list(image_indices)
    integrator = make_integrator(detector, integration_params)
    rocking_set = set(params.rocking_hkl)
    weight = scan.delta_deg / scan.n_substeps
    chunk = substeps_per_task(params, integration_params.method,
                              scan.n_substeps)

    tasks = []                  # (image_index, chunk_no, args)
    n_chunks = {}
    for idx in image_indices:
        angles = scan.substep_angles(idx)
        chunks = [angles[i:i + chunk]
                  for i in range(0, len(angles), chunk)]
        n_chunks[idx] = len(chunks)
        for c, angle_batch in enumerate(chunks):
            tasks.append((idx, c, (
                cif_file, geometry, simulator, detector, beam,
                params.base_thickness_A, angle_batch, rocking_set,
                integrator, weight,
            )))

    parts = {idx: {} for idx in image_indices}

    def finish(idx):
        """Merge an image's chunks in chunk order."""
        partials = [parts[idx][c] for c in range(n_chunks[idx])]
        del parts[idx]
        return _finish_image(partials, integrator, params,
                             integration_params, detector, scan, idx)

    if params.n_workers == 1:
        for idx, c, args in tasks:
            parts[idx][c] = _process_substep_batch(args)
            if len(parts[idx]) == n_chunks[idx]:
                result, rocking = finish(idx)
                extra = None
                if extra_task is not None:
                    function, extra_args = extra_task(result)
                    extra = function(*extra_args)
                yield idx, result, rocking, extra
        return

    # Keep this many tasks submitted, so the workers do not run
    # dry while the parent merges an image and writes its outputs.
    ahead = 4 * params.n_workers
    executor = ProcessPoolExecutor(
        max_workers=params.n_workers,
        initializer=_worker_init,
        initargs=(params.threads_per_worker,),
        mp_context=mp_context,
    )
    queue = iter(tasks)
    futures = {}
    waiting = {}                # image_index -> (result, rocking)

    def top_up():
        while len(futures) < ahead:
            try:
                idx, c, args = next(queue)
            except StopIteration:
                return
            future = executor.submit(_process_substep_batch, args)
            futures[future] = ("chunk", idx, c)

    try:
        top_up()
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                kind, idx, c = futures.pop(future)
                value = future.result()
                if kind == "chunk":
                    parts[idx][c] = value
                    if len(parts[idx]) < n_chunks[idx]:
                        continue
                    result, rocking = finish(idx)
                    if extra_task is None:
                        yield idx, result, rocking, None
                        continue
                    function, extra_args = extra_task(result)
                    futures[executor.submit(function, *extra_args)] = (
                        "extra", idx, None)
                    waiting[idx] = (result, rocking)
                else:
                    result, rocking = waiting.pop(idx)
                    yield idx, result, rocking, value
            top_up()
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown(wait=True)


def _finish_image(partials, integrator, params, integration_params,
                  detector, scan, image_index):
    """Merge one image's partial results into an ImageResult."""
    n_t = len(params.base_thickness_A)
    merged = [
        merge_states([states[t] for states, _rock in partials],
                     detector.npx, detector.npy)
        for t in range(n_t)
    ]
    image_rocking = _merge_rocking(partials)
    centre = scan.angles_deg[image_index] + 0.5 * scan.delta_deg
    image_result = _integrate_image(
        integrator, merged, params.base_thickness_A,
        params.intensity_cut, integration_params.method, centre,
        image_index,
    )
    return image_result, image_rocking


def run_image(cif_file, detector, beam, geometry, scan,
              image_index, simulator, params,
              integration_params):
    """
    Integrate a single image of the scan (run_images for one
    image).

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
    ((_idx, image_result, image_rocking, _extra),) = run_images(
        cif_file, detector, beam, geometry, scan, [image_index],
        simulator, params, integration_params,
    )
    return image_result, image_rocking
