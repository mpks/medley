"""
driver.py — scan-level driver above the single-image engine.

The engine integrates one image; the driver loops over the
selected images, calls the engine for each, writes that image's
outputs (NPZ, CBF, plots), and assembles the scan-wide rocking
curves.

Image selection lets a run cover only some images of the scan
(a single index, a range, or a list).  The rocking-curve axis
always spans the whole scan, with zeros at the substep angles of
images that were not simulated; a partial run therefore produces
a rocking file directly mergeable with a complementary run.

With export_SA set, each simulated image also gets its own
matrices NPZ holding the structure matrix A, the scattering
matrix S(t) and the projector diagonal M for one substep of that
image (see export_matrices).  This is a separate file, so the
per-image diffraction NPZ is unchanged.

With export_sg set (the default), each image's NPZ additionally
carries the signed excitation errors of its spots at the wedge
centre angle (see excitation).  This runs in the parent process
after the engine has returned, so it changes nothing about the
simulation itself; --no_sg reproduces the previous output byte
for byte.
"""

from __future__ import annotations

import os

import numpy as np

from dynamic.simulation.engine import run_image
from dynamic.simulation.excitation import (
    excitation_for_image,
)
from dynamic.simulation.export_npz import (
    save_image,
    save_rocking,
)
from dynamic.simulation.export_cbf import save_image_cbf
from dynamic.simulation.export_matrices import (
    export_matrices,
)
from dynamic.simulation.plotting import (
    plot_image,
    plot_detector_raster,
    plot_filename,
    raster_filename,
)


# ----------------------------------------------------------------
# Rocking-curve result (scan-wide)
# ----------------------------------------------------------------

class RockingResult:
    """
    Rocking curves for the tracked reflections across the scan.

    angles_deg : ndarray (M,)
        Full-scan substep angle axis.
    curves : dict (h, k, l) -> ndarray (M,)
        Intensity at each substep angle; zero where the image
        was not simulated.
    """

    def __init__(self, angles_deg, curves):
        self.angles_deg = angles_deg
        self.curves = curves


# ----------------------------------------------------------------
# Image selection
# ----------------------------------------------------------------

def parse_image_selection(text, n_images):
    """
    Parse an image-selection string into a sorted list of
    indices.

    Accepts single indices, ranges 'a-b' (inclusive), and
    comma-separated combinations, e.g.:

        '5'            -> [5]
        '0-9'          -> [0,1,...,9]
        '0,5,10'       -> [0,5,10]
        '0-9,20-29'    -> [0..9, 20..29]

    An empty or None string selects all images.

    Indices are validated against n_images.
    """
    if text is None or text.strip() == "":
        return list(range(n_images))

    indices = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            if lo > hi:
                lo, hi = hi, lo
            for i in range(lo, hi + 1):
                indices.add(i)
        else:
            indices.add(int(part))

    out = sorted(indices)
    for i in out:
        if i < 0 or i >= n_images:
            raise ValueError(
                f"Image index {i} out of range "
                f"[0, {n_images - 1}]."
            )
    return out


# ----------------------------------------------------------------
# Full-scan rocking axis
# ----------------------------------------------------------------

def build_full_axis(scan):
    """
    The full-scan substep angle axis: every image's substep
    angles concatenated in order.
    """
    angles = []
    for image_index in range(scan.n_images):
        sub = scan.substep_angles(image_index)
        angles.extend(float(a) for a in sub)
    return np.array(angles)


def _init_rocking(scan, rocking_hkl):
    """
    Initialise the scan-wide rocking storage: a full axis and a
    zero curve per tracked reflection.
    """
    axis = build_full_axis(scan)
    curves = {
        tuple(hkl): np.zeros(len(axis))
        for hkl in rocking_hkl
    }
    # Map angle -> index in the axis for fast fill-in.
    angle_to_idx = {
        float(a): i for i, a in enumerate(axis)
    }
    return axis, curves, angle_to_idx


def _fill_rocking(curves, angle_to_idx, image_rocking):
    """
    Place one image's per-substep intensities into the
    scan-wide curves at the matching angle positions.
    """
    for hkl, ang_map in image_rocking.items():
        if hkl not in curves:
            continue
        curve = curves[hkl]
        for angle, inten in ang_map.items():
            idx = angle_to_idx.get(float(angle))
            if idx is not None:
                curve[idx] = inten


# ----------------------------------------------------------------
# Per-image output
# ----------------------------------------------------------------

def _write_image_outputs(image_result, detector, beam, scan,
                         cbf_params, out_dir, tag,
                         write_cbf, write_plots,
                         sg=None, sg_geom=None):
    """Write NPZ, optionally CBF and plots, for one image."""
    save_image(image_result, detector, beam, scan,
               out_dir, tag, sg=sg, sg_geom=sg_geom)

    if write_cbf:
        save_image_cbf(image_result, detector, beam, scan,
                       cbf_params, out_dir, tag)

    if write_plots:
        idx = image_result.image_index
        method = image_result.method
        plot_image(
            image_result, detector,
            plot_filename(out_dir, method, tag, idx),
        )
        plot_detector_raster(
            image_result, detector, cbf_params,
            raster_filename(out_dir, method, tag, idx),
        )


# ----------------------------------------------------------------
# Top-level run
# ----------------------------------------------------------------

def run(cif_file, detector, beam, geometry, scan, simulator,
        engine_params, integration_params, cbf_params,
        out_dir, tag, image_selection=None, write_cbf=True,
        write_plots=True, export_SA=False,
        export_SA_substep=None, export_sg=True):
    """
    Run the scan over the selected images.

    Parameters
    ----------
    integration_params : IntegrationParams
        Passed through to the engine to select the integrator
        (vector or raster) and its psf_sigma / spot_percent.
    image_selection : str or None
        Selection string (see parse_image_selection); None or
        empty means all images.
    write_cbf, write_plots : bool
        Toggle the CBF and plot outputs.
    export_SA : bool
        Also write a matrices NPZ per simulated image, holding
        A, S(t) and M for one substep of that image.
    export_SA_substep : int or None
        Which substep the matrices are taken from; None means
        the middle substep of the image.
    export_sg : bool
        Also store the signed excitation errors of the image's
        spots, at the wedge centre angle, in that image's NPZ.
        Costs one beam-pool build per image (no eigen-
        decomposition) and runs in the parent process, so the
        simulation itself is unaffected.

    Returns
    -------
    (image_results, rocking_result)
      image_results : list of ImageResult for the simulated
                      images, in index order
      rocking_result : RockingResult on the full scan axis,
                       zeros where images were not simulated
    """
    os.makedirs(out_dir, exist_ok=True)

    selected = parse_image_selection(
        image_selection, scan.n_images
    )
    print(
        f"Selected {len(selected)} of {scan.n_images} "
        f"images: {selected[:10]}"
        + (" ..." if len(selected) > 10 else "")
    )

    axis, curves, angle_to_idx = _init_rocking(
        scan, engine_params.rocking_hkl
    )

    image_results = []

    for n, image_index in enumerate(selected):
        centre = (
            scan.angles_deg[image_index]
            + 0.5 * scan.delta_deg
        )
        print(
            f"[{n + 1}/{len(selected)}] image "
            f"{image_index} (centre {centre:.3f} deg)",
            flush=True,
        )

        image_result, image_rocking = run_image(
            cif_file, detector, beam, geometry, scan,
            image_index, simulator, engine_params,
            integration_params,
        )
        image_results.append(image_result)

        _fill_rocking(curves, angle_to_idx, image_rocking)

        sg = sg_geom = None
        if export_sg:
            sg, sg_geom, _info = excitation_for_image(
                cif_file, geometry, beam, simulator,
                image_result.angle_centre_deg,
                image_result.millers,
            )

        _write_image_outputs(
            image_result, detector, beam, scan,
            cbf_params, out_dir, tag,
            write_cbf, write_plots, sg, sg_geom,
        )

        if export_SA:
            export_matrices(
                cif_file=cif_file,
                geometry=geometry,
                scan=scan,
                simulator=simulator,
                engine_params=engine_params,
                image_index=image_index,
                out_dir=out_dir,
                tag=tag,
                substep_index=export_SA_substep,
            )

    rocking_result = RockingResult(axis, curves)
    save_rocking(rocking_result, out_dir, tag)

    print("Driver finished.")
    return image_results, rocking_result
