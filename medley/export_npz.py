"""
export_npz.py — default exporter writing the engine outputs to
NumPy NPZ files.

Two kinds of file are written:

  * one NPZ per image, holding the integrated signal and noise
    images, the spot list (Miller indices and detector
    centroids, used for labels), and all the geometry needed to
    reconstruct a CBF later without re-running the simulation;

  * one NPZ for the whole scan, holding the rocking curves
    (the substep angle axis and the raw intensity of each
    tracked reflection at every substep).

The integration method ("vector" or "raster") is recorded both
in the per-image filename and inside the file, so a downstream
tool knows how the signal and noise were built.

Per-image files are compressed (the images are mostly zeros).
"""

from __future__ import annotations

import os

import numpy as np


def _hkl_array(millers):
    """Stack a list of (h, k, l) into an (N, 3) int array."""
    if len(millers) == 0:
        return np.zeros((0, 3), dtype=np.int32)
    return np.array(millers, dtype=np.int32)


def image_filename(out_dir, method, tag, image_index):
    """
    Per-image NPZ path, including the integration method and a
    4-digit zero-padded index, e.g.
    image_vector_kmax02_0007.npz.
    """
    name = f"image_{method}_{tag}_{image_index:04d}.npz"
    return os.path.join(out_dir, name)


def save_image(image, detector, beam, scan, out_dir, tag,
               sg=None, sg_geom=None):
    """
    Save one ImageResult to a compressed NPZ file.

    Stored arrays/values:
      method        : "vector" or "raster"
      signal        : (npy, npx) float   integrated sharp signal
      hkl_indices   : (N, 3) int         spot list (labels)
      px, py        : (N,) float         spot centroids
      intensities   : (N,) float         integrated intensities
      image_index   : int
      angle_centre_deg, delta_deg : float
      npx, npy, pixel_size_mm, distance_mm : detector geometry
      beam_centre_px : (2,) float
      wavelength_A   : float

    Optionally, when supplied by the driver:
      sg            : (N,) float  signed excitation error at
                      angle_centre_deg, read out of the
                      Bloch-wave beam pool.  NaN where the
                      reflection is not in the pool at the
                      centre angle.  The sign sense is abTEM's;
                      compare against sg_geom.
      sg_geom       : (N,) float  the same quantity recomputed
                      from the oriented cell, with s_g > 0
                      meaning inside the Ewald sphere.

    Both are row-aligned to hkl_indices, so no downstream tool
    needs to re-derive the ordering.

    The noise halo is not stored; it is generated from the
    signal at render time, so its width (psf_sigma) and level
    (spot_percent) can be changed without re-simulating.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = image_filename(
        out_dir, image.method, tag, image.image_index
    )

    extra = {}
    if sg is not None:
        extra["sg"] = np.asarray(sg, dtype=float)
    if sg_geom is not None:
        extra["sg_geom"] = np.asarray(sg_geom, dtype=float)

    np.savez_compressed(
        path,
        method=image.method,
        signal=image.signal,
        hkl_indices=_hkl_array(image.millers),
        px=image.px,
        py=image.py,
        intensities=image.intensities,
        image_index=image.image_index,
        angle_centre_deg=image.angle_centre_deg,
        delta_deg=scan.delta_deg,
        npx=detector.npx,
        npy=detector.npy,
        pixel_size_mm=detector.pixel_size_mm,
        distance_mm=detector.distance_mm,
        beam_centre_px=np.array(detector.beam_centre_px),
        wavelength_A=beam.wavelength_A,
        **extra,
    )
    return path


def rocking_filename(out_dir, tag):
    """Build the scan-wide rocking-curve NPZ path."""
    return os.path.join(out_dir, f"rocking_{tag}.npz")


def save_rocking(rocking, out_dir, tag):
    """
    Save the scan-wide rocking curves to a single NPZ.

    Stored arrays:
      angles_deg  : (M,) float   substep angle axis
      hkl_indices : (K, 3) int   tracked reflections
      curves      : (K, M) float raw intensity per reflection
                                 per substep, row-aligned to
                                 hkl_indices

    The intensities are raw (unweighted); integrating a curve
    means multiplying by the angular step, which the angle axis
    provides.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = rocking_filename(out_dir, tag)

    hkls = sorted(rocking.curves.keys())
    hkl_arr = _hkl_array(hkls)
    if len(hkls) == 0:
        curves = np.zeros(
            (0, len(rocking.angles_deg)), dtype=float
        )
    else:
        curves = np.array(
            [rocking.curves[hkl] for hkl in hkls]
        )

    np.savez_compressed(
        path,
        angles_deg=rocking.angles_deg,
        hkl_indices=hkl_arr,
        curves=curves,
    )
    print(
        f"  wrote {os.path.basename(path)} "
        f"({len(hkls)} rocking curves, "
        f"{len(rocking.angles_deg)} substeps)",
        flush=True,
    )
    return path
