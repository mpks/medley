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

A run may simulate several crystal thicknesses.  Every per-image
array that depends on the thickness then has a leading thickness
axis of length T (also when T = 1), and thicknesses_A lists the
thicknesses in that order.  The spot list (hkl_indices) is the
same at every thickness.  load_image reads these files, and the
earlier single-thickness files, into one layout; image_at picks
one thickness out of it.

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

    Stored arrays/values (T thicknesses, N spots):
      method        : "vector" or "raster"
      thicknesses_A : (T,) float          base thicknesses (zero tilt)
      signal        : (T, npy, npx) float integrated sharp signal
      hkl_indices   : (N, 3) int          spot list, all thicknesses
      px, py        : (T, N) float        spot centroids
      intensities   : (T, N) float        integrated intensities
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
    needs to re-derive the ordering.  They are geometric, so the
    same at every thickness.

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
        thicknesses_A=np.asarray(image.thicknesses_A, dtype=float),
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

    Stored arrays (T thicknesses):
      thicknesses_A : (T,) float
      angles_deg    : (M,) float      substep angle axis
      hkl_indices   : (K, 3) int      tracked reflections
      curves        : (T, K, M) float raw intensity per thickness,
                                      reflection and substep,
                                      row-aligned to hkl_indices

    The intensities are raw (unweighted); integrating a curve
    means multiplying by the angular step, which the angle axis
    provides.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = rocking_filename(out_dir, tag)

    hkls = sorted(rocking.curves.keys())
    hkl_arr = _hkl_array(hkls)
    n_t = len(rocking.thicknesses_A)
    if len(hkls) == 0:
        curves = np.zeros(
            (n_t, 0, len(rocking.angles_deg)), dtype=float
        )
    else:
        # (K, T, M) -> (T, K, M): thickness axis first, as in
        # the per-image files
        curves = np.array(
            [rocking.curves[hkl] for hkl in hkls]
        ).transpose(1, 0, 2)

    np.savez_compressed(
        path,
        thicknesses_A=np.asarray(rocking.thicknesses_A, dtype=float),
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


# ----------------------------------------------------------------
# Reading per-image files, choosing a thickness
# ----------------------------------------------------------------

_PER_THICKNESS = ("signal", "px", "py", "intensities")


def load_image(path):
    """
    Read a per-image NPZ into a dict, always with a leading
    thickness axis on signal, px, py and intensities.

    Files from before thicknesses were stored get T = 1 and
    thicknesses_A = [nan] (the thickness was not recorded).
    """
    with np.load(path, allow_pickle=False) as d:
        rec = {k: d[k] for k in d.files}
    rec["method"] = str(rec["method"])
    rec["image_index"] = int(rec["image_index"])
    if "thicknesses_A" not in rec:
        for k in _PER_THICKNESS:
            rec[k] = rec[k][np.newaxis]
        rec["thicknesses_A"] = np.array([np.nan])
    rec["thicknesses_A"] = np.asarray(rec["thicknesses_A"], dtype=float)
    return rec


def image_at(rec, t_index):
    """
    One thickness of a load_image record: the same dict with
    signal (npy, npx), px, py and intensities (N,), and
    thickness_A the chosen thickness.
    """
    out = dict(rec)
    for k in _PER_THICKNESS:
        out[k] = rec[k][t_index]
    out["thickness_A"] = float(rec["thicknesses_A"][t_index])
    return out


def thickness_index(thicknesses_A, thickness=None):
    """
    Index of the requested thickness (A) in thicknesses_A, or 0
    (the first) when thickness is None.  Raises ValueError,
    listing the available ones, if it is not there.
    """
    thicknesses_A = np.asarray(thicknesses_A, dtype=float)
    if thickness is None:
        return 0
    match = np.flatnonzero(np.isclose(thicknesses_A, thickness,
                                      rtol=0.0, atol=1e-6))
    if match.size == 0:
        have = ", ".join(format_thickness(t) for t in thicknesses_A)
        raise ValueError(
            f"thickness {thickness:g} A is not in the file "
            f"(available: {have})")
    return int(match[0])


def format_thickness(t):
    """800.0 -> '800', 800.5 -> '800.5', nan -> 'unknown'."""
    if not np.isfinite(t):
        return "unknown"
    return f"{t:g}"


def thickness_suffix(t):
    """
    File-name suffix for one thickness, e.g. 800 ->
    '_thickness_00800_angs' (a fractional part is kept as
    '00800p5').
    """
    if float(t).is_integer():
        label = f"{int(t):05d}"
    else:
        label = f"{t:07.1f}".replace(".", "p")
    return f"_thickness_{label}_angs"


def with_thickness_suffix(path, t):
    """'out/L.hkl', 800 -> 'out/L_thickness_00800_angs.hkl'."""
    root, ext = os.path.splitext(path)
    return root + thickness_suffix(t) + ext
