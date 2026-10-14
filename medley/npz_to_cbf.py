"""
npz_to_cbf.py — re-render CBF images from per-image NPZ files
without re-running the simulation.

Each per-image NPZ stores the integrated signal and noise images
together with all the geometry the CBF header needs.  This tool
reads those files and renders CBFs with a fresh set of render
parameters (scale, background, beam blob), so the noise and
brightness can be changed cheaply, as many times as wanted.

The integration method ("vector" or "raster") is read from the
file; rendering is identical for both, since both store signal
and noise images.

Usage:
    python -m dynamic.simulation.npz_to_cbf IMAGE.npz [...] \\
        --scale 100000 --bg_intensity 1.0 --beam_intensity 200

A whole directory can be processed with a shell glob, e.g.
    python -m dynamic.simulation.npz_to_cbf out/image_*.npz

Pass --flip_y to reverse the slow (row / y) axis of each image
before writing, undoing a y-inverted signal.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from dynamic.simulation.export_cbf import (
    CbfParams,
    build_header,
    render_image,
    write_cbf,
)


def _load_npz(path):
    """Load a per-image NPZ into a plain dict of values."""
    d = np.load(path, allow_pickle=False)
    return {
        "method": str(d["method"]),
        "signal": d["signal"],
        "image_index": int(d["image_index"]),
        "angle_centre_deg": float(d["angle_centre_deg"]),
        "delta_deg": float(d["delta_deg"]),
        "pixel_size_mm": float(d["pixel_size_mm"]),
        "distance_mm": float(d["distance_mm"]),
        "beam_centre_px": tuple(d["beam_centre_px"]),
        "wavelength_A": float(d["wavelength_A"]),
    }


def _out_cbf_path(out_dir, method, tag, image_index):
    name = f"image_{method}_{tag}_{image_index:04d}.cbf"
    return os.path.join(out_dir, name)


def render_npz(path, params, out_dir, tag):
    """
    Render one per-image NPZ to a CBF with the given params.
    """
    rec = _load_npz(path)

    img = render_image(
        rec["signal"], rec["beam_centre_px"],
        params, rec["image_index"],
    )

    delta = rec["delta_deg"]
    start_angle = rec["angle_centre_deg"] - delta / 2.0
    header = build_header(
        rec["distance_mm"], rec["pixel_size_mm"],
        rec["beam_centre_px"], rec["wavelength_A"],
        start_angle, delta,
    )

    out_path = _out_cbf_path(
        out_dir, rec["method"], tag, rec["image_index"]
    )
    write_cbf(img, header, out_path)
    return out_path


def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Re-render CBF images from per-image NPZ files "
            "with new render parameters (no simulation)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "npz_files", nargs="+",
        help="Per-image NPZ files (e.g. out/image_*.npz)",
    )
    p.add_argument(
        "-o", "--out_dir", default=None,
        help=(
            "Output directory for CBFs (default: alongside "
            "each NPZ file)"
        ),
    )
    p.add_argument(
        "--tag", default="rerender",
        help="Run label used in the CBF filenames",
    )
    p.add_argument("--scale", type=float, default=10000.0)
    p.add_argument("--psf_sigma", type=float, default=1.0)
    p.add_argument(
        "--spot_percent", type=float, default=0.001
    )
    p.add_argument("--bg_intensity", type=float, default=1.0)
    p.add_argument("--bg_seed", type=int, default=0)
    p.add_argument("--beam_sigma", type=float, default=30.0)
    p.add_argument(
        "--beam_intensity", type=float, default=0.0
    )
    p.add_argument("--beam_seed", type=int, default=0)
    p.add_argument(
        "--flip_y", action="store_true",
        help=(
            "Reverse the slow (row / y) axis of each image "
            "before writing (undo a y-inverted signal)."
        ),
    )
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    params = CbfParams(
        scale=args.scale,
        psf_sigma=args.psf_sigma,
        spot_percent=args.spot_percent,
        bg_intensity=args.bg_intensity,
        bg_seed=args.bg_seed,
        beam_sigma=args.beam_sigma,
        beam_intensity=args.beam_intensity,
        beam_seed=args.beam_seed,
        flip_y=args.flip_y,
    )

    for path in args.npz_files:
        out_dir = args.out_dir or os.path.dirname(path) or "."
        os.makedirs(out_dir, exist_ok=True)
        out_path = render_npz(path, params, out_dir, args.tag)
        print(
            f"  {os.path.basename(path)} -> "
            f"{os.path.basename(out_path)}",
            flush=True,
        )
    print("Done.")


if __name__ == "__main__":
    main()
