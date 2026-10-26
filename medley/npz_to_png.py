"""
npz_to_png.py — render per-image NPZ files to greyscale PNG images,
for a quick look at a simulation without going through DIALS.

The detector image is built exactly as for a CBF (same scale, PSF,
spot noise, background, direct beam and flip options as
npz_to_cbf), then mapped to 8-bit grey, one PNG pixel per detector
pixel.  By default spots are dark on a white background.

Display mapping (PNG only, does not change the image itself):
  counts 0 .. vmax are mapped linearly to white .. black;
  --log maps log(1 + counts) instead, which shows weak spots;
  vmax defaults to the --percentile of the non-zero pixels.

Usage:
    medley.npz_to_png sim_out/image_vector_run1_*.npz -o png
    medley.npz_to_png image.npz --log --beam_intensity 200
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image

from medley.export_cbf import CbfParams, render_image
from medley.npz_to_cbf import _load_npz


def to_grey(img, vmax=None, percentile=99.5, log=False, invert=True):
    """
    Map detector counts to an 8-bit greyscale array.
    Returns (uint8 array, vmax used).
    """
    img = np.asarray(img, dtype=float)
    if vmax is None:
        nonzero = img[img > 0]
        vmax = float(np.percentile(nonzero, percentile)) if nonzero.size else 1.0
    vmax = max(vmax, 1e-12)
    if log:
        x = np.log1p(np.clip(img, 0, None)) / np.log1p(vmax)
    else:
        x = img / vmax
    x = np.clip(x, 0.0, 1.0)
    if invert:
        x = 1.0 - x
    return np.round(255 * x).astype(np.uint8), vmax


def png_path(out_dir, npz_path):
    base = os.path.splitext(os.path.basename(npz_path))[0]
    return os.path.join(out_dir, base + ".png")


def build_parser():
    p = argparse.ArgumentParser(
        description="Render per-image NPZ files to greyscale PNG "
                    "images (same rendering options as npz_to_cbf).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("npz_files", nargs="+",
                   help="Per-image NPZ files (e.g. out/image_*.npz)")
    p.add_argument("-o", "--out_dir", default=None,
                   help="Output directory (default: next to each NPZ)")
    # rendering: identical to npz_to_cbf
    p.add_argument("--scale", type=float, default=10000.0)
    p.add_argument("--psf_sigma", type=float, default=1.0)
    p.add_argument("--spot_percent", type=float, default=0.001)
    p.add_argument("--bg_intensity", type=float, default=1.0)
    p.add_argument("--bg_seed", type=int, default=0)
    p.add_argument("--beam_sigma", type=float, default=30.0)
    p.add_argument("--beam_intensity", type=float, default=0.0)
    p.add_argument("--beam_seed", type=int, default=0)
    p.add_argument("--flip_y", action="store_true",
                   help="Reverse the slow (row / y) axis")
    # display only
    p.add_argument("--vmax", type=float, default=None,
                   help="Counts shown as full black (default: the "
                        "--percentile of non-zero pixels)")
    p.add_argument("--percentile", type=float, default=99.5,
                   help="Percentile used for vmax when --vmax is not "
                        "given")
    p.add_argument("--log", action="store_true",
                   help="Logarithmic grey scale (shows weak spots)")
    p.add_argument("--white_spots", action="store_true",
                   help="White spots on black instead of black on white")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    params = CbfParams(
        scale=args.scale, psf_sigma=args.psf_sigma,
        spot_percent=args.spot_percent,
        bg_intensity=args.bg_intensity, bg_seed=args.bg_seed,
        beam_sigma=args.beam_sigma,
        beam_intensity=args.beam_intensity,
        beam_seed=args.beam_seed, flip_y=args.flip_y,
    )
    for path in args.npz_files:
        rec = _load_npz(path)
        img = render_image(rec["signal"], rec["beam_centre_px"],
                           params, rec["image_index"])
        grey, vmax = to_grey(img, vmax=args.vmax,
                             percentile=args.percentile, log=args.log,
                             invert=not args.white_spots)
        out_dir = args.out_dir or os.path.dirname(path) or "."
        os.makedirs(out_dir, exist_ok=True)
        out = png_path(out_dir, path)
        Image.fromarray(grey, mode="L").save(out)
        print(f"  {os.path.basename(path)} -> {os.path.basename(out)}"
              f"  (vmax {vmax:.4g} counts)", flush=True)
    print("Done.")


if __name__ == "__main__":
    main()
