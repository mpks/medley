"""
integrate_npz.py — sum per-image NPZ reflection tables into one
SHELX HKLF 4 file for the whole scan.

Each per-image NPZ written by Medley holds the intensity of every
reflection integrated over that image's angular range.  A
reflection recorded on several consecutive images (a partial) is
spread over those images, so summing over images gives its
intensity integrated over the full rotation.

Usage:
    medley-integrate sim_out/image_vector_L_*.npz -o L.hkl --auto_scale
    medley-integrate sim_out/image_vector_R_*.npz -o R.hkl --scale 1.43e5
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

# SHELX HKLF 4 writes intensities as F8.2: at most 99999.99, in
# steps of 0.01.
F82_MAX = 99999.99
F82_STEP = 0.01


def build_parser():
    p = argparse.ArgumentParser(
        description="Sum per-image NPZ reflection tables into one "
                    "SHELX HKLF 4 (.hkl) file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("npz_files", nargs="+",
                   help="Per-image NPZ files, e.g. "
                        "sim_out/image_vector_run1_*.npz")
    p.add_argument("-o", "--output", default="simulation.hkl",
                   help="Output .hkl file")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--scale", type=float, default=1.0,
                       help="Multiply all intensities by this factor. "
                            "Use the same value for datasets you want "
                            "to compare (e.g. both enantiomers).")
    group.add_argument("--auto_scale", action="store_true",
                       help="Choose the factor so the strongest "
                            "reflection is 99999.99 (the largest value "
                            "F8.2 can hold) and print it")
    return p


def sum_intensities(npz_files):
    """
    Return {(h, k, l): summed intensity} over all files.

    The direct beam (0, 0, 0) is left out: it is not a reflection,
    and in an HKLF 4 file a 0 0 0 line marks the end of the data.
    """
    totals = defaultdict(float)
    methods = set()
    for path in npz_files:
        with np.load(path) as data:
            methods.add(str(data["method"]))
            for hkl, inten in zip(data["hkl_indices"],
                                  data["intensities"]):
                key = tuple(int(x) for x in hkl)
                if key == (0, 0, 0):          # direct beam
                    continue
                totals[key] += float(inten)
    if len(methods) > 1:
        raise SystemExit(
            f"Files mix integration methods {sorted(methods)}; "
            "pass files from one run only."
        )
    return dict(totals)


def auto_scale(totals):
    """
    One multiplicative factor that puts the strongest reflection
    just below F82_MAX.  Rounded down to 3 significant figures, so
    the printed value can be reused exactly with --scale and never
    pushes the strongest reflection past 99999.99.
    """
    raw = F82_MAX / max(totals.values())
    p = 10.0 ** (np.floor(np.log10(raw)) - 2)
    return float(np.floor(raw / p) * p)


def write_shelx(totals, path, scale=1.0):
    """
    SHELX HKLF 4: 3I4, 2F8.2, terminated by a 0 0 0 line.

    sigma is written as 1.00 for every reflection: simulated data
    have no measurement error.
    """
    with open(path, "w") as f:
        for (h, k, l), inten in sorted(totals.items()):
            f.write(f"{h:4d}{k:4d}{l:4d}{inten * scale:8.2f}"
                    f"{1.0:8.2f}\n")
        f.write(f"{0:4d}{0:4d}{0:4d}{0.0:8.2f}{0.0:8.2f}\n")


def report(totals, scale):
    """Print the range and warn about values F8.2 cannot hold."""
    vals = np.array(list(totals.values())) * scale
    lost = np.sum(vals < F82_STEP / 2)          # print as 0.00
    print(f"Intensity range after scaling: {vals.min():.3g} "
          f"to {vals.max():.3g} (scale = {scale:.6g})")
    if vals.max() > F82_MAX:
        print("WARNING: intensities exceed 99999.99 and will not "
              "fit the F8.2 column; use a smaller --scale.")
    if lost:
        print(f"WARNING: {lost} of {len(vals)} reflections are "
              "below 0.005 and will be written as 0.00"
              + "; use a larger --scale or --auto_scale.")


def main(argv=None):
    args = build_parser().parse_args(argv)
    totals = sum_intensities(args.npz_files)
    scale = auto_scale(totals) if args.auto_scale else args.scale
    write_shelx(totals, args.output, scale)
    report(totals, scale)
    print(f"{len(args.npz_files)} files -> {len(totals)} reflections "
          f"-> {args.output}")
    if args.auto_scale:
        print(f"Scale factor used: {scale:.3g}  "
              f"(pass --scale {scale:.3g} to reuse it)")


if __name__ == "__main__":
    main()
