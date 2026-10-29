"""
npz_to_nexus.py — combine Medley's per-image NPZ files into one
NeXus/NXmx file with the geometry of a DIALS experiment.

The images are rendered exactly as for CBF (same scale, PSF, noise,
direct-beam and flip options as npz_to_cbf) and stored as one 3D
dataset.  DIALS imports the file with its generic NXmx reader, so this
works for any detector, not only those with a dedicated CBF format.

Usage:
    medley.npz_to_nexus refined.expt sim_out/image_raster_run1_*.npz \\
        -o simulation.nxs
    dials.import simulation.nxs

Files holding several thicknesses give one NeXus file per
thickness, named e.g. simulation_thickness_00800_angs.nxs, unless
--thickness (in A) picks one.
"""

from __future__ import annotations

import argparse

import numpy as np

from medley.export_cbf import CbfParams, render_image
from medley.export_npz import (
    format_thickness,
    load_image,
    thickness_index,
    with_thickness_suffix,
)
from medley.export_nexus import read_geometry, write_nxmx


def build_parser():
    p = argparse.ArgumentParser(
        description="Write per-image NPZ files as one NeXus/NXmx file "
                    "with the geometry of a DIALS experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("expt_file", help="DIALS .expt the simulation used")
    p.add_argument("npz_files", nargs="+",
                   help="Per-image NPZ files (a contiguous block)")
    p.add_argument("-o", "--output", default="simulation.nxs")
    p.add_argument("--experiment", type=int, default=0,
                   help="Experiment in the .expt to take the geometry from")
    p.add_argument("--probe", default="electron",
                   choices=["electron", "x-ray", "neutron"])
    p.add_argument("--scale", type=float, default=10000.0)
    p.add_argument("--psf_sigma", type=float, default=1.0)
    p.add_argument("--spot_percent", type=float, default=0.001)
    p.add_argument("--bg_intensity", type=float, default=1.0)
    p.add_argument("--bg_seed", type=int, default=0)
    p.add_argument("--beam_sigma", type=float, default=30.0)
    p.add_argument("--beam_intensity", type=float, default=0.0)
    p.add_argument("--beam_seed", type=int, default=0)
    p.add_argument("--flip_y", action="store_true",
                   help="Reverse the slow (row) axis of each image")
    p.add_argument("--thickness", type=float, default=None,
                   help="Thickness (A) to write, for files holding "
                        "several. Default: one file per thickness.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    params = CbfParams(
        scale=args.scale, psf_sigma=args.psf_sigma,
        spot_percent=args.spot_percent, bg_intensity=args.bg_intensity,
        bg_seed=args.bg_seed, beam_sigma=args.beam_sigma,
        beam_intensity=args.beam_intensity, beam_seed=args.beam_seed,
        flip_y=args.flip_y,
    )

    records = {}
    thicknesses = None
    for path in args.npz_files:
        with np.load(path) as d:
            if "signal" not in d.files:
                print(f"Skipping {path}: not a per-image NPZ")
                continue
        rec = load_image(path)
        if thicknesses is None:
            thicknesses = rec["thicknesses_A"]
        elif not np.array_equal(rec["thicknesses_A"], thicknesses,
                                equal_nan=True):
            raise SystemExit(f"{path} holds different thicknesses "
                             "from the other files.")
        records[rec["image_index"]] = rec
    if not records:
        raise SystemExit("No per-image NPZ files among the inputs.")
    indices = sorted(records)
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise SystemExit("The NPZ files must be a contiguous block of "
                         f"images; got {indices[0]}..{indices[-1]} with gaps.")

    if args.thickness is not None:
        try:
            chosen = [thickness_index(thicknesses, args.thickness)]
        except ValueError as e:
            raise SystemExit(str(e))
    else:
        chosen = list(range(len(thicknesses)))

    geom = read_geometry(args.expt_file, args.experiment)
    for t in chosen:
        output = args.output
        if len(chosen) > 1:
            output = with_thickness_suffix(args.output, thicknesses[t])
        stack = np.stack([
            render_image(records[i]["signal"][t],
                         tuple(records[i]["beam_centre_px"]), params, i)
            for i in indices
        ])
        Q = write_nxmx(output, stack, geom, indices, probe=args.probe)
        print(f"{len(indices)} images ({indices[0]}..{indices[-1]}), "
              f"thickness {format_thickness(thicknesses[t])} A -> "
              f"{output}")

    tilt = np.degrees(np.arccos(np.clip((np.trace(Q) - 1) / 2, -1, 1)))
    if tilt > 1e-6:
        print(f"Beam tilt of {tilt * 1000 / 57.29578:.3f} mrad moved onto z "
              "(the whole geometry is rotated with it).")


if __name__ == "__main__":
    main()
