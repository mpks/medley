"""
cli.py — command-line entry point

Two modes:

  experiment : geometry, detector and beam are read from a
               DIALS .expt file; a CIF supplies the atoms.
  synthetic  : geometry is generated from a random orientation
               seed, the detector distance from a desired
               corner resolution (or given directly), and the
               scan from start/delta/n_images.

Both modes build the same domain objects, then run the engine
with a BlochSimulator and export NPZ, CBF and plots.
"""

from __future__ import annotations

import argparse

from medley.simulation.experiment import load_experiment
from medley.simulation.experiment import Scan
from medley.simulation.synthetic_experiment import (
    build_synthetic_calibrated,
)
from medley.simulation.simulator import (
    BlochParams,
    BlochSimulator,
)
from medley.simulation.engine import EngineParams
from medley.simulation.integration import IntegrationParams
from medley.simulation.miller import read_miller_indices
from medley.simulation.export_cbf import CbfParams
from medley.simulation.driver import run as driver_run


# ----------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------

DEFAULTS = {
    "wavelength_A": 0.02851,        # 160 kV
    "npx": 512,
    "npy": 512,
    "pixel_size_mm": 0.075,
    "distance_mm": 578.3,
    "k_max": 2.0,
    "sg_max": 0.05,
    "num_phonon_configs": 1,
    "phonon_sigma": 0.0,
    "phonon_seed": 42,
    "base_thickness_A": 800.0,      # 80 nm
    "n_substeps": 10,
    "intensity_cut": 0.0,
    "n_workers": 1,
    # integration (simulation-time)
    "integration": "vector",
    "psf_sigma": 1.0,
    "spot_percent": 0.001,
    # CBF render (changeable without re-simulating)
    "scale": 10000.0,
    "bg_intensity": 1.0,
    "bg_seed": 0,
    "beam_sigma": 30.0,
    "beam_intensity": 0.0,
    "beam_seed": 0,
    # synthetic scan
    "start_deg": -30.0,
    "delta_deg": 0.5,
    "n_images": 60,
    "orientation_seed": 0,
}


# ----------------------------------------------------------------
# Argument parser
# ----------------------------------------------------------------

def build_parser():
    d = DEFAULTS

    p = argparse.ArgumentParser(
        description=(
            "Simulate electron diffraction by the Bloch-wave "
            "method, from a DIALS experiment or a synthetic "
            "random orientation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # -- common options shared by both modes -----------------
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("cif_file", help="CIF structure file")
    common.add_argument(
        "-o", "--out_dir", default="sim_out",
        help="Output directory",
    )
    common.add_argument(
        "--tag", default="kmax",
        help="Run label used in output filenames",
    )
    common.add_argument(
        "--k_max", type=float, default=d["k_max"],
    )
    common.add_argument(
        "--sg_max", type=float, default=d["sg_max"],
    )
    common.add_argument(
        "--num_phonon_configs", type=int,
        default=d["num_phonon_configs"],
    )
    common.add_argument(
        "--phonon_sigma", type=float,
        default=d["phonon_sigma"],
    )
    common.add_argument(
        "--phonon_seed", type=int,
        default=d["phonon_seed"],
    )
    common.add_argument(
        "--base_thickness_A", type=float,
        default=d["base_thickness_A"],
        help="Crystal thickness along the beam at zero tilt",
    )
    common.add_argument(
        "--n_substeps", type=int, default=d["n_substeps"],
    )
    common.add_argument(
        "--intensity_cut", type=float,
        default=d["intensity_cut"],
    )
    common.add_argument(
        "--n_workers", type=int, default=d["n_workers"],
    )
    common.add_argument(
        "--rocking_hkl_file", default=None,
        help=(
            "File of Miller indices (HKL or 3-column) to "
            "record rocking curves for"
        ),
    )
    common.add_argument(
        "--integration", choices=["vector", "raster"],
        default=d["integration"],
        help="Substep integration method",
    )
    common.add_argument(
        "--psf_sigma", type=float, default=d["psf_sigma"],
        help="PSF sigma (pixels) for the spot-noise image",
    )
    common.add_argument(
        "--spot_percent", type=float,
        default=d["spot_percent"],
        help=(
            "Spot-level noise as a fraction of spot intensity "
            "(e.g. 0.001 = 0.1%%)"
        ),
    )
    common.add_argument(
        "--scale", type=float, default=d["scale"],
        help="Counts at the strongest signal pixel",
    )
    common.add_argument(
        "--bg_intensity", type=float,
        default=d["bg_intensity"],
        help="Poisson background level (CBF render time)",
    )
    common.add_argument(
        "--bg_seed", type=int, default=d["bg_seed"],
        help="Master seed for the background field",
    )
    common.add_argument(
        "--beam_sigma", type=float, default=d["beam_sigma"],
        help="Direct-beam blob sigma (pixels)",
    )
    common.add_argument(
        "--beam_intensity", type=float,
        default=d["beam_intensity"],
        help="Direct-beam blob peak counts (0 disables)",
    )
    common.add_argument(
        "--beam_seed", type=int, default=d["beam_seed"],
        help="Master seed for the direct-beam blob",
    )
    common.add_argument(
        "--images", default=None,
        help=(
            "Images to simulate: a single index, a range "
            "'a-b', a list 'a,b,c', or combinations "
            "'0-9,20-29'. Default: all images."
        ),
    )
    common.add_argument(
        "--export_SA", action="store_true",
        help=(
            "Write a separate matrices NPZ per simulated "
            "image, holding the structure matrix A, the "
            "scattering matrix S(t), the projector diagonal "
            "M and the beam-pool Miller indices for one "
            "substep of that image"
        ),
    )
    common.add_argument(
        "--export_SA_substep", type=int, default=None,
        help=(
            "Substep index used by --export_SA; negative "
            "indices count from the end. Default: the middle "
            "substep of the image."
        ),
    )
    common.add_argument(
        "--no_sg", action="store_true",
        help=(
            "Skip the excitation-error export (sg and "
            "sg_geom) from the per-image NPZ"
        ),
    )
    common.add_argument(
        "--no_cbf", action="store_true",
        help="Skip writing CBF images",
    )
    common.add_argument(
        "--no_plots", action="store_true",
        help="Skip writing diagnostic plots",
    )

    # -- experiment mode -------------------------------------
    pe = sub.add_parser(
        "experiment", parents=[common],
        help="Read geometry from a DIALS .expt file",
    )
    pe.add_argument("expt_file", help="DIALS .expt file")

    # -- synthetic mode --------------------------------------
    ps = sub.add_parser(
        "synthetic", parents=[common],
        help="Generate a random synthetic experiment",
    )
    ps.add_argument(
        "--wavelength_A", type=float,
        default=d["wavelength_A"],
    )
    ps.add_argument("--npx", type=int, default=d["npx"])
    ps.add_argument("--npy", type=int, default=d["npy"])
    ps.add_argument(
        "--pixel_size_mm", type=float,
        default=d["pixel_size_mm"],
    )
    ps.add_argument(
        "--start_deg", type=float, default=d["start_deg"],
    )
    ps.add_argument(
        "--delta_deg", type=float, default=d["delta_deg"],
    )
    ps.add_argument(
        "--n_images", type=int, default=d["n_images"],
    )
    ps.add_argument(
        "--orientation_seed", type=int,
        default=d["orientation_seed"],
    )
    # distance / resolution: at most one
    ps.add_argument(
        "--distance_mm", type=float, default=None,
        help="Detector distance (mm). Overrides resolution.",
    )
    ps.add_argument(
        "--d_min", type=float, default=None,
        help="Desired corner resolution (A).",
    )
    ps.add_argument(
        "--g_max", type=float, default=None,
        help="Desired corner g_max (A^-1).",
    )

    return p


# ----------------------------------------------------------------
# Setup helpers
# ----------------------------------------------------------------

def _make_scan(angles, n_substeps):
    return Scan(angles_deg=angles, n_substeps=n_substeps)


def _load_rocking_hkl(path):
    if path is None:
        return ()
    return tuple(read_miller_indices(path))


def _setup_experiment(args):
    """Build domain objects for experiment mode."""
    detector, beam, geometry, angles = load_experiment(
        args.expt_file
    )
    scan = _make_scan(angles, args.n_substeps)
    return detector, beam, geometry, scan


def _setup_synthetic(args):
    """Build domain objects for synthetic mode."""
    dist = args.distance_mm
    d_min = args.d_min
    g_max = args.g_max
    n_given = sum(
        v is not None for v in (dist, d_min, g_max)
    )
    if n_given == 0:
        dist = DEFAULTS["distance_mm"]
    elif n_given > 1:
        raise SystemExit(
            "Give at most one of --distance_mm, --d_min, "
            "--g_max."
        )

    detector, beam, geometry, angles = (
        build_synthetic_calibrated(
            cif_file=args.cif_file,
            wavelength_A=args.wavelength_A,
            npx=args.npx, npy=args.npy,
            px_x_mm=args.pixel_size_mm,
            px_y_mm=args.pixel_size_mm,
            start_deg=args.start_deg,
            delta_deg=args.delta_deg,
            n_images=args.n_images,
            orientation_seed=args.orientation_seed,
            g_max=g_max, d_min=d_min, distance_mm=dist,
        )
    )
    scan = _make_scan(angles, args.n_substeps)
    return detector, beam, geometry, scan


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.mode == "experiment":
        detector, beam, geometry, scan = _setup_experiment(
            args
        )
    else:
        detector, beam, geometry, scan = _setup_synthetic(
            args
        )

    rocking_hkl = _load_rocking_hkl(args.rocking_hkl_file)

    bloch_params = BlochParams(
        k_max=args.k_max,
        sg_max=args.sg_max,
        num_phonon_configs=args.num_phonon_configs,
        phonon_sigma=args.phonon_sigma,
        phonon_seed=args.phonon_seed,
    )
    simulator = BlochSimulator(
        bloch_params, energy_eV=beam.energy_eV
    )

    engine_params = EngineParams(
        base_thickness_A=args.base_thickness_A,
        intensity_cut=args.intensity_cut,
        n_workers=args.n_workers,
        rocking_hkl=rocking_hkl,
    )
    integration_params = IntegrationParams(
        method=args.integration,
    )
    cbf_params = CbfParams(
        scale=args.scale,
        psf_sigma=args.psf_sigma,
        spot_percent=args.spot_percent,
        bg_intensity=args.bg_intensity,
        bg_seed=args.bg_seed,
        beam_sigma=args.beam_sigma,
        beam_intensity=args.beam_intensity,
        beam_seed=args.beam_seed,
    )

    print(f"Running {args.mode} scan: "
          f"{scan.n_images} images total, "
          f"{scan.n_substeps} substeps each "
          f"({args.integration} integration)")

    driver_run(
        cif_file=args.cif_file,
        detector=detector,
        beam=beam,
        geometry=geometry,
        scan=scan,
        simulator=simulator,
        engine_params=engine_params,
        integration_params=integration_params,
        cbf_params=cbf_params,
        out_dir=args.out_dir,
        tag=args.tag,
        image_selection=args.images,
        write_cbf=not args.no_cbf,
        write_plots=not args.no_plots,
        export_SA=args.export_SA,
        export_SA_substep=args.export_SA_substep,
        export_sg=not args.no_sg,
    )

    print("Done.")


if __name__ == "__main__":
    main()
