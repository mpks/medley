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
import dataclasses
from pathlib import Path

from medley.experiment import load_experiment
from medley.experiment import Scan
from medley.synthetic_experiment import (
    build_synthetic_calibrated,
)
from medley.simulator import (
    BlochParams,
    BlochSimulator,
)
from medley.engine import EngineParams
from medley.integration import IntegrationParams
from medley.miller import read_miller_indices
from medley.export_cbf import CbfParams
from medley.driver import run as driver_run
from medley.params import (
    MODES,
    default_params,
    load_params,
    params_to_argv,
    save_used_params,
    write_params,
)
from medley.optimise import (
    OptimiseParams,
    Runner,
    default_test_images,
    detector_resolution,
    g_length_function,
    make_beam_counter,
    make_evaluator,
    make_plan,
    optimise,
    substep_offset,
    thickest,
)
from medley.driver import parse_image_selection


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
    "base_thickness_A": [800.0],    # 80 nm; several values allowed
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
        "--base_thickness_A", type=float, nargs="+",
        default=d["base_thickness_A"],
        help=(
            "Crystal thickness(es) along the beam at zero tilt "
            "(A). Several values are simulated in one run, at "
            "little extra cost, and stored together in each "
            "image's NPZ."
        ),
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
        "--cbf", action="store_true",
        help=(
            "Also write CBF images during the run (first "
            "thickness only). They can be made later from the "
            "NPZ files with npz_to_cbf or npz_to_nexus."
        ),
    )
    common.add_argument(
        "--plots", action="store_true",
        help=(
            "Also write diagnostic plots during the run (first "
            "thickness only; slow on large detectors)"
        ),
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

    # -- params file -----------------------------------------
    pp = sub.add_parser(
        "params",
        help="Write a default parameter file (JSON)",
    )
    pp.add_argument(
        "params_mode", choices=MODES,
        help="Mode the parameter file is for",
    )
    pp.add_argument(
        "-o", "--output", default="params.json",
        help="Where to write the parameter file",
    )
    pp.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing file",
    )

    # -- run from params file --------------------------------
    pr = sub.add_parser(
        "run",
        help=(
            "Run from a parameter file; options given after "
            "the file override it"
        ),
    )
    pr.add_argument("params_file", help="JSON parameter file")
    pr.add_argument(
        "overrides", nargs=argparse.REMAINDER,
        help="Options overriding the file, e.g. --k_max 3",
    )

    # -- optimise settings for a params file ----------------
    o = OptimiseParams()
    po = sub.add_parser(
        "optimise", aliases=["optimize"],
        help=(
            "Choose n_substeps, k_max and sg_max for the "
            "experiment and thickness in a params file"
        ),
        description=(
            "Step 0 reads the detector resolution; step 1 "
            "estimates n_substeps from a rocking-curve model; "
            "steps 2-4 check n_substeps and sweep k_max and "
            "sg_max with Bloch runs on a few test images, "
            "comparing R1 of the spots inside the complete "
            "resolution ring. The n_substeps, k_max and sg_max "
            "in the params file are not used; everything else "
            "is (n_workers, ...; of several thicknesses, the "
            "thickest)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    po.add_argument("params_file", help="JSON parameter file")
    po.add_argument(
        "-o", "--output", default="params_optimised.json",
        help=(
            "Params file to write with the chosen settings; a "
            "report goes next to it as <name>_report.json"
        ),
    )
    po.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output files",
    )
    po.add_argument(
        "--tol", type=float, default=o.tol,
        help="R1 allowed against the most accurate run of a step",
    )
    po.add_argument(
        "--d_min", type=float, default=o.d_min,
        help=(
            "Resolution cutoff (A): k_max is not taken beyond "
            "2 / d_min, and finer spots are not compared"
        ),
    )
    po.add_argument(
        "--k_max_top", type=float, default=None,
        help="Largest k_max tried. Default: 2 / d_min.",
    )
    po.add_argument(
        "--k_steps", type=int, default=o.k_steps,
        help="Number of k_max values tried",
    )
    po.add_argument(
        "--sg_max_probe", type=float, default=o.sg_max_probe,
        help="sg_max used while n_substeps and k_max are chosen",
    )
    po.add_argument(
        "--sg_max_values",
        default=",".join(f"{v:g}" for v in o.sg_max_values),
        help="Comma-separated sg_max values tried in step 4",
    )
    po.add_argument(
        "--n_substeps", type=int, default=None,
        help="Skip the step 1 estimate and check this value",
    )
    po.add_argument(
        "--test_images", default=None,
        help=(
            "Images to test on, same syntax as --images. "
            "Default: see --n_test_images."
        ),
    )
    po.add_argument(
        "--n_test_images", type=int, default=4,
        help=(
            "Number of integrated test images (step 2 and the "
            "final timing) when --test_images is not given: the "
            "most tilted, the one nearest zero tilt, and the "
            "rest spread over the scan"
        ),
    )
    po.add_argument(
        "--n_orientations", type=int, default=o.n_orientations,
        help=(
            "Single orientations compared in steps 3 and 4 "
            "(k_max, sg_max): one substep at the centre of this "
            "many images, chosen as for --n_test_images"
        ),
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

def _build_simulation(args):
    """
    Build everything a simulation needs from parsed run args:
    (detector, beam, geometry, scan, bloch_params,
     engine_params, integration_params).
    """
    if args.mode == "experiment":
        detector, beam, geometry, scan = _setup_experiment(
            args
        )
    else:
        detector, beam, geometry, scan = _setup_synthetic(
            args
        )

    bloch_params = BlochParams(
        k_max=args.k_max,
        sg_max=args.sg_max,
        num_phonon_configs=args.num_phonon_configs,
        phonon_sigma=args.phonon_sigma,
        phonon_seed=args.phonon_seed,
    )
    engine_params = EngineParams(
        base_thickness_A=args.base_thickness_A,
        intensity_cut=args.intensity_cut,
        n_workers=args.n_workers,
        rocking_hkl=_load_rocking_hkl(args.rocking_hkl_file),
    )
    integration_params = IntegrationParams(
        method=args.integration,
    )
    return (detector, beam, geometry, scan, bloch_params,
            engine_params, integration_params)


def _value_list(text, kind):
    """'0.01,0.02' -> [0.01, 0.02] (kind = float or int)."""
    return [kind(v) for v in text.split(",") if v.strip()]


def _optimise(parser, args):
    """medley optimise: choose n_substeps, k_max and sg_max."""
    try:
        params = load_params(args.params_file)
        run_args = parser.parse_args(
            params_to_argv(parser, params)
        )
    except (OSError, ValueError) as e:
        raise SystemExit(
            f"medley optimise: {args.params_file}: {e}"
        )

    (detector, beam, geometry, scan, bloch_params,
     engine_params, integration_params) = _build_simulation(
        run_args
    )
    # Tune for the thickest crystal: it needs the finest angular
    # sampling and the most beams, so its settings also suit the
    # thinner ones.
    t_base = max(engine_params.base_thickness_A)
    engine_params = dataclasses.replace(
        engine_params, base_thickness_A=(t_base,)
    )

    if args.test_images is None:
        test_images = default_test_images(scan, args.n_test_images)
    else:
        test_images = parse_image_selection(
            args.test_images, scan.n_images
        )

    opt = OptimiseParams(
        tol=args.tol,
        d_min=args.d_min,
        k_max_top=args.k_max_top,
        k_steps=args.k_steps,
        sg_max_probe=args.sg_max_probe,
        sg_max_values=tuple(_value_list(args.sg_max_values, float)),
        n_orientations=args.n_orientations,
    )
    orientations = default_test_images(scan, args.n_orientations)
    g_ring, g_corner = detector_resolution(detector, beam)
    plan = make_plan(g_ring, g_corner, opt)
    tilt, t_max = thickest(scan, test_images, t_base)

    print(f"Medley optimise: thickness {t_base:g} A "
          f"at zero tilt, {scan.n_images} images of "
          f"{scan.delta_deg:g} deg, tolerance R1 <= {100 * opt.tol:g}%")
    print(f"  integrated test images {test_images}")
    print(f"  single orientations: centres of images {orientations}")
    print()
    print("Step 0  Geometry (from the experiment; k_max is the "
          "diameter of the disc of beams, so spots reach d = "
          "2 / k_max)")
    print(f"  {'':<28}{'d (A)':>8}{'k_max to reach it':>20}")
    for name, d in (("complete ring on detector", plan.d_ring),
                    ("detector corners", plan.d_corner),
                    ("resolution cutoff --d_min", opt.d_min)):
        print(f"  {name:<28}{d:>8.3f}{2.0 / d:>20.2f}")
    print(f"  spots compared: d >= {plan.d_cmp:.3f} A (present in "
          f"every run)")
    print("  k_max values for step 3: "
          + ", ".join(f"{k:g}" for k in plan.k_values))

    runner = Runner(
        make_evaluator(
            run_args.cif_file, detector, beam, geometry, scan,
            bloch_params, engine_params, integration_params,
            test_images,
        ),
        len(test_images),
        count_beams=make_beam_counter(
            run_args.cif_file, beam, geometry, scan, bloch_params,
            test_images[0],
        ),
    )
    orient_runner = Runner(
        make_evaluator(
            run_args.cif_file, detector, beam, geometry, scan,
            bloch_params, engine_params, integration_params,
            orientations,
        ),
        len(orientations),
        count_beams=make_beam_counter(
            run_args.cif_file, beam, geometry, scan, bloch_params,
            orientations[0],
        ),
    )
    report = optimise(
        runner, plan, opt, g_length_function(geometry.base_B()),
        t_max, tilt, scan.delta_deg, substep_offset(scan),
        scan.n_images, n_estimate=args.n_substeps,
        orient_runner=orient_runner,
    )
    report["test_images"] = test_images
    report["orientations"] = orientations

    tuned = dict(params)
    tuned.update(report["settings"])
    out = Path(args.output)
    report_path = out.with_name(out.stem + "_report.json")
    try:
        write_params(tuned, out, overwrite=args.force)
        write_params(report, report_path, overwrite=args.force)
    except FileExistsError as e:
        raise SystemExit(f"medley optimise: {e}")
    print()
    print(f"Wrote {out} and {report_path}. Run with: "
          f"medley run {out}")


def _write_default_params(parser, args):
    params = default_params(parser, args.params_mode)
    try:
        path = write_params(
            params, args.output, overwrite=args.force
        )
    except FileExistsError as e:
        raise SystemExit(f"medley params: {e}")
    print(f"Wrote {path}. Set the file paths, edit as "
          f"needed, then: medley run {path}")


def _args_from_params_file(parser, args):
    try:
        params = load_params(args.params_file)
        argv = params_to_argv(parser, params)
    except (OSError, ValueError) as e:
        raise SystemExit(f"medley run: {args.params_file}: {e}")
    return parser.parse_args(argv + args.overrides)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode == "params":
        _write_default_params(parser, args)
        return
    if args.mode in ("optimise", "optimize"):
        _optimise(parser, args)
        return
    if args.mode == "run":
        args = _args_from_params_file(parser, args)

    (detector, beam, geometry, scan, bloch_params,
     engine_params, integration_params) = _build_simulation(args)
    simulator = BlochSimulator(
        bloch_params, energy_eV=beam.energy_eV
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
          f"({args.integration} integration), thickness "
          + ", ".join(f"{t:g}" for t in engine_params.base_thickness_A)
          + " A")

    saved = save_used_params(args, args.out_dir)
    print(f"Parameters saved to {saved}")

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
        write_cbf=args.cbf,
        write_plots=args.plots,
        export_SA=args.export_SA,
        export_SA_substep=args.export_SA_substep,
        export_sg=not args.no_sg,
    )

    print("Done.")


if __name__ == "__main__":
    main()
