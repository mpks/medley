"""
optimise.py — choose n_substeps, k_max and sg_max for a given
experiment and thickness, step by step.

Settings
--------
  n_substeps  angular samples per image.
  k_max       structure-factor limit (1/A).  abTEM uses
              structure factors out to k_max but beams (and so
              spots) only out to k_max / 2: k_max is the DIAMETER
              of the disc of beams.  A spot at resolution d is
              simulated only if k_max >= 2 / d.
  sg_max      excitation-error limit for the beam pool (1/A).

Steps
-----
0. Geometry (from the experiment, no simulation).  The largest
   resolution ring that lies completely on the detector, d_ring,
   and the corners, d_corner.  Spots compared in all later steps
   are those with d >= d_cmp = max(d_ring, d_min), where d_min is
   a resolution cutoff: these spots are on the detector in every
   image and are simulated by every k_max tried.  k_max runs from
   k_min = 2 / d_cmp up to k_top (default 2 / d_min).

1. Angular sampling estimate (no Bloch runs).  A reflection's
   rocking curve is modelled kinematically,
       I(s) = sinc^2(t s),  s = g * (phi - phi0),
   cut at |s| > sg_max, with t the thickness of the thickest
   test image (base / cos tilt) and g = 1 / d_cmp, the outermost
   compared spot (narrowest curve).  The integral over one image
   computed from n substeps (placed as medley places them) is
   compared with the exact integral, for peaks at many positions
   in and around the image, and summarised as an R1.  The
   smallest n with R1 <= tol is the estimate.

2. Angular sampling check (Bloch).  n_est / 2, n_est and 2 n_est
   at k_min and sg_max_probe, each compared with the largest; the
   largest is doubled (up to twice) if n_est is not within tol.

3. k_max sweep (Bloch), n from step 2, sg_max_probe.  k_min to
   k_top in k_steps.  Each run is compared with the first (the
   convergence curve), the previous and the last.  A saturating
   curve R1 = R_inf (1 - exp(-(k - k_min) / kappa)) is fitted to
   "R1 vs first", as a guide to where the curve levels off.

4. sg_max sweep (Bloch), n and k_max from steps 2 and 3, over
   values below and above sg_max_probe; same table and fit.

In steps 2-4 the chosen value is the smallest whose R1 against
the most accurate run of that step is within tol.

R1 = sum |I_run - I_ref| / sum I_ref over the compared spots of
all test images (the direct beam 000 is left out; a spot missing
from one run counts as zero there).
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field

import numpy as np

from medley.engine import make_base_atoms, orient_atoms, run_image
from medley.projection import in_detector_bounds
from medley.simulator import BlochSimulator


@dataclass(frozen=True)
class OptimiseParams:
    """
    tol : float
        R1 allowed between a chosen value and the most accurate
        run of its step (also the target of the step 1 model).
    d_min : float
        Resolution cutoff (A): spots finer than this are not of
        interest, and k_max is not taken beyond 2 / d_min.
    k_max_top : float or None
        Largest k_max tried; None means 2 / d_min (or 1.5 x
        k_min if the detector does not reach d_min).
    k_steps : int
        Number of k_max values, evenly spaced from k_min.
    sg_max_probe : float
        sg_max used in steps 2 and 3.
    sg_max_values : tuple
        sg_max values tried in step 4.
    n_candidates : tuple
        n_substeps values the step 1 model chooses from.
    n_shells : int
        Resolution shells in the per-shell breakdown.
    """
    tol: float = 0.01
    d_min: float = 0.6
    k_max_top: float | None = None
    k_steps: int = 5
    sg_max_probe: float = 0.05
    sg_max_values: tuple = (0.0125, 0.025, 0.0375, 0.05, 0.075, 0.1)
    n_candidates: tuple = (4, 6, 8, 12, 16, 24, 32, 48, 64, 96,
                           128, 192, 256)
    n_shells: int = 5


# ----------------------------------------------------------------
# Step 0: geometry
# ----------------------------------------------------------------

def detector_resolution(detector, beam, n_per_side=64):
    """
    |g| (1/A) reaching the panel border, sampled along all four
    edges.  Returns (g_ring, g_corner):

      g_ring   : smallest |g| on the border, i.e. the largest
                 resolution ring that lies completely on the
                 detector
      g_corner : largest |g| on the border (a corner)

    |g| = |s1 - s0|, with s0 the incident and s1 the scattered
    wavevector, both 1/wavelength long.
    """
    d = np.column_stack([
        np.asarray(detector.fast_axis, dtype=float),
        np.asarray(detector.slow_axis, dtype=float),
        np.asarray(detector.origin, dtype=float),
    ])
    k0 = 1.0 / beam.wavelength_A
    direction = np.asarray(beam.direction, dtype=float)
    s0 = -k0 * direction / np.linalg.norm(direction)
    pix = detector.pixel_size_mm
    nx, ny = detector.npx, detector.npy

    t = np.linspace(0.0, 1.0, n_per_side + 1)
    border = (
        [(x * nx, 0.0) for x in t] + [(x * nx, ny) for x in t]
        + [(0.0, y * ny) for y in t] + [(nx, y * ny) for y in t]
    )
    g = []
    for px, py in border:
        pos = d @ np.array([px * pix, py * pix, 1.0])
        s1 = k0 * pos / np.linalg.norm(pos)
        g.append(float(np.linalg.norm(s1 - s0)))
    return min(g), max(g)


@dataclass(frozen=True)
class Plan:
    """What step 0 decides; all resolutions in A, k in 1/A."""
    d_ring: float
    d_corner: float
    d_cmp: float
    k_values: tuple
    g_cmp: float = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "g_cmp", 1.0 / self.d_cmp)

    @property
    def k_min(self):
        return self.k_values[0]


def make_plan(g_ring, g_corner, opt):
    """Spots to compare and k_max values to try (step 0)."""
    d_ring = 1.0 / g_ring
    d_cmp = max(d_ring, opt.d_min)
    k_min = 2.0 / d_cmp
    k_top = opt.k_max_top
    if k_top is None:
        k_top = 2.0 / opt.d_min if d_ring > opt.d_min else 1.5 * k_min
    k_top = max(k_top, k_min)
    k_values = sorted({round(float(k), 3) for k in
                       np.linspace(k_min, k_top, opt.k_steps)})
    return Plan(d_ring=d_ring, d_corner=1.0 / g_corner,
                d_cmp=d_cmp, k_values=tuple(k_values))


def default_test_images(scan, n=4):
    """
    n images: the one with the largest |tilt| (thickest crystal,
    most dynamical), the one closest to zero tilt, and the rest
    evenly spread over the scan.
    """
    centres = np.asarray(scan.angles_deg) + 0.5 * scan.delta_deg
    order = np.argsort(np.abs(centres))
    chosen = {int(order[-1]), int(order[0])}
    # fill in from evenly spaced positions, keeping the picks
    # at least half an even spacing apart
    gap = scan.n_images / (2.0 * n)
    for i in np.linspace(0, scan.n_images - 1, 2 * n + 1):
        if len(chosen) >= n:
            break
        i = int(round(i))
        if min(abs(i - c) for c in chosen) >= gap:
            chosen.add(i)
    return sorted(chosen)


def thickest(scan, test_images, base_thickness_A):
    """(tilt in degrees, thickness in A) of the thickest test image."""
    centres = (np.asarray(scan.angles_deg)[list(test_images)]
               + 0.5 * scan.delta_deg)
    tilt = float(np.max(np.abs(centres)))
    return tilt, base_thickness_A / math.cos(math.radians(tilt))


# ----------------------------------------------------------------
# Step 1: rocking-curve model of the angular sampling
# ----------------------------------------------------------------

def substep_offset(scan):
    """
    Where medley puts the substeps within their slice: 0.0 at
    the start of each slice (left edge), 0.5 in the middle.
    """
    two = dataclasses.replace(scan, n_substeps=2)
    first = float(two.substep_angles(0)[0])
    return (first - float(scan.angles_deg[0])) / (0.5 * scan.delta_deg)


def sampling_error(t, g, width_deg, n, offset=0.0, sg_max=0.05,
                   n_peaks=401, fine=40):
    """
    R1 between n-substep and exact integrals of one image, for
    a kinematic rocking curve I = sinc^2(t s), s = g (phi - phi0),
    cut at |s| > sg_max, with the peak phi0 at n_peaks positions
    from one image before to one image after this one (so
    partial spots at the image edges are included).
    """
    D = math.radians(width_deg)
    phi0 = np.linspace(-D, 2.0 * D, n_peaks)

    def intensity(phi):
        s = g * (phi[None, :] - phi0[:, None])
        out = np.sinc(t * s) ** 2
        out[np.abs(s) > sg_max] = 0.0
        return out

    width = 1.0 / (t * g)
    m = max(int(fine * D / width), 2000)
    exact = intensity((np.arange(m) + 0.5) * D / m).sum(1) * D / m
    sampled = intensity((np.arange(n) + offset) * D / n).sum(1) * D / n
    return float(np.abs(sampled - exact).sum() / exact.sum())


def estimate_substeps(t, g, width_deg, offset, sg_max, opt):
    """(n_est, {n: model R1}) for the n_candidates."""
    model = {}
    n_est = None
    for n in opt.n_candidates:
        model[n] = sampling_error(t, g, width_deg, n, offset, sg_max)
        if model[n] <= opt.tol:
            n_est = n
            break
    if n_est is None:
        n_est = opt.n_candidates[-1]
    return n_est, model


# ----------------------------------------------------------------
# Comparing runs
# ----------------------------------------------------------------

def is_direct_beam(key):
    """key = (image, h, k, l)."""
    return tuple(key[1:]) == (0, 0, 0)


def r1(run, ref, keep):
    """
    R1 = sum |I_run - I_ref| / sum I_ref over the spots keep(key)
    accepts, in either run (missing counts as zero).
    """
    keys = [k for k in set(run) | set(ref) if keep(k)]
    num = sum(abs(run.get(k, 0.0) - ref.get(k, 0.0)) for k in keys)
    den = sum(ref.get(k, 0.0) for k in keys)
    if den == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return num / den


def shell_r1(run, ref, keep, g_of, n_shells):
    """R1 per resolution shell (equal numbers of reference spots)."""
    keys = sorted((k for k in ref if keep(k)), key=g_of)
    rows = []
    if not keys:
        return rows
    for idx in np.array_split(np.arange(len(keys)),
                              min(n_shells, len(keys))):
        shell = [keys[i] for i in idx]
        inside = set(shell)
        rows.append({
            "d_from": 1.0 / g_of(shell[0]),
            "d_to": 1.0 / g_of(shell[-1]),
            "n_spots": len(shell),
            "r1": r1(run, ref, lambda k: k in inside),
        })
    return rows


def fit_saturation(x, r):
    """
    Fit R1 = R_inf (1 - exp(-(x - x0) / kappa)) to points that
    start at (x0, 0).  Returns (R_inf, kappa) or None.
    """
    x = np.asarray(x, float)
    r = np.asarray(r, float)
    if len(x) < 3 or r.max() <= 0:
        return None
    from scipy.optimize import curve_fit

    x0 = x[0]

    def model(xx, r_inf, kappa):
        return r_inf * (1.0 - np.exp(-(xx - x0) / kappa))

    try:
        (r_inf, kappa), _ = curve_fit(
            model, x, r, p0=(r.max(), (x[-1] - x0) / 2),
            bounds=([0.0, 1e-6], [np.inf, np.inf]), maxfev=10000,
        )
    except (RuntimeError, ValueError):
        return None
    return float(r_inf), float(kappa)


def fit_summary(x, r, tol):
    """
    Read the fitted curve: where it is within tol of its plateau,
    and how far the last value still is from it.
    """
    fit = fit_saturation(x, r)
    if fit is None:
        return None
    r_inf, kappa = fit
    x0 = x[0]
    if r_inf <= tol:
        x_tol = x0
    else:
        x_tol = x0 - kappa * math.log(tol / r_inf)
    remaining = r_inf * math.exp(-(x[-1] - x0) / kappa)
    return {"r_inf": r_inf, "kappa": kappa, "x_tol": x_tol,
            "remaining_at_last": remaining}


# ----------------------------------------------------------------
# Running the test images
# ----------------------------------------------------------------

def _simulator(bloch_params, beam, settings):
    params = dataclasses.replace(
        bloch_params,
        k_max=settings["k_max"],
        sg_max=settings["sg_max"],
    )
    return BlochSimulator(params, energy_eV=beam.energy_eV,
                          verbose=False)


def make_evaluator(cif_file, detector, beam, geometry, scan,
                   bloch_params, engine_params,
                   integration_params, test_images):
    """
    Return evaluate(settings) -> {(image, h, k, l): intensity}:
    each test image integrated over all its substeps.  Only
    spots whose centroid lands on the detector count.
    """
    def evaluate(settings):
        simulator = _simulator(bloch_params, beam, settings)
        test_scan = dataclasses.replace(
            scan, n_substeps=settings["n_substeps"]
        )
        out = {}
        for image_index in test_images:
            result, _rocking = run_image(
                cif_file, detector, beam, geometry, test_scan,
                image_index, simulator, engine_params,
                integration_params,
            )
            for hkl, px, py, intensity in zip(
                result.millers, result.px, result.py,
                result.intensities,
            ):
                if in_detector_bounds(px, py, detector.npx,
                                      detector.npy):
                    out[(image_index, *map(int, hkl))] = float(
                        intensity
                    )
        return out
    return evaluate


def make_beam_counter(cif_file, beam, geometry, scan,
                      bloch_params, image_index):
    """
    Return count(settings) -> number of beams at the centre of
    one image (no diffraction is calculated), or None if it
    cannot be worked out.
    """
    angle = scan.angles_deg[image_index] + 0.5 * scan.delta_deg

    def count(settings):
        try:
            base = make_base_atoms(cif_file, geometry)
            atoms = orient_atoms(base, geometry, angle, beam)
            simulator = _simulator(bloch_params, beam, settings)
            return len(simulator.build_bloch_waves(atoms))
        except Exception:
            return None
    return count


def g_length_function(B):
    """|g| = |B h| for a key (image, h, k, l)."""
    B = np.asarray(B, dtype=float)

    def g_of(key):
        return float(np.linalg.norm(B @ np.array(key[1:], float)))
    return g_of


class Runner:
    """Runs settings once each, timing them and counting beams."""

    def __init__(self, evaluate, n_test_images, count_beams=None,
                 out=print):
        self.evaluate = evaluate
        self.n_test_images = n_test_images
        self.count_beams = count_beams
        self.out = out
        self.cache = {}

    def run(self, settings):
        key = (settings["n_substeps"], settings["k_max"],
               settings["sg_max"])
        if key not in self.cache:
            beams = (self.count_beams(settings)
                     if self.count_beams else None)
            t0 = time.perf_counter()
            intensities = self.evaluate(settings)
            seconds = time.perf_counter() - t0
            self.cache[key] = {
                "settings": dict(settings),
                "intensities": intensities,
                "seconds_per_image": seconds / self.n_test_images,
                "beams": beams,
            }
            self.out(
                f"    ran n_substeps {settings['n_substeps']:<4d} "
                f"k_max {settings['k_max']:<6g} "
                f"sg_max {settings['sg_max']:<7g} "
                f"beams {beams if beams is not None else '?':<6} "
                f"{seconds:7.1f} s"
            )
        return self.cache[key]


# ----------------------------------------------------------------
# Steps 2-4
# ----------------------------------------------------------------

def _even(x):
    return max(2, 2 * int(round(x / 2.0)))


def check_substeps(runner, base, n_est, keep, tol, max_extra=2):
    """
    Step 2: n_est / 2, n_est, 2 n_est, compared with the largest;
    the largest is doubled while no smaller value is within tol
    (at most max_extra times).  Returns (chosen, rows, converged).
    """
    values = sorted({_even(n_est / 2), _even(n_est), _even(2 * n_est)})
    for attempt in range(max_extra + 1):
        runs = {n: runner.run({**base, "n_substeps": n})
                for n in values}
        top = runs[values[-1]]["intensities"]
        within = [n for n in values[:-1]
                  if r1(runs[n]["intensities"], top, keep) <= tol]
        if within:
            break
        if attempt == max_extra:
            break
        values.append(2 * values[-1])
    rows = [{
        "n_substeps": n,
        "beams": runs[n]["beams"],
        "seconds_per_image": runs[n]["seconds_per_image"],
        "r1_vs_last": r1(runs[n]["intensities"], top, keep),
    } for n in values]
    if within:
        return min(within), rows, True
    return values[-1], rows, False


def sweep(runner, name, values, base, keep, tol):
    """
    Steps 3 and 4: run each value of one setting with the others
    fixed, compare with the first, previous and last run, and
    fit the convergence curve.  Returns (chosen, rows, fit).
    """
    values = list(values)
    runs = [runner.run({**base, name: v}) for v in values]
    first = runs[0]["intensities"]
    last = runs[-1]["intensities"]
    rows = []
    for i, (v, run) in enumerate(zip(values, runs)):
        I = run["intensities"]
        rows.append({
            name: v,
            "beams": run["beams"],
            "seconds_per_image": run["seconds_per_image"],
            "r1_vs_first": r1(I, first, keep),
            "r1_vs_previous": (r1(I, runs[i - 1]["intensities"], keep)
                               if i else None),
            "r1_vs_last": r1(I, last, keep),
        })
    chosen = min(r[name] for r in rows if r["r1_vs_last"] <= tol)
    fit = fit_summary(values, [r["r1_vs_first"] for r in rows], tol)
    return chosen, rows, fit


# ----------------------------------------------------------------
# All steps
# ----------------------------------------------------------------

def optimise(runner, plan, opt, g_of, t_max, tilt_max, width_deg,
             offset, n_images, n_estimate=None, out=print):
    """
    Run steps 1-4 and return the report dict.  Prints each step's
    table as soon as it is done.
    """
    def keep(key):
        return (not is_direct_beam(key)
                and g_of(key) <= plan.g_cmp * (1 + 1e-9))

    tol = opt.tol
    report = {"tol": tol, "d_min": opt.d_min,
              "d_ring": plan.d_ring, "d_corner": plan.d_corner,
              "d_compared": plan.d_cmp, "k_values": list(plan.k_values),
              "thickest_tilt_deg": tilt_max, "thickness_A": t_max,
              "image_width_deg": width_deg, "substep_offset": offset}

    # Step 1
    out("")
    out("Step 1  Angular sampling estimate (rocking-curve model, "
        "no Bloch runs)")
    rock_deg = math.degrees(1.0 / (t_max * plan.g_cmp))
    rule = ("start of each slice" if offset < 0.25
            else "middle of each slice")
    out(f"  thickest test image: tilt {tilt_max:.1f} deg, "
        f"t = {t_max:.0f} A;  outermost compared spot d = "
        f"{plan.d_cmp:.3f} A")
    out(f"  rocking curve width ~ 1/(t g) = {rock_deg:.4f} deg;  "
        f"image width {width_deg:g} deg = "
        f"{width_deg / rock_deg:.1f} rocking widths")
    out(f"  substeps are placed at the {rule}")
    n_est, model = estimate_substeps(
        t_max, plan.g_cmp, width_deg, offset, opt.sg_max_probe, opt
    )
    out("  " + f"{'n_substeps':<12}" + "".join(
        f"{n:>8d}" for n in model))
    out("  " + f"{'model R1':<12}" + "".join(
        f"{100 * e:>7.2f}%" for e in model.values()))
    if n_estimate is not None:
        out(f"  estimate {n_est}; using {n_estimate} as requested")
        n_est = n_estimate
    else:
        out(f"  estimate: n_substeps = {n_est}")
    if offset < 0.25:
        mid = {n: sampling_error(t_max, plan.g_cmp, width_deg, n,
                                 0.5, opt.sg_max_probe)
               for n in (8, 16, 32)}
        out("  note: with substeps in the middle of each slice the "
            "model gives " + ", ".join(
                f"{100 * e:.2f}% at {n}" for n, e in mid.items()))
    report["step1"] = {"n_estimate": n_est,
                       "model_r1": {str(n): e for n, e in model.items()},
                       "rocking_width_deg": rock_deg}

    # Step 2
    base = {"k_max": plan.k_min, "sg_max": opt.sg_max_probe}
    out("")
    out(f"Step 2  Angular sampling check (Bloch; k_max {plan.k_min:g}, "
        f"sg_max {opt.sg_max_probe:g})")
    n_chosen, rows2, ok2 = check_substeps(runner, base, n_est, keep, tol)
    top = rows2[-1]["n_substeps"]
    out(f"  {'n_substeps':>10}{'beams':>8}{'s/image':>9}"
        f"{'R1 vs n=' + str(top):>14}{'model R1 vs exact':>19}")
    for r in rows2:
        m = sampling_error(t_max, plan.g_cmp, width_deg,
                           r["n_substeps"], offset, opt.sg_max_probe)
        r["model_r1"] = m
        out(f"  {r['n_substeps']:>10d}{_beams(r):>8}"
            f"{r['seconds_per_image']:>9.1f}"
            f"{_pct(r['r1_vs_last']):>14}{_pct(m):>19}")
    out(f"  chosen: n_substeps = {n_chosen}"
        + ("" if ok2 else "  (WARNING: no smaller value was within "
           "tol; this is the largest tried)"))
    report["step2"] = {"chosen": n_chosen, "converged": ok2,
                       "rows": rows2}

    # Step 3
    base = {"n_substeps": n_chosen, "sg_max": opt.sg_max_probe}
    out("")
    out(f"Step 3  k_max (Bloch; n_substeps {n_chosen}, sg_max "
        f"{opt.sg_max_probe:g}); R1 on spots with d >= "
        f"{plan.d_cmp:.3f} A")
    k_chosen, rows3, fit3 = sweep(runner, "k_max", plan.k_values,
                                  base, keep, tol)
    _print_sweep(out, "k_max", rows3, fit3, tol,
                 extra=("beams reach", lambda v: f"d={2.0 / v:.3f}"))
    out(f"  chosen: k_max = {k_chosen:g} (beams reach d = "
        f"{2.0 / k_chosen:.3f} A)")
    report["step3"] = {"chosen": k_chosen, "rows": rows3, "fit": fit3}

    # Step 4
    base = {"n_substeps": n_chosen, "k_max": k_chosen}
    out("")
    out(f"Step 4  sg_max (Bloch; n_substeps {n_chosen}, k_max "
        f"{k_chosen:g})")
    sg_chosen, rows4, fit4 = sweep(runner, "sg_max",
                                   sorted(opt.sg_max_values), base,
                                   keep, tol)
    _print_sweep(out, "sg_max", rows4, fit4, tol)
    out(f"  chosen: sg_max = {sg_chosen:g}")
    report["step4"] = {"chosen": sg_chosen, "rows": rows4, "fit": fit4}

    # Result
    final = runner.run({"n_substeps": n_chosen, "k_max": k_chosen,
                        "sg_max": sg_chosen})
    best = runner.run({"n_substeps": n_chosen, "k_max": k_chosen,
                       "sg_max": max(opt.sg_max_values)})
    shells = shell_r1(final["intensities"], best["intensities"], keep,
                      g_of, opt.n_shells)
    hours = final["seconds_per_image"] * n_images / 3600
    out("")
    out(f"Result: n_substeps {n_chosen}, k_max {k_chosen:g}, "
        f"sg_max {sg_chosen:g}")
    out(f"  beams {_beams(final)}, {final['seconds_per_image']:.1f} s "
        f"per image -> about {hours:.2f} h for {n_images} images")
    if shells:
        out(f"  R1 per resolution shell, chosen vs sg_max "
            f"{max(opt.sg_max_values):g}:")
        out(f"  {'d range (A)':<18}{'spots':>7}{'R1':>9}")
        for s in shells:
            d_range = f"{s['d_from']:.3f} - {s['d_to']:.3f}"
            out(f"  {d_range:<18}{s['n_spots']:>7d}{_pct(s['r1']):>9}")
    report["settings"] = {"n_substeps": n_chosen, "k_max": k_chosen,
                          "sg_max": sg_chosen}
    report["shells"] = shells
    report["seconds_per_image"] = final["seconds_per_image"]
    report["estimated_full_run_hours"] = hours
    report["n_compared_spots"] = sum(
        1 for k in best["intensities"] if keep(k))
    return report


def _pct(x):
    return "-" if x is None else f"{100 * x:.2f}%"


def _beams(r):
    return "?" if r.get("beams") is None else str(r["beams"])


def _print_sweep(out, name, rows, fit, tol, extra=None):
    last = rows[-1][name]
    head = f"  {name:>8}"
    if extra:
        head += f"{extra[0]:>13}"
    head += (f"{'beams':>8}{'s/image':>9}{'R1 vs first':>13}"
             f"{'vs previous':>13}{'vs ' + format(last, 'g'):>11}")
    out(head)
    for r in rows:
        line = f"  {r[name]:>8g}"
        if extra:
            line += f"{extra[1](r[name]):>13}"
        line += (f"{_beams(r):>8}{r['seconds_per_image']:>9.1f}"
                 f"{_pct(r['r1_vs_first']):>13}"
                 f"{_pct(r['r1_vs_previous']):>13}"
                 f"{_pct(r['r1_vs_last']):>11}")
        out(line)
    if fit is None:
        out("  fit: not enough points or no change to fit")
        return
    out(f"  fit of 'R1 vs first': levels off at "
        f"{_pct(fit['r_inf'])}; within {_pct(tol)} of that from "
        f"{name} ~ {fit['x_tol']:.3g}")
    if fit["remaining_at_last"] > tol:
        out(f"  WARNING: the fit says {name} = {last:g} is still "
            f"{_pct(fit['remaining_at_last'])} from the plateau; "
            f"the largest value tried may not be converged")
