import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

import medley.optimise as om
from medley.engine import ImageResult
from medley.experiment import Beam, Detector, Scan
from medley.optimise import (
    OptimiseParams,
    check_substeps,
    default_test_images,
    detector_resolution,
    estimate_substeps,
    fit_saturation,
    fit_summary,
    make_evaluator,
    make_plan,
    r1,
    sampling_error,
    shell_r1,
    substep_offset,
    sweep,
    thickest,
)
from medley.simulator import BlochParams


class FakeRunner:
    """Runner stand-in: intensities from a function of settings."""

    def __init__(self, intensities):
        self.intensities = intensities
        self.calls = []

    def run(self, settings):
        self.calls.append(dict(settings))
        return {"settings": dict(settings),
                "intensities": self.intensities(settings),
                "seconds_per_image": 1.0, "beams": 100}


def keep_all(key):
    return tuple(key[1:]) != (0, 0, 0)


def one_thickness_result(millers, px, py, intensities, n=10):
    """An ImageResult with a single thickness, for fake engines."""
    return ImageResult(
        millers=[list(m) for m in millers], thicknesses_A=(800.0,),
        px=np.asarray([px], float), py=np.asarray([py], float),
        intensities=np.asarray([intensities], float),
        signal=np.zeros((1, n, n)), method="vector",
        angle_centre_deg=0.0, image_index=0,
    )


# ----------------------------------------------------------------
# Step 0
# ----------------------------------------------------------------

def test_detector_resolution_flat_centred_panel():
    """Ring at the edge midpoints, maximum at the corners."""
    wl, L, pix, n = 0.02508, 500.0, 0.075, 512
    c = n / 2
    det = Detector(
        distance_mm=L, npx=n, npy=n, pixel_size_mm=pix,
        beam_centre_px=(c, c),
        fast_axis=np.array([1.0, 0.0, 0.0]),
        slow_axis=np.array([0.0, 1.0, 0.0]),
        origin=np.array([-c * pix, -c * pix, -L]),
    )
    beam = Beam(wavelength_A=wl, energy_eV=200e3,
                direction=np.array([0.0, 0.0, 1.0]))

    def g_at(r_mm):
        return 2.0 * np.sin(np.arctan(r_mm / L) / 2.0) / wl

    g_ring, g_corner = detector_resolution(det, beam)
    assert g_ring == pytest.approx(g_at(c * pix))
    assert g_corner == pytest.approx(g_at(np.hypot(c, c) * pix))


def test_plan_detector_coarser_than_cutoff():
    # complete ring at 0.843 A, corners 0.374 A, cutoff 0.6 A
    plan = make_plan(1 / 0.843, 1 / 0.374, OptimiseParams(d_min=0.6))
    assert plan.d_cmp == pytest.approx(0.843)
    assert plan.k_min == pytest.approx(2 / 0.843, abs=1e-3)
    assert plan.k_values[-1] == pytest.approx(2 / 0.6, abs=1e-3)
    assert len(plan.k_values) == 5


def test_plan_detector_finer_than_cutoff():
    # the ring (0.5 A) is already beyond the cutoff (0.8 A):
    # compare spots to 0.8 A, and still look 50% beyond in k_max
    plan = make_plan(1 / 0.5, 1 / 0.4, OptimiseParams(d_min=0.8))
    assert plan.d_cmp == pytest.approx(0.8)
    assert plan.k_min == pytest.approx(2.5)
    assert plan.k_values[-1] == pytest.approx(3.75)


def test_default_test_images():
    # centres -19.95, -19.45, ..., 9.55: most tilted is image 0,
    # nearest zero tilt is image 40 (centre 0.05)
    scan = Scan(angles_deg=-20.2 + 0.5 * np.arange(60), n_substeps=2)
    assert default_test_images(scan, n=2) == [0, 40]
    assert default_test_images(scan, n=4) == [0, 15, 30, 40]


def test_thickest():
    scan = Scan(angles_deg=-60.25 + 0.5 * np.arange(10), n_substeps=2)
    tilt, t = thickest(scan, [0, 5], 800.0)
    assert tilt == pytest.approx(60.0)
    assert t == pytest.approx(1600.0)


# ----------------------------------------------------------------
# Step 1: rocking-curve model
# ----------------------------------------------------------------

def test_substep_offset_matches_scan():
    scan = Scan(angles_deg=np.arange(5) * 0.25, n_substeps=8)
    # medley puts substeps in the middle of each slice
    assert substep_offset(scan) == pytest.approx(0.5)


def test_sampling_error_left_rule_falls_as_one_over_n():
    e32 = sampling_error(800, 1.19, 0.25, 32, offset=0.0)
    e64 = sampling_error(800, 1.19, 0.25, 64, offset=0.0)
    assert e32 / e64 == pytest.approx(2.0, rel=0.1)


def test_sampling_error_midpoint_much_smaller():
    left = sampling_error(800, 1.19, 0.25, 16, offset=0.0)
    mid = sampling_error(800, 1.19, 0.25, 16, offset=0.5)
    assert mid < left / 10
    assert sampling_error(800, 1.19, 0.25, 256, offset=0.5) < 1e-3


def test_estimate_substeps_first_within_tol():
    opt = OptimiseParams(tol=0.01)
    n, model = estimate_substeps(800, 1.19, 0.25, 0.5, 0.05, opt)
    assert model[n] <= 0.01
    smaller = [m for m in model if m < n]
    assert all(model[m] > 0.01 for m in smaller)


# ----------------------------------------------------------------
# Comparing runs
# ----------------------------------------------------------------

def test_r1_skips_direct_beam_and_counts_missing():
    ref = {(0, 0, 0, 0): 1e6, (0, 1, 0, 0): 100.0, (0, 2, 0, 0): 50.0}
    run = {(0, 0, 0, 0): 5e5, (0, 1, 0, 0): 110.0}     # 2 0 0 lost
    assert r1(run, ref, keep_all) == pytest.approx((10 + 50) / 150)


def test_r1_only_compared_spots():
    ref = {(0, 1, 0, 0): 100.0, (0, 9, 0, 0): 100.0}
    run = {(0, 1, 0, 0): 100.0, (0, 9, 0, 0): 50.0}
    inner = lambda k: k[1] < 5
    assert r1(run, ref, inner) == 0.0


def test_shell_r1():
    ref = {(0, h, 0, 0): 100.0 for h in range(1, 11)}
    run = dict(ref)
    run[(0, 10, 0, 0)] = 90.0
    rows = shell_r1(run, ref, keep_all, lambda k: 0.1 * k[1], 5)
    assert [r["n_spots"] for r in rows] == [2] * 5
    assert [r["r1"] for r in rows[:4]] == [0.0] * 4
    assert rows[-1]["r1"] == pytest.approx(10 / 200)
    assert rows[-1]["d_to"] == pytest.approx(1.0)


def test_fit_saturation_recovers_curve():
    x = np.linspace(2.0, 4.0, 6)
    r = 0.05 * (1 - np.exp(-(x - 2.0) / 0.4))
    r_inf, kappa = fit_saturation(x, r)
    assert r_inf == pytest.approx(0.05, rel=1e-3)
    assert kappa == pytest.approx(0.4, rel=1e-3)
    s = fit_summary(list(x), list(r), 0.01)
    assert s["x_tol"] == pytest.approx(2.0 - 0.4 * math.log(0.01 / 0.05))
    assert s["remaining_at_last"] == pytest.approx(
        0.05 * math.exp(-2.0 / 0.4), rel=1e-3)


# ----------------------------------------------------------------
# Steps 2-4
# ----------------------------------------------------------------

def test_check_substeps_doubles_until_within_tol():
    """
    Intensities off by 1/n: R1(n vs m) = (1/n - 1/m) / (1 + 1/m).
    From n_est 16: 16 vs 32 is 3.0%, so 64 is added; 32 vs 64 is
    1.5%, so 128 is added; 64 vs 128 is 0.78% -> 64.
    """
    runner = FakeRunner(lambda s: {(0, 1, 0, 0):
                                   100 * (1 + 1 / s["n_substeps"])})
    chosen, rows, ok = check_substeps(
        runner, {"k_max": 2.0, "sg_max": 0.05}, 16, keep_all, 0.01)
    assert ok
    assert chosen == 64
    assert [r["n_substeps"] for r in rows] == [8, 16, 32, 64, 128]
    assert rows[-1]["r1_vs_last"] == 0.0


def test_check_substeps_gives_up_after_two_doublings():
    runner = FakeRunner(lambda s: {(0, 1, 0, 0):
                                   100 * (1 + 10 / s["n_substeps"])})
    chosen, rows, ok = check_substeps(
        runner, {"k_max": 2.0, "sg_max": 0.05}, 4, keep_all, 0.001)
    assert not ok
    assert chosen == rows[-1]["n_substeps"] == 32


def test_sweep_k_max():
    """
    Inner spots change with k_max as 5% x exp(-(k - 2) / 0.3),
    so R1 vs k = 3.2 falls below 1% from k ~ 2.48 on.
    """
    def intensities(s):
        e = 0.05 * math.exp(-(s["k_max"] - 2.0) / 0.3)
        return {(0, h, 0, 0): 100 * (1 + e) for h in range(1, 6)}

    runner = FakeRunner(intensities)
    values = [2.0, 2.3, 2.6, 2.9, 3.2]
    chosen, rows, fit = sweep(runner, "k_max", values,
                              {"n_substeps": 16, "sg_max": 0.05},
                              keep_all, 0.01)
    assert chosen == 2.6
    assert rows[0]["r1_vs_first"] == 0.0
    assert rows[0]["r1_vs_previous"] is None
    assert rows[-1]["r1_vs_last"] == 0.0
    assert fit is not None and fit["r_inf"] > 0.04


# ----------------------------------------------------------------
# make_evaluator
# ----------------------------------------------------------------

def test_make_evaluator_keeps_on_detector_spots(monkeypatch):
    seen = []

    def fake_run_image(cif, det, beam, geom, scan, image_index,
                       simulator, engine_params, integ):
        seen.append((image_index, scan.n_substeps,
                     simulator.params.k_max,
                     simulator.params.sg_max, simulator.verbose))
        result = one_thickness_result(
            [[1, 0, 0], [0, 1, 0]],
            [5.0, 50.0],                   # second is off the panel
            [5.0, 5.0],
            [3.0, 4.0],
        )
        return result, {}

    monkeypatch.setattr(om, "run_image", fake_run_image)
    evaluate = make_evaluator(
        "x.cif",
        SimpleNamespace(npx=10, npy=10),
        SimpleNamespace(energy_eV=200e3),
        None,
        Scan(angles_deg=np.arange(4.0), n_substeps=10),
        BlochParams(k_max=1.0, sg_max=0.1),
        None, None,
        test_images=[0, 3],
    )
    out = evaluate({"k_max": 2.5, "sg_max": 0.02, "n_substeps": 4})
    assert out == {(0, 1, 0, 0): 3.0, (3, 1, 0, 0): 3.0}
    assert seen == [(0, 4, 2.5, 0.02, False), (3, 4, 2.5, 0.02, False)]


# ----------------------------------------------------------------
# medley optimise, end to end with the simulation faked
# ----------------------------------------------------------------

def test_optimise_command(tmp_path, monkeypatch, capsys):
    import medley.cli as cli
    from medley.engine import EngineParams
    from medley.integration import IntegrationParams

    n = 100
    # half-width 3.75 mm at 110 mm: complete ring near d = 0.84 A
    det = Detector(
        distance_mm=110.0, npx=n, npy=n, pixel_size_mm=0.075,
        beam_centre_px=(n / 2, n / 2),
        origin=np.array([-n / 2 * 0.075, -n / 2 * 0.075, -110.0]),
    )
    beam = Beam(wavelength_A=0.02851, energy_eV=160e3)
    scan = Scan(angles_deg=-30.0 + 0.25 * np.arange(240), n_substeps=8)

    def fake_build(args):
        geometry = SimpleNamespace(base_B=lambda: 0.1 * np.eye(3))
        return (det, beam, geometry, scan,
                BlochParams(k_max=args.k_max, sg_max=args.sg_max),
                EngineParams(base_thickness_A=800.0),
                IntegrationParams())

    def fake_run_image(cif, det_, beam_, geom, scan_, image_index,
                       simulator, engine_params, integ):
        k, sg = simulator.params.k_max, simulator.params.sg_max
        nsub = scan_.n_substeps
        I = {(0, 0, 0): 1e6}
        for h in range(1, 16):                       # |g| = 0.1 h
            if 0.1 * h > k / 2:
                continue
            e = (np.sin(h + image_index) / nsub
                 + 0.05 * math.exp(-(k - 2.0) / 0.3)
                 + 1e-5 / sg**2)
            I[(h, 0, 0)] = 100.0 * (1 + e)
        result = one_thickness_result(
            list(I), [10.0] * len(I), [10.0] * len(I),
            list(I.values()))
        return result, {}

    monkeypatch.setattr(cli, "_build_simulation", fake_build)
    monkeypatch.setattr(cli, "make_beam_counter",
                        lambda *a: (lambda s: 123))
    monkeypatch.setattr(om, "run_image", fake_run_image)

    params_file = tmp_path / "params.json"
    cli.main(["params", "synthetic", "-o", str(params_file)])
    params = json.loads(params_file.read_text())
    params["cif_file"] = "x.cif"
    params_file.write_text(json.dumps(params))

    out = tmp_path / "tuned.json"
    cli.main(["optimise", str(params_file), "-o", str(out)])
    printed = capsys.readouterr().out
    for step in ("Step 0", "Step 1", "Step 2", "Step 3", "Step 4",
                 "Result"):
        assert step in printed

    tuned = json.loads(out.read_text())
    report = json.loads((tmp_path / "tuned_report.json").read_text())
    plan_k = report["k_values"]
    assert plan_k[0] == pytest.approx(2 / report["d_ring"], abs=1e-3)
    assert tuned["k_max"] in plan_k
    assert tuned["sg_max"] in OptimiseParams().sg_max_values
    assert tuned["n_substeps"] == report["step2"]["chosen"]
    assert tuned["cif_file"] == "x.cif"
    assert report["n_compared_spots"] > 0
