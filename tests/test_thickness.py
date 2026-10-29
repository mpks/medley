"""
Several crystal thicknesses in one run.

The key property: simulating thicknesses (a, b) together gives,
at each thickness, the same result as simulating it alone.  The
engine is driven with a fake simulator and fake projection, so
these tests need neither abTEM nor ase.
"""

import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

import medley.driver as driver
import medley.engine as engine
from medley.engine import EngineParams, ImageResult, run_image
from medley.experiment import Scan, Spots
from medley.export_npz import (
    image_at,
    load_image,
    save_image,
    thickness_index,
    thickness_suffix,
    with_thickness_suffix,
)
from medley.integration import (
    IntegrationParams,
    IntegratorState,
    spot_table,
)
from medley.simulator import BlochParams, BlochSimulator


# ----------------------------------------------------------------
# A fake simulator: intensities depend on hkl, thickness and angle
# ----------------------------------------------------------------

HKLS = [(1, 0, 0), (0, 1, 0), (1, 1, 0), (2, 0, 0)]


def fake_intensity(hkl, t, angle):
    h, k, _ = hkl
    if hkl == (2, 0, 0) and t < 700:       # dark at the thinner one
        return 0.0
    return (1 + h + k) * (1.0 + math.sin(t / 97.0 + 7 * angle + h))


class FakeSimulator:
    """simulate(atoms, thicknesses): here atoms is the angle."""

    def simulate(self, atoms, thicknesses_A):
        angle = float(atoms)
        positions = np.array([[h, k, 0.0] for h, k, _ in HKLS])
        return [
            Spots(positions=positions,
                  millers=[list(m) for m in HKLS],
                  intensities=np.array([fake_intensity(m, t, angle)
                                        for m in HKLS]))
            for t in thicknesses_A
        ]


def fake_project(spots, detector, beam):
    """A moving spot: px drifts with the (smuggled) angle."""
    out = []
    for pos, hkl, inten in zip(spots.positions, spots.millers,
                               spots.intensities):
        h, k = int(hkl[0]), int(hkl[1])
        out.append({"hkl": tuple(int(x) for x in hkl),
                    "px": 10.0 * h + 4.0 + pos[2],
                    "py": 10.0 * k + 5.0,
                    "intensity": float(inten)})
    return out


@pytest.fixture
def fake_engine(monkeypatch):
    monkeypatch.setattr(engine, "make_base_atoms", lambda c, g: None)
    monkeypatch.setattr(engine, "orient_atoms",
                        lambda base, g, angle, beam: angle)
    monkeypatch.setattr(engine, "project_spots", fake_project)


def run(thicknesses, method="vector", rocking=()):
    scan = Scan(angles_deg=np.arange(4) * 0.5, n_substeps=6)
    detector = SimpleNamespace(npx=40, npy=40)
    params = EngineParams(base_thickness_A=thicknesses,
                          rocking_hkl=rocking)
    return run_image("x.cif", detector, None, None, scan, 1,
                     FakeSimulator(), params,
                     IntegrationParams(method=method))


# ----------------------------------------------------------------
# Engine
# ----------------------------------------------------------------

def test_engine_params_thickness_tuple():
    assert EngineParams(base_thickness_A=800).base_thickness_A == (800.0,)
    assert EngineParams(base_thickness_A=[500, 800]).base_thickness_A == (
        500.0, 800.0)
    with pytest.raises(ValueError):
        EngineParams(base_thickness_A=[800, 800])


@pytest.mark.parametrize("method", ["vector", "raster"])
def test_together_equals_separately(fake_engine, method):
    both, _ = run((500.0, 800.0), method)
    alone = {t: run((t,), method)[0] for t in (500.0, 800.0)}

    assert both.thicknesses_A == (500.0, 800.0)
    assert both.intensities.shape == (2, len(both.millers))
    assert both.signal.shape == (2, 40, 40)
    for i, t in enumerate((500.0, 800.0)):
        one = alone[t].at(0)
        idx = [both.millers.index(m) for m in one.millers]
        np.testing.assert_allclose(both.intensities[i, idx],
                                   one.intensities)
        np.testing.assert_allclose(both.px[i, idx], one.px)
        np.testing.assert_allclose(both.py[i, idx], one.py)
        np.testing.assert_allclose(both.signal[i], one.signal)


def test_spot_kept_if_bright_at_any_thickness(fake_engine):
    both, _ = run((500.0, 800.0))
    thin, _ = run((500.0,))
    assert (2, 0, 0) not in thin.millers              # dark when alone
    j = both.millers.index((2, 0, 0))
    assert both.intensities[0, j] == 0.0              # kept, at zero
    assert both.intensities[1, j] > 0.0
    # its centroid at the dark thickness comes from the bright one
    assert both.px[0, j] == pytest.approx(both.px[1, j])


def test_rocking_curves_per_thickness(fake_engine):
    _, rocking = run((500.0, 800.0), rocking=((1, 0, 0),))
    per_angle = rocking[(1, 0, 0)]
    assert len(per_angle) == 6                        # substeps
    for angle, values in per_angle.items():
        assert values == pytest.approx(
            [fake_intensity((1, 0, 0), t / math.cos(math.radians(angle)),
                            angle) for t in (500.0, 800.0)])


def test_image_at():
    r = ImageResult(millers=[(1, 0, 0)], thicknesses_A=(500.0, 800.0),
                    px=np.array([[1.0], [2.0]]),
                    py=np.array([[3.0], [4.0]]),
                    intensities=np.array([[5.0], [6.0]]),
                    signal=np.zeros((2, 4, 4)), method="vector",
                    angle_centre_deg=0.25, image_index=3)
    one = r.at(1)
    assert one.thickness_A == 800.0
    assert one.px.tolist() == [2.0] and one.intensities.tolist() == [6.0]
    assert one.signal.shape == (4, 4)
    assert one.image_index == 3


# ----------------------------------------------------------------
# Integration: shared spot table
# ----------------------------------------------------------------

def _state(values):
    s = IntegratorState(npx=10, npy=10)
    for hkl, (inten, px, py) in values.items():
        s.sum_I[hkl] = inten
        s.sum_I_px[hkl] = inten * px
        s.sum_I_py[hkl] = inten * py
    return s


def test_spot_table_single_thickness():
    s = _state({(1, 0, 0): (2.0, 3.0, 4.0), (0, 1, 0): (0.0, 1.0, 1.0)})
    millers, px, py, ints = spot_table([s], intensity_cut=0.0)
    assert millers == [(1, 0, 0)]                     # 0 is cut
    assert px.tolist() == [[3.0]] and py.tolist() == [[4.0]]
    assert ints.tolist() == [[2.0]]


def test_spot_table_two_thicknesses():
    a = _state({(1, 0, 0): (2.0, 3.0, 4.0)})
    b = _state({(1, 0, 0): (1.0, 5.0, 6.0), (0, 1, 0): (4.0, 7.0, 8.0)})
    millers, px, py, ints = spot_table([a, b], intensity_cut=0.0)
    assert millers == [(0, 1, 0), (1, 0, 0)]
    assert ints.tolist() == [[0.0, 2.0], [4.0, 1.0]]
    assert px.tolist() == [[7.0, 3.0], [7.0, 5.0]]    # fallback 7.0


# ----------------------------------------------------------------
# Simulator: one abTEM call per configuration, all thicknesses
# ----------------------------------------------------------------

def test_bloch_simulator_averages_per_thickness(monkeypatch):
    sim = BlochSimulator(BlochParams(k_max=2.0, sg_max=0.05),
                         energy_eV=200e3, verbose=False)
    configs = ["cfg1", "cfg2"]
    seen = []

    def fake_config(cfg, thicknesses):
        seen.append(list(thicknesses))
        if cfg == "cfg1":
            millers = [[1, 0, 0], [0, 1, 0]]
            ints = np.array([[1.0, 2.0], [3.0, 4.0]])   # (T, n)
        else:                                            # (0 1 0) missing
            millers = [[1, 0, 0]]
            ints = np.array([[5.0], [7.0]])
        positions = np.array([[float(m[0]), float(m[1]), 0.0]
                              for m in millers])
        return positions, millers, ints, len(millers)

    monkeypatch.setattr(sim, "_make_frozen_phonons", lambda atoms: configs)
    monkeypatch.setattr(sim, "_diffraction_for_config", fake_config)

    spots = sim.simulate(None, [500.0, 800.0])
    assert seen == [[500.0, 800.0], [500.0, 800.0]]   # one call each
    assert len(spots) == 2
    for s in spots:
        assert s.millers == [[0, 1, 0], [1, 0, 0]]
    # averaged over both configurations (missing counts as zero)
    assert spots[0].intensities.tolist() == [1.0, 3.0]   # 500 A
    assert spots[1].intensities.tolist() == [2.0, 5.0]   # 800 A


# ----------------------------------------------------------------
# NPZ files and the tools that read them
# ----------------------------------------------------------------

def two_thickness_result(image_index, scale=1.0):
    return ImageResult(
        millers=[(1, 0, 0), (0, 1, 0), (0, 0, 0)],
        thicknesses_A=(500.0, 800.0),
        px=np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.0]]),
        py=np.array([[4.0, 5.0, 6.0], [4.5, 5.5, 6.0]]),
        intensities=scale * np.array([[10.0, 20.0, 99.0],
                                      [30.0, 40.0, 99.0]]),
        signal=np.stack([np.full((4, 5), 1.0), np.full((4, 5), 2.0)]),
        method="vector", angle_centre_deg=0.25 + image_index,
        image_index=image_index,
    )


GEOM = dict(
    detector=SimpleNamespace(npx=5, npy=4, pixel_size_mm=0.075,
                             distance_mm=500.0,
                             beam_centre_px=(2.0, 2.0)),
    beam=SimpleNamespace(wavelength_A=0.025),
    scan=SimpleNamespace(delta_deg=1.0),
)


def write_two(tmp_path):
    paths = []
    for i in (0, 1):
        paths.append(save_image(two_thickness_result(i, scale=i + 1),
                                GEOM["detector"], GEOM["beam"],
                                GEOM["scan"], str(tmp_path), "run"))
    return paths


def test_npz_round_trip(tmp_path):
    path = write_two(tmp_path)[0]
    rec = load_image(path)
    assert rec["thicknesses_A"].tolist() == [500.0, 800.0]
    assert rec["signal"].shape == (2, 4, 5)
    assert rec["intensities"].shape == (2, 3)
    assert rec["hkl_indices"].shape == (3, 3)
    one = image_at(rec, 1)
    assert one["thickness_A"] == 800.0
    assert one["intensities"].tolist() == [30.0, 40.0, 99.0]
    assert one["signal"].shape == (4, 5)


def test_old_single_thickness_file_still_loads(tmp_path):
    path = tmp_path / "old.npz"
    np.savez(path, method="vector", signal=np.zeros((4, 5)),
             hkl_indices=np.array([[1, 0, 0]]), px=np.array([1.0]),
             py=np.array([2.0]), intensities=np.array([3.0]),
             image_index=0)
    rec = load_image(path)
    assert rec["signal"].shape == (1, 4, 5)
    assert np.isnan(rec["thicknesses_A"][0])
    assert image_at(rec, 0)["intensities"].tolist() == [3.0]


def test_thickness_names():
    assert thickness_suffix(800) == "_thickness_00800_angs"
    assert thickness_suffix(800.5) == "_thickness_00800p5_angs"
    assert with_thickness_suffix("out/L.hkl", 1200) == (
        "out/L_thickness_01200_angs.hkl")
    assert thickness_index([500.0, 800.0]) == 0
    assert thickness_index([500.0, 800.0], 800) == 1
    with pytest.raises(ValueError, match="available: 500, 800"):
        thickness_index([500.0, 800.0], 650)


def test_integrate_writes_one_file_per_thickness(tmp_path, capsys):
    from medley.integrate_npz import main
    files = write_two(tmp_path)
    out = tmp_path / "L.hkl"
    main(files + ["-o", str(out)])
    thin = (tmp_path / "L_thickness_00500_angs.hkl").read_text().split("\n")
    thick = (tmp_path / "L_thickness_00800_angs.hkl").read_text().split("\n")
    assert not out.exists()
    # summed over both images (scale 1 and 2); 000 left out
    assert thin[0] == "   0   1   0   60.00    1.00"
    assert thin[1] == "   1   0   0   30.00    1.00"
    assert thick[0] == "   0   1   0  120.00    1.00"


def test_integrate_one_thickness(tmp_path):
    from medley.integrate_npz import main
    files = write_two(tmp_path)
    out = tmp_path / "L.hkl"
    main(files + ["-o", str(out), "--thickness", "800"])
    assert out.read_text().split("\n")[0] == "   0   1   0  120.00    1.00"
    with pytest.raises(SystemExit, match="available"):
        main(files + ["-o", str(out), "--thickness", "650"])


def test_integrate_auto_scale_shared(tmp_path, capsys):
    from medley.integrate_npz import main
    files = write_two(tmp_path)
    main(files + ["-o", str(tmp_path / "L.hkl"), "--auto_scale"])
    thin = (tmp_path / "L_thickness_00500_angs.hkl").read_text()
    thick = (tmp_path / "L_thickness_00800_angs.hkl").read_text()
    # one factor (833, rounded down to 3 s.f.) for both files: the
    # strongest overall (120) reaches the top, the thinner one's
    # strongest (60) half of it
    assert "   0   1   099960.00" in thick
    assert "   0   1   049980.00" in thin


def test_npz_to_nexus_per_thickness(tmp_path, monkeypatch):
    import medley.npz_to_nexus as nx
    written = []
    monkeypatch.setattr(nx, "read_geometry", lambda expt, i: None)

    def fake_write(output, stack, geom, indices, probe):
        written.append((os.path.basename(output), stack.shape))
        return np.eye(3)

    monkeypatch.setattr(nx, "write_nxmx", fake_write)
    files = write_two(tmp_path)
    nx.main(["x.expt"] + files + ["-o", str(tmp_path / "sim.nxs")])
    assert written == [("sim_thickness_00500_angs.nxs", (2, 4, 5)),
                       ("sim_thickness_00800_angs.nxs", (2, 4, 5))]
    written.clear()
    nx.main(["x.expt"] + files + ["-o", str(tmp_path / "sim.nxs"),
                                  "--thickness", "500"])
    assert written == [("sim.nxs", (2, 4, 5))]


def test_npz_to_cbf_loads_chosen_thickness(tmp_path):
    from medley.npz_to_cbf import _load_npz
    path = write_two(tmp_path)[0]
    assert _load_npz(path)["thickness_A"] == 500.0
    rec = _load_npz(path, 800)
    assert rec["thickness_A"] == 800.0
    assert rec["signal"][0, 0] == 2.0


# ----------------------------------------------------------------
# Driver: NPZ by default, CBF and plots only when asked
# ----------------------------------------------------------------

def test_driver_writes_npz_only_by_default(tmp_path, monkeypatch):
    def fake_run_image(cif, det, beam, geom, scan, i, sim, params, integ):
        rocking = {(1, 0, 0): {float(a): np.array([1.0, 2.0])
                               for a in scan.substep_angles(i)}}
        return two_thickness_result(i), rocking

    monkeypatch.setattr(driver, "run_image", fake_run_image)
    scan = Scan(angles_deg=np.arange(3) * 1.0, n_substeps=2)
    params = EngineParams(base_thickness_A=(500.0, 800.0),
                          rocking_hkl=((1, 0, 0),))
    driver.run("x.cif", GEOM["detector"], GEOM["beam"], None, scan,
               None, params, IntegrationParams(), None,
               str(tmp_path), "run", export_sg=False)

    names = sorted(os.listdir(tmp_path))
    assert names == ["image_vector_run_0000.npz",
                     "image_vector_run_0001.npz",
                     "image_vector_run_0002.npz", "rocking_run.npz"]
    with np.load(tmp_path / "rocking_run.npz") as d:
        assert d["thicknesses_A"].tolist() == [500.0, 800.0]
        assert d["curves"].shape == (2, 1, 6)        # (T, K, M)
        assert d["curves"][1].tolist() == [[2.0] * 6]


# ----------------------------------------------------------------
# Params files with several thicknesses
# ----------------------------------------------------------------

def test_params_file_with_thickness_list():
    from medley.cli import build_parser
    from medley.params import default_params, params_to_argv
    parser = build_parser()
    params = default_params(parser, "synthetic")
    assert params["base_thickness_A"] == [800.0]
    params.update(cif_file="x.cif", base_thickness_A=[500, 800, 1100])
    args = parser.parse_args(params_to_argv(parser, params))
    assert args.base_thickness_A == [500.0, 800.0, 1100.0]
    assert args.cbf is False and args.plots is False
    json.dumps(params)                                  # stays JSON


# ----------------------------------------------------------------
# The real thing: abTEM with several thicknesses (skipped where
# abTEM or ase is not installed)
# ----------------------------------------------------------------

def test_abtem_together_equals_separately():
    pytest.importorskip("abtem")
    build = pytest.importorskip("ase.build")
    atoms = build.bulk("Si", "diamond", a=5.43, cubic=True)
    atoms.rotate(7.0, "x", rotate_cell=True)       # off a zone axis
    atoms.rotate(4.0, "y", rotate_cell=True)
    sim = BlochSimulator(BlochParams(k_max=2.0, sg_max=0.05),
                         energy_eV=200e3, verbose=False)

    both = sim.simulate(atoms, [500.0, 800.0])
    for i, t in enumerate((500.0, 800.0)):
        (alone,) = sim.simulate(atoms, [t])
        assert both[i].millers == alone.millers
        np.testing.assert_allclose(both[i].intensities,
                                   alone.intensities,
                                   rtol=1e-8, atol=1e-12)
    # and the two thicknesses really differ
    assert not np.allclose(both[0].intensities, both[1].intensities)
