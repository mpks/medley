"""
One pool of workers for all images, (image, chunk) tasks.

The key property: the results do not depend on how many workers
there are or which task finishes first.  These tests run the real
process pool, started with fork so the fake simulator and the
patched engine functions reach the workers; they need neither
abTEM nor ase.
"""

import math
import multiprocessing
from types import SimpleNamespace

import numpy as np
import pytest

import medley.engine as engine
from medley.engine import (
    EngineParams,
    _finish_image,
    _process_substep_batch,
    run_image,
    run_images,
    substeps_per_task,
)
from medley.experiment import Scan, Spots
from medley.integration import IntegrationParams, make_integrator

FORK = multiprocessing.get_context("fork")
HKLS = [(1, 0, 0), (0, 1, 0), (1, 1, 0), (2, 0, 0), (0, 2, 0)]


class FakeSimulator:
    """Intensities depend on hkl, thickness and angle (atoms)."""

    def simulate(self, atoms, thicknesses_A):
        angle = float(atoms)
        positions = np.array([[h, k, angle] for h, k, _ in HKLS])
        return [
            Spots(positions=positions,
                  millers=[list(m) for m in HKLS],
                  intensities=np.array([
                      (1 + h + k) * (1.0 + math.sin(t / 97.0 + 7 * angle + h))
                      for h, k, _ in HKLS]))
            for t in thicknesses_A
        ]


class FailingSimulator(FakeSimulator):
    def simulate(self, atoms, thicknesses_A):
        if float(atoms) > 1.0:
            raise RuntimeError("simulated failure")
        return super().simulate(atoms, thicknesses_A)


def fake_project(spots, detector, beam):
    """A moving spot: px drifts with the angle (positions[:, 2])."""
    return [{"hkl": tuple(int(x) for x in hkl),
             "px": 6.0 * hkl[0] + 3.0 + 2.0 * pos[2],
             "py": 6.0 * hkl[1] + 4.0,
             "intensity": float(inten)}
            for pos, hkl, inten in zip(spots.positions, spots.millers,
                                       spots.intensities)]


def summarise(image_index, total):
    """A module-level extra task (runs in the pool)."""
    return image_index, total


@pytest.fixture
def fake_engine(monkeypatch):
    monkeypatch.setattr(engine, "make_base_atoms", lambda c, g: None)
    monkeypatch.setattr(engine, "orient_atoms",
                        lambda base, g, angle, beam: angle)
    monkeypatch.setattr(engine, "project_spots", fake_project)


SCAN = Scan(angles_deg=np.arange(6) * 0.5, n_substeps=7)   # 7: uneven
DETECTOR = SimpleNamespace(npx=32, npy=32)


def run_all(n_workers, method="vector", simulator=None, chunk=None,
            images=range(6), extra_task=None):
    params = EngineParams(base_thickness_A=(500.0, 800.0),
                          n_workers=n_workers, substeps_per_task=chunk,
                          rocking_hkl=((1, 0, 0), (0, 2, 0)))
    out = {}
    for idx, result, rocking, extra in run_images(
            "x.cif", DETECTOR, None, None, SCAN, list(images),
            simulator or FakeSimulator(), params,
            IntegrationParams(method=method), extra_task=extra_task,
            mp_context=FORK):
        assert idx not in out                    # each image once
        out[idx] = (result, rocking, extra)
    return out


def assert_same(a, b, exact):
    close = (np.testing.assert_array_equal if exact else
             lambda x, y: np.testing.assert_allclose(x, y, rtol=1e-12,
                                                     atol=1e-12))
    assert sorted(a) == sorted(b)
    for idx in a:
        ra, rocka, _ = a[idx]
        rb, rockb, _ = b[idx]
        assert ra.millers == rb.millers
        assert ra.thicknesses_A == rb.thicknesses_A
        for name in ("px", "py", "intensities", "signal"):
            close(getattr(ra, name), getattr(rb, name))
        assert rocka.keys() == rockb.keys()
        for hkl in rocka:
            assert rocka[hkl].keys() == rockb[hkl].keys()
            for angle in rocka[hkl]:
                close(rocka[hkl][angle], rockb[hkl][angle])


# ----------------------------------------------------------------
# Results do not depend on the number of workers
# ----------------------------------------------------------------

def test_vector_any_number_of_workers(fake_engine):
    one = run_all(1)
    for n in (2, 3, 5):
        # same chunks (1 substep), merged in the same order: exact
        assert_same(run_all(n), one, exact=True)


def test_raster_any_number_of_workers(fake_engine):
    one = run_all(1, method="raster")
    # raster chunks follow n_workers, so sums group differently:
    # equal to rounding
    assert_same(run_all(3, method="raster"), one, exact=False)
    # with the chunk size fixed, exactly equal
    assert_same(run_all(3, method="raster", chunk=2),
                run_all(1, method="raster", chunk=2), exact=True)


def test_chunks_equal_one_batch_per_image(fake_engine):
    """Cutting an image into chunks gives what one batch gives."""
    params = EngineParams(base_thickness_A=(500.0, 800.0))
    integ = IntegrationParams()
    integrator = make_integrator(DETECTOR, integ)
    result = run_all(3)
    for idx in range(6):
        whole = _process_substep_batch((
            "x.cif", None, FakeSimulator(), DETECTOR, None,
            params.base_thickness_A, SCAN.substep_angles(idx), set(),
            integrator, SCAN.delta_deg / SCAN.n_substeps))
        expected, _ = _finish_image([whole], integrator, params, integ,
                                    DETECTOR, SCAN, idx)
        got = result[idx][0]
        assert got.millers == expected.millers
        np.testing.assert_allclose(got.intensities, expected.intensities,
                                   rtol=1e-12)
        np.testing.assert_allclose(got.px, expected.px, rtol=1e-12)


def test_every_substep_counted(fake_engine):
    """7 substeps on 3 workers: all 7 angles reach the rocking curves."""
    result = run_all(3)
    for idx in range(6):
        angles = sorted(result[idx][1][(1, 0, 0)])
        np.testing.assert_allclose(angles, SCAN.substep_angles(idx))


def test_image_selection(fake_engine):
    result = run_all(2, images=[4, 1])
    assert sorted(result) == [1, 4]
    assert result[4][0].image_index == 4


# ----------------------------------------------------------------
# Extra per-image task, errors, single image
# ----------------------------------------------------------------

@pytest.mark.parametrize("n_workers", [1, 3])
def test_extra_task_runs_per_image(fake_engine, n_workers):
    def extra_task(result):
        return summarise, (result.image_index,
                           float(result.intensities.sum()))

    result = run_all(n_workers, extra_task=extra_task)
    for idx, (res, _rock, extra) in result.items():
        assert extra == (idx, pytest.approx(float(res.intensities.sum())))


@pytest.mark.parametrize("n_workers", [1, 3])
def test_worker_error_is_raised(fake_engine, n_workers):
    with pytest.raises(RuntimeError, match="simulated failure"):
        run_all(n_workers, simulator=FailingSimulator())


def test_run_image_single(fake_engine):
    params = EngineParams(base_thickness_A=(500.0, 800.0), n_workers=1)
    result, rocking = run_image("x.cif", DETECTOR, None, None, SCAN, 2,
                                FakeSimulator(), params,
                                IntegrationParams())
    assert result.image_index == 2
    np.testing.assert_array_equal(result.intensities,
                                  run_all(1)[2][0].intensities)


# ----------------------------------------------------------------
# Settings
# ----------------------------------------------------------------

def test_substeps_per_task_default():
    p = EngineParams(base_thickness_A=800, n_workers=4)
    assert substeps_per_task(p, "vector", 10) == 1
    assert substeps_per_task(p, "raster", 10) == 3        # ceil(10/4)
    p = EngineParams(base_thickness_A=800, n_workers=4,
                     substeps_per_task=5)
    assert substeps_per_task(p, "vector", 10) == 5


def test_engine_params_checks():
    for bad in (dict(n_workers=0), dict(substeps_per_task=0),
                dict(threads_per_worker=0)):
        with pytest.raises(ValueError):
            EngineParams(base_thickness_A=800, **bad)


def test_base_atoms_read_once_per_process(monkeypatch):
    calls = []

    def counting(cif, geometry):
        calls.append(cif)
        return None

    monkeypatch.setattr(engine, "make_base_atoms", counting)
    monkeypatch.setattr(engine, "orient_atoms",
                        lambda base, g, angle, beam: angle)
    monkeypatch.setattr(engine, "project_spots", fake_project)
    monkeypatch.setattr(engine, "_BASE_ATOMS", {})
    geometry = SimpleNamespace(base_B=lambda: np.eye(3))
    params = EngineParams(base_thickness_A=800.0, n_workers=1)
    list(run_images("x.cif", DETECTOR, None, geometry, SCAN, [0, 1, 2],
                    FakeSimulator(), params, IntegrationParams()))
    assert calls == ["x.cif"]                 # 21 tasks, one read


def test_worker_init_limits_threads():
    threadpoolctl = pytest.importorskip("threadpoolctl")
    engine._worker_init(1)
    try:
        for pool in threadpoolctl.threadpool_info():
            assert pool["num_threads"] == 1
    finally:
        if engine._THREAD_LIMITS is not None:
            engine._THREAD_LIMITS.restore_original_limits()
