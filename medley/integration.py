"""
integration.py — combine the substeps of one image into a sharp
signal image and a spot list.

Two integration methods share one interface and differ only in
how the signal image is built:

  * vector : per substep, accumulate per Miller index the total
             intensity and the intensity-weighted detector
             position.  At the end each reflection is deposited
             as a single sharp pixel at its centroid.  A spot
             that rotates during the image collapses to one
             centroid.

  * raster : per substep, deposit the spots into the image at
             their integer pixels, so a moving spot spreads into
             a streak across the substeps.  Per Miller index
             sums are also kept so the same spot list (for
             labels) is available.

Both weight each substep by delta_deg / n_substeps so the
integrated intensity is invariant to the number of substeps.

The signal is the raw summed deposits — no Gaussian blur is
applied here.  The point-spread blur that produces the noise
halo is a render-time concern (plotting and CBF export); the
integrator never smears the spots.

Each integrator exposes partial state (the signal array and the
per-hkl dictionaries) so the work of one image can be split
across worker processes and merged by the parent.

With several thicknesses there is one state per thickness, all
filled from the same substeps.  spot_table then lines them up on
one shared list of reflections, so every thickness has the same
spots in the same order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------
# Parameters and state
# ----------------------------------------------------------------

@dataclass(frozen=True)
class IntegrationParams:
    """
    Simulation-time integration configuration.

    method : str
        "vector" or "raster".  This is the only integration
        choice; the point-spread blur and noise level are
        render-time parameters (see CbfParams), so they can be
        changed without re-running the simulation.
    """
    method: str = "vector"


@dataclass
class IntegratorState:
    """
    Partial accumulator for one image (per worker).

    signal : ndarray or None
        2D sharp signal image (raster builds it incrementally;
        vector leaves it None until finalize).
    sum_I, sum_I_px, sum_I_py : dict (h,k,l) -> float
        Per-reflection integrated intensity and weighted
        position sums.
    """
    npx: int
    npy: int
    signal: np.ndarray = None
    sum_I: dict = field(default_factory=dict)
    sum_I_px: dict = field(default_factory=dict)
    sum_I_py: dict = field(default_factory=dict)


# ----------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------

def _accumulate_dicts(state, projected, weight):
    """
    Update the per-reflection sums from one substep's projected
    spots, each weighted by weight (delta_deg / n_substeps).
    """
    for s in projected:
        hkl = s["hkl"]
        inten = s["intensity"] * weight
        px = s["px"]
        py = s["py"]
        state.sum_I[hkl] = state.sum_I.get(hkl, 0.0) + inten
        state.sum_I_px[hkl] = (
            state.sum_I_px.get(hkl, 0.0) + inten * px
        )
        state.sum_I_py[hkl] = (
            state.sum_I_py.get(hkl, 0.0) + inten * py
        )


def _deposit(image, projected, weight):
    """
    Deposit projected spot intensities (weighted) into image at
    integer pixel positions.  In-place.
    """
    npy, npx = image.shape
    for s in projected:
        ix = int(round(s["px"]))
        iy = int(round(s["py"]))
        if 0 <= ix < npx and 0 <= iy < npy:
            image[iy, ix] += s["intensity"] * weight


def spot_table(states, intensity_cut):
    """
    Line up per-thickness states on one shared spot list.

    A reflection is kept if its integrated intensity is above
    intensity_cut at ANY thickness, so all thicknesses share the
    same reflections.  Returns (millers, px, py, intensities),
    with px, py and intensities of shape (T, N) for T states.

    The centroid of a reflection with zero intensity at some
    thickness (but kept for another) is taken from all
    thicknesses together, since it is undefined on its own.
    """
    keys = sorted(set().union(*(s.sum_I.keys() for s in states)))
    millers = [
        hkl for hkl in keys
        if any(s.sum_I.get(hkl, 0.0) > intensity_cut for s in states)
    ]
    n_t, n = len(states), len(millers)
    ints = np.zeros((n_t, n))
    sum_px = np.zeros((n_t, n))
    sum_py = np.zeros((n_t, n))
    for t, s in enumerate(states):
        for j, hkl in enumerate(millers):
            ints[t, j] = s.sum_I.get(hkl, 0.0)
            sum_px[t, j] = s.sum_I_px.get(hkl, 0.0)
            sum_py[t, j] = s.sum_I_py.get(hkl, 0.0)

    total = ints.sum(axis=0)
    safe_total = np.where(total != 0.0, total, 1.0)
    fallback_px = sum_px.sum(axis=0) / safe_total
    fallback_py = sum_py.sum(axis=0) / safe_total
    nonzero = ints != 0.0
    safe = np.where(nonzero, ints, 1.0)
    px = np.where(nonzero, sum_px / safe, fallback_px)
    py = np.where(nonzero, sum_py / safe, fallback_py)
    return millers, px, py, ints


# ----------------------------------------------------------------
# Integrators
# ----------------------------------------------------------------

class VectorIntegrator:
    """Centroid-based integrator (see module docstring)."""

    def __init__(self, detector, params):
        self.detector = detector
        self.params = params

    def new_state(self):
        return IntegratorState(
            npx=self.detector.npx, npy=self.detector.npy
        )

    def add_substep(self, state, projected, weight):
        _accumulate_dicts(state, projected, weight)

    def build_signal(self, state, px, py, ints):
        """
        The sharp signal for one thickness: one pixel per
        reflection at its centroid (px, py, ints from
        spot_table for this thickness).
        """
        npy, npx = self.detector.npy, self.detector.npx
        signal = np.zeros((npy, npx), dtype=np.float64)
        for i in range(len(ints)):
            ix = int(round(px[i]))
            iy = int(round(py[i]))
            if 0 <= ix < npx and 0 <= iy < npy:
                signal[iy, ix] += ints[i]
        return signal


class RasterIntegrator:
    """Per-substep raster integrator (see module docstring)."""

    def __init__(self, detector, params):
        self.detector = detector
        self.params = params

    def new_state(self):
        npy, npx = self.detector.npy, self.detector.npx
        st = IntegratorState(npx=npx, npy=npy)
        st.signal = np.zeros((npy, npx), dtype=np.float64)
        return st

    def add_substep(self, state, projected, weight):
        # Per-hkl sums for the label spot list.
        _accumulate_dicts(state, projected, weight)
        # Sharp deposit for this substep; a moving spot lands in
        # different pixels across substeps, building a streak.
        _deposit(state.signal, projected, weight)

    def build_signal(self, state, px, py, ints):
        """
        The accumulated sharp signal (the streaks) for one
        thickness; the spot list is not needed.
        """
        return state.signal


# ----------------------------------------------------------------
# State merging (across worker processes)
# ----------------------------------------------------------------

def merge_states(states, npx, npy):
    """
    Merge partial IntegratorStates into one combined state.

    The signal arrays are summed (treating None as zero) and the
    per-hkl dictionaries are summed key by key.
    """
    combined = IntegratorState(npx=npx, npy=npy)
    has_arrays = any(s.signal is not None for s in states)
    if has_arrays:
        combined.signal = np.zeros((npy, npx), dtype=np.float64)

    for s in states:
        if s.signal is not None:
            combined.signal += s.signal
        for hkl, v in s.sum_I.items():
            combined.sum_I[hkl] = (
                combined.sum_I.get(hkl, 0.0) + v
            )
        for hkl, v in s.sum_I_px.items():
            combined.sum_I_px[hkl] = (
                combined.sum_I_px.get(hkl, 0.0) + v
            )
        for hkl, v in s.sum_I_py.items():
            combined.sum_I_py[hkl] = (
                combined.sum_I_py.get(hkl, 0.0) + v
            )
    return combined


def make_integrator(detector, params):
    """Factory: return the integrator for params.method."""
    if params.method == "vector":
        return VectorIntegrator(detector, params)
    if params.method == "raster":
        return RasterIntegrator(detector, params)
    raise ValueError(
        f"Unknown integration method {params.method!r}; "
        "use 'vector' or 'raster'."
    )
