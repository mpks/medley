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


def _spot_list(sum_I, sum_I_px, sum_I_py, intensity_cut):
    """
    Build (millers, px, py, intensities) from the per-hkl sums,
    computing intensity-weighted centroids and dropping spots
    at or below intensity_cut.
    """
    millers, px_l, py_l, int_l = [], [], [], []
    for hkl in sorted(sum_I.keys()):
        total = sum_I[hkl]
        if total <= intensity_cut:
            continue
        millers.append(hkl)
        px_l.append(sum_I_px[hkl] / total)
        py_l.append(sum_I_py[hkl] / total)
        int_l.append(total)
    return (millers, np.array(px_l), np.array(py_l),
            np.array(int_l))


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

    def finalize(self, state, intensity_cut):
        """
        Build the sharp signal (one pixel per reflection at its
        centroid) and the spot list.

        Returns (signal, millers, px, py, intensities).
        """
        millers, px, py, ints = _spot_list(
            state.sum_I, state.sum_I_px, state.sum_I_py,
            intensity_cut,
        )
        npy, npx = self.detector.npy, self.detector.npx
        signal = np.zeros((npy, npx), dtype=np.float64)
        for i in range(len(ints)):
            ix = int(round(px[i]))
            iy = int(round(py[i]))
            if 0 <= ix < npx and 0 <= iy < npy:
                signal[iy, ix] += ints[i]
        return signal, millers, px, py, ints


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

    def finalize(self, state, intensity_cut):
        """
        Return the accumulated sharp signal (the streaks) and
        the spot list.

        Returns (signal, millers, px, py, intensities).
        """
        millers, px, py, ints = _spot_list(
            state.sum_I, state.sum_I_px, state.sum_I_py,
            intensity_cut,
        )
        return state.signal, millers, px, py, ints


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
