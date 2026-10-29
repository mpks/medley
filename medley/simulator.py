"""
simulator.py — abstract simulator interface, the Bloch backend
parameters, and the Bloch-wave implementation.

A Simulator turns oriented atoms into diffraction Spots.  The
engine orients the atoms (via Geometry) and hands them, with the
thickness for that substep, to a Simulator; the Simulator is
purely "(oriented atoms, thickness) -> Spots" and is the only
place a particular diffraction backend (abTEM Bloch waves here,
multislice or FELIX in future) appears.

To add a new backend, subclass Simulator, give it its own
parameter dataclass, and implement simulate; nothing in the
engine, the domain data classes or the I/O changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from medley.experiment import Spots


# ----------------------------------------------------------------
# Abstract interface
# ----------------------------------------------------------------

class Simulator(ABC):
    """
    Abstract diffraction simulator.

    Concrete subclasses implement simulate, which takes an
    oriented ase.Atoms object and one or more propagation
    thicknesses and returns one Spots object per thickness
    (reciprocal-space positions in the lab frame, Miller
    indices, and intensities).
    """

    @abstractmethod
    def simulate(self, atoms, thicknesses_A) -> list:
        """
        Simulate diffraction from oriented atoms.

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms already oriented into the lab frame for the
            desired scan angle.
        thicknesses_A : sequence of float
            Crystal thicknesses propagated through, in Angstrom.

        Returns
        -------
        list of Spots, one per thickness, all with the same
        reflections in the same order.
        """
        raise NotImplementedError


# ----------------------------------------------------------------
# Bloch backend parameters
# ----------------------------------------------------------------

@dataclass(frozen=True)
class BlochParams:
    """
    Configuration for the Bloch-wave backend.

    k_max : float
        Structure-factor limit (A^-1).  abTEM uses beams (and
        so spots) only out to |g| = k_max / 2, i.e. spots reach
        d = 2 / k_max.
    sg_max : float
        Maximum excitation error for the beam pool (A^-1).
    num_phonon_configs : int
        Number of frozen-phonon configurations to average.
    phonon_sigma : float
        RMS thermal displacement (A) for the frozen phonons.
    phonon_seed : int
        Seed for the frozen-phonon displacements.
    parametrization : str
        Scattering-factor parametrization for abTEM.
    """
    k_max: float
    sg_max: float
    num_phonon_configs: int = 1
    phonon_sigma: float = 0.0
    phonon_seed: int = 42
    parametrization: str = "lobato"
    rocking_hkl: list = None
    n_workers: int = 2
    base_thickness_A: float = 20.0
    intensity_cut: float = 1.e-15


# ----------------------------------------------------------------
# Bloch-wave implementation
# ----------------------------------------------------------------

class BlochSimulator(Simulator):
    """
    Dynamical diffraction via the abTEM Bloch-wave method.

    Parameters
    ----------
    params : BlochParams
        Bloch backend configuration.
    energy_eV : float
        Electron energy, taken from the experiment Beam.
    verbose : bool
        Print the beam count and time of each calculation.
    """

    def __init__(self, params: BlochParams, energy_eV: float,
                 verbose: bool = True):
        self.params = params
        self.energy_eV = energy_eV
        self.verbose = verbose

    # -- internal helpers ----------------------------------------

    def _make_frozen_phonons(self, atoms):
        """Build the abTEM FrozenPhonons ensemble."""
        import abtem
        p = self.params
        frozen = abtem.FrozenPhonons(
            atoms,
            num_configs=p.num_phonon_configs,
            sigmas=p.phonon_sigma,
            seed=p.phonon_seed,
        )
        return frozen

    def _diffraction_for_config(self, atoms_cfg, thicknesses_A):
        """
        One Bloch-wave calculation for a single phonon
        configuration at all the given thicknesses.

        abTEM diagonalises the structure matrix once (cost ~N^3
        in the number of beams N) and then evaluates each
        thickness from the eigenvectors (~N^2 each), so extra
        thicknesses are cheap.

        Returns (positions, millers, intensities, n_beams), with
        intensities of shape (len(thicknesses_A), n_spots).
        """
        from abtem.bloch import BlochWaves, StructureFactor
        p = self.params
        sf = StructureFactor(
            atoms_cfg,
            k_max=p.k_max,
            parametrization=p.parametrization,
        )
        bw = BlochWaves(
            structure_factor=sf,
            energy=self.energy_eV,
            sg_max=p.sg_max,
        )
        calc = bw.calculate_diffraction_patterns
        patterns = calc(list(thicknesses_A))
        patterns = patterns.compute()
        first = patterns[0]
        ints = np.array([np.asarray(patterns[i].intensities)
                         for i in range(len(thicknesses_A))])
        return (first.positions, first.miller_indices,
                ints, len(bw))

    # -- public interface ----------------------------------------

    def build_bloch_waves(self, atoms, config_index=0):
        """
        Build the abTEM BlochWaves object for oriented atoms.

        Exposes the Bloch-wave problem itself (beam pool,
        structure matrix, scattering matrix) rather than just
        the resulting intensities, so a caller can export or
        inspect the matrices for one orientation.  All abTEM
        imports stay inside this module.

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms already oriented into the lab frame.
        config_index : int
            Which frozen-phonon configuration to use; the
            first one is the only one when
            num_phonon_configs == 1.

        Returns
        -------
        abtem.bloch.BlochWaves
        """
        from abtem.bloch import BlochWaves, StructureFactor
        p = self.params
        configs = list(self._make_frozen_phonons(atoms))
        sf = StructureFactor(
            configs[config_index],
            k_max=p.k_max,
            parametrization=p.parametrization,
        )
        return BlochWaves(
            structure_factor=sf,
            energy=self.energy_eV,
            sg_max=p.sg_max,
        )

    def simulate(self, atoms, thicknesses_A) -> list:
        """
        Average diffraction over frozen-phonon configurations,
        for every thickness at once.

        Intensities are accumulated keyed by Miller index so
        configurations with different surviving spot sets are
        combined correctly; low-intensity removal happens once,
        after averaging.  Returns one Spots per thickness.
        """
        thicknesses_A = [float(t) for t in np.atleast_1d(thicknesses_A)]
        n_t = len(thicknesses_A)
        frozen = self._make_frozen_phonons(atoms)

        accum_int = {}     # (h,k,l) -> (n_t,) summed intensity
        accum_pos = {}     # (h,k,l) -> position (kx,ky,kz)
        n_configs = 0
        printed_beams = False

        import time
        for atoms_cfg in frozen:

            t1 = time.time()
            (positions, millers, ints,
             n_beams) = self._diffraction_for_config(
                atoms_cfg, thicknesses_A
            )

            t2 = time.time()
            if self.verbose and not printed_beams:
                print(f"    Beams: {n_beams}  Time: {t2 - t1:.5f}")
                printed_beams = True

            for i in range(ints.shape[1]):
                hkl = (
                    int(millers[i][0]),
                    int(millers[i][1]),
                    int(millers[i][2]),
                )
                prev = accum_int.get(hkl, np.zeros(n_t))
                accum_int[hkl] = prev + ints[:, i]
                if hkl not in accum_pos:
                    accum_pos[hkl] = positions[i]
            n_configs += 1

        return self._finalize(accum_int, accum_pos, n_configs, n_t)

    def _finalize(self, accum_int, accum_pos, n_configs, n_t):
        """
        Build one Spots object per thickness from the accumulated
        dictionaries, averaging over the phonon configurations.
        All share the same reflections in the same order.  No
        intensity filtering is applied: the engine filters on
        integrated intensity after image integration.
        """
        hkl_keys = sorted(accum_int.keys())
        positions = np.array(
            [accum_pos[hkl] for hkl in hkl_keys]
        )
        millers = [list(hkl) for hkl in hkl_keys]
        mean_int = np.array(
            [accum_int[hkl] / n_configs for hkl in hkl_keys]
        ).reshape(len(hkl_keys), n_t)

        return [
            Spots(
                positions=positions,
                millers=millers,
                intensities=mean_int[:, t].copy(),
            )
            for t in range(n_t)
        ]
