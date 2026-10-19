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

from medley.simulation.experiment import Spots


# ----------------------------------------------------------------
# Abstract interface
# ----------------------------------------------------------------

class Simulator(ABC):
    """
    Abstract diffraction simulator.

    Concrete subclasses implement simulate, which takes an
    oriented ase.Atoms object and a propagation thickness and
    returns a Spots object (reciprocal-space positions in the
    lab frame, Miller indices, and intensities).
    """

    @abstractmethod
    def simulate(self, atoms, thickness_A) -> Spots:
        """
        Simulate diffraction from oriented atoms.

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms already oriented into the lab frame for the
            desired scan angle.
        thickness_A : float
            Crystal thickness propagated through, in Angstrom.

        Returns
        -------
        Spots
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
        Maximum scattering vector (A^-1); sets the structure
        factor resolution (d_min = 1 / k_max).
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
    """

    def __init__(self, params: BlochParams, energy_eV: float):
        self.params = params
        self.energy_eV = energy_eV

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

    def _diffraction_for_config(self, atoms_cfg, thickness_A):
        """
        One Bloch-wave calculation for a single phonon
        configuration at the given thickness.

        Returns (positions, millers, intensities, n_beams).
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
        patterns = calc([thickness_A])
        patterns = patterns.compute()
        spots = patterns[0]
        ints = np.array(spots.intensities)
        return (spots.positions, spots.miller_indices,
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

    def simulate(self, atoms, thickness_A) -> Spots:
        """
        Average diffraction over frozen-phonon configurations.

        Intensities are accumulated keyed by Miller index so
        configurations with different surviving spot sets are
        combined correctly; low-intensity removal happens once,
        after averaging.
        """
        frozen = self._make_frozen_phonons(atoms)

        accum_int = {}     # (h,k,l) -> summed intensity
        accum_pos = {}     # (h,k,l) -> position (kx,ky,kz)
        n_configs = 0
        printed_beams = False

        import time
        for atoms_cfg in frozen:
            
            t1 = time.time()
            (positions, millers, ints,
             n_beams) = self._diffraction_for_config(
                atoms_cfg, thickness_A
            )

            t2 = time.time()
            if not printed_beams:
                print(f"    Beams: {n_beams}  Time: {t2 - t1:.5f}")
                printed_beams = True

            for i in range(len(ints)):
                hkl = (
                    int(millers[i][0]),
                    int(millers[i][1]),
                    int(millers[i][2]),
                )
                prev = accum_int.get(hkl, 0.0)
                accum_int[hkl] = prev + ints[i]
                if hkl not in accum_pos:
                    accum_pos[hkl] = positions[i]
            n_configs += 1

        return self._finalize(accum_int, accum_pos, n_configs)

    def _finalize(self, accum_int, accum_pos, n_configs):
        """
        Build a Spots object from the accumulated dictionaries,
        averaging over the phonon configurations.  No intensity
        filtering is applied: the engine filters on integrated
        intensity after image integration.
        """
        hkl_keys = sorted(accum_int.keys())
        positions = np.array(
            [accum_pos[hkl] for hkl in hkl_keys]
        )
        millers = [list(hkl) for hkl in hkl_keys]
        mean_int = np.array(
            [accum_int[hkl] / n_configs for hkl in hkl_keys]
        )

        return Spots(
            positions=positions,
            millers=millers,
            intensities=mean_int,
        )
