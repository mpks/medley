"""
export_matrices.py — export the Bloch-wave structure matrix A
and scattering matrix S(t) for a single orientation of a scan.

The integrator sums many substeps into one image; this exporter
picks one substep of one image (by default the middle one, which
for an even n_substeps falls exactly on the image centre angle),
rebuilds the Bloch-wave problem for that orientation, and writes
A, S and the projector diagonal M to their own NPZ file.  Nothing
is added to the per-image NPZ, and nothing travels through the
worker processes.

The orientation, thickness and phonon configuration are built
exactly as the engine builds them, so the exported matrices are
the ones the engine used at that substep:

    angle       = scan.substep_angles(image)[substep]
    thickness   = base_thickness_A / cos(angle)
    atoms       = orient_atoms(base_atoms, geometry, angle, beam)

abTEM defines the matrices (abtem/bloch/dynamical.py) as

    A_gh = prefactor * V_{g-h} * M_g * M_h      (off-diagonal)
    A_gg = (2 * s_g / lambda) * M_g             (diagonal)
    S(z) = M @ expm(1j * pi * z * lambda * A) @ inv(M)
    M_g  = 1 / sqrt(1 + g_z / k0),   k0 = 1 / lambda

so the saved file is self-contained: with A, M (as the diagonal
vector), thickness_A and wavelength_A, S can be recomputed and
compared, e.g.

    from scipy.linalg import expm
    d = np.load("matrices_run_0030.npz")
    E = expm(1j * np.pi * d["thickness_A"]
             * d["wavelength_A"] * d["A"])
    S = (d["M"][:, None] * E) / d["M"][None, :]
    assert np.allclose(S, d["S"])

Row and column order of A and S is the order of the saved hkl
array, which is abTEM's beam-pool order (bw.hkl) for this
orientation.  That order is orientation-dependent: a different
substep excites a different subset of reflections, so hkl must
be read from each file rather than assumed.
"""

from __future__ import annotations

import os

import numpy as np

from medley.engine import (
    make_base_atoms,
    orient_atoms,
)


def matrices_filename(out_dir, tag, image_index):
    """Per-image matrix NPZ path, 4-digit zero-padded index."""
    name = f"matrices_{tag}_{image_index:04d}.npz"
    return os.path.join(out_dir, name)


def resolve_substep(scan, substep_index):
    """
    Validate a substep index, defaulting to the middle substep.

    With an even n_substeps the middle index n_substeps // 2
    lands on start + delta / 2, i.e. exactly the image centre
    angle recorded as angle_centre_deg.  With an odd
    n_substeps no substep sits on the centre and the index
    below it is used.
    """
    n = scan.n_substeps
    if substep_index is None:
        return n // 2
    idx = int(substep_index)
    if idx < 0:
        idx += n
    if not (0 <= idx < n):
        raise ValueError(
            f"Substep index {substep_index} out of range "
            f"[0, {n - 1}]."
        )
    return idx


def projector_diagonal(bw):
    """
    The diagonal of abTEM's projector matrix M for this beam
    pool, M_g = 1 / sqrt(1 + g_z / k0), returned as a vector
    of length N (M itself is diag(M_g)).

    Computed from the Bloch-wave object's own lab-frame g
    vectors, so it matches abTEM's calculate_M_matrix for the
    same orientation.
    """
    g = np.asarray(bw.g_vec, dtype=float)
    k0 = 1.0 / bw.wavelength
    return 1.0 / np.sqrt(1.0 + g[:, 2] / k0)


def compute_matrices(bw, thickness_A):
    """
    Build (A, S) for an existing BlochWaves object at a given
    thickness.

    A is computed eagerly once and reused for S, so the two
    saved matrices are guaranteed to correspond to the same
    structure factors (calling bw.calculate_scattering_matrix
    would rebuild A internally).
    """
    from abtem.bloch.dynamical import (
        calculate_scattering_matrix,
    )

    A = np.array(
        bw.calculate_structure_matrix(lazy=False), copy=True
    )
    S = calculate_scattering_matrix(
        A=A, hkl=bw.hkl, cell=bw.cell, z=thickness_A,
        energy=bw.energy,
    )
    return A, np.asarray(S)


def export_matrices(cif_file, geometry, beam, scan, simulator,
                    engine_params, image_index, out_dir, tag,
                    substep_index=None, report=True):
    """
    Export A, S and M for one substep of one image.

    Parameters
    ----------
    cif_file : str
    geometry : Geometry
    beam : Beam
        Needed to rotate the crystal into abTEM's frame.
    scan : Scan
    simulator : BlochSimulator
        Must expose build_bloch_waves(atoms).
    engine_params : EngineParams
        Supplies base_thickness_A.
    image_index : int
    out_dir, tag : str
        Output location and run label.
    substep_index : int or None
        Substep within the image; None selects the middle one.

    Returns
    -------
    str : path of the written NPZ.
    """
    os.makedirs(out_dir, exist_ok=True)

    idx = resolve_substep(scan, substep_index)
    angle = float(scan.substep_angles(image_index)[idx])
    tilt_rad = np.deg2rad(angle)
    thickness_A = (
        engine_params.base_thickness_A / np.cos(tilt_rad)
    )

    base_atoms = make_base_atoms(cif_file, geometry)
    atoms = orient_atoms(base_atoms, geometry, angle, beam)
    bw = simulator.build_bloch_waves(atoms)

    n_beams = int(bw.num_beams)
    if report:
        mb = n_beams ** 2 * 16 / 1e6
        print(
            f"  exporting matrices: image {image_index}, "
            f"substep {idx}, angle {angle:.4f} deg, "
            f"{n_beams} beams ({mb:.1f} MB per matrix)",
            flush=True,
        )

    A, S = compute_matrices(bw, thickness_A)
    M = projector_diagonal(bw)

    path = matrices_filename(out_dir, tag, image_index)
    np.savez_compressed(
        path,
        A=A,
        S=S,
        M=M,
        hkl=np.asarray(bw.hkl, dtype=np.int32),
        g_vec=np.asarray(bw.g_vec, dtype=float),
        sg=np.asarray(bw.excitation_errors(), dtype=float),
        thickness_A=thickness_A,
        angle_deg=angle,
        image_index=image_index,
        substep_index=idx,
        n_substeps=scan.n_substeps,
        delta_deg=scan.delta_deg,
        energy_eV=float(bw.energy),
        wavelength_A=float(bw.wavelength),
        k_max_pool=float(bw.k_max),
        sg_max=float(bw.sg_max),
        n_beams=n_beams,
    )
    if report:
        size_mb = os.path.getsize(path) / 1e6
        print(
            f"  wrote {os.path.basename(path)} "
            f"({size_mb:.1f} MB)",
            flush=True,
        )
    return path
