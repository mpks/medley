"""
excitation.py — signed excitation error s_g for the spots of one
image, evaluated at the wedge centre angle.

s_g is how far the reciprocal lattice point g sits from the Ewald
sphere, measured along the beam, in A^-1.  s_g = 0 is exact
Bragg; the sign says which side.  It is pure geometry: no
structure factors, no phases, so it is available at deployment
time as well as in simulation.

Two values are produced for every spot.

  sg        read out of abTEM's Bloch-wave object via
            bw.excitation_errors().  This is the number that
            actually entered the structure matrix diagonal
            (A_gg = 2 s_g M_g / lambda) and therefore produced
            the simulated intensity.

  sg_geom   recomputed here from the oriented cell alone:

                g   = inv(cell_rows) @ h
                s_g = -(|g|^2 + 2 s0.g) / (2 k)

            with k = 1 / lambda.

They are stored separately and compared, rather than one being
trusted.  A sign or frame error in either is silent — it would
still produce plausible-looking numbers, and everything built on
s_g later (w_g, and every 1/(2 k s_h) denominator in a Bethe sum)
would be wrong in a way no downstream check would catch.

FRAME NOTE.  abTEM computes with the beam along +z; the tilt onto
the experimental beam direction is applied to the scattered cone
afterwards, in projection.tilt_rotation.  The crystal handed to
abTEM is already in the lab frame via Geometry.oriented_cell.  So
the geometric check must use s0 = (0, 0, 1/lambda), NOT
beam.direction.  Using the experimental direction here produces a
mismatch that looks like a sign error but is a frame error.

SIGN CONVENTION of sg_geom: s_g > 0 means the reciprocal lattice
point lies inside the Ewald sphere.  Whether abTEM agrees is not
assumed — it is measured, and reported, on every image.

BEAM POOL.  The pool abTEM builds is orientation dependent (it is
cut by sg_max).  An image integrates many substeps, so its spot
list can contain reflections that are not excited at the centre
angle.  Those get NaN, and the count is reported.  They are not
silently dropped and not filled with zero: zero would read as
"exact Bragg", the opposite of the truth.
"""

from __future__ import annotations

import numpy as np

from dynamic.simulation.engine import (
    make_base_atoms,
    orient_atoms,
)


# ----------------------------------------------------------------
# Pure geometry
# ----------------------------------------------------------------

def reciprocal_vectors(cell_rows, millers):
    """
    Lab-frame scattering vectors g for a list of Miller indices.

    cell_rows : (3, 3) real-space cell, ASE row convention
                (rows are a, b, c), as returned by
                Geometry.oriented_cell.
    millers   : (N, 3) integer Miller indices.

    With C the row-convention cell, inv(C) has a*, b*, c* as its
    columns (since row_i(C) . col_j(inv C) = delta_ij), so
    g = inv(C) @ h.  Returned as (N, 3) with g in the rows.
    """
    C = np.asarray(cell_rows, dtype=float)
    h = np.asarray(millers, dtype=float).reshape(-1, 3)
    return h @ np.linalg.inv(C).T


def geometric_sg(cell_rows, millers, wavelength_A,
                 beam_axis=(0.0, 0.0, 1.0)):
    """
    Signed excitation error from the oriented cell alone.

        s_g = -(|g|^2 + 2 s0.g) / (2 k),   k = 1 / lambda

    beam_axis defaults to +z, matching abTEM's internal frame
    (see the module docstring).  Returns (N,) in A^-1.
    """
    g = reciprocal_vectors(cell_rows, millers)
    k = 1.0 / wavelength_A
    axis = np.asarray(beam_axis, dtype=float)
    s0 = k * axis / np.linalg.norm(axis)
    return -(np.sum(g * g, axis=1) + 2.0 * (g @ s0)) / (2.0 * k)


# ----------------------------------------------------------------
# Alignment of abTEM's beam pool onto the image spot list
# ----------------------------------------------------------------

def align_to_millers(pool_hkl, pool_values, millers):
    """
    Reorder per-beam values from abTEM's pool order onto the
    image's spot list.

    Reflections absent from the pool at this angle get NaN.
    Returns (values, n_missing).
    """
    lookup = {}
    for row, val in zip(np.asarray(pool_hkl, dtype=int),
                        np.asarray(pool_values, dtype=float)):
        lookup[(int(row[0]), int(row[1]), int(row[2]))] = val

    out = np.full(len(millers), np.nan, dtype=float)
    n_missing = 0
    for i, hkl in enumerate(millers):
        key = (int(hkl[0]), int(hkl[1]), int(hkl[2]))
        if key in lookup:
            out[i] = lookup[key]
        else:
            n_missing += 1
    return out, n_missing


# ----------------------------------------------------------------
# Agreement diagnostic
# ----------------------------------------------------------------

def compare_sg(sg, sg_geom):
    """
    Compare the abTEM read-out against the geometric
    recomputation over the spots where both are finite.

    Returns a dict with

      n          spots compared
      sign       +1 or -1, the best-fit scale rounded; +1 means
                 the conventions agree
      scale      least-squares scale sg ~ scale * sg_geom
      corr       Pearson correlation
      rms_resid  rms of (sg - scale * sg_geom), in A^-1

    A clean extraction gives scale close to +/-1, corr close to
    +/-1 and rms_resid small against the spread of s_g.  Anything
    else means the frames differ and nothing downstream should be
    trusted.
    """
    ok = np.isfinite(sg) & np.isfinite(sg_geom)
    n = int(np.sum(ok))
    if n < 2:
        return {'n': n, 'sign': 0, 'scale': np.nan,
                'corr': np.nan, 'rms_resid': np.nan}

    a = sg[ok]
    b = sg_geom[ok]
    denom = float(np.sum(b * b))
    scale = float(np.sum(a * b) / denom) if denom > 0 else np.nan
    resid = a - scale * b
    corr = float(np.corrcoef(a, b)[0, 1])

    return {
        'n': n,
        'sign': int(np.sign(scale)) if np.isfinite(scale) else 0,
        'scale': scale,
        'corr': corr,
        'rms_resid': float(np.sqrt(np.mean(resid ** 2))),
    }


# ----------------------------------------------------------------
# Top-level extraction for one image
# ----------------------------------------------------------------

def excitation_for_image(cif_file, geometry, beam, simulator,
                         angle_centre_deg, millers,
                         report=True):
    """
    Build the Bloch-wave problem at the wedge centre and return
    the excitation errors for that image's spots.

    The angle used is angle_centre_deg itself, taken straight
    from the ImageResult, so the value does not depend on
    n_substeps being even.  No diffraction patterns are
    computed: only the beam pool and its excitation errors are
    needed, so the eigendecomposition is skipped.

    Returns (sg, sg_geom, info) where sg and sg_geom are (N,)
    arrays aligned to millers and info is the compare_sg dict
    with 'n_missing' added.  Returns (None, None, None) if the
    simulator cannot expose a Bloch-wave object.
    """
    if not hasattr(simulator, 'build_bloch_waves'):
        if report:
            print('  s_g export skipped: simulator has no '
                  'build_bloch_waves')
        return None, None, None

    if len(millers) == 0:
        empty = np.zeros(0, dtype=float)
        return empty, empty, {'n': 0, 'n_missing': 0,
                              'sign': 0, 'scale': np.nan,
                              'corr': np.nan,
                              'rms_resid': np.nan}

    angle = float(angle_centre_deg)
    base_atoms = make_base_atoms(cif_file, geometry)
    atoms = orient_atoms(base_atoms, geometry, angle)

    bw = simulator.build_bloch_waves(atoms)
    pool_hkl = np.asarray(bw.hkl, dtype=int)
    pool_sg = np.asarray(bw.excitation_errors(), dtype=float)

    sg, n_missing = align_to_millers(pool_hkl, pool_sg, millers)

    cell_rows = geometry.oriented_cell(angle)
    sg_geom = geometric_sg(cell_rows, millers,
                           beam.wavelength_A)

    info = compare_sg(sg, sg_geom)
    info['n_missing'] = n_missing

    if report:
        _report(info, len(millers))

    return sg, sg_geom, info


def _report(info, n_spots):
    """Print the per-image agreement line."""
    if info['n_missing']:
        print(f"  s_g: {info['n_missing']} of {n_spots} spots "
              "not in the centre-angle beam pool (NaN)")
    if info['n'] < 2:
        print('  s_g: too few spots to check agreement')
        return
    clean = abs(abs(info['corr']) - 1.0) < 1e-3
    flag = '' if clean else '  <-- CHECK'
    print(f"  s_g: abTEM vs geometric  scale {info['scale']:+.4f}"
          f"  corr {info['corr']:+.6f}"
          f"  rms {info['rms_resid']:.3e} A^-1{flag}")
