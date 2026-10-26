"""
invert_cif.py — write the inverted structure (r -> -r) of a CIF.

Every atom position is replaced by its inverse through the origin,
so a chiral molecule becomes its mirror image (the other
enantiomer).  The cell is unchanged.

The CIF is first expanded to the full unit cell with its symmetry
operators (ASE does this when reading), and the inverted structure
is written in P1.  This also handles chiral space groups: e.g. a
P3_1 structure becomes the corresponding P3_2 structure.

Usage:
    medley.invert_cif histidine_L.cif histidine_D.cif
"""

from __future__ import annotations

import argparse

import ase.io


def invert_atoms(atoms):
    """Return a copy with every fractional coordinate negated."""
    inverted = atoms.copy()
    inverted.set_scaled_positions(-atoms.get_scaled_positions())
    inverted.wrap()
    return inverted


def build_parser():
    p = argparse.ArgumentParser(
        description="Write the inverted (mirror-image) structure "
                    "of a CIF file, in P1.")
    p.add_argument("cif_in", help="Input CIF")
    p.add_argument("cif_out", help="Output CIF (inverted, P1)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    atoms = ase.io.read(args.cif_in)
    ase.io.write(args.cif_out, invert_atoms(atoms), format="cif")
    print(f"{len(atoms)} atoms inverted: {args.cif_in} -> {args.cif_out}")


if __name__ == "__main__":
    main()
