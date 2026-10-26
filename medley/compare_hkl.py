"""
compare_hkl.py — agreement statistics between two HKLF 4 files.

File A is the reference (normally the experiment), file B is
compared against it (normally a simulation).  Only reflections
present in both files are used; the counts show how many that is.
B is put on A's scale with one least-squares factor before any
R value is computed, so the two files may have arbitrary scales.

Statistics (over the common reflections):

  scale   k minimising sum w (I_A - k I_B)^2, applied to B
  R1      sum | F_A - sqrt(k) F_B | / sum F_A,   F = sqrt(max(I, 0))
  wR2     sqrt( sum w (I_A - k I_B)^2 / sum w I_A^2 )
  R_I     sum | I_A - k I_B | / sum | I_A |
  CC      Pearson correlation of I_A and I_B (scale independent)
  CC_rank Spearman rank correlation of I_A and I_B

Weights are w = 1 / sigma_A^2 (sigma from the reference file), or 1
for every reflection with --unit_weights.  Note that wR2 here uses
these weights, not the SHELXL weighting scheme, so it is meant for
comparing files with each other rather than with SHELXL output.

Usage:
    medley.compare_hkl exp.hkl sim.hkl --point_group 2
    medley.compare_hkl exp.hkl sim.hkl --header      # with column names
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import spearmanr

from medley.compare_pairs import (
    POINT_GROUPS,
    apply_op,
    best_alignment,
    read_hkl,
)

COLUMNS = ["file_a", "file_b", "n_a", "n_b", "n_common", "align",
           "scale", "R1", "wR2", "R_I", "CC", "CC_rank"]


def compare_hkl(file_a, file_b, point_group="1", signal=None,
                unit_weights=False):
    """Return a dict with the statistics listed in COLUMNS."""
    int_a, sig_a = read_hkl(file_a, signal=signal)
    int_b, _ = read_hkl(file_b)

    name, signs, _, _ = best_alignment(int_b, set(int_a),
                                       POINT_GROUPS[point_group])
    int_b = apply_op(int_b, signs)

    common = sorted(set(int_a) & set(int_b))
    out = dict(file_a=file_a, file_b=file_b, n_a=len(int_a),
               n_b=len(int_b), n_common=len(common), align=name)
    nan = float("nan")
    out.update(scale=nan, R1=nan, wR2=nan, R_I=nan, CC=nan,
               CC_rank=nan)
    if len(common) < 2:
        return out

    ia = np.array([int_a[h] for h in common])
    ib = np.array([int_b[h] for h in common])
    sa = np.array([sig_a[h] for h in common])

    if unit_weights or np.any(sa <= 0):
        w = np.ones_like(ia)
    else:
        w = 1.0 / sa ** 2

    k = float(np.sum(w * ia * ib) / np.sum(w * ib * ib))
    fa = np.sqrt(np.clip(ia, 0, None))
    fb = np.sqrt(np.clip(ib, 0, None))

    out["scale"] = k
    out["R1"] = float(np.sum(np.abs(fa - np.sqrt(k) * fb)) / np.sum(fa))
    out["wR2"] = float(np.sqrt(np.sum(w * (ia - k * ib) ** 2)
                               / np.sum(w * ia ** 2)))
    out["R_I"] = float(np.sum(np.abs(ia - k * ib)) / np.sum(np.abs(ia)))
    out["CC"] = float(np.corrcoef(ia, ib)[0, 1])
    out["CC_rank"] = float(spearmanr(ia, ib).statistic)
    return out


def format_line(stats):
    vals = []
    for c in COLUMNS:
        v = stats[c]
        if isinstance(v, float):
            vals.append(f"{v:.6g}" if c == "scale" else f"{v:.4f}")
        else:
            vals.append(str(v))
    return "\t".join(vals)


def build_parser():
    p = argparse.ArgumentParser(
        description="R1, wR2 and correlations between two HKLF 4 "
                    "files, over the reflections they share.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("hkl_a", help="Reference HKLF 4 file (e.g. experiment)")
    p.add_argument("hkl_b", help="HKLF 4 file compared with it "
                                 "(e.g. simulation)")
    p.add_argument("--point_group", choices=sorted(POINT_GROUPS),
                   default="1",
                   help="Point group whose proper rotations are used to "
                        "bring B into A's setting")
    p.add_argument("--signal", type=float, default=None,
                   help="Use only reflections of A with I/sigma above this")
    p.add_argument("--unit_weights", action="store_true",
                   help="Weight all reflections equally instead of "
                        "1/sigma_A^2")
    p.add_argument("--header", action="store_true",
                   help="Print a line of column names first")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    stats = compare_hkl(args.hkl_a, args.hkl_b,
                        point_group=args.point_group,
                        signal=args.signal,
                        unit_weights=args.unit_weights)
    if args.header:
        print("\t".join(COLUMNS))
    print(format_line(stats))


if __name__ == "__main__":
    main()
