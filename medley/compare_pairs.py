"""
compare_pairs.py — compare Friedel-pair intensity differences
between simulation and experiment.

For every Friedel pair (h, -h) the sign of I(h) - I(-h) in the
experiment is compared with the sign predicted by the simulation.
The result is the percentage of pairs whose signs agree:

  well above 50 %  the data match the hand used in the simulation;
  well below 50 %  the data match the opposite hand;
  near 50 %        the comparison cannot tell the hands apart.

One or more simulation files can be given (typically the same
crystal at several thicknesses).  Only pairs whose simulated sign
is the same across those files ("sign-stable" pairs) are used.
With a single simulation file every simulated pair is used.

Each simulation is first relabelled into the experiment's setting
by the proper rotation of the point group that best matches the
two sets of reflections.  Proper rotations only relabel
reflections and never change the hand, so this cannot bias the
result.

Usage:
    medley.compare_pairs exp.hkl sim_0100.hkl sim_0200.hkl ... \\
        --point_group 2

Input files are SHELX HKLF 4 (3I4, 2F8.2), as written by
medley.integrate or by DIALS / XDS exports.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


# ----------------------------------------------------------------
# Reading
# ----------------------------------------------------------------

def _parse_line(line):
    """
    Parse one HKLF 4 line into (h, k, l, I, sigma), or None.

    The format is fixed width (3I4, 2F8.2), and large values run
    into each other with no space between them, so columns are cut
    by position.  Lines that are not in that layout fall back to
    whitespace splitting.
    """
    s = line.rstrip("\n")
    if not s.strip() or s.lstrip().startswith("#"):
        return None
    try:
        h, k, l = int(s[0:4]), int(s[4:8]), int(s[8:12])
        inten, sig = float(s[12:20]), float(s[20:28])
    except ValueError:
        vals = s.split()
        if len(vals) < 5:
            return None
        try:
            h, k, l = (int(v) for v in vals[:3])
            inten, sig = float(vals[3]), float(vals[4])
        except ValueError:
            return None
    return h, k, l, inten, sig


def read_hkl(path, signal=None):
    """
    Read an HKLF 4 file into ({hkl: I}, {hkl: sigma}).

    Reading stops at the 0 0 0 terminator.  With signal set, only
    reflections with sigma > 0 and I / sigma > signal are kept.
    """
    intens, sigmas = {}, {}
    with open(path) as fh:
        for line in fh:
            rec = _parse_line(line)
            if rec is None:
                continue
            h, k, l, inten, sig = rec
            if h == 0 and k == 0 and l == 0:
                break
            if signal is not None and not (sig > 0 and inten / sig > signal):
                continue
            intens[(h, k, l)] = inten
            sigmas[(h, k, l)] = sig
    return intens, sigmas


# ----------------------------------------------------------------
# Friedel bookkeeping
# ----------------------------------------------------------------

def mate(hkl):
    h, k, l = hkl
    return (-h, -k, -l)


def canonical(hkl):
    """One representative per Friedel pair."""
    return min(hkl, mate(hkl))


def friedel_pairs(intens):
    """{canonical hkl: (I(h), I(-h))} for pairs present in full."""
    pairs = {}
    for hkl in intens:
        key = canonical(hkl)
        if key in pairs or mate(hkl) not in intens:
            continue
        pairs[key] = (intens[key], intens[mate(key)])
    return pairs


# ----------------------------------------------------------------
# Setting alignment by proper point-group rotations
# ----------------------------------------------------------------

POINT_GROUPS = {
    "1": {
        "I": (1, 1, 1),
    },
    "2": {                      # monoclinic, b unique
        "I": (1, 1, 1),
        "2_b": (-1, 1, -1),
    },
    "222": {                    # orthorhombic
        "I": (1, 1, 1),
        "2_a": (1, -1, -1),
        "2_b": (-1, 1, -1),
        "2_c": (-1, -1, 1),
    },
}


def apply_op(intens, signs):
    sx, sy, sz = signs
    return {(sx * h, sy * k, sz * l): v for (h, k, l), v in intens.items()}


def best_alignment(intens, ref_keys, ops):
    """
    The operation that maximises the overlap of reflection sets.
    Returns (name, signs, overlap, runner_up_overlap).
    """
    ref = set(ref_keys)
    scored = sorted(
        ((len(set(apply_op(intens, s)) & ref), name, s)
         for name, s in ops.items()),
        key=lambda t: t[0], reverse=True,
    )
    overlap, name, signs = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0
    return name, signs, overlap, second


# ----------------------------------------------------------------
# Sign-stable pairs from the simulations
# ----------------------------------------------------------------

def sign_stable_pairs(sim_pairs_list, min_frac=1.0, min_valid=2,
                      sim_asym_min=0.0):
    """
    Pairs whose simulated sign of I(h) - I(-h) is stable across the
    simulations.  Returns {canonical hkl: (sign, strength)}, with
    strength the mean of |I+ - I-| / (I+ + I-).

    A pair is kept if it is present (both intensities > 0) in at
    least min_valid simulations, the majority sign has a fraction
    of at least min_frac, there is no tie, and the strength is at
    least sim_asym_min.
    """
    candidates = set()
    for pairs in sim_pairs_list:
        candidates |= set(pairs)

    stable = {}
    for key in candidates:
        signs, asyms = [], []
        for pairs in sim_pairs_list:
            if key not in pairs:
                continue
            i_pos, i_neg = pairs[key]
            if i_pos <= 0 or i_neg <= 0:
                continue
            signs.append(np.sign(i_pos - i_neg))
            asyms.append(abs(i_pos - i_neg) / (i_pos + i_neg))
        if len(signs) < min_valid:
            continue
        signs = np.array(signs)
        n_pos = int(np.sum(signs > 0))
        n_neg = int(np.sum(signs < 0))
        if n_pos == n_neg:
            continue
        if max(n_pos, n_neg) / len(signs) < min_frac:
            continue
        strength = float(np.mean(asyms))
        if strength < sim_asym_min:
            continue
        stable[key] = (1.0 if n_pos > n_neg else -1.0, strength)
    return stable


# ----------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------

def compare(exp_file, sim_files, point_group="1", signal=5.0,
            z_min=1.5, min_frac=1.0, min_valid=None,
            sim_asym_min=0.0, weight=False, sig_call=2.0, n_min=30):
    """Run the comparison; return a dict of results and statistics."""
    if min_valid is None:
        min_valid = min(2, len(sim_files))
    ops = POINT_GROUPS[point_group]

    exp_all, _ = read_hkl(exp_file)
    exp_i, exp_s = read_hkl(exp_file, signal=signal)
    exp_pairs = friedel_pairs(exp_i)

    stats = {
        "experiment": exp_file,
        "exp_reflections": len(exp_all),
        "exp_reflections_after_signal_cut": len(exp_i),
        "exp_friedel_pairs": len(exp_pairs),
        "simulations": [],
    }

    ref_keys = set(exp_all)
    sim_pairs_list = []
    for path in sim_files:
        sim_i, _ = read_hkl(path)
        name, signs, overlap, second = best_alignment(sim_i, ref_keys, ops)
        aligned = apply_op(sim_i, signs)
        pairs = friedel_pairs(aligned)
        sim_pairs_list.append(pairs)
        stats["simulations"].append({
            "file": path,
            "reflections": len(sim_i),
            "friedel_pairs": len(pairs),
            "alignment": name,
            "overlap": overlap,
            "runner_up_overlap": second,
        })

    stable = sign_stable_pairs(sim_pairs_list, min_frac=min_frac,
                               min_valid=min_valid,
                               sim_asym_min=sim_asym_min)
    stats["sim_candidate_pairs"] = len(set().union(*sim_pairs_list))
    stats["sim_stable_pairs"] = len(stable)

    in_both = [k for k in stable if k in exp_pairs]
    stats["pairs_in_both"] = len(in_both)

    agree, wts = [], []
    n_nonpositive = n_zcut = 0
    for key in in_both:
        s_sim, strength = stable[key]
        i_pos, i_neg = exp_pairs[key]
        if i_pos <= 0 or i_neg <= 0:
            n_nonpositive += 1
            continue
        if z_min is not None:
            sig_d = np.hypot(exp_s[key], exp_s[mate(key)])
            if sig_d <= 0 or abs(i_pos - i_neg) / sig_d < z_min:
                n_zcut += 1
                continue
        agree.append(1.0 if np.sign(i_pos - i_neg) == s_sim else 0.0)
        wts.append(strength)

    stats["rejected_nonpositive"] = n_nonpositive
    stats["rejected_z_cut"] = n_zcut
    n = len(agree)
    stats["pairs_used"] = n
    stats["pairs_matched"] = int(sum(agree))

    result = {"percent": float("nan"), "stderr": float("nan"),
              "n_eff": 0.0, "significance": float("nan"),
              "call": "undet"}
    if n > 0:
        agree = np.array(agree)
        w = np.array(wts) if (weight and sum(wts) > 0) else np.ones(n)
        sw = float(np.sum(w))
        p = float(np.sum(w * agree) / sw)
        n_eff = sw ** 2 / float(np.sum(w ** 2))
        perc = 100.0 * p
        err = 100.0 * np.sqrt(p * (1.0 - p) / n_eff)
        sig = abs(perc - 50.0) / err if err > 0 else float("inf")
        call = "undet"
        if n >= n_min and sig >= sig_call:
            call = "same" if perc > 50 else "opposite"
        result.update(percent=perc, stderr=err, n_eff=n_eff,
                      significance=sig, call=call)
    stats.update(result)
    stats["settings"] = dict(point_group=point_group, signal=signal,
                             z_min=z_min, min_frac=min_frac,
                             min_valid=min_valid,
                             sim_asym_min=sim_asym_min,
                             weight=weight, sig_call=sig_call,
                             n_min=n_min)
    return stats


# ----------------------------------------------------------------
# Output
# ----------------------------------------------------------------

def print_text(st, out=sys.stdout):
    p = lambda *a: print(*a, file=out)
    p(f"Experiment  {st['experiment']}")
    p(f"  reflections               {st['exp_reflections']}")
    p(f"  after I/sigma > {st['settings']['signal']:<9g} "
      f"{st['exp_reflections_after_signal_cut']}")
    p(f"  Friedel pairs             {st['exp_friedel_pairs']}")
    p("Simulations")
    p(f"  {'file':<30} {'refl':>7} {'pairs':>7}  {'align':>5} "
      f"{'overlap':>8} {'2nd':>8}")
    for s in st["simulations"]:
        p(f"  {s['file']:<30} {s['reflections']:7d} "
          f"{s['friedel_pairs']:7d}  {s['alignment']:>5} "
          f"{s['overlap']:8d} {s['runner_up_overlap']:8d}")
    p(f"  candidate pairs           {st['sim_candidate_pairs']}")
    p(f"  sign-stable pairs         {st['sim_stable_pairs']}")
    p("Comparison")
    p(f"  stable pairs also in exp  {st['pairs_in_both']}")
    p(f"  rejected (I <= 0)         {st['rejected_nonpositive']}")
    p(f"  rejected (z < {st['settings']['z_min']})        "
      f"{st['rejected_z_cut']}")
    p(f"  pairs used                {st['pairs_used']}")
    p(f"  pairs matched             {st['pairs_matched']}")
    if st["pairs_used"]:
        p(f"  match                     {st['percent']:.1f} % "
          f"+/- {st['stderr']:.1f}  (n_eff {st['n_eff']:.1f})")
        p(f"  significance              {st['significance']:.1f} "
          f"sigma from 50 %")
    p(f"  hand                      {st['call']}")


def print_tsv(st, header, out=sys.stdout):
    cols = ["experiment", "percent", "stderr", "significance", "call",
            "pairs_used", "pairs_matched", "n_eff", "sim_stable_pairs",
            "exp_friedel_pairs", "pairs_in_both"]
    if header:
        print("\t".join(cols), file=out)
    vals = []
    for c in cols:
        v = st[c]
        vals.append(f"{v:.3f}" if isinstance(v, float) else str(v))
    print("\t".join(vals), file=out)


def build_parser():
    p = argparse.ArgumentParser(
        description="Percentage of Friedel pairs whose intensity "
                    "difference has the same sign in experiment "
                    "and simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("exp_hkl", help="Experimental HKLF 4 file")
    p.add_argument("sim_hkl", nargs="+",
                   help="Simulated HKLF 4 file(s), e.g. one per "
                        "thickness")
    p.add_argument("--point_group", choices=sorted(POINT_GROUPS),
                   default="1",
                   help="Point group whose proper rotations are used "
                        "to bring each simulation into the "
                        "experiment's setting ('2' = monoclinic, "
                        "b unique)")
    p.add_argument("--signal", type=float, default=5.0,
                   help="Keep experimental reflections with "
                        "I/sigma above this")
    p.add_argument("--z_min", type=float, default=1.5,
                   help="Keep experimental pairs whose |I+ - I-| is "
                        "at least this many sigma")
    p.add_argument("--min_frac", type=float, default=1.0,
                   help="Fraction of simulations that must agree on "
                        "a pair's sign")
    p.add_argument("--min_valid", type=int, default=None,
                   help="Simulations a pair must appear in "
                        "(default: 2, or 1 with a single file)")
    p.add_argument("--sim_asym_min", type=float, default=0.0,
                   help="Minimum simulated |I+ - I-| / (I+ + I-)")
    p.add_argument("--weight", action="store_true",
                   help="Weight each pair by its simulated asymmetry")
    p.add_argument("--sig_call", type=float, default=2.0,
                   help="Sigma from 50 %% needed to call the hand")
    p.add_argument("--n_min", type=int, default=30,
                   help="Pairs needed to call the hand")
    p.add_argument("--format", choices=["text", "tsv", "json"],
                   default="text", help="Output format")
    p.add_argument("--no_header", action="store_true",
                   help="With --format tsv: omit the header line")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    st = compare(args.exp_hkl, args.sim_hkl,
                 point_group=args.point_group, signal=args.signal,
                 z_min=args.z_min, min_frac=args.min_frac,
                 min_valid=args.min_valid,
                 sim_asym_min=args.sim_asym_min, weight=args.weight,
                 sig_call=args.sig_call, n_min=args.n_min)
    if args.format == "json":
        print(json.dumps(st, indent=2))
    elif args.format == "tsv":
        print_tsv(st, header=not args.no_header)
    else:
        print_text(st)


if __name__ == "__main__":
    main()
