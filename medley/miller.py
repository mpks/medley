"""
miller.py — read Miller indices from HKL or plain text files.

Two formats are supported and auto-detected per line:

  * SHELX HKL : 'h k l intensity sigma'  (>= 5 columns;
                only the first three are used; a trailing
                '0 0 0' terminator and the (0,0,0) reflection
                are skipped)
  * plain     : 'h k l'                  (exactly 3 columns)

Columns may be separated by whitespace or commas.  Lines that
are blank or start with '#' are ignored.

The result is a list of (h, k, l) integer tuples, in file
order, with duplicates removed (first occurrence kept).
"""

from __future__ import annotations


def _split_fields(line):
    """Split a line on commas and/or whitespace into tokens."""
    return line.replace(",", " ").split()


def _parse_hkl_line(line):
    """
    Parse one line into an (h, k, l) tuple, or return None if
    the line is blank, a comment, the SHELX terminator, or the
    (0, 0, 0) direct beam.

    Both 'h k l' (3 columns) and SHELX 'h k l I sigma'
    (>= 5 columns) are accepted; only the first three integer
    fields are used.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    fields = _split_fields(stripped)
    if len(fields) < 3:
        return None

    try:
        h = int(round(float(fields[0])))
        k = int(round(float(fields[1])))
        l = int(round(float(fields[2])))     # noqa: E741
    except ValueError:
        return None

    if h == 0 and k == 0 and l == 0:
        return None         # terminator / direct beam

    return (h, k, l)


def read_miller_indices(path):
    """
    Read Miller indices from an HKL or plain 3-column file.

    Parameters
    ----------
    path : str

    Returns
    -------
    list of (h, k, l) integer tuples, duplicates removed
    (first occurrence kept), in file order.
    """
    seen = set()
    result = []
    with open(path) as fh:
        for line in fh:
            hkl = _parse_hkl_line(line)
            if hkl is None:
                continue
            if hkl in seen:
                continue
            seen.add(hkl)
            result.append(hkl)
    return result


def parse_hkl_string(text):
    """
    Parse a single 'h,k,l' or 'h k l' string into a tuple.

    Useful for command-line arguments.  Raises ValueError if
    the string does not contain exactly three integers.
    """
    fields = _split_fields(text)
    if len(fields) != 3:
        raise ValueError(
            f"Expected three indices 'h k l', got {text!r}"
        )
    h = int(round(float(fields[0])))
    k = int(round(float(fields[1])))
    l = int(round(float(fields[2])))         # noqa: E741
    return (h, k, l)
