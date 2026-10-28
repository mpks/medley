"""
params.py — JSON parameter files for the Medley CLI

    medley params synthetic  -o params.json   # write the defaults
    medley params experiment -o params.json
    medley run params.json                   # run it
    medley run params.json --k_max 3         # flags override the file

The JSON keys are the CLI option names without the leading
dashes, plus "mode" and the positional file paths (cif_file,
and expt_file in experiment mode). A null value means "not
set", the same as leaving the option off the command line.

The defaults are read from the argparse parser itself, so the
parser stays the single source of truth: an option added to
cli.build_parser shows up in new params files automatically.

A params file is turned back into a command line and parsed
by the same parser, so type checks, choices and the existing
validation all apply unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

MODES = ("experiment", "synthetic")

# Positional arguments per mode, in the order argparse wants them.
POSITIONALS = {
    "experiment": ("cif_file", "expt_file"),
    "synthetic": ("cif_file",),
}

# Written into a fresh params file; must be replaced before a run.
PLACEHOLDERS = {
    "cif_file": "path/to/structure.cif",
    "expt_file": "path/to/experiment.expt",
}

# Keys holding file paths; made absolute in the saved copy so
# the saved file can be re-run from any directory.
PATH_KEYS = ("cif_file", "expt_file", "rocking_hkl_file")
SAVED_NAME = "medley_params.json"


def default_params(parser, mode):
    """Return every parameter for `mode` with its default value."""
    if mode not in MODES:
        msg = f'"mode" must be one of {", ".join(MODES)}, '
        msg += f"got {mode!r}"
        raise ValueError(msg)
    argv = [mode] + [PLACEHOLDERS[k] for k in POSITIONALS[mode]]
    values = vars(parser.parse_args(argv))
    values.pop("mode")

    # mode and file paths first, then options in parser order
    params = {"mode": mode}
    for key in POSITIONALS[mode]:
        params[key] = values.pop(key)
    params.update(values)
    return params


def params_to_argv(parser, params):
    """Turn a params dict into an argv list for `parser`."""
    params = dict(params)
    mode = params.pop("mode", None)
    known = default_params(parser, mode)

    unknown = sorted(set(params) - set(known))
    if unknown:
        msg = f"unknown parameter(s) for {mode} mode: "
        msg += f"{', '.join(unknown)}"
        raise ValueError(msg)

    argv = [mode]
    for key in POSITIONALS[mode]:
        value = params.pop(key, None)
        if value is None or value == PLACEHOLDERS[key]:
            raise ValueError(f'"{key}" must be set')
        argv.append(str(value))

    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, bool):
            # store_true flags: present when true, absent when false
            if value:
                argv.append(f"--{key}")
            continue
        # "--key=value" so negative numbers are not read as flags
        argv.append(f"--{key}={value}")
    return argv


def load_params(path):
    """Read a params file."""
    with open(path) as f:
        params = json.load(f)
    if not isinstance(params, dict):
        raise ValueError(f"{path} must hold a JSON object")
    return params


def write_params(params, path, overwrite=False):
    """Write a params file, refusing to clobber unless asked."""
    path = Path(path)
    if path.exists() and not overwrite:
        msg = f"{path} already exists (use --force to overwrite)"
        raise FileExistsError(msg)
    path.write_text(json.dumps(params, indent=2) + "\n")
    return path


def save_used_params(args, out_dir):
    """Record the resolved parameters of a run in out_dir.

    The result is itself a valid params file, so
    `medley run <out_dir>/medley_params.json` repeats the run.
    """
    values = dict(vars(args))
    mode = values.pop("mode")
    params = {"mode": mode}
    for key in POSITIONALS[mode]:
        params[key] = values.pop(key)
    params.update(values)
    for key in PATH_KEYS:
        if params.get(key) is not None:
            params[key] = str(Path(params[key]).resolve())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return write_params(params, out / SAVED_NAME, overwrite=True)
