import json

import pytest

from medley.cli import build_parser, main
from medley.params import (
    PLACEHOLDERS,
    default_params,
    load_params,
    params_to_argv,
    save_used_params,
)


@pytest.mark.parametrize("mode", ["experiment", "synthetic"])
def test_defaults_round_trip(mode):
    """Defaults -> argv -> parse gives back the same values."""
    parser = build_parser()
    params = default_params(parser, mode)
    params["cif_file"] = "x.cif"
    if mode == "experiment":
        params["expt_file"] = "x.expt"

    args = parser.parse_args(params_to_argv(parser, params))
    assert vars(args) == params


def test_edited_values_reach_args():
    parser = build_parser()
    params = default_params(parser, "synthetic")
    params.update(
        cif_file="x.cif", k_max=3.5, start_deg=-45.0,
        d_min=0.8, no_cbf=True, integration="raster",
    )
    args = parser.parse_args(params_to_argv(parser, params))
    assert args.k_max == 3.5
    assert args.start_deg == -45.0      # negative value survives
    assert args.d_min == 0.8
    assert args.no_cbf is True
    assert args.integration == "raster"
    assert args.distance_mm is None     # null stays unset


def test_unknown_key_rejected():
    parser = build_parser()
    params = default_params(parser, "synthetic")
    params["cif_file"] = "x.cif"
    params["kmax"] = 2.0                # typo for k_max
    with pytest.raises(ValueError, match="kmax"):
        params_to_argv(parser, params)


def test_placeholder_path_rejected():
    parser = build_parser()
    params = default_params(parser, "experiment")
    params["cif_file"] = "x.cif"
    assert params["expt_file"] == PLACEHOLDERS["expt_file"]
    with pytest.raises(ValueError, match="expt_file"):
        params_to_argv(parser, params)


def test_bad_mode_rejected():
    parser = build_parser()
    with pytest.raises(ValueError, match="mode"):
        params_to_argv(parser, {"mode": "nope"})


def test_params_command_writes_file(tmp_path):
    out = tmp_path / "p.json"
    main(["params", "synthetic", "-o", str(out)])
    params = load_params(out)
    assert params["mode"] == "synthetic"
    assert params["k_max"] == default_params(
        build_parser(), "synthetic"
    )["k_max"]


def test_params_command_refuses_overwrite(tmp_path):
    out = tmp_path / "p.json"
    out.write_text("{}")
    with pytest.raises(SystemExit):
        main(["params", "synthetic", "-o", str(out)])
    main(["params", "synthetic", "-o", str(out), "--force"])
    assert load_params(out)["mode"] == "synthetic"


def test_saved_params_are_rerunnable(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        ["synthetic", "x.cif", "--k_max", "2.5",
         "-o", str(tmp_path)]
    )
    path = save_used_params(args, tmp_path)
    saved = json.loads(path.read_text())
    again = parser.parse_args(params_to_argv(parser, saved))
    assert again.k_max == 2.5
    assert again.cif_file == saved["cif_file"]   # absolute path
