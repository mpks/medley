import pytest

from medley.miller import parse_hkl_string


def test_parse_space_separated():
    assert parse_hkl_string("1 2 3") == (1, 2, 3)


def test_parse_comma_separated():
    assert parse_hkl_string("1,-2,3") == (1, -2, 3)


def test_parse_rejects_wrong_count():
    with pytest.raises(ValueError):
        parse_hkl_string("1 2")
