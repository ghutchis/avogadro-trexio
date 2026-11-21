# SPDX-License-Identifier: BSD-3-Clause

"""shellTypes sign (spherical vs Cartesian) and primitive grouping."""

import pytest

from conftest import read_cjson


def test_spherical_fixture_has_negative_shell_types_for_l_ge_2(fixture_paths):
    cjson = read_cjson(fixture_paths, "h2o_sto3g_spherical")
    shell_types = cjson["basisSet"]["shellTypes"]
    # shells are s, p, d, s, s (see make_fixtures._write_sto3g_like_basis)
    assert shell_types == [0, 1, -2, 0, 0]
    for t in shell_types:
        if abs(t) >= 2:
            assert t < 0
        else:
            assert t >= 0


def test_cartesian_fixture_has_no_negative_shell_types(fixture_paths):
    cjson = read_cjson(fixture_paths, "h2o_cartesian")
    shell_types = cjson["basisSet"]["shellTypes"]
    assert shell_types == [0, 1, 2, 0, 0]
    assert all(t >= 0 for t in shell_types)


def test_primitive_grouping_is_correct_with_shuffled_shell_index(fixture_paths):
    # make_fixtures.py deliberately interleaves shell 0 and shell 1's
    # primitives in basis_shell_index (order: 0,1,0,1,0,1,2,3,4) rather
    # than laying them out contiguously by shell. If reader.py assumed
    # contiguous/sorted primitives instead of bucketing by shell index,
    # shell 0 and shell 1 would end up with each other's exponents.
    cjson = read_cjson(fixture_paths, "h2o_sto3g_spherical")
    basis = cjson["basisSet"]

    # Shell 0 (s on O) exponents, in the *unshuffled* original order they
    # were written in for that shell.
    shell0_exponents = basis["exponents"][0:3]
    assert shell0_exponents == pytest.approx([130.70932, 23.808861, 6.4436083])

    # Shell 1 (p on O) exponents.
    shell1_start = basis["primitivesPerShell"][0]
    shell1_exponents = basis["exponents"][shell1_start : shell1_start + 3]
    assert shell1_exponents == pytest.approx([5.0331513, 1.1695961, 0.3803890])
