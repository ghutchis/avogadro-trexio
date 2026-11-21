# SPDX-License-Identifier: BSD-3-Clause

"""CJSON contract tests, parametrized over the generated fixture corpus."""

import json
import math

import numpy as np
import pytest

from conftest import ALL_MOLECULE_FIXTURES, read_cjson, strict_json_loads


def _all_finite(obj):
    """Recursively check that no float in obj is NaN/Infinity.

    reader.read() already drops any key containing a non-finite value
    (utils.sanitize), so this should always be true for reader output --
    this test exists to catch a regression in that guarantee.
    """
    if isinstance(obj, dict):
        return all(_all_finite(v) for v in obj.values())
    if isinstance(obj, list):
        return all(_all_finite(v) for v in obj)
    if isinstance(obj, float):
        return math.isfinite(obj)
    return True


@pytest.mark.parametrize("name", ALL_MOLECULE_FIXTURES)
def test_cjson_contract(fixture_paths, name):
    cjson = read_cjson(fixture_paths, name)

    assert cjson["chemicalJson"] == 1

    n_atoms = len(cjson["atoms"]["elements"]["number"])
    assert n_atoms == 3  # every fixture is a 3-atom "water-shaped" system

    coords = cjson["atoms"]["coords"]["3d"]
    assert len(coords) == 3 * n_atoms

    assert _all_finite(cjson)

    # The whole dict must survive a strict JSON round trip that rejects
    # NaN/Infinity, and must contain no non-JSON-serializable types.
    serialized = json.dumps(cjson, allow_nan=False)
    round_tripped = strict_json_loads(serialized)
    assert round_tripped == cjson


@pytest.mark.parametrize("name", ALL_MOLECULE_FIXTURES)
def test_oh_distance_is_point_96_angstrom(fixture_paths, name):
    # Every fixture is built by make_fixtures.py with an O-H bond length of
    # 0.96 Angstrom (see _write_water_nucleus's default bond_angstrom).
    # Atom 0 is O, atom 1 is the first H in every fixture.
    cjson = read_cjson(fixture_paths, name)
    coords = np.asarray(cjson["atoms"]["coords"]["3d"]).reshape(-1, 3)
    distance = np.linalg.norm(coords[0] - coords[1])
    assert distance == pytest.approx(0.96, abs=1e-6)


@pytest.mark.parametrize("name", ALL_MOLECULE_FIXTURES)
def test_elements_are_water_like(fixture_paths, name):
    cjson = read_cjson(fixture_paths, name)
    numbers = cjson["atoms"]["elements"]["number"]
    # O, H, H in every fixture (element 8 = oxygen, 1 = hydrogen) -- even
    # the ecp fixture, which has no nucleus_label, must reach the same
    # element numbers via the nucleus_charge (+ ecp_z_core) fallback.
    assert numbers == [8, 1, 1]


def test_properties_present_where_expected(fixture_paths):
    cjson = read_cjson(fixture_paths, "h2o_sto3g_spherical")
    props = cjson["properties"]
    assert props["totalSpinMultiplicity"] == 1
    assert props["totalCharge"] == 0
    assert props["pointGroup"] == "C2v"
    assert props["trexioCode"] == "FAKE-SCF"
    assert "totalEnergy" in props


def test_ecp_reduces_to_correct_elements_and_core_electrons(fixture_paths):
    cjson = read_cjson(fixture_paths, "ecp")
    assert cjson["atoms"]["elements"]["number"] == [8, 1, 1]
    props = cjson["properties"]
    assert props["coreElectrons"] == [2.0, 0.0, 0.0]
    assert props["totalCoreElectrons"] == 2.0


def test_periodic_emits_unit_cell(fixture_paths):
    cjson = read_cjson(fixture_paths, "periodic")
    assert "unitCell" in cjson
    vectors = cjson["unitCell"]["cellVectors"]
    assert len(vectors) == 9
    assert "a" not in cjson["unitCell"]  # cellVectors only, never both forms


@pytest.mark.parametrize("name", ["h2o_sto3g_spherical", "h2o_cartesian", "ecp"])
def test_non_periodic_fixtures_have_no_unit_cell(fixture_paths, name):
    cjson = read_cjson(fixture_paths, name)
    assert "unitCell" not in cjson
