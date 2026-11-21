# SPDX-License-Identifier: BSD-3-Clause

"""Non-Gaussian / no-basis fixtures: the molecule must still load cleanly."""

from conftest import read_cjson


def test_slater_basis_is_skipped_but_molecule_loads(fixture_paths, capsys):
    cjson = read_cjson(fixture_paths, "slater")
    assert "basisSet" not in cjson
    assert "orbitals" not in cjson
    assert cjson["atoms"]["elements"]["number"] == [8, 1, 1]
    # reader.read() itself doesn't touch stderr (only the CLI layer and the
    # basis-skip warning do) -- the warning was already printed during the
    # read() call above; just make sure it explains why, for anyone running
    # this outside the test suite and wondering where their basis went.
    captured = capsys.readouterr()
    assert "Slater" in captured.err
    assert "skip" in captured.err.lower()


def test_no_basis_fixture_loads_without_basis_or_orbitals(fixture_paths):
    cjson = read_cjson(fixture_paths, "no_basis")
    assert "basisSet" not in cjson
    assert "orbitals" not in cjson
    assert cjson["atoms"]["elements"]["number"] == [8, 1, 1]
    assert "properties" in cjson
    assert cjson["properties"]["totalSpinMultiplicity"] == 1
