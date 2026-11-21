# SPDX-License-Identifier: BSD-3-Clause

"""orbitals: closed vs open shell shapes, eV conversion, k-point filtering."""

import pytest
import trexio

from avogadro_trexio.utils import HARTREE_TO_EV
from conftest import read_cjson


def test_closed_shell_emits_flat_mo_coefficients_and_energies(fixture_paths):
    cjson = read_cjson(fixture_paths, "h2o_sto3g_spherical")
    orbitals = cjson["orbitals"]
    assert "moCoefficients" in orbitals
    assert "energies" in orbitals
    assert "alphaCoefficients" not in orbitals
    assert "betaCoefficients" not in orbitals

    ao_num = len(cjson["basisSet"]["shellToAtomMap"])  # not directly ao_num
    # moCoefficients length must be a whole multiple of ao_num (mo_num * ao_num).
    with trexio.File(str(fixture_paths["h2o_sto3g_spherical"]), "r", trexio.TREXIO_HDF5) as f:
        real_ao_num = trexio.read_ao_num(f)
        mo_num = trexio.read_mo_num(f)
    assert len(orbitals["moCoefficients"]) == mo_num * real_ao_num
    assert len(orbitals["energies"]) == mo_num


def test_open_shell_emits_alpha_beta_pairs(fixture_paths):
    cjson = read_cjson(fixture_paths, "uhf_open_shell")
    orbitals = cjson["orbitals"]
    assert "moCoefficients" not in orbitals
    assert "alphaCoefficients" in orbitals
    assert "betaCoefficients" in orbitals
    assert "alphaEnergies" in orbitals
    assert "betaEnergies" in orbitals

    with trexio.File(str(fixture_paths["uhf_open_shell"]), "r", trexio.TREXIO_HDF5) as f:
        ao_num = trexio.read_ao_num(f)
        mo_spin = trexio.read_mo_spin(f)
    n_alpha_mo = int((mo_spin == 0).sum())
    n_beta_mo = int((mo_spin == 1).sum())

    assert len(orbitals["alphaCoefficients"]) == n_alpha_mo * ao_num
    assert len(orbitals["betaCoefficients"]) == n_beta_mo * ao_num
    assert len(orbitals["alphaEnergies"]) == n_alpha_mo
    assert len(orbitals["betaEnergies"]) == n_beta_mo

    symmetries = orbitals["symmetries"]
    assert len(symmetries) == 2
    assert len(symmetries[0]) == n_alpha_mo
    assert len(symmetries[1]) == n_beta_mo


def test_orbital_energies_are_in_ev(fixture_paths):
    cjson = read_cjson(fixture_paths, "h2o_sto3g_spherical")
    energies_ev = cjson["orbitals"]["energies"]

    with trexio.File(str(fixture_paths["h2o_sto3g_spherical"]), "r", trexio.TREXIO_HDF5) as f:
        energies_hartree = trexio.read_mo_energy(f)

    for ev, hartree in zip(energies_ev, energies_hartree):
        assert ev == pytest.approx(hartree * HARTREE_TO_EV)
        assert ev == pytest.approx(hartree * 27.2114, rel=1e-4)


def test_periodic_keeps_only_gamma_point_mos(fixture_paths):
    cjson = read_cjson(fixture_paths, "periodic")
    orbitals = cjson["orbitals"]

    with trexio.File(str(fixture_paths["periodic"]), "r", trexio.TREXIO_HDF5) as f:
        ao_num = trexio.read_ao_num(f)
        k_point = trexio.read_mo_k_point(f)
    n_gamma = int((k_point == 0).sum())

    assert len(orbitals["moCoefficients"]) == n_gamma * ao_num
    assert len(orbitals["energies"]) == n_gamma


def test_electron_count_matches_up_plus_dn(fixture_paths):
    cjson = read_cjson(fixture_paths, "uhf_open_shell")
    with trexio.File(str(fixture_paths["uhf_open_shell"]), "r", trexio.TREXIO_HDF5) as f:
        up = trexio.read_electron_up_num(f)
        dn = trexio.read_electron_dn_num(f)
    assert cjson["orbitals"]["electronCount"] == up + dn
