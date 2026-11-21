# SPDX-License-Identifier: BSD-3-Clause

"""Build small synthetic TREXIO HDF5 fixtures for the test suite.

There is no public corpus of small, permissively-licensed TREXIO files, so
we generate our own directly with the trexio Python API -- see
conftest.py's ``fixtures_dir`` session fixture, which calls
``build_all(tmp_dir)`` once per test session. Nothing here is committed to
the repository (binary .h5 files do not belong in git); every fixture is
built fresh into a pytest tmp dir.

Values are deliberately small and partly fictitious (these are not real
converged wavefunctions) -- they only need to be internally self-consistent
enough to exercise the reader's unit conversions, normalization math, and
optional-group handling.
"""

import math

import numpy as np
import trexio

BOHR_TO_ANGSTROM = 0.529177210903


def _new(path):
    return trexio.File(str(path), "w", trexio.TREXIO_HDF5)


def _write_water_nucleus(f, bond_angstrom=0.96, angle_deg=104.5, labels=None, charges=None):
    """A single water molecule, O at the origin, both O-H bonds equal.

    Returns the nucleus_coord array (bohr) actually written, so callers can
    independently check e.g. the O-H distance in a test.
    """
    bond_bohr = bond_angstrom / BOHR_TO_ANGSTROM
    half_angle = math.radians(angle_deg) / 2.0
    coord = np.array(
        [
            [0.0, 0.0, 0.0],
            [bond_bohr * math.sin(half_angle), bond_bohr * math.cos(half_angle), 0.0],
            [-bond_bohr * math.sin(half_angle), bond_bohr * math.cos(half_angle), 0.0],
        ]
    )
    trexio.write_nucleus_num(f, 3)
    trexio.write_nucleus_coord(f, coord)
    trexio.write_nucleus_charge(f, charges or [8.0, 1.0, 1.0])
    trexio.write_nucleus_label(f, labels or ["O", "H", "H"])
    trexio.write_nucleus_point_group(f, "C2v")
    return coord


def _write_water_electrons(f):
    trexio.write_electron_up_num(f, 5)
    trexio.write_electron_dn_num(f, 5)


def _write_water_metadata(f):
    trexio.write_metadata_code_num(f, 1)
    trexio.write_metadata_code(f, ["FAKE-SCF"])
    trexio.write_metadata_description(f, "synthetic fixture for avogadro-trexio tests")
    trexio.write_metadata_author_num(f, 1)
    trexio.write_metadata_author(f, ["avogadro-trexio test suite"])


def _write_sto3g_like_basis(f, ao_cartesian, shell_factor_second_shell=1.7):
    """A minimal s/p/d Gaussian basis, one shell of each on the oxygen atom.

    Shell 0: s on O (3 primitives)
    Shell 1: p on O (3 primitives) -- shell_factor deliberately != 1 here,
        so a normalization test that dropped basis_shell_factor would fail.
    Shell 2: d on O (1 primitive), to exercise the l=2 shellTypes sign.
    Shell 3, 4: s on H1, H2 (1 primitive each)

    Primitive order is deliberately shuffled across shells in
    basis_shell_index (see the interleaving below) rather than laid out
    shell-by-shell, so that a reader which assumed contiguous/sorted
    primitives would produce a visibly wrong basis.
    """
    shell_ang_mom = [0, 1, 2, 0, 0]
    shell_nucleus_index = [0, 0, 0, 1, 2]
    shell_factor = [1.0, shell_factor_second_shell, 1.0, 1.0, 1.0]

    # Raw (shell, exponent, coefficient, prim_factor) tuples, listed in a
    # deliberately shuffled order (interleaving shells 0 and 1's
    # primitives) to exercise non-contiguous basis_shell_index grouping.
    raw = [
        (0, 130.70932, 0.15432897, 1.0),   # shell 0, prim 0
        (1, 5.0331513, 0.15591627, 1.3),   # shell 1, prim 0 (factor != 1)
        (0, 23.808861, 0.53532814, 1.0),   # shell 0, prim 1
        (1, 1.1695961, 0.60768372, 1.3),   # shell 1, prim 1
        (0, 6.4436083, 0.44463454, 1.0),   # shell 0, prim 2
        (1, 0.3803890, 0.39195739, 1.3),   # shell 1, prim 2
        (2, 0.8000000, 1.00000000, 1.0),   # shell 2 (d), single primitive
        (3, 3.4252509, 0.15432897, 1.0),   # shell 3 (s on H1)
        (4, 3.4252509, 0.15432897, 1.0),   # shell 4 (s on H2)
    ]
    basis_shell_index = [r[0] for r in raw]
    basis_exponent = [r[1] for r in raw]
    basis_coefficient = [r[2] for r in raw]
    basis_prim_factor = [r[3] for r in raw]

    trexio.write_basis_type(f, "Gaussian")
    trexio.write_basis_shell_num(f, len(shell_ang_mom))
    trexio.write_basis_prim_num(f, len(raw))
    trexio.write_basis_nucleus_index(f, shell_nucleus_index)
    trexio.write_basis_shell_ang_mom(f, shell_ang_mom)
    trexio.write_basis_shell_factor(f, shell_factor)
    trexio.write_basis_shell_index(f, basis_shell_index)
    trexio.write_basis_exponent(f, basis_exponent)
    trexio.write_basis_coefficient(f, basis_coefficient)
    trexio.write_basis_prim_factor(f, basis_prim_factor)

    def _ao_count(l):
        # s,p give the same AO count either way (2l+1 == (l+1)(l+2)/2 for
        # l=0,1); only d and above differ between cartesian and spherical.
        return (l + 1) * (l + 2) // 2 if ao_cartesian else 2 * l + 1

    ao_num = sum(_ao_count(l) for l in shell_ang_mom)
    trexio.write_ao_num(f, ao_num)
    trexio.write_ao_cartesian(f, 1 if ao_cartesian else 0)

    return {
        "shell_ang_mom": shell_ang_mom,
        "shell_nucleus_index": shell_nucleus_index,
        "shell_factor": shell_factor,
        "basis_shell_index": basis_shell_index,
        "basis_exponent": basis_exponent,
        "basis_coefficient": basis_coefficient,
        "basis_prim_factor": basis_prim_factor,
        "ao_num": ao_num,
    }


def _write_closed_shell_mo(f, ao_num, n_occ, energy_hartree=None):
    mo_num = ao_num
    rng = np.random.default_rng(42)
    coeff = rng.normal(size=(mo_num, ao_num))
    if energy_hartree is None:
        energy_hartree = np.linspace(-20.0, 5.0, mo_num)
    occupation = [2.0 if i < n_occ else 0.0 for i in range(mo_num)]
    symmetry = [f"A{i % 2 + 1}" for i in range(mo_num)]

    trexio.write_mo_num(f, mo_num)
    trexio.write_mo_coefficient(f, coeff)
    trexio.write_mo_energy(f, energy_hartree)
    trexio.write_mo_occupation(f, occupation)
    trexio.write_mo_symmetry(f, symmetry)
    return coeff, energy_hartree


def build_h2o_sto3g_spherical(path):
    with _new(path) as f:
        _write_water_nucleus(f)
        _write_water_electrons(f)
        _write_water_metadata(f)
        basis_info = _write_sto3g_like_basis(f, ao_cartesian=False)
        _write_closed_shell_mo(f, basis_info["ao_num"], n_occ=5)
        trexio.write_state_energy(f, -75.983)
    return basis_info


def build_h2o_cartesian(path):
    with _new(path) as f:
        _write_water_nucleus(f)
        _write_water_electrons(f)
        _write_water_metadata(f)
        basis_info = _write_sto3g_like_basis(f, ao_cartesian=True)
        _write_closed_shell_mo(f, basis_info["ao_num"], n_occ=5)
        trexio.write_state_energy(f, -75.982)
    return basis_info


def build_uhf_open_shell(path):
    """A small doublet (e.g. OH-radical-like: 5 alpha, 4 beta electrons)."""
    with _new(path) as f:
        _write_water_nucleus(f, labels=["O", "H", "H"])
        trexio.write_electron_up_num(f, 5)
        trexio.write_electron_dn_num(f, 4)
        _write_water_metadata(f)
        basis_info = _write_sto3g_like_basis(f, ao_cartesian=False)
        ao_num = basis_info["ao_num"]

        n_alpha_mo = ao_num
        n_beta_mo = ao_num
        mo_num = n_alpha_mo + n_beta_mo
        rng = np.random.default_rng(7)
        coeff = rng.normal(size=(mo_num, ao_num))
        energy = np.concatenate(
            [np.linspace(-20.0, 5.0, n_alpha_mo), np.linspace(-19.5, 5.5, n_beta_mo)]
        )
        spin = [0] * n_alpha_mo + [1] * n_beta_mo
        occupation = [1.0 if i < 5 else 0.0 for i in range(n_alpha_mo)] + [
            1.0 if i < 4 else 0.0 for i in range(n_beta_mo)
        ]
        symmetry = [f"A{i % 2 + 1}" for i in range(mo_num)]

        trexio.write_mo_num(f, mo_num)
        trexio.write_mo_coefficient(f, coeff)
        trexio.write_mo_energy(f, energy)
        trexio.write_mo_spin(f, spin)
        trexio.write_mo_occupation(f, occupation)
        trexio.write_mo_symmetry(f, symmetry)
        trexio.write_state_energy(f, -75.5)
    return basis_info, n_alpha_mo, n_beta_mo


def build_periodic(path):
    """A tiny fake periodic system with 2 k-points (Gamma + one other)."""
    with _new(path) as f:
        _write_water_nucleus(f)
        _write_water_electrons(f)
        _write_water_metadata(f)
        basis_info = _write_sto3g_like_basis(f, ao_cartesian=False)
        ao_num = basis_info["ao_num"]

        trexio.write_cell_a(f, [10.0, 0.0, 0.0])
        trexio.write_cell_b(f, [0.0, 10.0, 0.0])
        trexio.write_cell_c(f, [0.0, 0.0, 10.0])
        trexio.write_pbc_periodic(f, 1)

        n_per_kpoint = ao_num
        mo_num = 2 * n_per_kpoint
        rng = np.random.default_rng(99)
        coeff = rng.normal(size=(mo_num, ao_num))
        energy = np.concatenate(
            [
                np.linspace(-20.0, 5.0, n_per_kpoint),
                np.linspace(-18.0, 6.0, n_per_kpoint),
            ]
        )
        k_point = [0] * n_per_kpoint + [1] * n_per_kpoint
        occupation = [2.0 if i < 5 else 0.0 for i in range(n_per_kpoint)] * 2

        trexio.write_mo_num(f, mo_num)
        trexio.write_mo_coefficient(f, coeff)
        trexio.write_mo_energy(f, energy)
        trexio.write_mo_k_point(f, k_point)
        trexio.write_mo_occupation(f, occupation)
    return basis_info, n_per_kpoint


def build_ecp(path):
    """An ECP on the oxygen: nucleus_charge is reduced, ecp_z_core restores it.

    No nucleus_label is written at all -- this fixture exists specifically
    to exercise the charge+z_core element-number fallback path (label
    lookup is unavailable, so the reader must fall back to
    round(nucleus_charge + z_core)).
    """
    with _new(path) as f:
        bond_bohr = 0.96 / BOHR_TO_ANGSTROM
        half_angle = math.radians(104.5) / 2.0
        coord = np.array(
            [
                [0.0, 0.0, 0.0],
                [bond_bohr * math.sin(half_angle), bond_bohr * math.cos(half_angle), 0.0],
                [-bond_bohr * math.sin(half_angle), bond_bohr * math.cos(half_angle), 0.0],
            ]
        )
        trexio.write_nucleus_num(f, 3)
        trexio.write_nucleus_coord(f, coord)
        # Oxygen with a 2-electron core replaced by an ECP: real Z=8,
        # z_core=2, so the *reduced* nucleus_charge TREXIO stores is 6.
        # Hydrogens carry no ECP (z_core=0, charge unchanged).
        trexio.write_nucleus_charge(f, [6.0, 1.0, 1.0])
        trexio.write_electron_up_num(f, 4)
        trexio.write_electron_dn_num(f, 4)
        _write_water_metadata(f)
        trexio.write_ecp_z_core(f, [2, 0, 0])
        basis_info = _write_sto3g_like_basis(f, ao_cartesian=False)
        _write_closed_shell_mo(f, basis_info["ao_num"], n_occ=4)
    return coord


def build_slater(path):
    """basis_type = Slater -- the basis must be skipped, not crash the reader."""
    with _new(path) as f:
        _write_water_nucleus(f)
        _write_water_electrons(f)
        _write_water_metadata(f)
        trexio.write_basis_type(f, "Slater")
        trexio.write_basis_shell_num(f, 1)
        trexio.write_basis_prim_num(f, 1)
        trexio.write_basis_nucleus_index(f, [0])
        trexio.write_basis_shell_ang_mom(f, [0])
        trexio.write_basis_shell_index(f, [0])
        trexio.write_basis_exponent(f, [1.0])
        trexio.write_basis_coefficient(f, [1.0])


def build_no_basis(path):
    """Nucleus + electron groups only -- no basis, no mo group at all."""
    with _new(path) as f:
        _write_water_nucleus(f)
        _write_water_electrons(f)


def build_bad_magic(path):
    """Not HDF5 at all: right extension, wrong (in fact absent) magic bytes."""
    with open(path, "wb") as fh:
        fh.write(b"this is not an hdf5 file, just plain text\n" * 4)


def build_hdf5_no_nucleus(path):
    """A file that opens fine with the TREXIO/HDF5 back end but never had
    anything TREXIO-shaped written to it -- no nucleus group at all. We
    write metadata only, which requires nothing else, to get a valid HDF5
    file via trexio's own writer without ever touching the nucleus group.
    This exercises the "no nucleus group" sniffing branch without needing
    an h5py dependency to fabricate a foreign HDF5 file from scratch.
    """
    with _new(path) as f:
        trexio.write_metadata_code_num(f, 1)
        trexio.write_metadata_code(f, ["not-a-real-trexio-file"])


def build_all(tmp_dir):
    """Build every fixture into tmp_dir and return {name: path}."""
    paths = {}

    def make(name, builder):
        path = tmp_dir / name
        builder(path)
        paths[name] = path

    make("h2o_sto3g_spherical.h5", build_h2o_sto3g_spherical)
    make("h2o_cartesian.h5", build_h2o_cartesian)
    make("uhf_open_shell.h5", lambda p: build_uhf_open_shell(p))
    make("periodic.h5", lambda p: build_periodic(p))
    make("ecp.h5", build_ecp)
    make("slater.h5", build_slater)
    make("no_basis.h5", build_no_basis)
    make("bad_magic.h5", build_bad_magic)
    make("hdf5_no_nucleus.h5", build_hdf5_no_nucleus)

    return paths
