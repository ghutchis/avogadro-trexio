# SPDX-License-Identifier: BSD-3-Clause

"""The important test: does the CJSON coefficient conversion round-trip?

For every shell in every Gaussian-basis fixture, we independently
recompute TREXIO's own radial function directly from the raw TREXIO
arrays (opened here with our own trexio.File call, not through
reader.py), and separately reconstruct the radial function from what
reader.read() actually emitted in basisSet. If reader.py's grouping,
defaults, or normalization division were wrong, these two would disagree.

Fixture h2o_sto3g_spherical (and h2o_cartesian) is deliberately built with
a shell_factor != 1 on its p shell (1.7) and a prim_factor != 1 on every
one of that shell's primitives (1.3) -- see make_fixtures.py -- so this
test would fail if either factor were silently dropped.
"""

import numpy as np
import pytest
import trexio

from conftest import GAUSSIAN_BASIS_FIXTURES, read_cjson
from avogadro_trexio.utils import gto_normalization

RADII = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]


def _raw_basis(path):
    """Read every raw basis array from the fixture, defaults applied.

    Returns a dict of numpy arrays/scalars, independent of anything in
    reader.py (this test would be pointless if it re-used reader.py's own
    reading code).
    """
    with trexio.File(str(path), "r", trexio.TREXIO_HDF5) as f:
        shell_ang_mom = np.asarray(trexio.read_basis_shell_ang_mom(f), dtype=int)
        shell_num = shell_ang_mom.shape[0]
        shell_nucleus = np.asarray(trexio.read_basis_nucleus_index(f), dtype=int)
        shell_index = np.asarray(trexio.read_basis_shell_index(f), dtype=int)
        exponent = np.asarray(trexio.read_basis_exponent(f), dtype=float)
        coefficient = np.asarray(trexio.read_basis_coefficient(f), dtype=float)
        prim_num = exponent.shape[0]
        shell_factor = (
            np.asarray(trexio.read_basis_shell_factor(f), dtype=float)
            if trexio.has_basis_shell_factor(f)
            else np.ones(shell_num)
        )
        prim_factor = (
            np.asarray(trexio.read_basis_prim_factor(f), dtype=float)
            if trexio.has_basis_prim_factor(f)
            else np.ones(prim_num)
        )
        ao_cartesian = (
            bool(trexio.read_ao_cartesian(f)) if trexio.has_ao_cartesian(f) else False
        )
    return {
        "shell_ang_mom": shell_ang_mom,
        "shell_nucleus": shell_nucleus,
        "shell_index": shell_index,
        "exponent": exponent,
        "coefficient": coefficient,
        "shell_factor": shell_factor,
        "prim_factor": prim_factor,
        "ao_cartesian": ao_cartesian,
        "shell_num": shell_num,
        "prim_num": prim_num,
    }


def _bucket_by_shell(shell_index, shell_num):
    """Group primitive indices by shell, preserving original relative order.

    Written independently from reader._read_basis_set's identical-looking
    loop -- both happen to be "the obvious way to do it", which is the
    point: if reader.py instead assumed contiguity/sortedness (a
    plausible bug for hand-rolled TREXIO writers that usually do write
    shells contiguously), this test's fixture data deliberately
    interleaves shells 0 and 1's primitives (see make_fixtures.py) so
    that bug would show up as a mismatch here.
    """
    buckets = [[] for _ in range(shell_num)]
    for k, s in enumerate(shell_index):
        buckets[int(s)].append(k)
    return buckets


@pytest.mark.parametrize("name", GAUSSIAN_BASIS_FIXTURES)
def test_shell_radial_function_round_trips(fixture_paths, name):
    raw = _raw_basis(fixture_paths[name])
    cjson = read_cjson(fixture_paths, name)
    basis = cjson["basisSet"]

    buckets = _bucket_by_shell(raw["shell_index"], raw["shell_num"])

    assert basis["primitivesPerShell"] == [len(b) for b in buckets]

    # Walk shells in the same order reader.py concatenates them into
    # exponents/coefficients (i.e. shell order, primitives within a shell
    # in their bucketed order).
    cursor = 0
    for s in range(raw["shell_num"]):
        l = int(raw["shell_ang_mom"][s])
        prim_indices = buckets[s]
        n = len(prim_indices)

        emitted_exponents = basis["exponents"][cursor : cursor + n]
        emitted_coefficients = basis["coefficients"][cursor : cursor + n]
        cursor += n

        norm_s = float(raw["shell_factor"][s])

        def r_trexio(r, prim_indices=prim_indices, norm_s=norm_s):
            total = 0.0
            for k in prim_indices:
                alpha = float(raw["exponent"][k])
                a_k = float(raw["coefficient"][k])
                f_k = float(raw["prim_factor"][k])
                total += f_k * a_k * np.exp(-alpha * r * r)
            return norm_s * total

        def r_cjson(r, coeffs=emitted_coefficients, exps=emitted_exponents, l=l):
            return sum(
                c * gto_normalization(g, l) * np.exp(-g * r * r)
                for c, g in zip(coeffs, exps)
            )

        for r in RADII:
            assert r_trexio(r) == pytest.approx(r_cjson(r), abs=1e-10), (
                f"{name} shell {s} (l={l}) radial function mismatch at r={r}"
            )

    assert cursor == len(basis["exponents"]) == len(basis["coefficients"])


@pytest.mark.parametrize("name", GAUSSIAN_BASIS_FIXTURES)
def test_basis_array_lengths_and_atom_indices(fixture_paths, name):
    cjson = read_cjson(fixture_paths, name)
    basis = cjson["basisSet"]
    n_atoms = len(cjson["atoms"]["elements"]["number"])

    assert len(basis["exponents"]) == len(basis["coefficients"])
    assert len(basis["exponents"]) == sum(basis["primitivesPerShell"])
    assert (
        len(basis["primitivesPerShell"])
        == len(basis["shellToAtomMap"])
        == len(basis["shellTypes"])
    )
    assert min(basis["shellToAtomMap"]) >= 0
    assert max(basis["shellToAtomMap"]) < n_atoms


def test_shell_factor_and_prim_factor_are_not_silently_dropped(fixture_paths):
    # h2o_sto3g_spherical's p shell (index 1) has shell_factor=1.7 and every
    # primitive's prim_factor=1.3 (see make_fixtures.py). If reader.py
    # ignored either one, the round-trip test above would already have
    # failed -- this test just makes that specific fixture choice explicit
    # and documented, and independently pins the raw values so a change to
    # make_fixtures.py that accidentally set them back to 1.0 would be
    # caught here too.
    raw = _raw_basis(fixture_paths["h2o_sto3g_spherical"])
    p_shell_index = 1
    assert raw["shell_ang_mom"][p_shell_index] == 1
    assert raw["shell_factor"][p_shell_index] == pytest.approx(1.7)
    buckets = _bucket_by_shell(raw["shell_index"], raw["shell_num"])
    p_shell_prims = buckets[p_shell_index]
    assert len(p_shell_prims) == 3
    for k in p_shell_prims:
        assert raw["prim_factor"][k] == pytest.approx(1.3)


def test_normalization_matches_avogadros_hard_coded_constants():
    """N(alpha, l) must reproduce GaussianSet's own literal constants.

    The round-trip test above imports gto_normalization, so on its own it
    would only prove reader.py is self-consistent -- if N(alpha, l) were
    the wrong formula entirely, both sides would shift together and the
    test would still pass. This one closes that gap by checking the
    formula against the constants avogadrolibs actually hard-codes in
    GaussianSet::initCalculation (gaussianset.cpp:333-378), which is the
    code that consumes our coefficients:

        S:  m_gtoC[j] * pow(m_gtoA[j], 0.75) * 0.71270547
        P:  m_gtoC[j] * pow(m_gtoA[j], 1.25) * 1.425410941
        D5: m_gtoC[j] * pow(2048 * pow(m_gtoA[j], 7.0) / (9 * pi^3), 0.25)

    i.e. N(alpha, 0) = alpha^0.75 * 0.71270547, and so on. The D5 case is
    written in Avogadro's own closed form rather than as a
    coefficient * power, which is why it is spelled out differently here.
    """
    for alpha in (0.1, 0.8, 3.0, 42.0):
        assert gto_normalization(alpha, 0) == pytest.approx(
            alpha**0.75 * 0.71270547, rel=1e-8
        )
        assert gto_normalization(alpha, 1) == pytest.approx(
            alpha**1.25 * 1.425410941, rel=1e-8
        )
        assert gto_normalization(alpha, 2) == pytest.approx(
            (2048.0 * alpha**7 / (9.0 * np.pi**3)) ** 0.25, rel=1e-12
        )
