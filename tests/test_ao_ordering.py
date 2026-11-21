# SPDX-License-Identifier: BSD-3-Clause

"""AO component ordering within a shell.

TREXIO and Avogadro lay out MO coefficients shell by shell, but they order
the components *inside* a shell differently in two places:

  * spherical p, where TREXIO's (0, +1, -1) means (pz, px, py) while
    Avogadro's pointP evaluates x, y, z; and
  * every Cartesian shell with l >= 2, where TREXIO uses the alphabetical
    ordering and Avogadro uses a Gaussian/Molden-style table.

Neither is detectable downstream -- CJSON has no "-1" shell type, so a
spherical p shell and a Cartesian one both arrive as shellType 1 -- so if
the reader does not permute the coefficient columns, the orbitals are
silently wrong. These tests pin both permutations down against expected
index sequences written out by hand from the two conventions, rather than
against the reader's own helper (which would only prove self-consistency).
"""

import numpy as np
import pytest
import trexio

from avogadro_trexio.reader import read
from avogadro_trexio.utils import ao_permutation, shell_ao_count

# Every fixture basis is built by make_fixtures._write_sto3g_like_basis with
# these shells, in this order: s, p, d, s, s.
FIXTURE_SHELLS = [0, 1, 2, 0, 0]

# Hand-derived expected column order, as indices into TREXIO's AO layout.
#
# Spherical, block offsets s@0, p@1, d@4, s@9, s@10:
#   s  -> 0
#   p  -> TREXIO (z, x, y) read into Avogadro (x, y, z) = TREXIO 1, 2, 0
#         -> absolute 2, 3, 1
#   d  -> both are 0, +1, -1, +2, -2; unchanged
EXPECTED_SPHERICAL = [0] + [2, 3, 1] + [4, 5, 6, 7, 8] + [9] + [10]

# Cartesian, block offsets s@0, p@1, d@4, s@10, s@11:
#   p  -> both are x, y, z; unchanged
#   d  -> TREXIO (xx, xy, xz, yy, yz, zz) read into Avogadro
#         (xx, yy, zz, xy, xz, yz) = TREXIO 0, 3, 5, 1, 2, 4
#         -> absolute 4, 7, 9, 5, 6, 8
EXPECTED_CARTESIAN = [0] + [1, 2, 3] + [4, 7, 9, 5, 6, 8] + [10] + [11]


def _raw_mo_coefficients(path):
    """MO coefficients straight from the file, in TREXIO's own AO order."""
    f = trexio.File(str(path), "r", trexio.TREXIO_HDF5)
    try:
        return np.asarray(trexio.read_mo_coefficient(f), dtype=float)
    finally:
        f.close()


@pytest.mark.parametrize(
    "fixture_name, expected",
    [
        ("h2o_sto3g_spherical", EXPECTED_SPHERICAL),
        ("h2o_cartesian", EXPECTED_CARTESIAN),
    ],
)
def test_mo_coefficients_are_reordered_to_avogadro_layout(
    fixture_paths, fixture_name, expected
):
    """Every emitted MO's coefficients follow Avogadro's component order."""
    raw = _raw_mo_coefficients(fixture_paths[fixture_name])
    cjson = read(str(fixture_paths[fixture_name]))
    emitted = np.asarray(cjson["orbitals"]["moCoefficients"], dtype=float)

    mo_num, ao_num = raw.shape
    assert ao_num == len(expected), "fixture basis changed; update the expectation"
    assert emitted.size == mo_num * ao_num

    # moCoefficients is the [mo_num, ao_num] matrix flattened row-major.
    emitted = emitted.reshape(mo_num, ao_num)
    for mo in range(mo_num):
        np.testing.assert_allclose(
            emitted[mo], raw[mo][expected], rtol=0, atol=1e-15
        )


def test_spherical_p_is_actually_permuted():
    """Guard the guard: the spherical expectation must not be the identity.

    If EXPECTED_SPHERICAL were ever "fixed" into the identity, the test
    above would pass against an unpermuted reader and the bug it exists to
    catch would come straight back.
    """
    assert EXPECTED_SPHERICAL != list(range(len(EXPECTED_SPHERICAL)))
    assert EXPECTED_CARTESIAN != list(range(len(EXPECTED_CARTESIAN)))


@pytest.mark.parametrize("l", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("cartesian", [False, True])
def test_permutation_is_a_permutation(l, cartesian):
    """Whatever the ordering, no component may be dropped or duplicated."""
    perm = ao_permutation(l, cartesian)
    if perm is None:
        return
    assert sorted(perm) == list(range(shell_ao_count(l, cartesian)))


def test_only_expected_shells_need_reordering():
    """Spherical d/f/g and Cartesian s/p already agree; keep it that way.

    A future change that starts permuting these would be reordering
    something that was already correct.
    """
    assert ao_permutation(0, False) is None
    assert ao_permutation(1, False) == [1, 2, 0]
    for l in (2, 3, 4):
        assert ao_permutation(l, False) is None, f"spherical l={l} must be identity"

    assert ao_permutation(0, True) is None
    assert ao_permutation(1, True) is None
    assert ao_permutation(2, True) == [0, 3, 5, 1, 2, 4]
    assert ao_permutation(3, True) == [0, 6, 9, 3, 1, 2, 5, 8, 7, 4]
    assert ao_permutation(4, True) == [
        0, 10, 14, 1, 2, 6, 11, 9, 13, 3, 5, 12, 4, 7, 8,
    ]


def test_ao_counts():
    assert [shell_ao_count(l, False) for l in range(5)] == [1, 3, 5, 7, 9]
    assert [shell_ao_count(l, True) for l in range(5)] == [1, 3, 6, 10, 15]
