# SPDX-License-Identifier: BSD-3-Clause

"""Small, dependency-free helpers shared by reader.py.

Everything here is pure Python/NumPy -- no trexio calls except the generic
``has_*``/``read_*`` dispatch helpers, which take an already-open
``trexio.File`` and just save reader.py from writing
``trexio.has_foo(f) and trexio.read_foo(f)`` a hundred times over.
"""

import math
import re

import trexio

# ---------------------------------------------------------------------------
# Unit conversions. TREXIO stores everything in atomic units; CJSON (as read
# by Avogadro) wants Angstrom for coordinates, eV for orbital energies, and
# kJ/mol for the scalar total energy (matching how avogadro-cclib reports
# energies, e.g. HARTREE_TO_J_MOL / 1000 there).
# ---------------------------------------------------------------------------
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_TO_KJ_MOL = 2625.499639
HARTREE_TO_EV = 27.211386245988


# ---------------------------------------------------------------------------
# Generic has_x/read_x dispatch
# ---------------------------------------------------------------------------
def has(trexio_file, name):
    """True if the TREXIO file has the ``name`` entry (e.g. "nucleus_num").

    Thin wrapper around ``trexio.has_<name>``. Written once here because
    almost everything in a TREXIO file is optional, and every single read
    below must be guarded by the matching has_* check -- doing that inline
    a hundred times would be error-prone (easy to guard the wrong name, or
    forget one entirely).
    """
    return getattr(trexio, f"has_{name}")(trexio_file)


def read_if(trexio_file, name, default=None):
    """Return ``trexio.read_<name>(trexio_file)`` if present, else ``default``.

    This is the one-line form of "check has_x, then read_x" for values that
    are simple to default (an optional array or scalar with an obvious
    fallback). Do not use it where the caller needs to distinguish "absent"
    from "present but equal to the default" -- use has()/read separately.
    """
    if has(trexio_file, name):
        return getattr(trexio, f"read_{name}")(trexio_file)
    return default


# ---------------------------------------------------------------------------
# Element symbol -> atomic number
# ---------------------------------------------------------------------------
# Index 0 is the dummy/unknown placeholder ("Xx"); index Z holds element Z's
# symbol. TREXIO (like many quantum chemistry codes) allows dummy centres
# that carry basis functions but no real nucleus, hence element 0 is a
# legitimate output value, not just an error sentinel.
_ELEMENT_SYMBOLS = (
    "Xx", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)

_SYMBOL_TO_NUMBER = {symbol.upper(): z for z, symbol in enumerate(_ELEMENT_SYMBOLS)}

# TREXIO nucleus labels are free-form strings that quantum chemistry codes
# often decorate to distinguish otherwise-identical centres, e.g. "C1",
# "O_2", "H37", "Fe_a". Strip a trailing run of underscores/digits (in
# either order, e.g. "C_1", "C1_") before looking the symbol up.
_LABEL_SUFFIX = re.compile(r"[_0-9]+$")


def symbol_to_number(label):
    """Best-effort element symbol -> atomic number, or None if unrecognised.

    Case-insensitive; strips a trailing digit/underscore decoration (see
    _LABEL_SUFFIX) so labels like "C1" or "O_2" resolve to carbon/oxygen.
    Returns None (never 0) when nothing usable remains, so the caller can
    tell "no label" apart from "label really means the dummy element" and
    fall back to nucleus_charge as directed.
    """
    if not label:
        return None
    stripped = _LABEL_SUFFIX.sub("", label.strip())
    if not stripped:
        return None
    return _SYMBOL_TO_NUMBER.get(stripped.upper())


# ---------------------------------------------------------------------------
# Gaussian primitive normalization
# ---------------------------------------------------------------------------
def _double_factorial(n):
    """(n)!! for odd non-negative n, i.e. n * (n-2) * (n-4) * ... * 1.

    Only ever called here with n = 2*l - 1 for l = 0..4, i.e. n = -1, 1, 3,
    5, 7. By convention (-1)!! = 1 (needed for l = 0, where the "(2l-1)!!"
    factor in the normalization formula must vanish to 1).
    """
    if n <= 0:
        return 1
    result = 1
    while n > 0:
        result *= n
        n -= 2
    return result


def gto_normalization(alpha, l):
    """Standard normalization constant N(alpha, l) for a primitive GTO shell.

    N(alpha, l) = (2*alpha/pi)^(3/4) * (4*alpha)^(l/2) / sqrt((2l-1)!!)

    This is the normalization Avogadro's GaussianSet bakes into every CJSON
    coefficient before use (avogadrolibs gaussianset.cpp:333-378) -- verified
    against its hard-coded S, P and D5 base constants (0.71270547,
    1.425410941 and 1.645922781 respectively fall straight out of this
    formula). It normalizes only the radial part combined with a properly
    normalized real solid harmonic angular part; it does NOT capture the
    extra per-AO-component factor GaussianSet applies for individual
    Cartesian/spherical components within a shell (TREXIO's
    ``ao_normalization``, aka N'_i) -- that per-component detail has no home
    in CJSON's one-coefficient-per-primitive-per-shell format, so it is
    deliberately dropped. See README.md for the practical consequence.
    """
    return (2.0 * alpha / math.pi) ** 0.75 * (4.0 * alpha) ** (l / 2.0) / math.sqrt(
        _double_factorial(2 * l - 1)
    )


# ---------------------------------------------------------------------------
# Non-finite sanitization
# ---------------------------------------------------------------------------
def _sanitize_value(value):
    """Convert numpy scalars to native types, reporting any non-finite float.

    Returns ``(converted_value, saw_non_finite)`` in a single pass over
    ``value`` (recursing through nested lists), the same shape as
    avogadro-cclib's helper of the same name in its reader.py.
    """
    if isinstance(value, list):
        converted = []
        dirty = False
        for item in value:
            item_value, item_dirty = _sanitize_value(item)
            dirty = dirty or item_dirty
            converted.append(item_value)
        return converted, dirty
    if hasattr(value, "item"):
        # numpy scalar (numpy.float64, numpy.int64, numpy.bool_, ...) --
        # unwrap to a native Python type. numpy.float64 is a subclass of
        # the built-in float, so this must be checked before isinstance(...,
        # float).
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return value, True
    return value, False


def sanitize(obj):
    """Make ``obj`` safe to pass to ``json.dumps``.

    A single bare NaN/Infinity anywhere in a CJSON document makes nlohmann's
    parser return is_discarded, which throws away the ENTIRE document on
    Avogadro's side -- not just the offending key. So any key whose value
    (scalar or list, however deeply nested) contains a non-finite float is
    dropped entirely (with a note to stderr) rather than substituting a
    placeholder number, which would fabricate data. Numpy scalar types are
    coerced to plain Python int/float so the result is always
    JSON-serializable.
    """
    import sys

    sanitized = {}
    for key, value in obj.items():
        if isinstance(value, dict):
            # Nested dicts (e.g. properties, orbitals) are structure, not a
            # leaf value -- recurse instead of testing them as a whole.
            sanitized[key] = sanitize(value)
            continue
        converted, dirty = _sanitize_value(value)
        if dirty:
            print(
                f"avogadro-trexio: dropping non-finite values found in '{key}'",
                file=sys.stderr,
            )
            continue
        sanitized[key] = converted
    return sanitized


# ---------------------------------------------------------------------------
# AO component ordering within a shell
# ---------------------------------------------------------------------------
# TREXIO and Avogadro both lay MO coefficients out shell by shell, but they
# disagree on the order of the components *inside* a shell, so the
# coefficient columns have to be permuted or every affected orbital comes out
# scrambled. Two independent disagreements:
#
#  * Spherical p. TREXIO orders real solid harmonics 0, +1, -1, +2, -2, ...
#    which for l=1 means (pz, px, py) -- the TREXIO spec calls this out
#    explicitly. Avogadro's pointP evaluates x, y, z in that order
#    (gaussiansettools.cpp, "values[moIndex + i] = components[i] * delta[i]").
#    CJSON has no "-1" shell type, so spherical and Cartesian p both arrive
#    as shellType 1 and there is no way for Avogadro to tell them apart --
#    we must reorder here. For l >= 2 the two conventions agree exactly
#    (Avogadro's D5/F7/G9 components are documented as 0, +1, -1, +2, -2, ...
#    matching TREXIO), so only l == 1 needs a permutation.
#
#  * Cartesian shells. TREXIO uses the canonical/alphabetical ordering
#    (xx, xy, xz, yy, yz, zz); Avogadro uses the Gaussian/Molden-style
#    tables below. These differ for every l >= 2.
#
# Each entry is Avogadro's component order as (nx, ny, nz) exponent tuples,
# transcribed from gaussiansettools.cpp's componentsD/componentsF/componentsG
# arrays.
_AVOGADRO_CARTESIAN_ORDER = {
    0: [(0, 0, 0)],
    1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    # xx, yy, zz, xy, xz, yz
    2: [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)],
    # xxx, yyy, zzz, xyy, xxy, xxz, xzz, yzz, yyz, xyz
    3: [
        (3, 0, 0), (0, 3, 0), (0, 0, 3), (1, 2, 0), (2, 1, 0),
        (2, 0, 1), (1, 0, 2), (0, 1, 2), (0, 2, 1), (1, 1, 1),
    ],
    # xxxx, yyyy, zzzz, xxxy, xxxz, yyyx, yyyz, zzzx, zzzy,
    # xxyy, xxzz, yyzz, xxyz, yyxz, zzxy
    4: [
        (4, 0, 0), (0, 4, 0), (0, 0, 4), (3, 1, 0), (3, 0, 1),
        (1, 3, 0), (0, 3, 1), (1, 0, 3), (0, 1, 3), (2, 2, 0),
        (2, 0, 2), (0, 2, 2), (2, 1, 1), (1, 2, 1), (1, 1, 2),
    ],
}


def _trexio_cartesian_order(l):
    """TREXIO's canonical ("alphabetical") Cartesian order for shell ``l``.

    Iterating nx from l down to 0, then ny from (l - nx) down to 0, yields
    xx, xy, xz, yy, yz, zz for l=2 -- exactly the ordering the TREXIO spec
    documents.
    """
    return [
        (nx, ny, l - nx - ny)
        for nx in range(l, -1, -1)
        for ny in range(l - nx, -1, -1)
    ]


def ao_permutation(l, cartesian):
    """Map Avogadro's AO component slots onto TREXIO's, for one shell.

    Returns a list ``perm`` of length equal to the shell's AO count, where
    Avogadro's component ``j`` must take TREXIO's component ``perm[j]``.
    Returns None when no reordering is needed, so callers can skip the work
    in the overwhelmingly common case.
    """
    if cartesian:
        if l < 2:
            # l=0 is a single function; TREXIO and Avogadro both order
            # Cartesian p as x, y, z.
            return None
        trexio_order = _trexio_cartesian_order(l)
        return [trexio_order.index(term) for term in _AVOGADRO_CARTESIAN_ORDER[l]]

    # Spherical: only p disagrees. TREXIO (0, +1, -1) == (z, x, y), so
    # Avogadro's x, y, z takes TREXIO components 1, 2, 0.
    if l == 1:
        return [1, 2, 0]
    return None


def shell_ao_count(l, cartesian):
    """Number of AO components in a shell of angular momentum ``l``."""
    return (l + 1) * (l + 2) // 2 if cartesian else 2 * l + 1
