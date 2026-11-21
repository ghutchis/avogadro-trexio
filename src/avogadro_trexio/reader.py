"""
/******************************************************************************
  This source file is part of the Avogadro project.
  This source code is released under the 3-Clause BSD License, (see "LICENSE").
******************************************************************************/
"""

import sys

import numpy as np
import trexio

from .utils import (
    BOHR_TO_ANGSTROM,
    HARTREE_TO_EV,
    HARTREE_TO_KJ_MOL,
    ao_permutation,
    gto_normalization,
    has,
    read_if,
    sanitize,
    shell_ao_count,
    symbol_to_number,
)

# The first 8 bytes of every HDF5 file, regardless of what is inside it.
# TREXIO's HDF5 back end is just an HDF5 file with a particular internal
# layout, so this only rules out "not HDF5 at all" -- it does not tell us
# the file is a TREXIO file (see _require_trexio_structure below).
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"

# CJSON (and Avogadro's GaussianSet, avogadrolibs cjsonformat.cpp:880-910)
# only understands shell angular momenta up to g (l=4); anything higher
# falls through to GaussianSet::UU ("unknown/unsupported") and silently
# breaks the whole basis set. We check this ourselves and skip the basis
# cleanly rather than emitting something Avogadro would half-parse.
_MAX_SHELL_ANG_MOM = 4


def read(filename):
    """Read the TREXIO HDF5 file at ``filename`` and return a CJSON dict.

    Raises ValueError for anything that means "this is not a readable
    TREXIO file" (wrong magic bytes, a trexio-library open failure, or an
    HDF5 file that opens fine but has no nucleus group) -- these are the
    three sniffing failure modes we must not let hijack an unrelated .h5/
    .hdf5 file. Every other problem (e.g. a basis set CJSON cannot
    represent) is non-fatal: the molecule is still returned, just missing
    that piece, with an explanation on stderr.
    """
    with open(filename, "rb") as fh:
        magic = fh.read(len(_HDF5_MAGIC))
    if magic != _HDF5_MAGIC:
        raise ValueError(f"{filename}: not an HDF5 file")

    try:
        trexio_file = trexio.File(filename, "r", trexio.TREXIO_HDF5)
    except Exception as exc:
        # trexio.Error (or, in principle, some lower-level failure) --
        # either way this is "the trexio library could not open the file",
        # which we report as one clean, filename-specific message rather
        # than whatever internal exception type trexio happened to raise.
        raise ValueError(f"{filename}: not a readable TREXIO file ({exc})") from exc

    try:
        # A generic .h5/.hdf5 file (e.g. some other program's HDF5 output)
        # will typically open fine here -- HDF5 itself doesn't know about
        # TREXIO's schema. The nucleus group is present in every TREXIO
        # file that describes an actual system, so its absence is our
        # sniff for "this is HDF5, but not TREXIO".
        if not (has(trexio_file, "nucleus_num") and has(trexio_file, "nucleus_coord")):
            raise ValueError(
                f"{filename}: this HDF5 file is not a TREXIO file "
                "(no nucleus group)"
            )
        cjson = _read_molecule(trexio_file)
    finally:
        trexio_file.close()

    return sanitize(cjson)


def _read_molecule(f):
    cjson = {"chemicalJson": 1, "atoms": {}}

    nucleus_num = trexio.read_nucleus_num(f)
    coords_bohr = np.asarray(trexio.read_nucleus_coord(f), dtype=float)
    coords_angstrom = (coords_bohr * BOHR_TO_ANGSTROM).flatten().tolist()

    labels = read_if(f, "nucleus_label")
    charges = read_if(f, "nucleus_charge")
    z_core = read_if(f, "ecp_z_core")

    numbers = []
    for i in range(nucleus_num):
        label = labels[i] if labels is not None else None
        number = symbol_to_number(label) if label else None
        if number is None:
            # No usable label -- fall back to nucleus_charge. Under an
            # ECP, nucleus_charge is the *reduced* charge (core electrons
            # already subtracted), so it must be corrected back up by
            # z_core before rounding to an atomic number.
            if charges is not None:
                charge = float(charges[i])
                if z_core is not None:
                    charge += float(z_core[i])
                number = int(round(charge))
            else:
                # No label, no charge: nothing to identify this centre by.
                # TREXIO allows dummy centres that carry basis functions
                # but no real nucleus, so element 0 is a legitimate,
                # deliberate output here -- not a failure.
                number = 0
        numbers.append(number)

    cjson["atoms"]["coords"] = {"3d": coords_angstrom}
    cjson["atoms"]["elements"] = {"number": numbers}
    if labels is not None:
        cjson["atoms"]["labels"] = list(labels)

    properties = {}

    electron_count = None
    if has(f, "electron_up_num") and has(f, "electron_dn_num"):
        up = trexio.read_electron_up_num(f)
        dn = trexio.read_electron_dn_num(f)
        electron_count = up + dn
        properties["totalSpinMultiplicity"] = up - dn + 1
        if charges is not None:
            properties["totalCharge"] = int(
                round(float(np.sum(charges)) - electron_count)
            )

    if has(f, "state_energy"):
        properties["totalEnergy"] = trexio.read_state_energy(f) * HARTREE_TO_KJ_MOL

    if has(f, "nucleus_point_group"):
        properties["pointGroup"] = trexio.read_nucleus_point_group(f)

    if z_core is not None:
        # CJSON has no real ECP representation -- this is informational
        # only (see README.md). A flat numeric array under "properties"
        # becomes an Nx1 matrix on Avogadro's side (cjsonformat.cpp:1185),
        # which is the closest thing CJSON has to "one number per atom".
        core_electrons = [float(x) for x in z_core]
        properties["coreElectrons"] = core_electrons
        properties["totalCoreElectrons"] = float(np.sum(z_core))

    if has(f, "metadata_code"):
        codes = trexio.read_metadata_code(f)
        if codes:
            properties["trexioCode"] = ", ".join(codes)
    if has(f, "metadata_description"):
        description = trexio.read_metadata_description(f)
        if description:
            properties["trexioDescription"] = description

    if properties:
        cjson["properties"] = properties

    unit_cell = _read_unit_cell(f)
    if unit_cell is not None:
        cjson["unitCell"] = unit_cell

    basis_set, shell_ang_mom, ao_cartesian = _read_basis_set(f)
    if basis_set is not None:
        cjson["basisSet"] = basis_set
        orbitals = _read_orbitals(f, electron_count, shell_ang_mom, ao_cartesian)
        if orbitals is not None:
            cjson["orbitals"] = orbitals

    return cjson


def _read_unit_cell(f):
    """Return a CJSON unitCell dict, or None if there is nothing to emit.

    Only emitted when the cell group is present AND the structure is
    actually periodic (pbc_periodic absent is treated as "yes, periodic"
    -- TREXIO files that bother to write cell vectors at all are
    periodic in practice; pbc_periodic only needs checking when present
    and it says otherwise).
    """
    if not has(f, "cell"):
        return None
    if has(f, "pbc_periodic") and not trexio.read_pbc_periodic(f):
        return None

    cell_a = np.asarray(trexio.read_cell_a(f), dtype=float) * BOHR_TO_ANGSTROM
    cell_b = np.asarray(trexio.read_cell_b(f), dtype=float) * BOHR_TO_ANGSTROM
    cell_c = np.asarray(trexio.read_cell_c(f), dtype=float) * BOHR_TO_ANGSTROM

    # Avogadro prefers cellVectors over a/b/c/angles when both could be
    # derived (cjsonformat.cpp:799) -- so that is the only form we emit.
    return {
        "cellVectors": (
            cell_a.tolist() + cell_b.tolist() + cell_c.tolist()
        )
    }


def _read_basis_set(f):
    """Return (basisSet dict, shell_ang_mom array, ao_cartesian), else Nones.

    Also returns the per-shell angular momentum array so the caller does
    not need to re-derive it, but the important thing this function
    decides is whether the basis is representable in CJSON at all --
    Gaussian-only, l<=4, and no non-zero radial power (a plain GTO).
    Anything else means "skip basisSet/orbitals entirely, keep everything
    else", with a one-line explanation on stderr.
    """
    if not has(f, "basis_type"):
        return None, None, None

    basis_type = trexio.read_basis_type(f)
    if basis_type.strip().lower() != "gaussian":
        print(
            f"avogadro-trexio: basis_type='{basis_type}' cannot be represented "
            "in CJSON (only Gaussian basis sets can); skipping basisSet and "
            "orbitals",
            file=sys.stderr,
        )
        return None, None, None

    shell_ang_mom = np.asarray(trexio.read_basis_shell_ang_mom(f), dtype=int)
    if shell_ang_mom.size and int(shell_ang_mom.max()) > _MAX_SHELL_ANG_MOM:
        print(
            "avogadro-trexio: basis contains shells with angular momentum "
            "above g (l=4), which CJSON cannot represent; skipping basisSet "
            "and orbitals",
            file=sys.stderr,
        )
        return None, None, None

    if has(f, "basis_r_power"):
        r_power = np.asarray(trexio.read_basis_r_power(f))
        if np.any(r_power != 0):
            print(
                "avogadro-trexio: basis has a non-zero radial power (not a "
                "plain GTO), which CJSON cannot represent; skipping "
                "basisSet and orbitals",
                file=sys.stderr,
            )
            return None, None, None

    shell_num = shell_ang_mom.shape[0]
    shell_nucleus = np.asarray(trexio.read_basis_nucleus_index(f), dtype=int)
    shell_index_per_prim = np.asarray(trexio.read_basis_shell_index(f), dtype=int)
    exponents_raw = np.asarray(trexio.read_basis_exponent(f), dtype=float)
    coefficients_raw = np.asarray(trexio.read_basis_coefficient(f), dtype=float)
    prim_num = exponents_raw.shape[0]

    shell_factor = (
        np.asarray(trexio.read_basis_shell_factor(f), dtype=float)
        if has(f, "basis_shell_factor")
        else np.ones(shell_num)
    )
    prim_factor = (
        np.asarray(trexio.read_basis_prim_factor(f), dtype=float)
        if has(f, "basis_prim_factor")
        else np.ones(prim_num)
    )

    # ao_cartesian is a single flag for the whole basis (not per-shell):
    # 1 = Cartesian AOs, 0 (or absent) = spherical.
    ao_cartesian = bool(trexio.read_ao_cartesian(f)) if has(f, "ao_cartesian") else False

    # Bucket primitives by shell index. basis_shell_index is *not*
    # guaranteed to be contiguous or sorted by shell -- some TREXIO
    # writers interleave primitives across shells -- so we group them
    # ourselves, preserving each primitive's original relative order
    # within its shell.
    buckets = [[] for _ in range(shell_num)]
    for k in range(prim_num):
        buckets[int(shell_index_per_prim[k])].append(k)

    shell_to_atom_map = shell_nucleus.tolist()
    shell_types = []
    primitives_per_shell = []
    exponents = []
    coefficients = []

    for s in range(shell_num):
        l = int(shell_ang_mom[s])
        if l <= 1:
            shell_types.append(l)
        else:
            # Spherical -> negative (CJSON/Avogadro convention for D5/F7/
            # G9); Cartesian -> positive.
            shell_types.append(l if ao_cartesian else -l)

        prim_indices = buckets[s]
        primitives_per_shell.append(len(prim_indices))
        norm_s = float(shell_factor[s])
        for k in prim_indices:
            alpha = float(exponents_raw[k])
            a_k = float(coefficients_raw[k])
            f_k = float(prim_factor[k])
            # The crux of the whole conversion: TREXIO's shell radial
            # function is R_s(r) = norm_s * sum_k f_k * a_k * exp(-alpha
            # r^2), but Avogadro's GaussianSet multiplies each CJSON
            # coefficient by its OWN primitive normalization N(alpha, l)
            # before use. So the CJSON coefficient must be the TREXIO
            # coefficient (with its shell/primitive factors folded in)
            # divided by that same N(alpha, l), so the two cancel out and
            # GaussianSet reconstructs exactly R_s(r). See
            # utils.gto_normalization for the formula and its provenance.
            coefficients.append(norm_s * f_k * a_k / gto_normalization(alpha, l))
            exponents.append(alpha)

    basis_set = {
        "shellToAtomMap": shell_to_atom_map,
        "shellTypes": shell_types,
        "primitivesPerShell": primitives_per_shell,
        "exponents": exponents,
        "coefficients": coefficients,
    }
    return basis_set, shell_ang_mom, ao_cartesian


def _ao_column_permutation(shell_ang_mom, cartesian, ao_num):
    """Column order that maps TREXIO's AO layout onto Avogadro's.

    Walks the shells in order, converting each shell's within-shell
    permutation into absolute AO column indices. Returns None when no shell
    needs reordering (the common case for an all-spherical s/d/f/g basis),
    so the caller can skip the copy entirely.

    Also returns None -- deliberately leaving the coefficients untouched --
    if the AO count implied by the shells disagrees with the file's actual
    ao_num. That means our picture of the basis is wrong (a mixed
    spherical/Cartesian file, say), and reordering on a wrong picture would
    scramble the orbitals far more thoroughly than leaving them alone.
    """
    order = []
    offset = 0
    needed = False
    for l in shell_ang_mom:
        count = shell_ao_count(int(l), cartesian)
        perm = ao_permutation(int(l), cartesian)
        if perm is None:
            order.extend(range(offset, offset + count))
        else:
            needed = True
            order.extend(offset + index for index in perm)
        offset += count

    if offset != ao_num:
        print(
            "avogadro-trexio: the shells describe "
            f"{offset} atomic orbitals but the file reports {ao_num}; "
            "leaving MO coefficients in their original order",
            file=sys.stderr,
        )
        return None

    return order if needed else None


def _read_orbitals(f, electron_count, shell_ang_mom, ao_cartesian):
    """Return a CJSON orbitals dict, or None if there is no mo group."""
    if not (has(f, "mo_num") and has(f, "mo_coefficient")):
        return None

    mo_num = trexio.read_mo_num(f)
    # Reshaped to [mo_num, ao_num] by trexio itself.
    mo_coefficient = np.asarray(trexio.read_mo_coefficient(f), dtype=float)

    # TREXIO and Avogadro order the AO components *within* each shell
    # differently (spherical p, and every Cartesian shell with l >= 2), so
    # the coefficient columns must be permuted -- see utils.ao_permutation.
    # Without this every p orbital in an ordinary spherical file comes out
    # silently scrambled.
    column_order = _ao_column_permutation(
        shell_ang_mom, ao_cartesian, mo_coefficient.shape[1]
    )
    if column_order is not None:
        mo_coefficient = mo_coefficient[:, column_order]

    keep = np.ones(mo_num, dtype=bool)
    if has(f, "mo_k_point"):
        # A periodic file's mo_num sums MOs over every k-point; CJSON has
        # no notion of k-points at all, so we keep only the Gamma point
        # (k_point == 0) and say how many we dropped.
        k_point = np.asarray(trexio.read_mo_k_point(f))
        keep = k_point == 0
        dropped = int(mo_num - keep.sum())
        if dropped:
            print(
                f"avogadro-trexio: dropped {dropped} MO(s) from non-Gamma "
                "k-points; CJSON has no representation for k-points",
                file=sys.stderr,
            )

    mo_coefficient = mo_coefficient[keep]

    mo_energy = (
        np.asarray(trexio.read_mo_energy(f), dtype=float)[keep]
        if has(f, "mo_energy")
        else None
    )
    mo_occupation = (
        np.asarray(trexio.read_mo_occupation(f), dtype=float)[keep]
        if has(f, "mo_occupation")
        else None
    )
    mo_symmetry = (
        [s for s, k in zip(trexio.read_mo_symmetry(f), keep) if k]
        if has(f, "mo_symmetry")
        else None
    )
    mo_spin = (
        np.asarray(trexio.read_mo_spin(f), dtype=int)[keep]
        if has(f, "mo_spin")
        else None
    )

    orbitals = {}

    # An open-shell TREXIO file counts spin-orbitals in mo_num, with
    # mo_spin telling apart alpha (0) from beta (1); Avogadro reads split
    # alpha/beta coefficients and energies for that case
    # (cjsonformat.cpp:953-1007). A closed-shell file (mo_spin absent, or
    # present but all zero) uses the single moCoefficients/energies form.
    open_shell = mo_spin is not None and bool(np.any(mo_spin == 1))

    if open_shell:
        alpha_mask = mo_spin == 0
        beta_mask = mo_spin == 1
        orbitals["alphaCoefficients"] = mo_coefficient[alpha_mask].flatten().tolist()
        orbitals["betaCoefficients"] = mo_coefficient[beta_mask].flatten().tolist()
        if mo_energy is not None:
            orbitals["alphaEnergies"] = (mo_energy[alpha_mask] * HARTREE_TO_EV).tolist()
            orbitals["betaEnergies"] = (mo_energy[beta_mask] * HARTREE_TO_EV).tolist()
        if mo_symmetry is not None:
            alpha_syms = [s for s, keep_a in zip(mo_symmetry, alpha_mask) if keep_a]
            beta_syms = [s for s, keep_b in zip(mo_symmetry, beta_mask) if keep_b]
            orbitals["symmetries"] = [alpha_syms, beta_syms]
    else:
        orbitals["moCoefficients"] = mo_coefficient.flatten().tolist()
        if mo_energy is not None:
            orbitals["energies"] = (mo_energy * HARTREE_TO_EV).tolist()
        if mo_symmetry is not None:
            orbitals["symmetries"] = list(mo_symmetry)

    if mo_occupation is not None:
        if open_shell:
            # Avogadro reads open-shell occupations only from the split
            # alphaOccupations/betaOccupations keys -- a flat "occupations"
            # array is ignored entirely on the open-shell path
            # (cjsonformat.cpp:1019-1035).
            orbitals["alphaOccupations"] = [
                float(x) for x in mo_occupation[alpha_mask]
            ]
            orbitals["betaOccupations"] = [
                float(x) for x in mo_occupation[beta_mask]
            ]
        else:
            orbitals["occupations"] = [float(x) for x in mo_occupation]

    if electron_count is not None:
        orbitals["electronCount"] = int(electron_count)

    return orbitals
