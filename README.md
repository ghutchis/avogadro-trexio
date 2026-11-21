# avogadro-trexio

Avogadro 2 script plugin that reads [TREXIO](https://trex-coe.github.io/trexio/)
HDF5 wave function files and emits CJSON, using the
[trexio](https://pypi.org/project/trexio/) Python package.

## Requirements

- Avogadro **2.0.0 or newer**. This plugin uses `input-mode = "file"`
  (see `pyproject.toml`), which hands the plugin a filename to open
  directly rather than streaming file content over stdin. That is a
  hard requirement here, not an optimization: an HDF5 file is a
  random-access binary container read via internal seeks, so there is
  no way to feed trexio "the bytes of the file" the way avogadro-cclib
  feeds cclib a `StringIO`. Older Avogadro releases that only support
  stdin-byte-stream ("string mode") input cannot use this plugin at
  all; running it there produces a clear stderr message rather than a
  crash or a hang.
- No decompression of any kind is implemented (unlike avogadro-cclib):
  since the file is opened directly by path, decompressing it into an
  in-memory copy first would gain nothing, and HDF5 files are usually
  already compressed internally on a per-dataset basis besides.

## File extensions

Registered for `.h5`, `.hdf5` and `.trexio`. The first two are generic
HDF5 suffixes, so the reader sniffs before claiming a file: it checks the
HDF5 magic bytes, then requires a `nucleus` group. A plain HDF5 file from
some other program fails that second check and is reported as "not a
TREXIO file" rather than being half-parsed. Note that Avogadro matches
only the final suffix, so `foo.trexio.gz` will not reach this plugin.

## Platform support

`pyproject.toml`'s `[tool.pixi.workspace]` deliberately omits `win-64`.
The `trexio` package publishes no Windows wheel on PyPI, and
conda-forge has no `win-64` build for it either -- there is currently
nothing to install on Windows. CI accordingly only runs on
`ubuntu-latest` and `macos-latest`.

## Coverage

- **Atoms**: coordinates (bohr -> Angstrom), element numbers (from
  `nucleus_label` when recognisable, else from `nucleus_charge`,
  corrected for `ecp_z_core` when present), and nucleus labels.
- **Properties**: total charge, spin multiplicity, total energy
  (`state_energy`, hartree -> kJ/mol), point group, and metadata
  (`trexioCode`, `trexioDescription`). `ecp_z_core` is surfaced as
  `properties.coreElectrons` (informational only -- see "ECPs" below).
- **Unit cell**: `cellVectors` only (never `a`/`b`/`c`/angles), only
  when the `cell` group is present and the structure is periodic.
- **Basis set**: Gaussian (contracted GTO) basis sets only, with shell
  angular momentum up to g (l=4) and zero radial power -- i.e. exactly
  what CJSON/Avogadro's `GaussianSet` can represent. Slater, numerical
  (grid), plane-wave, and higher-l basis sets are not representable;
  when encountered, the basis (and any orbitals) are skipped with a
  one-line explanation on stderr, but the rest of the molecule (atoms,
  properties, unit cell) still loads normally.
- **Orbitals**: closed-shell (`moCoefficients`/`energies`) and
  open-shell (`alphaCoefficients`/`betaCoefficients` etc., split via
  `mo_spin`) MOs, energies converted hartree -> eV, occupations, and
  symmetry labels. For a periodic file (`mo_k_point` present), only
  Gamma-point (`k_point == 0`) MOs are kept; everything else is
  dropped with a count on stderr, since CJSON has no notion of
  k-points at all.

## Coefficient normalization

TREXIO defines a shell's radial function as

```
R_s(r) = shell_factor * sum_k prim_factor_k * coefficient_k * exp(-exponent_k * r^2)
```

while Avogadro's `GaussianSet` multiplies every CJSON coefficient by
its own per-primitive normalization constant

```
N(alpha, l) = (2*alpha/pi)^(3/4) * (4*alpha)^(l/2) / sqrt((2l-1)!!)
```

before use. So the CJSON coefficient this plugin emits for each
primitive is `shell_factor * prim_factor * coefficient / N(exponent, l)`
-- dividing out exactly the factor `GaussianSet` will multiply back in,
so the reconstructed radial function matches TREXIO's to floating-point
precision. See `src/avogadro_trexio/utils.py:gto_normalization` and
`tests/test_normalization.py` for the derivation and a from-scratch
round-trip check.

This conversion does **not** capture TREXIO's `ao_normalization`
(sometimes called N'_i): a further, per-AO-*component* normalization
factor that can differ between e.g. an `xy` and an `xx` Cartesian
component of the same d shell, or between different `m` values of a
spherical shell. CJSON has exactly one coefficient per primitive *per
shell* (not per AO component), so this detail has no home in the
format -- it is silently dropped. This matches Avogadro's existing
behaviour for Cartesian shells read via avogadro-cclib, which has the
same limitation for the same structural reason.

## AO component ordering

TREXIO and Avogadro both lay MO coefficients out shell by shell, but
they disagree on the order of the components *inside* a shell, so this
plugin permutes the coefficient columns on the way out. Two separate
disagreements:

- **Spherical p.** TREXIO orders real solid harmonics
  `0, +1, -1, +2, -2, ...`, which for l=1 means `(pz, px, py)` -- the
  TREXIO specification calls this out explicitly. Avogadro's `pointP`
  evaluates `x, y, z` in that order. For l >= 2 the two conventions
  agree exactly (Avogadro's D5/F7/G9 components are documented in the
  same `0, +1, -1, +2, -2, ...` order), so only p needs reordering.
- **Cartesian shells with l >= 2.** TREXIO uses the canonical
  ("alphabetical") ordering `xx, xy, xz, yy, yz, zz`; Avogadro uses a
  Gaussian/Molden-style table, `xx, yy, zz, xy, xz, yz`, and similarly
  for f and g.

Neither is detectable downstream: CJSON has no `-1` shell type, so a
spherical p shell and a Cartesian one both arrive at Avogadro as
`shellTypes` entry `1`. If the reader did not permute, the affected
orbitals would be silently wrong rather than visibly broken -- which
is why `tests/test_ao_ordering.py` pins both permutations against
index sequences derived by hand from the two conventions, rather than
against the reader's own helper.

## ECPs

CJSON has no real representation for effective core potentials. When
`ecp_z_core` is present, this plugin reports it as
`properties.coreElectrons` (one value per atom) and
`properties.totalCoreElectrons` (their sum) purely as informational
metadata -- Avogadro will not actually apply the ECP to anything (e.g.
the basis set is still read and used as-is, with no core potential
term). `nucleus_charge` under an ECP is the *reduced* charge (core
electrons already subtracted), so `ecp_z_core` is also used to correct
the element-number fallback back to the real atomic number when no
usable `nucleus_label` is available.

## Testing

```bash
pip install -e ".[test]"
pytest
```

There is no small, permissively-licensed corpus of real TREXIO files to
check in, and binary `.h5` files do not belong in git regardless -- see
`tests/README.md` for how the test fixtures are generated instead.
