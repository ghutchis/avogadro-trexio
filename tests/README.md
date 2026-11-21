# Test data

Unlike avogadro-cclib, this project has no `tests/data/` corpus of
committed real-world files. There is no small, permissively-licensed
corpus of TREXIO HDF5 files to draw on, and binary `.h5` files do not
belong in git regardless.

Instead, `make_fixtures.py` builds a small set of synthetic TREXIO
files directly with the `trexio` Python API (water-molecule geometries,
a minimal s/p/d Gaussian basis, closed- and open-shell MOs, a periodic
cell with two k-points, an ECP, a Slater basis, and a bare
nucleus-only file). `conftest.py`'s session-scoped `fixtures_dir`/
`fixture_paths` fixtures build these once per test run into a pytest
tmp directory -- nothing is written into the repository.

To add a new fixture, add a `build_*` function to `make_fixtures.py`
and register it in `build_all()`, then reference it by its logical
name (filename without `.h5`) from a test via the `fixture_paths`
fixture.
