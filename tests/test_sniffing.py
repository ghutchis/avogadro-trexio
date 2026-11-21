# SPDX-License-Identifier: BSD-3-Clause

"""Sniffing: avogadro-trexio must not hijack a foreign .h5/.hdf5 file.

.h5/.hdf5 are generic extensions -- any program's HDF5 output could carry
one. reader.read() must fail cleanly (a plain ValueError, never an
unhandled trexio/HDF5 exception) for anything that is not actually a
TREXIO file, and the CLI must turn that into a readable exit(1) message.

We can reach two of the three documented sniffing branches without an
h5py dependency:
  1. bad magic bytes (not HDF5 at all)
  2. a file trexio itself can open (built with trexio.File(..., "w", ...))
     but that never had a nucleus group written to it

The third branch -- a real foreign HDF5 file (e.g. written by h5py, with
valid HDF5 structure but no TREXIO groups at all) -- is not exercised here
because this project has no h5py dependency to fabricate one; the
"nucleus group missing" check (branch 2) already covers the same code
path in reader.py (has_nucleus_num()/has_nucleus_coord() both false), just
via a TREXIO-library-created file rather than an arbitrary one. This is a
known, accepted gap in fixture coverage -- see also the note in
tests/make_fixtures.py:build_hdf5_no_nucleus.
"""

import json

import pytest

from avogadro_trexio.reader import read
from conftest import run_cli


def test_bad_magic_raises_clean_value_error(fixture_paths):
    with pytest.raises(ValueError, match="not an HDF5 file"):
        read(str(fixture_paths["bad_magic"]))


def test_hdf5_without_nucleus_group_raises_clean_value_error(fixture_paths):
    with pytest.raises(ValueError, match="no nucleus group"):
        read(str(fixture_paths["hdf5_no_nucleus"]))


def test_cli_exits_nonzero_on_bad_magic(fixture_paths):
    payload = json.dumps(
        {"operation": "read", "filename": str(fixture_paths["bad_magic"])}
    ).encode("utf-8")
    result = run_cli(["trexio", "--read"], payload)
    assert result.returncode != 0
    assert result.stdout == b""
    stderr_text = result.stderr.decode("utf-8", errors="replace")
    assert "not an HDF5 file" in stderr_text


def test_cli_exits_nonzero_on_hdf5_without_nucleus_group(fixture_paths):
    payload = json.dumps(
        {"operation": "read", "filename": str(fixture_paths["hdf5_no_nucleus"])}
    ).encode("utf-8")
    result = run_cli(["trexio", "--read"], payload)
    assert result.returncode != 0
    assert result.stdout == b""
    stderr_text = result.stderr.decode("utf-8", errors="replace")
    assert "nucleus group" in stderr_text


def test_nonexistent_file_fails_cleanly(tmp_path):
    missing = tmp_path / "does_not_exist.h5"
    with pytest.raises((ValueError, OSError)):
        read(str(missing))
