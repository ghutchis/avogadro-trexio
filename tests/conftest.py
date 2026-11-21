# SPDX-License-Identifier: BSD-3-Clause

"""Shared fixtures for the avogadro-trexio test suite.

Unlike avogadro-cclib (which reads a corpus of real, committed output
files), there is no small permissively-licensed corpus of TREXIO HDF5
files to draw on, and binary .h5 files do not belong in git anyway. So the
whole fixture corpus is generated on the fly by make_fixtures.py, once per
test session, into a pytest tmp dir -- see fixtures_dir() below.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import make_fixtures  # noqa: E402  (path insert must come first)

# Avogadro spawns ``avogadro-trexio <identifier> --read [--debug] --lang
# <locale>`` as a subprocess and writes a small JSON envelope to its
# stdin (this plugin only supports file-input mode -- see __init__.py).
# We reproduce that here with ``sys.executable -c "from avogadro_trexio
# import main; main()" ...`` so the tests work the same way on Windows
# (where the installed console-script is a .exe wrapper) as everywhere
# else -- mirrors avogadro-cclib's tests/conftest.py.
MAIN_SNIPPET = "from avogadro_trexio import main; main()"


@pytest.fixture(scope="session")
def fixtures_dir(tmp_path_factory):
    """Build every synthetic TREXIO fixture once per test session.

    Returns the directory containing them; use the `fixture_paths` fixture
    below to get a {name: Path} mapping instead of listing the directory
    yourself.
    """
    tmp_dir = tmp_path_factory.mktemp("trexio-fixtures")
    make_fixtures.build_all(tmp_dir)
    return tmp_dir


@pytest.fixture(scope="session")
def fixture_paths(fixtures_dir):
    """{logical name (no .h5 suffix): Path} for every generated fixture."""
    return {p.stem: p for p in sorted(fixtures_dir.iterdir())}


# Fixture names that build_all() always produces; used to parametrize tests
# without depending on directory iteration order and to catch a fixture
# silently failing to be generated.
GAUSSIAN_BASIS_FIXTURES = [
    "h2o_sto3g_spherical",
    "h2o_cartesian",
    "uhf_open_shell",
    "periodic",
    "ecp",
]
ALL_MOLECULE_FIXTURES = GAUSSIAN_BASIS_FIXTURES + ["slater", "no_basis"]


def read_cjson(fixture_paths, name):
    """Return the CJSON dict for a fixture, running the real reader.read()."""
    from avogadro_trexio.reader import read

    return read(str(fixture_paths[name]))


def run_cli(args, input_bytes, env=None):
    cmd = [sys.executable, "-c", MAIN_SNIPPET, *args]
    return subprocess.run(cmd, input=input_bytes, capture_output=True, env=env)


def strict_json_loads(text):
    """json.loads that rejects bare NaN/Infinity/-Infinity tokens."""

    def _reject(token):
        raise ValueError(f"non-finite constant in JSON output: {token}")

    return json.loads(text, parse_constant=_reject)
