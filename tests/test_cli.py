# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end CLI tests: invoke the plugin exactly the way Avogadro does.

Avogadro spawns ``avogadro-trexio <identifier> --read [--debug] --lang
<locale>`` as a subprocess and writes a small JSON envelope to its stdin
(this plugin only understands file-input mode). We reproduce that here
with ``sys.executable -c "from avogadro_trexio import main; main()" ...``
so the tests work the same way on Windows (where the installed
console-script is a .exe wrapper) as everywhere else -- mirrors
avogadro-cclib's tests/test_cli.py.
"""

import json

from conftest import ALL_MOLECULE_FIXTURES, run_cli, strict_json_loads


def _payload(path):
    return json.dumps({"operation": "read", "filename": str(path)}).encode("utf-8")


def test_cli_read_produces_valid_cjson(fixture_paths):
    for name in ALL_MOLECULE_FIXTURES:
        result = run_cli(["trexio", "--read"], _payload(fixture_paths[name]))
        assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
        cjson = strict_json_loads(result.stdout.decode("utf-8"))
        assert cjson["chemicalJson"] == 1


def test_stdout_is_bare_cjson_with_no_leaked_diagnostics(fixture_paths):
    # The "slater" fixture is guaranteed to print a stderr diagnostic
    # (basis skipped) -- exactly the case where a bug might leak that text
    # onto stdout instead. stdout must be exactly one JSON document.
    result = run_cli(["trexio", "--read"], _payload(fixture_paths["slater"]))
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")

    stdout_text = result.stdout.decode("utf-8")
    assert stdout_text.startswith("{")
    decoder = json.JSONDecoder()
    obj, end = decoder.raw_decode(stdout_text)
    assert (
        stdout_text[end:].strip() == ""
    ), "trailing content found after the JSON document"
    assert obj["chemicalJson"] == 1

    # ... but the diagnostic must still have gone *somewhere* useful.
    assert "Slater" in result.stderr.decode("utf-8", errors="replace")


def test_periodic_kpoint_diagnostic_stays_on_stderr(fixture_paths):
    result = run_cli(["trexio", "--read"], _payload(fixture_paths["periodic"]))
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    stdout_text = result.stdout.decode("utf-8")
    cjson = strict_json_loads(stdout_text)
    assert cjson["chemicalJson"] == 1
    assert "dropped" in result.stderr.decode("utf-8", errors="replace")


def test_bare_non_json_stdin_exits_nonzero_and_mentions_avogadro_version():
    # Simulates an older Avogadro that still uses "string" mode and simply
    # writes the file's raw bytes to stdin -- this plugin only supports
    # file-input mode, so that must fail loudly with an explanation, not
    # crash or hang.
    result = run_cli(["trexio", "--read"], b"not json at all, just raw bytes")
    assert result.returncode != 0
    assert result.stdout == b""
    stderr_text = result.stderr.decode("utf-8", errors="replace")
    assert "2.0.0" in stderr_text


def test_missing_filename_key_also_fails_with_version_message():
    payload = json.dumps({"operation": "read"}).encode("utf-8")
    result = run_cli(["trexio", "--read"], payload)
    assert result.returncode != 0
    assert result.stdout == b""
    assert "2.0.0" in result.stderr.decode("utf-8", errors="replace")
