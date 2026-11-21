"""
/******************************************************************************
  This source file is part of the Avogadro project.
  This source code is released under the 3-Clause BSD License, (see "LICENSE").
******************************************************************************/
"""

import argparse
import json
import sys
import traceback


def main():
    # Avogadro calls the plugin as:
    #   avogadro-trexio <identifier> [--read] [--write] [--lang <locale>] [--debug]
    #
    # Unlike avogadro-cclib, this plugin supports only "file" input mode
    # (input-mode = "file" in pyproject.toml): stdin carries a small JSON
    # object, e.g. {"operation": "read", "filename": "/path/to/file.h5"}.
    # There is no "string" mode fallback here, and deliberately so -- an
    # HDF5 file is a random-access binary container (it is read via seeks
    # into its own internal b-tree/heap structures, not a byte stream), so
    # there is no way to hand trexio "the bytes of the file" over stdin the
    # way avogadro-cclib hands cclib a StringIO. TREXIO needs a real path
    # on disk, which is exactly what file-input mode provides. Only
    # Avogadro >= 2.0.0 understands input-mode = "file"
    # (avogadrolibs/avogadro/qtplugins/scriptfileformats/fileformatscript.cpp:62),
    # which is why pyproject.toml declares that as the minimum version, and
    # why a bare (non-JSON, or JSON-without-filename) payload below is
    # treated as "this Avogadro is too old for this plugin" rather than as
    # ordinary garbage input.
    parser = argparse.ArgumentParser()
    parser.add_argument("feature")
    parser.add_argument("--lang", nargs="?", default="en")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--read", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    # Read stdin as bytes -- never let the platform locale decide the
    # encoding (this is the source of mojibake / UnicodeDecodeError on
    # Windows, where stdin would otherwise be decoded as cp1252 or cp932).
    raw = sys.stdin.buffer.read()

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        payload = None

    if not isinstance(payload, dict) or not (
        "filename" in payload or "file" in payload
    ):
        print(
            "avogadro-trexio: this format requires Avogadro 2.0.0 or newer "
            "(file input mode) -- no filename was supplied on stdin",
            file=sys.stderr,
        )
        sys.exit(1)

    filename = payload.get("filename", payload.get("file"))

    output = None
    try:
        match args.feature:
            case "trexio":
                if args.read:
                    from .reader import read

                    cjson = read(filename)
                    output = json.dumps(cjson, ensure_ascii=True)
    except Exception as exc:  # noqa: BLE001 - report cleanly, never crash
        print(f"avogadro-trexio: {exc}", file=sys.stderr)
        if args.debug:
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    if output is None:
        print(
            f"avogadro-trexio: no output produced for feature='{args.feature}'",
            file=sys.stderr,
        )
        sys.exit(1)

    # stdout must be nothing but the raw CJSON document -- no envelope, no
    # extra text -- since it is passed directly to Avogadro's
    # CjsonFormat::readString().
    sys.stdout.buffer.write(output.encode("utf-8"))
    sys.stdout.buffer.flush()
