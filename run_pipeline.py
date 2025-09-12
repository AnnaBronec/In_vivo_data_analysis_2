#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

def _guess_output_csv(session_dir: Path, explicit_out: str | None) -> Path:
    if explicit_out:
        p = Path(explicit_out).expanduser().resolve()
        return p if p.suffix == ".csv" else p.with_suffix(".csv")
    return (session_dir / f"{session_dir.name}.csv").resolve()

def _import_attr_by_module_name(module_name: str, attr_name: str):
    mod = importlib.import_module(module_name)
    return getattr(mod, attr_name), mod

def _import_attr_by_file_path(file_path: Path, attr_name: str):
    spec = importlib.util.spec_from_file_location(file_path.stem, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[file_path.stem] = mod
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return getattr(mod, attr_name), mod

def _import_attr(module: str | None, path: str | None, func: str, kind: str):
    """
    kind is 'converter' or 'analysis' (nur für Fehlermeldungen).
    1) Wenn path gesetzt -> lade per Pfad
    2) sonst lade per Modulname
    """
    if path:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"{kind} file not found: {p}")
        try:
            return _import_attr_by_file_path(p, func)
        except Exception as e:
            raise ImportError(f"Failed to import {func} from file {p}: {e}") from e
    if not module:
        raise ValueError(f"No {kind} module provided")
    try:
        return _import_attr_by_module_name(module, func)
    except Exception as e:
        raise ImportError(f"Failed to import {func} from module '{module}': {e}") from e

def main():
    ap = argparse.ArgumentParser(description="Convert Neuralynx -> CSV, then run analysis main_safe.")
    ap.add_argument("session_dir", help="Path to Neuralynx session directory")
    ap.add_argument("--out-csv", default=None, help="Output CSV path (default: <session>/<folder>.csv)")
    ap.add_argument("--skip-convert-if-exists", action="store_true", help="Skip conversion if target CSV exists")

    # Converter
    ap.add_argument("--converter-module", default="neuralynx_rawio_to_csv_pulsed",
                    help="Module name that defines converter function (default: neuralynx_rawio_to_csv_pulsed)")
    ap.add_argument("--converter-path", default=None,
                    help="Alternative: path to converter .py file (overrides --converter-module)")
    ap.add_argument("--converter-func", default="main",
                    help="Converter function name (default: main(session_dir, out_csv=None))")

    # Analysis
    ap.add_argument("--analysis-module", default="main_safe",
                    help="Module name that defines analysis function (default: main_safe)")
    ap.add_argument("--analysis-path", default=None,
                    help="Alternative: path to analysis .py file (overrides --analysis-module)")
    ap.add_argument("--analysis-func", default="main_safe",
                    help="Analysis function name (default: main_safe(base_path, lfp_filename))")

    args = ap.parse_args()

    session_dir = Path(args.session_dir).expanduser().resolve()
    if not session_dir.is_dir():
        print(f"[ERROR] Session directory not found: {session_dir}", file=sys.stderr)
        sys.exit(1)

    out_csv_path = _guess_output_csv(session_dir, args.out_csv)

    # 1) Converter laden + ausführen
    if args.skip_convert_if_exists and out_csv_path.exists():
        print(f"[INFO] CSV already exists -> skipping conversion: {out_csv_path}")
    else:
        print("[STEP 1/2] Converting Neuralynx -> CSV …")
        try:
            converter_func, _ = _import_attr(
                module=args.converter_module,
                path=args.converter_path,
                func=args.converter_func,
                kind="converter"
            )
        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            sys.exit(2)

        try:
            converter_func(str(session_dir), str(out_csv_path) if args.out_csv else None)
        except SystemExit as e:
            if int(e.code) != 0:
                print(f"[ERROR] Converter exited with code {e.code}", file=sys.stderr)
                sys.exit(e.code)
        except Exception as e:
            print(f"[ERROR] Converter raised an exception: {e}", file=sys.stderr)
            sys.exit(1)

        if not out_csv_path.exists():
            fallback = (session_dir / f"{session_dir.name}.csv").resolve()
            if fallback.exists():
                out_csv_path = fallback
            else:
                print(f"[ERROR] Expected CSV not found. Looked for: {out_csv_path}", file=sys.stderr)
                sys.exit(1)
        print(f"[OK] Wrote CSV: {out_csv_path}")

    # 2) Analyse laden + ausführen
    print(f"[STEP 2/2] Running analysis …")
    try:
        analysis_func, _ = _import_attr(
            module=args.analysis_module,
            path=args.analysis_path,
            func=args.analysis_func,
            kind="analysis"
        )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)

    base_path = str(session_dir)
    lfp_filename = out_csv_path.name

    try:
        analysis_func(base_path, lfp_filename)
    except TypeError as e:
        print(f"[ERROR] Calling analysis function failed: {e}", file=sys.stderr)
        print(f"Hint: Ensure {args.analysis_func}(BASE_PATH: str, LFP_FILENAME: str) exists.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Analysis raised an exception: {e}", file=sys.stderr)
        sys.exit(1)

    print("[DONE] Conversion + analysis finished successfully.")

if __name__ == "__main__":
    main()
