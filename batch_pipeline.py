#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch: Neuralynx -> CSV -> Analyse über viele Sessions.

Für jede Session (Unterordner) wird:
  1) Converter main(session_dir, out_csv=None)
  2) Analyse  main_safe(base_path, lfp_filename)

Features:
  - erkennt Session-Ordner (mit .ncs oder .nse/.ntt/.nst) automatisch
  - --skip-convert-if-exists: CSV nicht neu erzeugen
  - --recursive: rekursiv unterhalb des Root-Ordners suchen
  - --max-workers N: mehrere Sessions parallel (vorsichtig mit I/O/CPU)
  - --dry-run: nur anzeigen, was passieren würde
  - Summary am Ende (Erfolge/Fehler); optional CSV-Report

Beispiel:
  python batch_pipeline.py "/path/to/DRD cross" \
    --converter-path "/…/Source/neuralynx_rawio_to_csv.py" \
    --analysis-path  "/…/Source/analysis_main.py" \
    --analysis-func main_safe \
    --skip-convert-if-exists \
    --recursive \
    --max-workers 1
"""

import argparse
import importlib
import importlib.util
import sys
import csv  
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ---------- Import-Helpers ----------

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

# ---------- Discovery ----------
_NEURALYNX_EXTS = {".ncs", ".nse", ".ntt", ".nst"}

def _has_neuralynx_raw(p: Path) -> bool:
    try:
        for f in p.iterdir():
            if f.is_file() and f.suffix.lower() in _NEURALYNX_EXTS:
                return True
    except PermissionError:
        pass
    return False

def _has_xdat_pair(p: Path) -> bool:
    try:
        for dp in p.glob("*_data.xdat"):
            ts = dp.with_name(dp.stem.replace("_data", "_timestamp") + ".xdat")
            if ts.is_file():
                return True
    except PermissionError:
        pass
    return False

def _has_session_csv_exact(p: Path) -> bool:
    return (p / f"{p.name}.csv").is_file()

def _has_any_csv(p: Path) -> bool:
    try:
        return any(f.is_file() and f.suffix.lower()==".csv" for f in p.iterdir())
    except PermissionError:
        return False

def _looks_like_session_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    # Neuralynx ODER XDAT-Paar ODER eine passende/fertige CSV im Ordner
    return _has_neuralynx_raw(p) or _has_xdat_pair(p) or _has_session_csv_exact(p) or _has_any_csv(p)

def _find_sessions(root: Path, recursive: bool) -> list[Path]:
    out: list[Path] = []
    if _looks_like_session_dir(root):
        out.append(root)
    if recursive:
        for sub in root.rglob("*"):
            if _looks_like_session_dir(sub):
                out.append(sub)
    else:
        for d in root.iterdir():
            if _looks_like_session_dir(d):
                out.append(d)
    return sorted(set(out))



   
# ---------- CSV-Pfad ----------

def _default_csv_for_session(session_dir: Path) -> Path:
    return (session_dir / f"{session_dir.name}.csv").resolve()

# ---------- Single-Job ----------

def _process_one_session(
    session_dir: Path,
    converter_func,
    analysis_func,
    out_csv: str | None,
    skip_convert_if_exists: bool,
    dry_run: bool
) -> tuple[str, bool, str]:
    """
    Return: (session_dir_name, success, message_or_error)
    """
    try:
        csv_path = Path(out_csv).expanduser().resolve() if out_csv else _default_csv_for_session(session_dir)
        # Wenn wir eine bereits vorhandene CSV akzeptieren sollen, aber der Defaultname nicht existiert:
        if skip_convert_if_exists and not csv_path.exists():
            # nimm eine vorhandene CSV im Session-Ordner (falls genau eine da ist)
            csvs = sorted([f for f in session_dir.iterdir() if f.is_file() and f.suffix.lower()==".csv"])
            if len(csvs) == 1:
                csv_path = csvs[0].resolve()
            # wenn mehrere CSVs da sind, präferiere die mit Ordnernamen drin
            elif len(csvs) > 1:
                preferred = [f for f in csvs if session_dir.name in f.name]
                if preferred:
                    csv_path = preferred[0].resolve()

        # Step 1: Convert
        if skip_convert_if_exists and csv_path.exists():
            msg = f"[SKIP] CSV exists: {csv_path.name}"
            print(f"{session_dir.name}: {msg}")
        else:
            print(f"{session_dir.name}: [1/2] Convert -> {csv_path.name}")
            if not dry_run:
                converter_func(str(session_dir), str(csv_path) if out_csv else None)
                # Fallback, falls Converter seinen Default-Namen nutzt
                if not csv_path.exists():
                    fallback = _default_csv_for_session(session_dir)
                    if fallback.exists():
                        csv_path = fallback
                    else:
                        return (session_dir.name, False, "CSV not created")

        # Step 2: Analyse
        print(f"{session_dir.name}: [2/2] Analysis on {csv_path.name}")
        if not dry_run:
            analysis_func(str(session_dir), csv_path.name)

        return (session_dir.name, True, "OK (converted+analyzed)" if not dry_run else "DRY-RUN")
    except SystemExit as e:
        return (session_dir.name, False, f"SystemExit {int(e.code)}")
    except Exception as e:
        return (session_dir.name, False, f"{type(e).__name__}: {e}")

# # ---------- Discovery ----------

# _NEURALYNX_EXTS = {".ncs", ".nse", ".ntt", ".nst"}  # Neuralynx-Dateiendungen

# def _looks_like_session_dir(p: Path) -> bool:
#     """Heuristik: Session-Ordner enthält mind. eine Neuralynx-Datei ODER schon eine CSV."""
#     if not p.is_dir():
#         return False
#     try:
#         has_neuralynx = any(f.is_file() and f.suffix.lower() in _NEURALYNX_EXTS for f in p.iterdir())
#         has_csv       = any(f.is_file() and f.suffix.lower() == ".csv" for f in p.iterdir())
#         return has_neuralynx or has_csv
#     except PermissionError:
#         return False

# def _find_sessions(root: Path, recursive: bool) -> list[Path]:
#     """Finde Session-Ordner unterhalb von root (rekursiv optional)."""
#     if not recursive:
#         return [d for d in root.iterdir() if _looks_like_session_dir(d)]
#     out: list[Path] = []
#     for sub in root.rglob("*"):
#         if _looks_like_session_dir(sub):
#             out.append(sub)
#     # Duplikate/verschachtelte Mehrtreffer vermeiden
#     return sorted(set(out))



# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Batch convert+analyze all sessions in a folder.")
    ap.add_argument("root_dir", help="Folder that contains many session subfolders (e.g., 'DRD cross')")
    ap.add_argument("--recursive", action="store_true", help="Search sessions recursively")
    ap.add_argument("--out-csv", default=None, help="Optional explicit CSV filename for ALL sessions (usually omit)")
    ap.add_argument("--skip-convert-if-exists", action="store_true", help="Do not reconvert if CSV already exists")
    ap.add_argument("--dry-run", action="store_true", help="Print actions, do not execute")
    ap.add_argument("--max-workers", type=int, default=1, help="Parallel sessions (default 1; increase with care)")

    # Converter
    ap.add_argument("--converter-module", default=None, help="Module that defines converter func (e.g., neuralynx_rawio_to_csv)")
    ap.add_argument("--converter-path", default=None, help="Path to converter .py file")
    ap.add_argument("--converter-func", default="main", help="Converter function name (default: main)")

    # Analysis
    ap.add_argument("--analysis-module", default=None, help="Module that defines analysis func (e.g., analysis_main)")
    ap.add_argument("--analysis-path", default=None, help="Path to analysis .py file")
    ap.add_argument("--analysis-func", default="main_safe", help="Analysis function name (default: main_safe)")

    # Reporting
    ap.add_argument("--report-csv", default=None, help="Optional path to write a batch summary CSV")

    args = ap.parse_args()

    root = Path(args.root_dir).expanduser().resolve()
    if not root.is_dir():
        print(f"[ERROR] Root folder not found: {root}", file=sys.stderr)
        sys.exit(1)

    # Import converter/analysis entrypoints
    try:
        converter_func, _ = _import_attr(
            module=args.converter_module, path=args.converter_path,
            func=args.converter_func, kind="converter"
        )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)

    try:
        analysis_func, _ = _import_attr(
            module=args.analysis_module, path=args.analysis_path,
            func=args.analysis_func, kind="analysis"
        )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)

    # Find sessions
    sessions = _find_sessions(root, recursive=args.recursive)
    if not sessions:
        print("[INFO] No sessions found.")
        sys.exit(0)

    summary_path = root / "upstate_summary.csv"
    rows = []
    for s in sessions:
        experiment_name = s.name
        parent_folder   = s.parent.name
        row = {
            "Parent": parent_folder,
            "Experiment": experiment_name,
            "Dauer [s]": None,
            "Samplingrate [Hz]": None,
            "Kanäle": None,
            "Pulse count 1": None,
            "Pulse count 2": None,
            "Upstates total": None,
            "triggered": None,
            "spon": None,
            "associated": None,
            "Downstates total": None,
            "UP/DOWN ratio": None,
            "Mean UP Dauer [s]": None,
            "Mean UP Dauer Triggered [s]": None,
            "Mean UP Dauer Spontaneous [s]": None,
            "Datum Analyse": None,
        }
        rows.append(row)

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Master-Liste geschrieben: {summary_path}")

    print(f"[INFO] Found {len(sessions)} session(s) under {root}")
    if args.dry_run:
        for s in sessions:
            target_csv = Path(args.out_csv).expanduser().resolve() if args.out_csv else _default_csv_for_session(s)
            print(f"  - {s} -> CSV: {target_csv.name}")
        print("[DRY-RUN] Nothing executed.")
        sys.exit(0)

    results = []
    if args.max_workers and args.max_workers > 1:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            fut2sess = {
                ex.submit(
                    _process_one_session, s, converter_func, analysis_func,
                    args.out_csv, args.skip_convert_if_exists, args.dry_run
                ): s for s in sessions
            }
            for fut in as_completed(fut2sess):
                results.append(fut.result())
    else:
        for s in sessions:
            results.append(
                _process_one_session(
                    s, converter_func, analysis_func,
                    args.out_csv, args.skip_convert_if_exists, args.dry_run
                )
            )

    # Summary
    ok = sum(1 for _, success, _ in results if success)
    fail = len(results) - ok
    print("\n=== BATCH SUMMARY ===")
    print(f"Total: {len(results)}  OK: {ok}  FAIL: {fail}")
    for name, success, msg in results:
        status = "OK " if success else "ERR"
        print(f"[{status}] {name}: {msg}")

    # Optional CSV report
    if args.report_csv:
        report_path = Path(args.report_csv).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "root", "session", "status", "message"])
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for name, success, msg in results:
                w.writerow([ts, str(root), name, "OK" if success else "ERR", msg])
        print(f"[INFO] Wrote report: {report_path}")

if __name__ == "__main__":
    main()
