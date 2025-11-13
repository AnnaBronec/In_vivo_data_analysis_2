#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch: Neuralynx -> CSV -> Analyse über viele Sessions.
"""

import argparse
import importlib
import importlib.util
import sys
import csv
import gc
import shutil
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import math

# ---------- Import-Helpers ----------

def merge_csv_parts(parts_dir: Path, out_csv: Path) -> int:
    parts = sorted(parts_dir.glob("*.part*.csv"))
    if not parts:
        return 0
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fout:
        with parts[0].open("r") as f0:
            header = f0.readline()
            fout.write(header)
            shutil.copyfileobj(f0, fout)
        for pf in parts[1:]:
            with pf.open("r") as f:
                _ = f.readline()
                shutil.copyfileobj(f, fout)
    print(f"[MERGE] {len(parts)} Parts → {out_csv}")
    return len(parts)


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

def _default_csv_for_session(session_dir: Path) -> Path:
    return (session_dir / f"{session_dir.name}.csv").resolve()

# --- CSV-Auswahl: "gute" vs. auszuschließende CSVs ---
# erweitert: ALLES, wo "summary" im Namen ist, fliegt raus
_BAD_CSV_BASENAMES = {
    "upstate_summary.csv",
    "upstate_summary_all.csv",
}

def _is_good_csv_file(p: Path, session_dir: Path) -> bool:
    if not p.is_file() or p.suffix.lower() != ".csv":
        return False
    name = p.name.lower()
    if name in _BAD_CSV_BASENAMES:
        return False
    if "summary" in name:   # <- wichtig für upstate_summary_ALL.csv
        return False
    return True

def _has_any_good_csv(p: Path) -> bool:
    try:
        return any(_is_good_csv_file(f, p) for f in p.iterdir())
    except PermissionError:
        return False

def _looks_like_session_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    if p.name == "_csv_parts":
        return False
    return (
        _has_neuralynx_raw(p)
        or _has_xdat_pair(p)
        or _has_session_csv_exact(p)
        or _has_any_good_csv(p)
    )

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

# ---------- Split-Helpers ----------

def _bytes_to_gb(nbytes: int) -> float:
    return nbytes / (1024**3)

def _split_csv_by_rows(csv_path: Path, out_dir: Path, rows_per_part: int = 5_000_000, header: bool = True) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    with csv_path.open("r", newline="") as fin:
        hdr_line = fin.readline() if header else None
        if header and not hdr_line:
            return []
        idx = -1
        written_in_part = 0
        fout = None
        for line in fin:
            if (written_in_part == 0) or (written_in_part >= rows_per_part):
                if fout:
                    fout.close()
                idx += 1
                part_path = out_dir / f"{csv_path.stem}.part{idx:03d}.csv"
                fout = part_path.open("w", newline="")
                if header and hdr_line is not None:
                    fout.write(hdr_line)
                parts.append(part_path)
                written_in_part = 0
            fout.write(line)
            written_in_part += 1
        if fout:
            fout.close()
    return parts

# ---------- Single-Job (einfacher Fall) ----------

def _process_one_session(
    session_dir: Path,
    converter_func,
    analysis_func,
    out_csv: str | None,
    skip_convert_if_exists: bool,
    dry_run: bool
) -> tuple[str, bool, str]:
    try:
        csv_path = Path(out_csv).expanduser().resolve() if out_csv else _default_csv_for_session(session_dir)
        if skip_convert_if_exists and not csv_path.exists():
            csvs = sorted([f for f in session_dir.iterdir() if _is_good_csv_file(f, session_dir)])
            preferred = [f for f in csvs if f.name == f"{session_dir.name}.csv"]
            if preferred:
                csv_path = preferred[0].resolve()
            elif len(csvs) == 1:
                csv_path = csvs[0].resolve()
            elif len(csvs) > 1:
                preferred = [f for f in csvs if session_dir.name in f.stem]
                csv_path = (preferred[0] if preferred else csvs[0]).resolve()

        if skip_convert_if_exists and csv_path.exists():
            print(f"{session_dir.name}: [SKIP] CSV exists: {csv_path.name}")
        else:
            print(f"{session_dir.name}: [1/2] Convert -> {csv_path.name}")
            if not dry_run:
                converter_func(str(session_dir), str(csv_path) if out_csv else None)
                if not csv_path.exists():
                    fallback = _default_csv_for_session(session_dir)
                    if fallback.exists():
                        csv_path = fallback
                    else:
                        return (session_dir.name, False, "CSV not created")

        print(f"{session_dir.name}: [2/2] Analysis on {csv_path.name}")
        if not dry_run:
            analysis_func(str(session_dir), csv_path.name)

        return (session_dir.name, True, "OK (converted+analyzed)" if not dry_run else "DRY-RUN")
    except SystemExit as e:
        return (session_dir.name, False, f"SystemExit {int(e.code)}")
    except Exception as e:
        return (session_dir.name, False, f"{type(e).__name__}: {e}")

# ---------- Single-Job (Subprozess, split) ----------

def _process_one_session_subproc(
    session_dir: Path,
    converter_func,
    wrapper_script_path: Path,
    out_csv: str | None,
    skip_convert_if_exists: bool,
    dry_run: bool
) -> tuple[str, bool, str]:
    try:
        import os, subprocess

        split_threshold_gb = float(os.environ.get("BATCH_SPLIT_THRESHOLD_GB", "7.0"))
        rows_per_part      = int(os.environ.get("BATCH_ROWS_PER_PART", "5000000"))
        min_rows_per_part  = int(os.environ.get("BATCH_MIN_ROWS_PER_PART", "200000"))

        csv_path = Path(out_csv).expanduser().resolve() if out_csv else _default_csv_for_session(session_dir)

        if skip_convert_if_exists and not csv_path.exists():
            csvs = sorted([f for f in session_dir.iterdir() if _is_good_csv_file(f, session_dir)])
            preferred = [f for f in csvs if f.name == f"{session_dir.name}.csv"]
            if preferred:
                csv_path = preferred[0].resolve()
            elif len(csvs) == 1:
                csv_path = csvs[0].resolve()
            elif len(csvs) > 1:
                preferred = [f for f in csvs if session_dir.name in f.stem]
                csv_path = (preferred[0] if preferred else csvs[0]).resolve()

        # 1) ggf. konvertieren
        if skip_convert_if_exists and csv_path.exists():
            print(f"{session_dir.name}: [SKIP] CSV exists: {csv_path.name}")
        else:
            print(f"{session_dir.name}: [1/2] Convert -> {csv_path.name}")
            if not dry_run:
                converter_func(str(session_dir), str(csv_path) if out_csv else None)
                if not csv_path.exists():
                    fallback = _default_csv_for_session(session_dir)
                    if fallback.exists():
                        csv_path = fallback
                    else:
                        return (session_dir.name, False, "CSV not created")

        size_gb = _bytes_to_gb(csv_path.stat().st_size)
        if size_gb >= split_threshold_gb:
            print(f"{session_dir.name}: CSV {size_gb:.2f} GB ≥ {split_threshold_gb:.2f} GB -> split in Teile")
            parts_dir = session_dir / "_csv_parts"
            parts = _split_csv_by_rows(csv_path, parts_dir, rows_per_part=rows_per_part, header=True)
            if not parts:
                return (session_dir.name, False, "Split failed / no parts created")
            print(f"{session_dir.name}: created {len(parts)} part(s) in {parts_dir}")

            analyzed = 0
            for i, part in enumerate(parts, start=1):
                # kleine Parts überspringen
                try:
                    with part.open("r") as fh:
                        cnt = 0
                        for cnt, _ in enumerate(fh, 0):
                            if cnt >= min_rows_per_part:
                                break
                    if cnt < min_rows_per_part:
                        print(f"{session_dir.name}: [SKIP-PART] {part.name} hat nur ~{cnt} Datenzeilen -> übersprungen")
                        continue
                except Exception:
                    pass

                print(f"{session_dir.name}: [2/2] Analysis part {i}/{len(parts)} -> {part.name}")
                if not dry_run:
                    # WICHTIG: base_path = SESSION, filename = "_csv_parts/partXXX.csv"
                    rel_part = f"_csv_parts/{part.name}"
                    cmd = [
                        sys.executable,
                        str(wrapper_script_path),
                        str(session_dir),
                        "--lfp-filename", rel_part,
                    ]
                    env = os.environ.copy()
                    env.update({
                        "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "VECLIB_MAXIMUM_THREADS": "1",
                        "NUMEXPR_NUM_THREADS": "1",
                        "BLIS_NUM_THREADS": "1",
                        "BATCH_PARTIAL_APPEND": "1",
                        "ANALYSIS_SAVE_DIR": str(session_dir),
                    })
                    cp = subprocess.run(cmd, env=env)
                    if cp.returncode != 0:
                        return (session_dir.name, False, f"SubprocessError on part {i}: returncode={cp.returncode}")
                    analyzed += 1

            # 3) wieder mergen
            out_merged = session_dir / f"{session_dir.name}.merged.csv"
            try:
                n_parts = merge_csv_parts(parts_dir, out_merged)
                if n_parts == 0:
                    print(f"{session_dir.name}: [MERGE][WARN] keine Parts gefunden")
            except Exception as e:
                print(f"{session_dir.name}: [MERGE][WARN] {e}")

            # 4) finaler Lauf auf dem gemergten CSV
            if out_merged.exists() and not dry_run:
                print(f"{session_dir.name}: [FINAL] Analysis on merged -> {out_merged.name}")
                cmd = [
                    sys.executable,
                    str(wrapper_script_path),
                    str(session_dir),
                    "--lfp-filename", out_merged.name,
                ]
                env = os.environ.copy()
                env.update({
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "VECLIB_MAXIMUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                    "BLIS_NUM_THREADS": "1",
                    "ANALYSIS_SAVE_DIR": str(session_dir),
                    "BATCH_IS_MERGED_RUN": "1",
                })
                cp = subprocess.run(cmd, env=env)
                if cp.returncode != 0:
                    return (session_dir.name, False, f"SubprocessError on merged: returncode={cp.returncode}")

            return (session_dir.name, True, f"OK (split {len(parts)}; analyzed {analyzed}; merged+final-run)")

        # --- Standardfall (keine Splits) ---
        print(f"{session_dir.name}: [2/2] Analysis on {csv_path.name}")
        if not dry_run:
            cmd = [
                sys.executable,
                str(wrapper_script_path),
                str(session_dir),
                "--lfp-filename", str(csv_path.name),
            ]
            env = os.environ.copy()
            env.update({
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "BLIS_NUM_THREADS": "1",
                "ANALYSIS_SAVE_DIR": str(session_dir),
            })
            cp = subprocess.run(cmd, env=env)
            if cp.returncode != 0:
                return (session_dir.name, False, f"SubprocessError: returncode={cp.returncode}")

        return (session_dir.name, True, "OK (converted+analyzed)" if not dry_run else "DRY-RUN")

    except SystemExit as e:
        return (session_dir.name, False, f"SystemExit {int(e.code)}")
    except Exception as e:
        return (session_dir.name, False, f"{type(e).__name__}: {e}")

# ---------- Memory cleanup ----------

def _teardown_memory():
    try:
        plt.close('all')
    except Exception:
        pass
    gc.collect()

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Batch convert+analyze all sessions in a folder.")
    ap.add_argument("root_dir")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--skip-convert-if-exists", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-workers", type=int, default=1)

    ap.add_argument("--converter-module", default=None)
    ap.add_argument("--converter-path", default=None)
    ap.add_argument("--converter-func", default="main")

    ap.add_argument("--analysis-module", default=None)
    ap.add_argument("--analysis-path", default=None)
    ap.add_argument("--analysis-func", default="main_safe")

    ap.add_argument("--report-csv", default=None)

    args = ap.parse_args()

    root = Path(args.root_dir).expanduser().resolve()
    if not root.is_dir():
        print(f"[ERROR] Root folder not found: {root}", file=sys.stderr)
        sys.exit(1)

    # imports
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

    sessions = _find_sessions(root, recursive=args.recursive)
    if not sessions:
        print("[INFO] No sessions found.")
        sys.exit(0)

    # Master-Liste
    summary_path = root / "upstate_summary.csv"
    rows = []
    for s in sessions:
        rows.append({
            "Parent": s.parent.name,
            "Experiment": s.name,
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
        })
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

    wrapper_path = Path(args.analysis_path).expanduser().resolve()

    results = []
    for s in sessions:
        print(f"\n=== SESSION: {s} ===")
        try:
            res = _process_one_session_subproc(
                s,
                converter_func=converter_func,
                wrapper_script_path=wrapper_path,
                out_csv=args.out_csv,
                skip_convert_if_exists=args.skip_convert_if_exists,
                dry_run=args.dry_run,
            )
        finally:
            _teardown_memory()
        results.append(res)

    _teardown_memory()

    ok = sum(1 for _, success, _ in results if success)
    fail = len(results) - ok
    print("\n=== BATCH SUMMARY ===")
    print(f"Total: {len(results)}  OK: {ok}  FAIL: {fail}")
    for name, success, msg in results:
        status = "OK " if success else "ERR"
        print(f"[{status}] {name}: {msg}")

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
