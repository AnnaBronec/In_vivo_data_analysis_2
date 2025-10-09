#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drop-in wrapper for batch_pipeline.py

- Executes Main_safe.py with BASE_PATH / LFP_FILENAME set as globals
- Preserves all original behavior (plots, PDFs, CSVs)
- Aggressive teardown after each session:
    * closes all matplotlib figures
    * forces garbage collection
"""

# analysis_wrapper.py (ganz oben)
import os
# BLAS/MKL/OpenBLAS/Accelerate/NumExpr strikt auf 1 Thread
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")



from pathlib import Path
import runpy
import sys
import gc
import traceback
from contextlib import suppress

# Headless backend for batch usage
import matplotlib
matplotlib.use("Agg")          # must be set before importing pyplot
import matplotlib.pyplot as plt

# Path to your existing analysis script
MAIN_FILE = Path(__file__).parent / "Main_safe.py"


def _teardown_memory():
    """Close all Matplotlib figures and run GC."""
    with suppress(Exception):
        plt.close("all")
    gc.collect()


def main_safe(base_path: str, lfp_filename: str | None = None):
    """
    Entry point used by batch_pipeline.py
    Returns: (session_name, success_bool, message)
    """
    session_name = Path(base_path).name
    try:
        base_path = str(Path(base_path).expanduser().resolve())
        if lfp_filename is None:
            lfp_filename = f"{session_name}.csv"

        init_globals = {
            "BASE_PATH": base_path,
            "LFP_FILENAME": str(lfp_filename),
        }

        # Run your original analysis (produces all plots/CSVs/PDFs)
        runpy.run_path(str(MAIN_FILE), init_globals=init_globals)

        return session_name, True, "OK (converted+analyzed)"

    except Exception as e:
        # Print full traceback for logs; return concise message upward
        traceback.print_exc()
        return session_name, False, f"{type(e).__name__}: {e}"

    finally:
        _teardown_memory()


# Optional CLI: allows running one session in a fully isolated subprocess
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run ONE analysis session (isolated, sequential).")
    p.add_argument("base_path", help="Session folder")
    p.add_argument("--lfp-filename", default=None, help="CSV filename (defaults to <foldername>.csv)")
    args = p.parse_args()

    # <-- Debug-Ausgabe hier einfügen:
    print(f"[WRAPPER] start PID={os.getpid()} base_path={args.base_path}")

    name, ok, msg = main_safe(args.base_path, args.lfp_filename)
    print(f"[WRAPPER] {name}: {'OK' if ok else 'ERR'} - {msg}")
    sys.exit(0 if ok else 1)
