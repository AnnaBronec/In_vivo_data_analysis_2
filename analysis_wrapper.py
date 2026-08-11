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


def _maybe_export_deep_probe_html(base_path: str, lfp_filename: str):
    """Optional extra export: deep Port-A channel vs. deep Port-C channel
    overlay with independently detected upstates (see
    deep_channel_dual_probe_html.py). Runs by default after every session;
    opt out with DEEP_PROBE_HTML=0. Channels auto-detect to the deepest
    connected electrode per port unless overridden with DEEP_PROBE_CH_A /
    DEEP_PROBE_CH_C (raw pri_/chNN index), same pattern as MAIN_UP_CH/SWR_CH.
    Sessions without both a Port A and Port C probe (or without an
    *.xdat.json) are skipped without failing the session.
    """
    if str(os.environ.get("DEEP_PROBE_HTML", "1")).strip().lower() in ("0", "false", "no", "off"):
        return
    try:
        import deep_channel_dual_probe_html as _deep_probe
        ch_a_env = os.environ.get("DEEP_PROBE_CH_A", "").strip()
        ch_c_env = os.environ.get("DEEP_PROBE_CH_C", "").strip()
        ch_a = int(ch_a_env) if ch_a_env else None
        ch_c = int(ch_c_env) if ch_c_env else None
        _deep_probe.main(base_path, lfp_filename, ch_a, ch_c)
    except SystemExit as e:
        print(f"[DEEP-PROBE-HTML][SKIP] {e}")
    except Exception as e:
        print(f"[DEEP-PROBE-HTML][WARN] skipped: {type(e).__name__}: {e}")
        traceback.print_exc()


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
        runpy.run_path(str(MAIN_FILE), init_globals=init_globals, run_name="__main__")

        _maybe_export_deep_probe_html(base_path, lfp_filename)

        return session_name, True, "OK (converted+analyzed)"

    except Exception as e:
        # Print full traceback for logs; return concise message upward
        traceback.print_exc()
        return session_name, False, f"{type(e).__name__}: {e}"

    finally:
        _teardown_memory()


# allows running one session in a fully isolated subprocess
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run ONE analysis session (isolated, sequential).")
    p.add_argument("base_path", help="Session folder")
    p.add_argument("--lfp-filename", default=None, help="CSV filename (defaults to <foldername>.csv)")
    p.add_argument(
        "--experiment", default=None,
        help="Named experiment profile from experiments.env (section name); "
             "sets ANALYSIS_EXPERIMENT for this run.",
    )
    args = p.parse_args()

    if args.experiment:
        os.environ["ANALYSIS_EXPERIMENT"] = args.experiment
        print(f"[WRAPPER] using experiment profile: {args.experiment}")

    print(f"[WRAPPER] start PID={os.getpid()} base_path={args.base_path}")

    name, ok, msg = main_safe(args.base_path, args.lfp_filename)
    print(f"[WRAPPER] {name}: {'OK' if ok else 'ERR'} - {msg}")
    sys.exit(0 if ok else 1)
