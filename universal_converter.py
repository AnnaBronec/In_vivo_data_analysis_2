#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path
import numpy as np
import pandas as pd

# ---- Import der vorhandenen Converter ----
# Neuralynx-Converter: erwartet (session_dir: str, out_csv: Optional[str])
from neuralynx_rawio_to_csv import main as nlx_convert_main

# Deine XDAT-Konvertierung als Funktion
def _xdat_convert_pair(prefix: Path, out_csv: Path, fs: float | None = None, chunk_rows: int = 200_000):
    data_path = prefix.with_name(prefix.name + "_data.xdat")
    ts_path   = prefix.with_name(prefix.name + "_timestamp.xdat")
    if not data_path.is_file() or not ts_path.is_file():
        raise FileNotFoundError(f"XDAT-Paar fehlt unter {prefix.parent}")

    # --- timestamps (uint64 LE)
    ts = np.fromfile(ts_path, dtype="<u8")
    n = ts.size
    if n == 0:
        raise RuntimeError("timestamp.xdat ist leer")

    # --- data (float32 LE)
    data = np.fromfile(data_path, dtype="<f4")
    if data.size % n != 0:
        raise RuntimeError(f"Datenlänge passt nicht zu Timestamps: {data.size} floats / {n} timestamps")
    n_ch = data.size // n

    header = ["time"] + [f"ch{c:02d}" for c in range(n_ch)]
    out_tmp = out_csv.with_suffix(out_csv.suffix + ".part")

    # Header schreiben
    out_tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tmp, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\n")

    # chunked write
    for start in range(0, n, chunk_rows):
        end = min(n, start + chunk_rows)
        block = data[start*n_ch:end*n_ch].reshape(end - start, n_ch)
        t = ts[start:end].astype(np.float64)
        if fs and fs > 0:
            t = t / float(fs)
        df = pd.DataFrame(block, columns=header[1:])
        df.insert(0, "time", t)
        df.to_csv(out_tmp, mode="a", header=False, index=False)

    out_tmp.replace(out_csv)
    print(f"[OK] XDAT->CSV: {out_csv}  ({n} rows, {n_ch} channels)")

def _has_neuralynx_raw(p: Path) -> bool:
    exts = {".ncs", ".nse", ".ntt", ".nst"}
    try:
        for f in p.iterdir():
            if f.is_file() and f.suffix.lower() in exts:
                return True
    except PermissionError:
        pass
    return False

def _find_xdat_prefix(p: Path) -> Path | None:
    # nimmt das erste *_data.xdat und leitet das Prefix ab
    for dp in sorted(p.glob("*_data.xdat")):
        pref = dp.with_name(dp.name[:-len("_data.xdat")])
        ts = pref.with_name(pref.name + "_timestamp.xdat")
        if ts.is_file():
            return pref
    return None

def main(session_dir: str, out_csv: str | None = None, *, fs_xdat: float | None = None):
    """
    Universal-Entry:
      - wenn SESSION/SESSION.csv existiert -> nichts tun
      - wenn XDAT-Paar vorhanden -> XDAT->CSV (Name: SESSION/SESSION.csv)
      - sonst: Neuralynx->CSV (Name: SESSION/SESSION.csv)
    """
    sdir = Path(session_dir).expanduser().resolve()
    if not sdir.is_dir():
        raise FileNotFoundError(f"Session not found: {sdir}")

    target_csv = Path(out_csv).expanduser().resolve() if out_csv else (sdir / f"{sdir.name}.csv")

    # 1) Fertige CSV schon da?
    if target_csv.is_file():
        print(f"[SKIP] CSV existiert bereits: {target_csv.name}")
        return

    # 2) XDAT vorhanden?
    xpref = _find_xdat_prefix(sdir)
    if xpref is not None:
        print(f"[INFO] XDAT erkannt unter {sdir.name} -> konvertiere zu {target_csv.name}")
        _xdat_convert_pair(xpref, target_csv, fs=fs_xdat)
        return

    # 3) Neuralynx-Rohdaten?
    if _has_neuralynx_raw(sdir):
        print(f"[INFO] Neuralynx erkannt unter {sdir.name} -> konvertiere zu {target_csv.name}")
        # dein bestehender Converter kann den Zielnamen übernehmen:
        nlx_convert_main(str(sdir), str(target_csv))
        return

    # 4) Nichts gefunden
    raise RuntimeError(f"Keine unterstützten Quelldaten in {sdir} gefunden (weder CSV, XDAT noch Neuralynx).")

if __name__ == "__main__":
    # Direktaufruf (optional):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--fs-xdat", type=float, default=None, help="Samplingrate für XDAT (Hz)")
    args = ap.parse_args()
    main(args.session_dir, args.out_csv, fs_xdat=args.fs_xdat)
