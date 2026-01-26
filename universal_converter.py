#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path
import numpy as np
import pandas as pd
from neuralynx_rawio_to_csv import main as nlx_convert_main

# Deine XDAT-Konvertierung als Funktion
def _xdat_convert_pair(prefix: Path, out_csv: Path, fs: float | None = None, chunk_rows: int = 200_000):
    data_path = prefix.with_name(prefix.name + "_data.xdat")
    ts_path   = prefix.with_name(prefix.name + "_timestamp.xdat")
    if not data_path.is_file() or not ts_path.is_file():
        raise FileNotFoundError(f"XDAT-Paar fehlt unter {prefix.parent}")

    # --- timestamps (uint64 LE) via memmap (kein Voll-Load!)
    ts = np.memmap(ts_path, dtype="<u8", mode="r")
    n_ts = ts.size
    if n_ts == 0:
        raise RuntimeError("timestamp.xdat ist leer")

    data = np.memmap(data_path, dtype="<f4", mode="r")
    n_vals = data.size

    # --- Sanity + Auto-Heal: Kanalzahl herleiten, ggf. Tail robust abschneiden ---
    force_ch = os.environ.get("BATCH_FORCE_CHANNELS")
    if force_ch:
        try:
            n_ch = int(force_ch)
            if n_ch <= 0:
                raise ValueError
        except Exception:
            raise RuntimeError(f"BATCH_FORCE_CHANNELS ungültig: {force_ch!r}")
    else:
        ratio = n_vals / max(1, n_ts)
        n_ch  = int(round(ratio))
        if n_ch <= 0:
            raise RuntimeError(f"Konnte Kanalzahl nicht schätzen (n_vals={n_vals}, n_ts={n_ts}, ratio={ratio:.6f})")

    needed = n_ts * n_ch
    if n_vals != needed:
        # zwei Heuristiken: (A) exakt auf n_ts Frames bringen, (B) auf ganze Frames runden
        tail_vals = n_vals - needed
        frac_tail = abs(tail_vals) / max(1, n_vals)

        # bis 2% Abweichung schneiden wir automatisch
        if frac_tail <= 0.02 and needed > 0:
            if tail_vals > 0:
                # Daten länger -> überschüssiges Tail am Ende abschneiden
                data = data[:needed]
            else:
                # Timestamps länger -> Timestamps auf vorhandene Datenlänge kürzen
                n_ts_new = n_vals // n_ch
                ts = ts[:n_ts_new]
                n_ts = ts.size
                needed = n_ts * n_ch
                data = data[:needed]
            print(f"[XDAT][WARN] Tail korrigiert: ratio={n_vals/max(1, n_ts):.6f} -> n_ch={n_ch}, tail={tail_vals} Werte (≤2%) abgeschnitten.")
        else:
            # alternative kleine Korrektur: auf ganze Frames runden, wenn <2%
            cut_vals = n_vals - (n_vals // n_ch) * n_ch
            if abs(cut_vals) / max(1, n_vals) <= 0.02:
                keep = (n_vals // n_ch) * n_ch
                data = data[:keep]
                n_ts = keep // n_ch
                ts = ts[:n_ts]
                needed = n_ts * n_ch
                print(f"[XDAT][WARN] Frames gerundet: n_ch={n_ch}, abgeschnitten={cut_vals} Werte (≤2%).")
            else:
                raise RuntimeError(
                    "XDAT inkonsistent: "
                    f"{n_vals} Werte für {n_ts} Timestamps -> ratio={n_vals/max(1, n_ts):.6f}, "
                    "kein ganzzahliges n_ch. Hinweise: Datei evtl. korrupt/abgebrochen; "
                    "oder falsche Kanalzahl. Workaround: ENV BATCH_FORCE_CHANNELS=<N> setzen."
                )

    # ab hier passen Längen garantiert:
    assert data.size == n_ts * n_ch, "interner Längenfehler nach Tail-Fix"

    header = ["time"] + [f"ch{c:02d}" for c in range(n_ch)]
    out_tmp = out_csv.with_suffix(out_csv.suffix + ".part")

    # Header schreiben
    out_tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tmp, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\n")

    # chunked write (FIX: n -> n_ts)
    for start in range(0, n_ts, chunk_rows):
        end = min(n_ts, start + chunk_rows)
        block = np.reshape(data[start * n_ch : end * n_ch], (end - start, n_ch))
        t = ts[start:end].astype(np.float64, copy=False)
        if fs and fs > 0:
            t = t / float(fs)
        df = pd.DataFrame(block, columns=header[1:])
        df.insert(0, "time", t)
        df.to_csv(out_tmp, mode="a", header=False, index=False)

    out_tmp.replace(out_csv)
    print(f"[OK] XDAT->CSV: {out_csv}  ({n_ts} rows, {n_ch} channels)")

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
