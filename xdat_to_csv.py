#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse
import numpy as np
import pandas as pd

def read_header_like(template_csv: str):
    """Liest NUR die Kopfzeile des Templates ein und liefert die Spaltenliste."""
    df = pd.read_csv(template_csv, nrows=1)
    cols = list(df.columns)
    if "time" not in cols:
        raise RuntimeError(f"Template '{template_csv}' enthält keine 'time'-Spalte.")
    return cols

def convert_pair(data_path, ts_path, out_csv, fs=None, chunk_rows=200_000, like_header=None):
    # ---- timestamps (meist uint64 sample-index)
    ts = np.fromfile(ts_path, dtype="<u8")
    n = ts.size
    if n == 0:
        raise RuntimeError("timestamp.xdat ist leer")

    # ---- daten (float32, interleaved)
    data = np.fromfile(data_path, dtype="<f4")
    if data.size % n != 0:
        raise RuntimeError(f"Datenlänge passt nicht zu Timestamps: {data.size} floats / {n} timestamps")
    n_ch = data.size // n

    # ---- Zeitvektor (Sekunden, falls fs gegeben)
    t = ts.astype(np.float64)
    if fs is not None and fs > 0:
        t = t / float(fs)

    # ---- Header bauen
    if like_header is None:
        # dein bisheriges Format
        header = ["time"] + [f"ch{c:02d}" for c in range(n_ch)]
        # einfacher, direkter Dump
        out_tmp = out_csv + ".part"
        with open(out_tmp, "w", encoding="utf-8", newline="") as f:
            f.write(",".join(header) + "\n")
        for start in range(0, n, chunk_rows):
            end = min(n, start + chunk_rows)
            block = data[start*n_ch:end*n_ch].reshape(end - start, n_ch)
            df = pd.DataFrame(block, columns=header[1:])
            df.insert(0, "time", t[start:end])
            df.to_csv(out_tmp, mode="a", header=False, index=False)
        os.replace(out_tmp, out_csv)
        print(f"[OK] wrote {out_csv}  ({n} rows, {n_ch} channels)  [plain chXX-header]")
        return

    # ---- Template-Header übernehmen
    tpl_cols = read_header_like(like_header)
    # Hilfslisten
    meta_cols = {"time", "stim", "din_1", "din_2"}
    # alle Kanalsäulen im Template in der gewünschten Reihenfolge:
    tpl_ch_cols = [c for c in tpl_cols if c not in meta_cols]
    # unsere Standard-Kanalnamen (Quelle)
    src_ch_cols = [f"ch{c:02d}" for c in range(n_ch)]

    # Wir mappen der Reihe nach: jede „Template-Kanalspalte“ bekommt
    # die nächste Quelle (ch00, ch01, …). Wenn Template mehr Kanäle hat,
    # füllen wir mit NaN. Wenn Template weniger Kanäle hat, schneiden wir ab.
    out_tmp = out_csv + ".part"
    with open(out_tmp, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(tpl_cols) + "\n")

    for start in range(0, n, chunk_rows):
        end = min(n, start + chunk_rows)
        block = data[start*n_ch:end*n_ch].reshape(end - start, n_ch)
        df_src = pd.DataFrame(block, columns=src_ch_cols)

        # Ziel-DF in Template-Reihenfolge aufbauen
        df_out = pd.DataFrame(index=np.arange(end - start))

        # time
        df_out["time"] = t[start:end]

        # stim/din_* ggf. als Nullen
        if "stim" in tpl_cols:  df_out["stim"]  = 0
        if "din_1" in tpl_cols: df_out["din_1"] = 0
        if "din_2" in tpl_cols: df_out["din_2"] = 0

        # Kanäle mappen
        src_idx = 0
        for col in tpl_ch_cols:
            if src_idx < len(src_ch_cols):
                df_out[col] = df_src[src_ch_cols[src_idx]]
                src_idx += 1
            else:
                # Template verlangt mehr Kanäle als vorhanden → mit NaN füllen
                df_out[col] = np.nan

        # exakt in Template-Spaltenreihenfolge schreiben
        df_out = df_out[tpl_cols]
        df_out.to_csv(out_tmp, mode="a", header=False, index=False)

    os.replace(out_tmp, out_csv)
    print(f"[OK] wrote {out_csv}  ({n} rows, mapped to template header from '{like_header}')")

def main():
    ap = argparse.ArgumentParser(description="Convert *_data.xdat + *_timestamp.xdat to CSV")
    ap.add_argument("prefix", help="Pfadpräfix ohne Suffix, z.B. /path/sbpro_2__uid0118-17-50-11")
    ap.add_argument("--fs", type=float, default=None,
                    help="Samplingrate in Hz (optional). Wenn nicht gesetzt, bleibt 'time' in Samples.")
    ap.add_argument("-o", "--out", default=None, help="Ausgabe-CSV (optional)")
    args = ap.parse_args()

    data_path = args.prefix + "_data.xdat"
    ts_path   = args.prefix + "_timestamp.xdat"
    if not os.path.isfile(data_path):
         raise SystemExit(f"Nicht gefunden: {data_path}")
    if not os.path.isfile(ts_path):
         raise SystemExit(f"Nicht gefunden: {ts_path}")

    # ---> Hier: Parent-Ordnername als Basis nehmen
    # ---> RICHTIG: Session-Ordner (der Ordner, in dem die xdat-Paare liegen)
    session_dir  = os.path.dirname(data_path)         # .../Rasgrf2-ChR2inj-mouse2
    session_name = os.path.basename(session_dir)      # Rasgrf2-ChR2inj-mouse2
    out_csv = args.out or os.path.join(session_dir, session_name + ".csv")


    convert_pair(data_path, ts_path, out_csv, fs=args.fs)


if __name__ == "__main__":
    main()


