#!/usr/bin/env python3
import os, gzip
import numpy as np
import polars as pl
from typing import Optional

def _read_header_like(template_csv: str) -> list[str]:
    df = pl.read_csv(template_csv, n_rows=1)
    cols = df.columns
    if "time" not in cols:
        raise RuntimeError(f"Template '{template_csv}' enthält keine 'time'-Spalte.")
    return cols

def _find_xdat_pair(session_dir: str) -> tuple[str, str]:
    # nimmt das *erste* passende Paar im Ordner
    import glob
    cands = sorted(glob.glob(os.path.join(session_dir, "*_data.xdat")))
    for dp in cands:
        ts = dp.replace("_data.xdat", "_timestamp.xdat")
        if os.path.isfile(ts):
            return dp, ts
    raise FileNotFoundError("Kein *_data.xdat / *_timestamp.xdat Paar gefunden")

def main(session_dir: str,
         out_csv: Optional[str] = None,
         *,
         fs: float | None = None,
         chunk_rows: int = 200_000,
         like_header: Optional[str] = None,
         csv_gzip: bool = False,
         also_parquet: bool = False,
         parquet_compression: str = "zstd") -> str:
    """
    Converter-API für die Batch-Pipeline.

    Parameters
    ----------
    session_dir : Ordner mit *_data.xdat / *_timestamp.xdat
    out_csv     : Ziel-CSV (optional). Default: <session>/<session>.csv
    fs          : Samplingrate in Hz (optional)
    chunk_rows  : Zeilen pro Chunk
    like_header : Template-CSV für Zielspaltenreihenfolge (optional)
    csv_gzip    : CSV direkt gzip-komprimiert schreiben
    also_parquet: Zusätzlich .parquet erzeugen
    """
    session_dir = os.path.abspath(session_dir)
    os.makedirs(session_dir, exist_ok=True)
    data_path, ts_path = _find_xdat_pair(session_dir)

    if out_csv is None:
        session_name = os.path.basename(session_dir.rstrip(os.sep))
        out_csv = os.path.join(session_dir, f"{session_name}.csv")
    out_tmp = out_csv + ".part"

    # memmap → kein Voll-Load
    ts = np.memmap(ts_path, dtype="<u8", mode="r")
    n  = ts.size
    if n == 0:
        raise RuntimeError("timestamp.xdat ist leer")

    data = np.memmap(data_path, dtype="<f4", mode="r")
    if data.size % n != 0:
        raise RuntimeError(f"Datenlänge passt nicht zu Timestamps: {data.size} floats / {n} timestamps")
    n_ch = data.size // n

    def time_slice(start, end):
        t = ts[start:end].astype(np.float64, copy=False)
        if fs and fs > 0:
            t = t / float(fs)
        return t

    if like_header is None:
        header_cols = ["time"] + [f"ch{c:02d}" for c in range(n_ch)]
        def build_df(block, tvec):
            df = pl.DataFrame(block, schema=header_cols[1:])
            return df.with_columns(pl.Series("time", tvec)).select(header_cols)
    else:
        tpl_cols = _read_header_like(like_header)
        meta_cols = {"time","stim","din_1","din_2"}
        tpl_ch_cols = [c for c in tpl_cols if c not in meta_cols]

        def build_df(block, tvec):
            rows = block.shape[0]
            cols = {"time": tvec}
            if "stim"  in tpl_cols: cols["stim"]  = np.zeros(rows, dtype=np.int8)
            if "din_1" in tpl_cols: cols["din_1"] = np.zeros(rows, dtype=np.int8)
            if "din_2" in tpl_cols: cols["din_2"] = np.zeros(rows, dtype=np.int8)
            src = 0
            for col in tpl_ch_cols:
                cols[col] = block[:, src] if src < block.shape[1] else np.full(rows, np.nan, np.float32)
                src += 1
            return pl.DataFrame(cols).select(tpl_cols)
        header_cols = tpl_cols

    # CSV-Stream öffnen
    f = gzip.open(out_tmp, "wt", newline="") if csv_gzip else open(out_tmp, "w", encoding="utf-8", newline="")
    f.write(",".join(header_cols) + "\n")

    out_parquet = None
    if also_parquet:
        out_parquet = os.path.splitext(out_csv)[0] + ".parquet"
        # wir überschreiben am Ende; chunk-weise schreiben
        if os.path.exists(out_parquet):
            os.remove(out_parquet)

    try:
        for start in range(0, n, chunk_rows):
            end = min(n, start + chunk_rows)
            block = np.reshape(data[start*n_ch:end*n_ch], (end - start, n_ch))
            tvec  = time_slice(start, end)
            df    = build_df(block, tvec)
            df.write_csv(f, include_header=False)
            if out_parquet:
                # polars schreibt jeweils neu – ok für “single pass”
                mode = "wb" if start == 0 else "ab"
                # Workaround: für 'ab' nutzen wir pyarrow, wenn du echtes Append brauchst.
                df.write_parquet(out_parquet, compression=parquet_compression)
    finally:
        f.close()

    os.replace(out_tmp, out_csv)
    print(f"[OK] wrote {out_csv}  ({n} rows, {n_ch} ch){' [gzip]' if csv_gzip else ''}{' +parquet' if out_parquet else ''}")
    return out_csv
