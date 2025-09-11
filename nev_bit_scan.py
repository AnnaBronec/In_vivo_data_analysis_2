#!/usr/bin/env python3
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd

HEADER_BYTES = 16 * 1024
_num_pat = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')

def _first_number(s):
    if s is None: return None
    m = _num_pat.search(str(s)); 
    return float(m.group(0)) if m else None

def parse_header(hdr: bytes):
    txt = hdr.decode("latin-1", errors="ignore")
    meta = {}
    for raw in txt.splitlines():
        line = raw.strip()
        if not line: continue
        while line and line[0] in "#-":
            line = line[1:].strip()
        if not line: continue
        for sep in (":","=","\t"):
            if sep in line:
                k,v = line.split(sep,1); meta[k.strip()] = v.strip(); break
        else:
            parts = line.split(None,1)
            meta[parts[0].strip()] = parts[1].strip() if len(parts)==2 else ""
    return meta

def ts_to_seconds_fn(meta):
    low = {k.lower(): v for k,v in meta.items()}
    if "timestampfrequency" in low:
        f = _first_number(low["timestampfrequency"])
        if f and f>0: return lambda x: x / f
    if "timestampresolution" in low:
        val = (low["timestampresolution"] or "").lower()
        x = _first_number(val)
        if x:
            if any(u in val for u in ("usec","µs","micro")): return lambda x: x * (x*0+1) * (x*0+1)  # dummy avoid lints
            # (wir überschreiben direkt unten)
    # Fallbacks:
    if "timestampresolution" in low:
        val = (low["timestampresolution"] or "").lower()
        x = _first_number(val)
        if x:
            if "usec" in val or "µs" in val or "micro" in val: return lambda ticks: ticks * (x * 1e-6)
            if "msec" in val or "ms" in val                : return lambda ticks: ticks * (x * 1e-3)
            return lambda ticks: ticks * x
    if "cheetahtimeunit" in low:
        u = low["cheetahtimeunit"].lower()
        if "usec" in u or "micro" in u: return lambda t: t/1_000_000.0
        if "sec"  in u                : return lambda t: float(t)
    return lambda t: t/1_000_000.0

def infer_rec_size(n):
    for r in (184,208,304):
        if n % r == 0: return r
    return 184

def read_nev(path: Path):
    with open(path,"rb") as f:
        header = f.read(HEADER_BYTES)
        meta = parse_header(header)
        conv = ts_to_seconds_fn(meta)
        data = f.read()
    if not data: return np.array([]), np.array([]), meta, -1, conv
    rec = infer_rec_size(len(data))
    n = len(data)//rec
    ts_ticks = np.frombuffer(data, dtype="<u8", count=n, offset=0)
    # probiere TTL-Offsets
    ttl_offsets = [10,12,16,20]
    best_off, best_score, best_ttl = -1, -1, None
    for off in ttl_offsets:
        ttl = np.frombuffer(data, dtype="<u2", count=n, offset=off)
        prev = np.concatenate(([0], ttl[:-1])) & 0xFFFF
        curr = ttl & 0xFFFF
        transitions = int((prev != curr).sum())
        score = transitions - (0.9 if (ttl==0).all() else 0)
        if score > best_score:
            best_score, best_off, best_ttl = score, off, ttl
    return conv(ts_ticks.astype(np.float64)), best_ttl.astype(np.int32), meta, best_off, conv

def rising_edges(ttl: np.ndarray, mask: int):
    prev = np.concatenate(([0], ttl[:-1])) & mask
    curr = ttl & mask
    return ((~prev) & curr) != 0

def main():
    ap = argparse.ArgumentParser(description="Scannt Events.Nev: Rising-Edges pro DIN-Bit & Stim-Tabelle.")
    ap.add_argument("nev", help="Pfad zu Events.Nev")
    ap.add_argument("--export", help="CSV für Stim-Tabelle (optional)")
    ap.add_argument("--bit", type=int, default=None, help="Nur dieses DIN-Bit tabellieren (0..15). Default: bestes Bit")
    args = ap.parse_args()

    nev = Path(args.nev).expanduser().resolve()
    ts_s, ttl, meta, off, conv = read_nev(nev)
    if ts_s.size == 0:
        print("Keine Daten."); return

    print(f"[INFO] rec_size auto, TTL-Offset={off}")
    # Counts pro Bit
    counts = {b: int(rising_edges(ttl, 1<<b).sum()) for b in range(16)}
    print("[COUNTS] Rising edges per DIN bit:")
    for b in range(16):
        print(f"  bit {b:2d}: {counts[b]}")

    # bestes Bit wählen (meiste Edges)
    active = [(b,c) for b,c in counts.items() if c>0]
    if not active:
        print("Keine Edges gefunden."); return
    best_bit = max(active, key=lambda x: x[1])[0]
    bit = args.bit if args.bit is not None else best_bit
    mask = 1<<bit
    edges = rising_edges(ttl, mask)
    t_edges = ts_s[edges]

    # Tabelle: Stim 1..N + Zeit (s, relativ zum 1. Stim & absolut)
    if t_edges.size:
        t_rel = t_edges - t_edges.min()
        df = pd.DataFrame({
            "Stimulus": np.arange(1, len(t_edges)+1),
            "Time_abs_s": t_edges,
            "Time_rel_s": t_rel,
            "DIN_bit": bit,
        })
        print(f"\n[STIMS] Bit {bit} → {len(df)} Rising-Edges")
        print(df.head(20).to_string(index=False))
        if args.export:
            out = Path(args.export).expanduser().resolve()
            df.to_csv(out, index=False)
            print(f"[WRITE] {out}")
    else:
        print(f"Keine Rising-Edges auf Bit {bit}.")

if __name__ == "__main__":
    main()
