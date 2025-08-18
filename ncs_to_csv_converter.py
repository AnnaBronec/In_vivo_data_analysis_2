#!/usr/bin/env python3
import argparse
import struct
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

HEADER_BYTES = 16 * 1024
RECORD_SAMPLES = 512
RECORD_STRUCT = struct.Struct("<QIII" + "h"*RECORD_SAMPLES)

def parse_header(hdr_bytes: bytes) -> Dict[str, str]:
    txt = hdr_bytes.decode("latin-1", errors="ignore")
    meta: Dict[str, str] = {}
    for raw in txt.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("########") or line.startswith("##"):
            continue
        if line.startswith("-"):
            parts = line[1:].strip().split(None, 1)
            if len(parts) == 2:
                k, v = parts
                meta[k.strip()] = v.strip()
            elif len(parts) == 1:
                meta[parts[0].strip()] = ""
            continue
        if " " in line:
            k, v = line.split(None, 1)
            meta[k.strip()] = v.strip()
        else:
            meta[line] = ""
    return meta

def _infer_nev_record_size(data_len: int) -> int:
    for rec in (184, 208, 304):
        if data_len % rec == 0:
            return rec
    return 184

def read_nev(filepath: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, str]]:
    with open(filepath, "rb") as f:
        header = f.read(HEADER_BYTES)
        meta = parse_header(header)
        ts_res = None
        for key in ("TimeStampResolution", "TimeStampFrequency", "CheetahTimeUnit"):
            if key in meta:
                try:
                    ts_res = float(meta[key])
                    break
                except ValueError:
                    pass
        if ts_res is None:
            ts_res = 1_000_000.0
        data = f.read()
    if not data:
        return np.array([]), np.array([]), np.array([]), meta
    rec_size = _infer_nev_record_size(len(data))
    n_rec = len(data) // rec_size
    ts_list: List[float] = []
    ttl_list: List[float] = []
    idx_list: List[int] = []
    for i in range(n_rec):
        base = i * rec_size
        ts_ticks = int.from_bytes(data[base:base+8], byteorder='little', signed=False)
        ts_sec = ts_ticks / ts_res
        evt_str_bytes = data[base + rec_size - 128: base + rec_size]
        try:
            evt_str = evt_str_bytes.split(b'\x00', 1)[0].decode('latin-1', errors='ignore')
        except Exception:
            evt_str = ""
        ttl_val = np.nan
        lower = evt_str.lower()
        for key in ("value ", "ttl "):
            if key in lower:
                try:
                    ttl_val = float(''.join(ch for ch in lower.split(key)[-1] if ch.isdigit()))
                except Exception:
                    ttl_val = ttl_val
                break
        if np.isnan(ttl_val) and rec_size >= 12:
            ttl_field = int.from_bytes(data[base+10:base+12], byteorder='little', signed=False)
            if 0 <= ttl_field <= 65535:
                ttl_val = float(ttl_field)
        ts_list.append(ts_sec)
        ttl_list.append(ttl_val)
        idx_list.append(i)
    return np.asarray(ts_list), np.asarray(ttl_list), np.asarray(idx_list), meta

def build_folder_csv(folder: Path, out_dir: Path) -> Path:
    event_files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower()==".nev" and ("event" in p.name.lower())]
    if not event_files:
        return None
    nev_path = sorted(event_files)[0]
    ts, ttl_values, rec_idx, _ = read_nev(nev_path)
    if ts.size > 0:
        # Output ALL event timestamps as light pulses (no TTL filtering)
        ts_pulses = ts
        stim_df = pd.DataFrame({"lightpulses": ts_pulses})
        out_dir.mkdir(parents=True, exist_ok=True)
        stim_csv = out_dir / f"{folder.name}_Stimuli.csv"
        stim_df.to_csv(stim_csv, index=False)
        return stim_csv
    return None

def find_leaf_folders_with_nev(root: Path) -> List[Path]:
    folders = set()
    # Walk all files and pick those whose suffix (case-insensitive) is .nev
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".nev":
            folders.add(p.parent)
    return sorted(folders)

def main():
    ap = argparse.ArgumentParser(description="Extract light pulses from events.nev into CSV with single column 'lightpulses'.")
    ap.add_argument("root", type=str, help="Root directory to scan")
    ap.add_argument("--out-dir", type=str, default=None, help="Directory to write CSVs")
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root path not found: {root}")
    global_out = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    if global_out:
        global_out.mkdir(parents=True, exist_ok=True)
    folders = find_leaf_folders_with_nev(root)
    if not folders:
        raise SystemExit("No folders with .nev files found.")
    for folder in folders:
        out_dir = global_out if global_out else folder
        try:
            stim_csv = build_folder_csv(folder, out_dir)
            print(f"[OK] {folder} -> stim: {stim_csv}")
        except Exception as e:
            print(f"[FAIL] {folder}: {e}")
if __name__ == "__main__":
    main()
