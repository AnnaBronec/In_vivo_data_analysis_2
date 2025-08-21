#!/usr/bin/env python3
import argparse
import struct
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import re

# --- Konstanten ---
HEADER_BYTES = 16 * 1024
NCS_SAMPLES_PER_REC = 512
NCS_RECORD_STRUCT = struct.Struct("<QIII" + "h"*NCS_SAMPLES_PER_REC)  # ts, ch, fs, nvalid, samples
_num_pat = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')

def _first_number(s: str, default: Optional[float] = None) -> Optional[float]:
    if s is None:
        return default
    m = _num_pat.search(str(s))
    return float(m.group(0)) if m else default

# -------- Header-Parsing & Zeitbasis --------
def parse_header(hdr_bytes: bytes) -> Dict[str, str]:
    txt = hdr_bytes.decode("latin-1", errors="ignore")
    meta: Dict[str, str] = {}
    for raw in txt.splitlines():
        line = raw.strip()
        if not line:
            continue
        while line and (line[0] in "#-"):
            line = line[1:].strip()
        if not line:
            continue
        for sep in (":", "=", "\t"):
            if sep in line:
                k, v = line.split(sep, 1)
                meta[k.strip()] = v.strip()
                break
        else:
            parts = line.split(None, 1)
            if len(parts) == 2:
                meta[parts[0].strip()] = parts[1].strip()
            else:
                meta[parts[0].strip()] = ""
    return meta

def _ts_to_seconds_fn(meta: Dict[str, str]):
    low = {k.lower(): v for k, v in meta.items()}
    if "timestampfrequency" in low:
        f = _first_number(low["timestampfrequency"])
        if f and f > 0:
            return lambda ticks: ticks / f
    if "timestampresolution" in low:
        val = low["timestampresolution"].lower()
        x = _first_number(val)
        if x:
            if "usec" in val or "µs" in val or "micro" in val:
                return lambda ticks: ticks * (x * 1e-6)
            if "msec" in val or "ms" in val:
                return lambda ticks: ticks * (x * 1e-3)
            return lambda ticks: ticks * x
    if "cheetahtimeunit" in low:
        unit = low["cheetahtimeunit"].lower()
        if "usec" in unit or "micro" in unit:
            return lambda ticks: ticks / 1_000_000.0
        if "sec" in unit:
            return lambda ticks: float(ticks)
    return lambda ticks: ticks / 1_000_000.0

# -------- Events (.nev) --------
def _infer_nev_record_size(data_len: int) -> int:
    for rec in (184, 208, 304):
        if data_len % rec == 0:
            return rec
    return 184

def read_nev(filepath: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, str]]:
    with open(filepath, "rb") as f:
        header = f.read(HEADER_BYTES)
        meta = parse_header(header)
        ts_to_s = _ts_to_seconds_fn(meta)
        data = f.read()

    if not data:
        return np.array([]), np.array([]), np.array([]), meta

    rec_size = _infer_nev_record_size(len(data))
    n_rec = len(data) // rec_size

    # Kandidaten-Offsets (variieren je nach NEV-Version)
    ttl_offsets = [10, 12, 16, 20]

    def extract_ttl_series(off: int) -> np.ndarray:
        ttl_list = np.empty(n_rec, dtype=np.int32)
        for i in range(n_rec):
            base = i * rec_size
            if base + off + 2 > len(data):
                ttl_list[i] = 0
                continue
            ttl_list[i] = int.from_bytes(data[base+off:base+off+2], byteorder="little", signed=False)
        return ttl_list

    ts_list = np.empty(n_rec, dtype=np.float64)
    for i in range(n_rec):
        base = i * rec_size
        ts_ticks = int.from_bytes(data[base:base+8], byteorder="little", signed=False)
        ts_list[i] = ts_to_s(ts_ticks)

    best_off, best_score, best_ttl = None, -1.0, None
    for off in ttl_offsets:
        ttl = extract_ttl_series(off)
        prev = np.concatenate(([0], ttl[:-1])) & 0xFFFF
        curr = ttl & 0xFFFF
        transitions = np.count_nonzero(prev != curr)
        uniq = np.unique(ttl)
        penalty = 0 if np.any(ttl > 0) else 0.9
        score = transitions - penalty
        if score > best_score:
            best_score, best_off, best_ttl = score, off, ttl

    # Debug
    print(f"[DEBUG] NEV record size={rec_size} | TTL-Offset={best_off} | nonzero={np.count_nonzero(best_ttl)}")

    idx = np.arange(n_rec, dtype=np.int32)
    return ts_list, best_ttl, idx, meta

def rising_edges_any_bit(ttl: np.ndarray, bitmask: int = 0xFFFF) -> np.ndarray:
    ttl = ttl.astype(np.int32)
    prev = np.concatenate(([0], ttl[:-1])) & bitmask
    curr = ttl & bitmask
    return ((~prev) & curr) != 0

# -------- LFP (.ncs) --------
def read_ncs(filepath: Path, decimate: int = 1) -> Tuple[np.ndarray, np.ndarray, Dict[str, str], float]:
    with open(filepath, "rb") as f:
        header = f.read(HEADER_BYTES)
        meta = parse_header(header)
        data = f.read()

    if not data:
        return np.array([]), np.array([]), meta, np.nan

    ts_to_s = _ts_to_seconds_fn(meta)
    rec_size = NCS_RECORD_STRUCT.size
    if len(data) % rec_size != 0:
        n_rec = len(data) // rec_size
        data = data[: n_rec * rec_size]
    n_rec = len(data) // rec_size

    fs_header = None
    for key in ("SamplingFrequency", "ADFrequency"):
        if key in meta:
            fs_header = _first_number(meta[key])
            break

    times, vals = [], []
    last_valid_fs = float(fs_header) if fs_header else np.nan

    for i in range(n_rec):
        base = i * rec_size
        rec = NCS_RECORD_STRUCT.unpack_from(data, base)
        ts_ticks = rec[0]
        fs_rec = float(rec[2]) if rec[2] > 0 else (float(fs_header) if fs_header else np.nan)
        if np.isfinite(fs_rec) and fs_rec > 0:
            last_valid_fs = fs_rec
        nvalid = int(rec[3])
        samples = np.frombuffer(data, dtype="<i2",
                                offset=base + 20, count=NCS_SAMPLES_PER_REC)
        if nvalid <= 0:
            continue
        samples = samples[:nvalid]
        if not np.isfinite(last_valid_fs) or last_valid_fs <= 0:
            continue
        t0 = ts_to_s(ts_ticks)
        dt = 1.0 / last_valid_fs
        t = t0 + np.arange(nvalid, dtype=np.float64) * dt
        if decimate > 1:
            t = t[::decimate]
            samples = samples[::decimate]
        times.append(t)
        vals.append(samples)

    if not times:
        return np.array([]), np.array([]), meta, float(fs_header) if fs_header else np.nan

    times_sec = np.concatenate(times)
    values = np.concatenate(vals)
    return times_sec, values, meta, float(last_valid_fs)

# -------- Datei-/Ordnerlogik --------
def find_session_folders(root: Path) -> List[Path]:
    folders = set()
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".ncs", ".nev"):
            folders.add(p.parent)
    return sorted(folders)

def get_events_nev_path(folder: Path) -> Optional[Path]:
    # Case-insensitive Suche nach Events.nev
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() == ".nev" and p.stem.lower() == "events":
            return p
    return None

# -------- Build-Funktionen --------
def build_session_combined(folder: Path, out_dir: Path, bitmask: int, decimate: int) -> Optional[Path]:
    """
    Liest alle .ncs in 'folder', merged sie zu einem breiten DataFrame (time + Kanäle),
    schreibt optional Stimulus-Spalte 'stim' basierend auf Events.nev.
    """
    print(f"\n[SESSION] {folder}")
    events_path = get_events_nev_path(folder)
    ncs_files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".ncs"])
    print(f"  Events.nev: {'gefunden' if events_path else '(nicht gefunden)'}")
    print(f"  NCS-Dateien: {len(ncs_files)}")

    if len(ncs_files) == 0:
        print("[SKIP] Keine NCS-Dateien.")
        return None

    # ---- LFP laden und als dict channel->DataFrame ablegen
    channel_dfs: Dict[str, pd.DataFrame] = {}
    base_time: Optional[np.ndarray] = None
    fs_ref: Optional[float] = None

    for ncs_path in ncs_files:
        ch_name = ncs_path.stem  # z.B. CSC1
        t, v, meta, fs = read_ncs(ncs_path, decimate=decimate)
        if t.size == 0:
            print(f"  [NO DATA] {ncs_path.name}")
            continue
        if base_time is None:
            base_time = t
            fs_ref = fs
            df = pd.DataFrame({"time": t, ch_name: v.astype(np.int16)})
        else:
            # Versuche Alignment: wenn gleiche Länge, prüfe Abweichung; sonst asof-merge
            if len(t) == len(base_time):
                diff = np.max(np.abs(t - base_time))
                if not np.isfinite(diff):
                    diff = np.inf
                if diff < (1.0 / (fs_ref or fs or 32000)) * 0.75:
                    df = pd.DataFrame({ch_name: v.astype(np.int16)})
                else:
                    # leichte Drift -> asof-merge
                    print(f"  [ALIGN] asof-merge für {ch_name} (Δmax={diff:.6f}s)")
                    tmp = pd.DataFrame({"time": t, ch_name: v.astype(np.int16)}).sort_values("time")
                    df = tmp
            else:
                print(f"  [ALIGN] unterschiedliche Länge für {ch_name} -> asof-merge")
                tmp = pd.DataFrame({"time": t, ch_name: v.astype(np.int16)}).sort_values("time")
                df = tmp

        # Mergen in Master
        if "time" in df.columns:
            if "time" not in channel_dfs:
                channel_dfs["time"] = df[["time"]].copy()
            master = channel_dfs["time"].merge(df, on="time", how="outer", sort=True)
            channel_dfs["time"] = master[["time"]]
            for c in master.columns:
                if c != "time":
                    channel_dfs[c] = master[[c]]
        else:
            # einfacher Spalten-append (gleiche Länge)
            channel_dfs[ch_name] = df[[ch_name]]

    if "time" not in channel_dfs:
        # Fall: alle hatten gleiche Länge → baue time aus base_time
        if base_time is None:
            print("[FAIL] Keine gültigen LFP-Samples.")
            return None
        channel_dfs["time"] = pd.DataFrame({"time": base_time})

    # kombiniere zu einem DataFrame
    cols = ["time"] + sorted([c for c in channel_dfs.keys() if c not in ("time")])
    combined = channel_dfs["time"].copy()
    for c in cols:
        if c == "time":
            continue
        combined[c] = channel_dfs[c].reindex(combined.index).values

    # ---- Events → stim-Spalte (0/1)
    if events_path:
        try:
            ts, ttl_values, rec_idx, meta = read_nev(events_path)
            if ts.size > 0:
                mask = rising_edges_any_bit(ttl_values, bitmask=bitmask)
                ts_pulses = ts[mask]
                print(f"  [EVENTS] pulses={ts_pulses.size} (bitmask={bin(bitmask)})")
                if ts_pulses.size > 0:
                    # weise Pulszeiten dem nächsten Sample zu (nearest)
                    # Annahme: time monotonic
                    time_arr = combined["time"].to_numpy()
                    stim = np.zeros_like(time_arr, dtype=np.int8)
                    idxs = np.searchsorted(time_arr, ts_pulses, side="left")
                    # clamp
                    idxs[idxs == len(time_arr)] = len(time_arr) - 1
                    # evtl. näherer Nachbar rechts/links wählen
                    left = np.clip(idxs - 1, 0, len(time_arr) - 1)
                    choose_left = (np.abs(time_arr[left] - ts_pulses) < np.abs(time_arr[idxs] - ts_pulses))
                    idxs[choose_left] = left[choose_left]
                    stim[idxs] = 1
                    combined["stim"] = stim
        except Exception as e:
            print(f"  [EVENTS FAIL] {e}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{folder.name}_combined.csv"
    combined.to_csv(out_path, index=False)
    print(f"[WRITE] {out_path}  (rows={len(combined)}, cols={combined.shape[1]})")
    return out_path

# -------- CLI --------
def main():
    ap = argparse.ArgumentParser(
        description="Konvertiert alle *.ncs einer Session zu EINER CSV (time + Kanäle) und schreibt optional Stimuli (stim=0/1) aus Events.nev in dieselbe Datei."
    )
    ap.add_argument("root", type=str, nargs="?", default=".", help="Wurzelverzeichnis (Standard: aktuelles Verzeichnis)")
    ap.add_argument("--out-dir", type=str, default=None, help="Globales Ausgabeverzeichnis (optional)")
    ap.add_argument("--bitmask", type=lambda s: int(s, 0), default=0b11,
                    help="Bitmaske für DIN-Bits (z.B. 0b01, 0b10, 0x3). Default 0b11.")
    ap.add_argument("--decimate", type=int, default=1,
                    help="LFP-Decimation-Faktor (z.B. 10 speichert jeden 10. Sample). Default 1 = keine Decimation.")
    ap.add_argument("--list-only", action="store_true",
                    help="Nur anzeigen, was gefunden würde; nichts schreiben.")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root path not found: {root}")

    global_out = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    if global_out:
        global_out.mkdir(parents=True, exist_ok=True)

    sessions = find_session_folders(root)
    if not sessions:
        raise SystemExit("Keine Ordner mit .nev/.ncs gefunden.")

    print(f"[INFO] Scanne {len(sessions)} Ordner unter {root}")

    for folder in sessions:
        out_dir = global_out if global_out else folder
        if args.list_only:
            events_path = get_events_nev_path(folder)
            ncs_files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".ncs"])
            print(f"\n[SESSION] {folder}")
            print(f"  Events.nev: {'gefunden' if events_path else '(nicht gefunden)'}")
            print(f"  NCS-Dateien: {len(ncs_files)}")
            for p in ncs_files:
                print(f"    - {p.name}")
            continue

        try:
            build_session_combined(folder, out_dir, bitmask=args.bitmask, decimate=args.decimate)
        except Exception as e:
            print(f"[FAIL] {folder}: {e}")

if __name__ == "__main__":
    main()
