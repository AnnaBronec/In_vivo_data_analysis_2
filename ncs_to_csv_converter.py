#!/usr/bin/env python3
import argparse
import struct
import re
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd

# ----------------- Konstanten -----------------
HEADER_BYTES = 16 * 1024
NCS_SAMPLES_PER_REC = 512
NCS_RECORD_STRUCT = struct.Struct("<QIII" + "h" * NCS_SAMPLES_PER_REC)  # ts, ch, fs, nvalid, samples
_num_pat = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')

# ----------------- Utils -----------------
def _first_number(s: str, default: Optional[float] = None) -> Optional[float]:
    if s is None:
        return default
    m = _num_pat.search(str(s))
    return float(m.group(0)) if m else default

def parse_header(hdr: bytes) -> Dict[str, str]:
    txt = hdr.decode("latin-1", errors="ignore")
    meta: Dict[str, str] = {}
    for raw in txt.splitlines():
        line = raw.strip()
        if not line:
            continue
        while line and line[0] in "#-":
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
            meta[parts[0].strip()] = parts[1].strip() if len(parts) == 2 else ""
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
            if any(u in val for u in ("usec", "µs", "micro")):
                return lambda t: t * (x * 1e-6)
            if any(u in val for u in ("msec", "ms")):
                return lambda t: t * (x * 1e-3)
            return lambda t: t * x
    if "cheetahtimeunit" in low:
        u = low["cheetahtimeunit"].lower()
        if "usec" in u or "micro" in u:
            return lambda t: t / 1_000_000.0
        if "sec" in u:
            return lambda t: float(t)
    return lambda t: t / 1_000_000.0  # Fallback: µs

# ----------------- NEV -----------------
def _infer_nev_record_size(n: int) -> int:
    for rec in (184, 208, 304):
        if n % rec == 0:
            return rec
    return 184

def read_nev(filepath: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, str], int]:
    with open(filepath, "rb") as f:
        header = f.read(HEADER_BYTES)
        meta = parse_header(header)
        ts_to_s = _ts_to_seconds_fn(meta)
        data = f.read()

    if not data:
        return np.array([]), np.array([]), meta, -1

    rec_size = _infer_nev_record_size(len(data))
    n_rec = len(data) // rec_size

    # Timestamps (immer am Anfang, 8 Byte LE)
    ts = np.empty(n_rec, dtype=np.float64)
    for i in range(n_rec):
        base = i * rec_size
        ts_ticks = int.from_bytes(data[base:base + 8], "little", signed=False)
        ts[i] = ts_to_s(ts_ticks)

    # Kandidaten-Offsets für TTL-Feld
    ttl_offsets = [10, 12, 16, 20]

    best_off, best_score, best_ttl = -1, -1.0, None
    for off in ttl_offsets:
        ttl = np.empty(n_rec, dtype=np.int32)
        for i in range(n_rec):
            base = i * rec_size
            ttl[i] = int.from_bytes(data[base + off: base + off + 2], "little", signed=False) if base + off + 2 <= len(data) else 0
        prev = np.concatenate(([0], ttl[:-1])) & 0xFFFF
        curr = ttl & 0xFFFF
        transitions = int((prev != curr).sum())
        score = transitions - (0.9 if (ttl <= 0).all() else 0.0)
        if score > best_score:
            best_score, best_off, best_ttl = score, off, ttl

    print(f"[DEBUG] NEV: rec={rec_size}B | TTL-Offset gewählt: {best_off} | transitions={best_score:.1f}")
    return ts, best_ttl, meta, best_off

def rising_edges_any_bit(ttl: np.ndarray, bitmask: int = 0xFFFF) -> np.ndarray:
    ttl = ttl.astype(np.int32)
    prev = np.concatenate(([0], ttl[:-1])) & bitmask
    curr = ttl & bitmask
    return ((~prev) & curr) != 0

# ----------------- NCS -----------------
def read_ncs(filepath: Path, decimate: int = 1) -> Tuple[np.ndarray, np.ndarray, Dict[str, str], float]:
    with open(filepath, "rb") as f:
        header = f.read(HEADER_BYTES)
        meta = parse_header(header)
        data = f.read()
    if not data:
        return np.array([]), np.array([]), meta, np.nan

    tconv = _ts_to_seconds_fn(meta)
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
    last_fs = float(fs_header) if fs_header else np.nan

    for i in range(n_rec):
        base = i * rec_size
        rec = NCS_RECORD_STRUCT.unpack_from(data, base)
        ts_ticks = rec[0]
        fs_rec = float(rec[2]) if rec[2] > 0 else (float(fs_header) if fs_header else np.nan)
        if np.isfinite(fs_rec) and fs_rec > 0:
            last_fs = fs_rec
        nvalid = int(rec[3])
        if nvalid <= 0 or not (np.isfinite(last_fs) and last_fs > 0):
            continue
        samples = np.frombuffer(data, dtype="<i2", offset=base + 20, count=NCS_SAMPLES_PER_REC)[:nvalid]
        t0 = tconv(ts_ticks)
        dt = 1.0 / last_fs
        t = t0 + np.arange(nvalid, dtype=np.float64) * dt
        if decimate > 1:
            t = t[::decimate]
            samples = samples[::decimate]
        times.append(t)
        vals.append(samples)

    if not times:
        return np.array([]), np.array([]), meta, float(fs_header) if fs_header else np.nan
    return np.concatenate(times), np.concatenate(vals), meta, float(last_fs)

# ----------------- Kombinieren -----------------
def build_combined(session_dir: Path, out_path: Optional[Path], bitmask: int, decimate: int) -> Optional[Path]:
    print(f"[SESSION] {session_dir}")
    ncs_files = sorted([p for p in session_dir.iterdir() if p.is_file() and p.suffix.lower() == ".ncs"])
    if not ncs_files:
        print("  ⛔ keine .ncs gefunden.")
        return None

    # 1) Kanäle laden & auf gemeinsame Zeit mergen
    # 2) Events → stim (auto DIN-Bit Wahl)
    ev = next((q for q in session_dir.iterdir() if q.is_file() and q.suffix.lower()==".nev" and q.stem.lower()=="events"), None)
    if ev is not None:
        ts, ttl, meta, ttl_off = read_nev(ev)
        if ts.size and ttl.size:
            ta = combined["time"].to_numpy()
            # Zeitachse sicherstellen (monoton & finite)
            if not (np.all(np.isfinite(ta)) and (np.diff(ta) > 0).all()):
                combined.dropna(subset=["time"], inplace=True)
                combined.sort_values("time", inplace=True, kind="mergesort", ignore_index=True)
                ta = combined["time"].to_numpy()

            tmin, tmax = float(ta.min()), float(ta.max())
            lfp_span = tmax - tmin
            dt_est = float(np.median(np.diff(ta)))

            def rising_edges(ttl_arr: np.ndarray, mask: int) -> np.ndarray:
                prev = np.concatenate(([0], ttl_arr[:-1])) & mask
                curr = ttl_arr & mask
                return ((~prev) & curr) != 0

            # Heuristik: bestes Bit wählen (Anzahl, IPI, Verteilung)
            candidates = []
            for b in range(16):
                edges = rising_edges(ttl, 1 << b)
                t_b = ts[edges]
                if t_b.size < 2:
                    continue
                duration = float(t_b.max() - t_b.min())
                ipi = np.diff(t_b)
                med_ipi = float(np.median(ipi)) if ipi.size else np.nan

                # Score: viele Pulse, über den LFP-Span verteilt, IPI ~ 5..60 s (dein Protokoll ~15 s)
                score = 0.0
                score += min(t_b.size, 40) * 2.0
                score += min(duration / max(lfp_span, 1e-9), 1.0) * 50.0
                if 5.0 <= med_ipi <= 60.0:
                    score += 50.0

                candidates.append((score, b, t_b, med_ipi, duration))

            if candidates:
                candidates.sort(reverse=True, key=lambda x: x[0])
                best_score, chosen_bit, tp, med_ipi, duration = candidates[0]
                print(f"  [EVENTS] TTL-Offset={ttl_off} | chosen DIN bit={chosen_bit} | pulses(raw)={tp.size} | med_IPI≈{med_ipi:.3f}s | spread≈{duration:.3f}s")
            else:
                # Fallback: any-bit
                mask_any = rising_edges_any_bit(ttl, 0xFFFF)
                tp = ts[mask_any]
                chosen_bit = -1
                print(f"  [EVENTS] TTL-Offset={ttl_off} | pulses(raw any-bit)={tp.size}")

            if tp.size:
                # Immer relativ zum ersten Puls; ggf. µs→s falls Range >> LFP-Span
                tp_rel = tp - tp.min()
                if tp_rel.max() > lfp_span * 10:
                    tp_rel = tp_rel / 1e6
                # Sicherheitscheck: wenn IPI << dt, ebenfalls µs→s
                if tp_rel.size > 1:
                    med_dtp = float(np.median(np.diff(tp_rel)))
                    if med_dtp < (dt_est / 10.0):
                        tp_rel = tp_rel / 1e6

                # An LFP-Start andocken und in Range schneiden
                tp2 = tmin + tp_rel
                tp2 = tp2[(tp2 >= tmin) & (tp2 <= tmax)]

                # Nearest-Sample markieren (ein Sample pro Puls)
                stim = np.zeros_like(ta, dtype=np.int8)
                if tp2.size:
                    idx = np.searchsorted(ta, tp2, side="left")
                    idx[idx == len(ta)] = len(ta) - 1
                    left = np.clip(idx - 1, 0, len(ta) - 1)
                    choose_left = (np.abs(ta[left] - tp2) < np.abs(ta[idx] - tp2))
                    idx[choose_left] = left[choose_left]
                    idx_unique = np.unique(idx)
                    stim[idx_unique] = 1

                combined["stim"] = stim
                print(f"  [MAP] LFP span(s)={lfp_span:.6f}, dt≈{dt_est:.6g}")
                print(f"  [CHECK] stim=1 samples: {int(stim.sum())}")


    # 3) Schreiben
    if out_path is None:
        out_path = session_dir / f"{session_dir.name}_combined.csv"
    combined.to_csv(out_path, index=False)
    print(f"[WRITE] {out_path}  (rows={len(combined)}, cols={combined.shape[1]})")
    return out_path

# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser(description="Mergt *.ncs zu einer CSV (time + Kanäle); optional stim aus Events.nev.")
    ap.add_argument("session", help="Pfad zum Session-Ordner (enthält *.ncs und optional Events.nev)")
    ap.add_argument("--out", help="Pfad zur Ausgabedatei (optional)")
    ap.add_argument("--bitmask", type=lambda s: int(s, 0), default=0b11, help="DIN-Bitmaske (nur für Debug; Auto-Choice aktiv).")
    ap.add_argument("--decimate", type=int, default=1, help="Jedes n-te Sample behalten (Tempo/Vorschau).")
    args = ap.parse_args()

    session_dir = Path(args.session).expanduser().resolve()
    if not session_dir.exists() or not session_dir.is_dir():
        raise SystemExit(f"Session-Ordner nicht gefunden: {session_dir}")

    out_path = Path(args.out).expanduser().resolve() if args.out else None
    build_combined(session_dir, out_path, bitmask=args.bitmask, decimate=args.decimate)

if __name__ == "__main__":
    main()
