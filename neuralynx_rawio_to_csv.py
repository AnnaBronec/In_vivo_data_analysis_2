#!/usr/bin/env python3
import os
import sys, re
from pathlib import Path
from functools import reduce
import numpy as np
import pandas as pd
from neo.rawio import NeuralynxRawIO
from loader_old import read_nev_timestamps_and_ttl, ttl_to_on_off


def _dec(x):
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="ignore")
    return x

def _streams_from_header(rr):
    """neo 0.14.x: header['signal_streams'] is a numpy structured array."""
    ss = rr.header.get("signal_streams", None)
    out = []
    if ss is None or len(ss) == 0:
        out.append({"id": "0", "name": "signals"})
        return out
    for s in ss:
        out.append({"id": _dec(s["id"]), "name": _dec(s["name"])})
    return out

def _channels_from_header(rr):
    sc = rr.header.get("signal_channels", [])
    chans = []
    for c in sc:
        chans.append({
            "id": _dec(c["id"]),
            "name": _dec(c["name"]),
            "stream_id": _dec(c["stream_id"]),
            "dtype": str(c["dtype"]),
            "units": _dec(c["units"]),
            "gain": float(c["gain"]),
            "offset": float(c["offset"]),
        })
    return chans

def _sig_size(rr, si):
    return rr.get_signal_size(block_index=0, seg_index=0, stream_index=si)

def _sig_sr(rr, si):
    return float(rr.get_signal_sampling_rate(stream_index=si))

def _sig_t0(rr, si):
    return float(rr.get_signal_t_start(block_index=0, seg_index=0, stream_index=si))

def _get_chunk(rr, si, start, stop, chan_idx):
  
    return rr.get_analogsignal_chunk(block_index=0, seg_index=0,
                                     stream_index=si, i_start=start, i_stop=stop,
                                     channel_indexes=chan_idx)


_hex_re = re.compile(r"ttl\s*value\s*:\s*0x([0-9a-fA-F]+)", flags=re.I)
_pulse_ms_re = re.compile(r"onePulse\s*([0-9]+(?:\.[0-9]+)?)\s*ms", flags=re.I)
_any_ms_re = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*ms", flags=re.I)

def _ttl_from_label(lbl: str):
    m = _hex_re.search(lbl)
    if not m:
        return None
    try:
        return int(m.group(1), 16)
    except Exception:
        return None

def _infer_mask_from_sequence(ttls):
    """OR of XORs across consecutive TTLs => bits that changed at least once."""
    if len(ttls) < 2:
        return 0
    x = 0
    prev = ttls[0]
    for v in ttls[1:]:
        x |= (prev ^ v)
        prev = v
    return x


def _infer_pulse_ms_from_session_name(session_name: str):
    """
    Tries to extract pulse width from names like:
      '...onePulse150ms...'
      '...onePulse 200 ms...'
    Returns float ms or None.
    """
    if not session_name:
        return None
    s = str(session_name)
    m = _pulse_ms_re.search(s)
    if m:
        try:
            val = float(m.group(1))
            if np.isfinite(val) and val > 0:
                return val
        except Exception:
            pass
    # fallback: any '<number>ms' token (e.g. '..._500ms_20%')
    cand = []
    for mm in _any_ms_re.finditer(s):
        try:
            v = float(mm.group(1))
        except Exception:
            continue
        if np.isfinite(v) and 0.5 <= v <= 5000:
            cand.append(v)
    if len(cand) == 1:
        return cand[0]
    if len(cand) > 1:
        # Prefer realistic stimulation widths 10..1000 ms if available.
        mid = [v for v in cand if 10 <= v <= 1000]
        if len(mid) == 1:
            return mid[0]
        if mid:
            return sorted(mid)[0]
        return sorted(cand)[0]
    return None

def _build_stim_from_ttls(event_times_s, event_ttls, time, mask=None, active_low=True):
    if len(event_times_s) == 0:
        return np.zeros_like(time, dtype=np.uint8), 0

    order = np.argsort(event_times_s)
    e_t = np.asarray(event_times_s, dtype=np.float64)[order]
    e_v = np.asarray(event_ttls, dtype=np.int64)[order]

    if mask is None or mask == 0:
        mask = _infer_mask_from_sequence(e_v)
        if mask == 0:
            mask = 0x0001

    idxs = np.searchsorted(e_t, time, side="right")
    stim = np.zeros(time.shape, dtype=np.uint8)
    for i in range(time.size):
        cur = e_v[idxs[i]-1] if idxs[i] > 0 else e_v[0]
        bit = 1 if (cur & mask) != 0 else 0
        stim[i] = (1 - bit) if active_low else bit
    return stim, int(mask)

def _build_stim_by_pulses(event_times_s, time, pulse_ms=5.0, durations_s=None):
    """
    Fallback when TTL values don't change:
    mark stim=1 for [t, t+dur] for each event, where dur = durations_s[i] if provided else pulse_ms.
    Efficient via prefix-sum trick.
    """
    stim = np.zeros(time.shape, dtype=np.uint8)
    if len(event_times_s) == 0:
        return stim

    starts = np.asarray(event_times_s, dtype=np.float64)
    if durations_s is not None:
        durs = np.asarray(durations_s, dtype=np.float64)
        durs[durs < 0] = 0.0
        ends = starts + durs
    else:
        width = float(pulse_ms) / 1000.0
        ends = starts + width

    # prefix sum method
    add = np.zeros(time.size + 1, dtype=np.int32)
    s_idx = np.searchsorted(time, starts, side="left")
    e_idx = np.searchsorted(time, ends,   side="right")
    s_idx = np.clip(s_idx, 0, time.size)
    e_idx = np.clip(e_idx, 0, time.size)
    for a, b in zip(s_idx, e_idx):
        if a < b:
            add[a] += 1
            add[b] -= 1
        else:
            # if event collapses onto one sample, at least mark that sample
            if a < time.size:
                add[a] += 1
                add[a+1 if a+1 <= time.size else time.size] -= 1
    stim = (np.cumsum(add[:-1]) > 0).astype(np.uint8)
    return stim


def _build_stim_onset_impulses(event_times_s, time):
    """
    Onset-only representation: mark only the sample at each event onset with 1.
    This avoids inventing pulse durations when OFF/duration is unavailable.
    """
    stim = np.zeros(time.shape, dtype=np.uint8)
    if len(event_times_s) == 0:
        return stim
    idx = np.searchsorted(time, np.asarray(event_times_s, dtype=np.float64), side="left")
    idx = np.clip(idx, 0, len(time) - 1)
    stim[np.unique(idx)] = 1
    return stim


def _build_stim_from_on_off(on_s, off_s, time):
    stim = np.zeros(time.shape, dtype=np.uint8)
    on_s = np.asarray(on_s, dtype=float)
    off_s = np.asarray(off_s, dtype=float)
    if on_s.size == 0 or off_s.size == 0:
        return stim
    m = min(on_s.size, off_s.size)
    on_s = on_s[:m]
    off_s = off_s[:m]
    add = np.zeros(time.size + 1, dtype=np.int32)
    s_idx = np.searchsorted(time, on_s, side="left")
    e_idx = np.searchsorted(time, off_s, side="right")
    s_idx = np.clip(s_idx, 0, time.size)
    e_idx = np.clip(e_idx, 0, time.size)
    for a, b in zip(s_idx, e_idx):
        if b > a:
            add[a] += 1
            add[b] -= 1
    stim = (np.cumsum(add[:-1]) > 0).astype(np.uint8)
    return stim

def build_stim_vector(rr, time, mask=None, active_low=True, pulse_ms=5.0, verbose=True):
    """
    1) Try TTL-bit method (if multiple distinct hex TTLs are present).
    2) If TTLs are constant, fall back to pulse windows (use durations if available else pulse_ms).
    """
    # events
    try:
        ts, dur, labels = rr.get_event_timestamps(block_index=0, seg_index=0)
    except TypeError:
        ts, dur, labels = rr.get_event_timestamps()
    if ts is None or labels is None or len(ts) == 0:
        if verbose:
            print("No events found; stim will be all zeros.")
        return np.zeros_like(time, dtype=np.uint8)

    e_times = rr.rescale_event_timestamp(ts, dtype="float64")  # neo 0.14.x: no 'unit'
    ttl_vals, kept_times = [], []
    for t, lab in zip(e_times, labels):
        v = _ttl_from_label(_dec(lab) or "")
        if v is not None:
            ttl_vals.append(v)
            kept_times.append(float(t))

    if len(kept_times) == 0:
        if verbose:
            print("Events present, but no 'TTL Value: 0x...' labels found; using pulse fallback.")
        return _build_stim_by_pulses(e_times, time, pulse_ms=pulse_ms, durations_s=None)

    uniq = sorted(set(ttl_vals))
    if verbose:
        print(f"TTL unique values (hex): {[hex(u) for u in uniq]}")

    if len(uniq) > 1:
        stim, used_mask = _build_stim_from_ttls(kept_times, ttl_vals, time, mask=mask, active_low=active_low)
        if verbose:
            print(f"Inferred/used mask: {hex(used_mask)}  (active_low={active_low})")
            print(f"Event time range: [{min(kept_times):.6f}, {max(kept_times):.6f}] s; "
                  f"Signal time range: [{float(time[0]):.6f}, {float(time[-1]):.6f}] s")
        return stim
    else:
        # TTLs constant -> use durations if present; otherwise onset-only impulses.
        durations_s = None
        if dur is not None and np.any(dur > 0):
            try:
                durations_s = rr.rescale_event_timestamp(dur, dtype="float64")
            except Exception:
                durations_s = np.asarray(dur, dtype=np.float64) / 1e6
        if durations_s is not None:
            stim = _build_stim_by_pulses(kept_times, time, pulse_ms=pulse_ms, durations_s=durations_s)
            if verbose:
                print("TTL values are constant -> using durations from file.")
        else:
            stim = _build_stim_onset_impulses(kept_times, time)
            if verbose:
                print("TTL values are constant and no durations available -> using onset-only impulses.")
        if verbose:
            print(f"Event time range: [{min(kept_times):.6f}, {max(kept_times):.6f}] s; "
                  f"Signal time range: [{float(time[0]):.6f}, {float(time[-1]):.6f}] s")
        return stim

# --------------- main ----------------

def main(session_dir, out_csv=None):
    p = Path(session_dir).expanduser().resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"Directory not found: {p}")

    skip_suffixes = {'.nse', '.ntt', '.nst'}
    exclude_list = [f.name for f in Path(p).iterdir()
                    if f.is_file() and f.suffix.lower() in skip_suffixes]

    print("Excluding files:", exclude_list)  # optional zum Debuggen

    rr = NeuralynxRawIO(
        dirname=str(p),
        exclude_filename=exclude_list,   # << genau hier!
        keep_original_times=False        # oder True, falls du absolute Zeiten willst
    )
    rr.parse_header()

    streams = _streams_from_header(rr)
    chans   = _channels_from_header(rr)

    # one DF per stream, then outer-merge
    dfs = []
    total_ch = 0

    for si, stream in enumerate(streams):
        sid = stream["id"]
        ch_idx = [i for i, ch in enumerate(chans) if ch["stream_id"] == sid]
        if not ch_idx and len(streams) == 1:
            ch_idx = list(range(len(chans)))
        if not ch_idx:
            continue

        n  = _sig_size(rr, si)
        sr = _sig_sr(rr, si)
        t0 = _sig_t0(rr, si)
        time = t0 + np.arange(n, dtype=np.float64) / sr

        # NEU: lokale Indizes für Neo (0..n_stream_ch-1)
        ch_idx_local = list(range(len(ch_idx)))

        chunk = _get_chunk(rr, si, 0, n, ch_idx_local)  # shape (n, n_ch)

        # Column names (robust gegen UNKNOWN: erst sinnvollen Namen versuchen, sonst CSC1..N)
        names = []
        unk_set = {"", "UNKNOWN", "NA", "UNTITLED", "NULL", None}
        

        def _clean(x):
            x = (x or "").strip()
            return x.upper()

        # Optional: versuche, eine CSC-Nummer aus file_origin oder id zu parsen
        import re
        def _extract_csc_num(j):
            # Versuch 1: aus file_origin (falls vorhanden, oft "CSC5.ncs")
            fo = rr.header.get("signal_channels", [])[j].get("file_origin", None) if isinstance(rr.header.get("signal_channels", []), list) else None
            if fo:
                m = re.search(r"CSC\s*([0-9]+)", str(fo), flags=re.I)
                if m:
                    return int(m.group(1))
            # Versuch 2: aus id/name selbst
            for key in ("id", "name"):
                val = chans[j].get(key, "")
                m = re.search(r"([0-9]+)", str(val))
                if m:
                    return int(m.group(1))
            return None

        for pos_in_stream, j in enumerate(ch_idx):
            raw_name = _clean(chans[j].get("name"))
            raw_id   = _clean(chans[j].get("id"))

            # gültigen Namen nehmen, wenn nicht UNKNOWN
            if raw_name not in unk_set:
                base = raw_name
            elif raw_id not in unk_set:
                base = raw_id
            else:
                # Versuche echte CSC-Nummer zu rekonstruieren, sonst Reihenfolge
                cnum = _extract_csc_num(j)
                if cnum is not None:
                    base = f"CSC{cnum}"
                else:
                    base = f"CSC{pos_in_stream+1}"

            # Einzigartigkeit sicherstellen
            col = base
            k = 2
            while col in names:
                col = f"{base}_{k}"
                k += 1
            names.append(col)


        df = pd.DataFrame({"time": time})
        for j, nm in enumerate(names):
            df[nm] = chunk[:, j]
        dfs.append(df)
        total_ch += len(names)

    if not dfs:
        raise RuntimeError("No analog signals (.Ncs) were read.")

    df_all = reduce(lambda a, b: pd.merge(a, b, on="time", how="outer"), dfs)
    df_all.sort_values("time", kind="mergesort", inplace=True)
    df_all.reset_index(drop=True, inplace=True)

    # Try precise pulse ON/OFF from Events.nev first (case-insensitive).
    event_nev = None
    for c in sorted(p.glob("*.nev")) + sorted(p.glob("*.Nev")) + sorted(p.glob("*.NEV")):
        if c.name.lower() == "events.nev":
            event_nev = c
            break
    if event_nev is None:
        evs = sorted(p.glob("*.nev")) + sorted(p.glob("*.Nev")) + sorted(p.glob("*.NEV"))
        if evs:
            event_nev = evs[0]

    used_nev = False
    if event_nev is not None and event_nev.exists():
        try:
            ts_us, ttl_words, _ = read_nev_timestamps_and_ttl(str(event_nev))
            best = None
            for bit in range(16):
                on_us, off_us = ttl_to_on_off(ts_us, ttl_words, bit=bit)
                n_pair = min(len(on_us), len(off_us))
                score = (n_pair, len(on_us), -bit)
                if best is None or score > best[0]:
                    best = (score, bit, on_us, off_us)
            if best is not None:
                _, best_bit, on_us, off_us = best
                t = df_all["time"].to_numpy(dtype=np.float64)
                t0 = float(t[0]) if t.size else 0.0
                on_s = np.asarray(on_us, dtype=float) / 1e6
                off_s = np.asarray(off_us, dtype=float) / 1e6
                # align NEV to current timeline (if absolute mismatch)
                if on_s.size and not ((np.nanmax(on_s) >= t0) and (np.nanmin(on_s) <= float(t[-1]))):
                    ref0 = float(on_s[0])
                    on_s = on_s - ref0 + t0
                    if off_s.size:
                        off_s = off_s - ref0 + t0
                df_all["stim"] = _build_stim_from_on_off(on_s, off_s, t)
                df_all["stim_on"] = _build_stim_onset_impulses(on_s, t)
                df_all["stim_off"] = _build_stim_onset_impulses(off_s, t) if off_s.size else np.zeros_like(df_all["stim"])
                used_nev = True
                print(f"[NEV->CSV] {event_nev.name}: selected bit={best_bit}, on={len(on_s)}, off={len(off_s)}")
        except Exception as e:
            print(f"[NEV->CSV][WARN] parse failed: {e}")

    # ----- build stim -----
    USER_MASK  = None   # e.g., 0x1000 if you know your TTL bit; otherwise leave None
    ACTIVE_LOW = True   # for the TTL-bit method (ignored in pulse fallback)
    # used only if TTL values are constant and durations are missing
    # Priority: ENV override > default
    pulse_ms_env = os.environ.get("NCS_PULSE_MS", "").strip()
    PULSE_MS = None
    if pulse_ms_env:
        try:
            v = float(pulse_ms_env)
            if np.isfinite(v) and v > 0:
                PULSE_MS = v
        except Exception:
            pass
    if PULSE_MS is None:
        PULSE_MS = 5.0
    print(f"[NCS] pulse width fallback = {PULSE_MS:.3f} ms")

    if not used_nev:
        df_all["stim"] = build_stim_vector(
            rr, df_all["time"].to_numpy(dtype=np.float64),
            mask=USER_MASK, active_low=ACTIVE_LOW, pulse_ms=PULSE_MS, verbose=True
        )

        # Session-Ordner-Name als Basis nehmen
    folder_name = p.name  # z.B. "2017-8-9_13-52-30onePulse200msX20per15s"
    out = Path(out_csv) if out_csv else (p / f"{folder_name}.csv")
    df_all.to_csv(out, index=False)


    print(f"\nWrote: {out}")
    print(f"Rows: {len(df_all)}")
    print(f"Analog channels written: {total_ch}")
    print("First columns:", ", ".join(df_all.columns[:10]), "..." if len(df_all.columns) > 10 else "")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python neuralynx_rawio_to_csv_pulsed.py /path/to/session_dir", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
