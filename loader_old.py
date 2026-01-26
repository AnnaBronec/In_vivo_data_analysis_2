# loader_fixed.py
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List, Dict
# Source/nev_events.py
#from __future__ import annotations
import os
import struct
import numpy as np


# -------------------------
# Zeit-/Einheiten-Helfer
# -------------------------

def _infer_time_seconds(t: np.ndarray) -> Tuple[np.ndarray, str, float, float]:
    """Gibt (t_rel, unit, t0_raw, scale) zurück. unit ∈ {'s','us'}."""
    t = np.asarray(t)
    if t.size == 0:
        return t.astype(float), "s", 0.0, 1.0
    if t.size < 2:
        return (t - float(t[0])), "s", float(t[0]), 1.0
    dt = float(np.median(np.diff(t[: min(len(t), 1000)])))
    unit, scale = ("us", 1e6) if dt > 1e2 else ("s", 1.0)
    t0 = float(t[0])
    return (t - t0) / scale, unit, t0, scale

def _infer_unit_from_dt(t: np.ndarray) -> str:
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        return "s"
    dt = float(np.median(np.diff(t[: min(1000, t.size)])))
    return "us" if dt > 1e2 else "s"

def _to_seconds(x: np.ndarray, unit: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x / 1e6 if unit == "us" else x

# -------------------------
# LFP CSV Loader
# -------------------------

def load_LFP_new(BASE_PATH: str, LFP_FILENAME: str) -> Tuple[pd.DataFrame, List[str], Dict]:
    """
    Lädt LFP aus CSV und akzeptiert beide Layouts:
      1) ALT:   * _timestamps / * _values - Paare
      2) NEU:   time + eine Spalte pro Kanal (+ optional stim)
    Gibt zurück:
      - LFP_df: DataFrame mit Spalten ['time', <kanäle> (, 'stim' optional)]
      - ch_names: Liste der Kanalspalten
      - lfp_meta: {'fs_est': float | None}
    """
    csv_path = Path(BASE_PATH) / LFP_FILENAME
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV nicht gefunden: {csv_path}")

    df = pd.read_csv(csv_path)

        # --- DROP-IN: akzeptiere 'timesamples' oder 'timestamps' als Zeitspalte ---
    # Dein CSV hat 'timesamples' (wir nehmen das als Sekunden-Zeitachse)
    if "time" not in df.columns:
        if "timesamples" in df.columns:
            df = df.rename(columns={"timesamples": "time"})
        elif "timestamps" in df.columns:
            df = df.rename(columns={"timestamps": "time"})


    # --- Fall A: neues Layout (time + Kanäle) ---
    if "time" in df.columns:
        # Kanalspalten: alles außer 'time' und ggf. 'stim'
                # --- DROP-IN: Meta-Spalten NICHT als Kanäle behandeln ---
        NON_CH = {"time", "stim", "timestamps", "timesamples"}
        ch_names = [c for c in df.columns if c not in NON_CH]

        # Sicherstellen: numeric
        for c in ch_names:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # Optional: stim als int
        if "stim" in df.columns:
            df["stim"] = pd.to_numeric(df["stim"], errors="coerce").fillna(0).astype(np.int8)

        # Samplingrate grob schätzen
        fs_est = None
        if len(df) > 5:
            dt = np.median(np.diff(df["time"].to_numpy()))
            if np.isfinite(dt) and dt > 0:
                fs_est = 1.0 / dt

        return df, ch_names, {"fs_est": fs_est, "layout": "new"}

    # --- Fall B: altes Layout (*_timestamps / *_values) ---
    # Sammle alle Paare
    ts_cols = [c for c in df.columns if c.endswith("_timestamps")]
    val_cols = [c for c in df.columns if c.endswith("_values")]

    if not ts_cols or not val_cols:
        raise ValueError("Erwarte *_values und *_timestamps Spalten ODER 'time' + Kanäle.")

    # Kanalnamen aus Paaren ableiten (Schnittmenge)
    chans_ts = set([c[: -len("_timestamps")] for c in ts_cols])
    chans_v  = set([c[: -len("_values")] for c in val_cols])
    chans = sorted(list(chans_ts.intersection(chans_v)))
    if not chans:
        raise ValueError("Keine übereinstimmenden *_timestamps/_values Paare gefunden.")

    # Referenzzeit nehmen (erste Kanalzeitreihe) und alle Kanäle asof-mergen
    ref = chans[0]
    time = pd.to_numeric(df[f"{ref}_timestamps"], errors="coerce").to_numpy()
    out = pd.DataFrame({"time": time})

    for ch in chans:
        t = pd.to_numeric(df[f"{ch}_timestamps"], errors="coerce")
        v = pd.to_numeric(df[f"{ch}_values"], errors="coerce")
        tmp = pd.DataFrame({"time": t, ch: v}).dropna().sort_values("time")
        # asof-Join auf Referenzzeit
        out = pd.merge_asof(out.sort_values("time"), tmp, on="time", direction="nearest")

    # Optional: stim rekonstruieren, falls als zusätzl. Paar vorhanden
    if "stim_timestamps" in df.columns and "stim_values" in df.columns:
        st = pd.to_numeric(df["stim_timestamps"], errors="coerce")
        sv = pd.to_numeric(df["stim_values"], errors="coerce").fillna(0).astype(np.int8)
        stim_df = pd.DataFrame({"time": st, "stim": sv}).dropna().sort_values("time")
        out = pd.merge_asof(out.sort_values("time"), stim_df, on="time", direction="nearest")
        out["stim"] = out["stim"].fillna(0).astype(np.int8)

    ch_names = chans

    # fs schätzen
    fs_est = None
    if len(out) > 5:
        dt = np.median(np.diff(out["time"].to_numpy()))
        if np.isfinite(dt) and dt > 0:
            fs_est = 1.0 / dt

    return out, ch_names, {"fs_est": fs_est, "layout": "old"}

# -------------------------
# Stimuli-Finder & -Reader
# -------------------------

_CANDIDATE_COLS = ("lightpulses", "Light pulses", "Light_pulses", "LightPulses")

def _find_stimuli_csv(base_path: str, light_csv_rel: Optional[str] = None) -> str:
    """Sucht Stimuli-CSV:
      1) expliziter relativer Pfad
      2) *_Stimuli.csv (rekursiv)
      3) irgendeine CSV mit passender Spalte
    """
    if light_csv_rel:
        cand = os.path.join(base_path, light_csv_rel)
        if os.path.isfile(cand):
            return cand

    hits = glob.glob(os.path.join(base_path, "**", "*_Stimuli.csv"), recursive=True)
    if hits:
        return sorted(hits)[0]

    csvs = glob.glob(os.path.join(base_path, "**", "*.csv"), recursive=True)
    for path in sorted(csvs):
        try:
            head = pd.read_csv(path, nrows=3)
            if any(col in head.columns for col in _CANDIDATE_COLS):
                return path
        except Exception:
            pass

    raise FileNotFoundError("Keine Stimuli-CSV gefunden. Erwarte '*_Stimuli.csv' oder eine CSV mit Spalte 'lightpulses'.")

def _read_lightpulses_csv(csv_path: str) -> Tuple[np.ndarray, str]:
    df = pd.read_csv(csv_path, sep=None, engine="python", comment="#")
    col = next((c for c in _CANDIDATE_COLS if c in df.columns), None)
    series = df.iloc[:, 0] if col is None else df[col]
    raw = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    return raw, (col if col else getattr(series, "name", "column0"))

# -------------------------
# Light pulses Loader
# -------------------------

# --- in loader_old.py ersetzen ---
import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple

_CANDIDATE_COLS = ("lightpulses", "Light pulses", "Light_pulses", "LightPulses")

def _read_light_csv(csv_path: str) -> np.ndarray:
    # auto-sep
    try:
        df = pd.read_csv(csv_path, sep=None, engine="python", comment="#")
    except Exception:
        # deutsch: Semikolon + Dezimal-Komma als Fallback
        df = pd.read_csv(csv_path, sep=";", engine="python", decimal=",", comment="#")

    col = next((c for c in _CANDIDATE_COLS if c in df.columns), None)
    series = df.iloc[:, 0] if col is None else df[col]
    raw = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    return raw

def _guess_unit_from_dt_and_scale(x: np.ndarray) -> str:
    """
    Heuristik:
      - dt > 100  -> 'us'
      - oder Werte mit |x| > 1e5 -> sehr wahrscheinlich 'us' (µs-Zeitstempel)
      - sonst 's'
    """
    if x.size < 2:
        return "s"
    dt = float(np.median(np.diff(x[: min(1000, x.size)])))
    if dt > 100:
        return "us"
    if np.nanmax(np.abs(x)) > 1e5:
        return "us"
    return "s"

def _to_seconds(x: np.ndarray, unit: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x / 1e6 if unit == "us" else x

def load_lightpulses_simple(base_path: str,
                            filename: Optional[str] = None,
                            lfp_meta: Optional[dict] = None) -> np.ndarray:
    """
    Liest eine einspaltige Stimuli-CSV und liefert Pulse **in Sekunden relativ zum LFP-Start**,
    wenn lfp_meta übergeben wurde. Andernfalls relativ zum ersten Puls (Start=0).
    """
    if filename is None:
        raise ValueError("Bitte 'filename' für die Stimuli-CSV angeben.")
    csv_path = os.path.join(base_path, filename)
    raw = _read_light_csv(csv_path)

    if raw.size == 0:
        return np.array([], dtype=float)

    # 1) Einheit robust schätzen
    p_unit = _guess_unit_from_dt_and_scale(raw)
    pulses_abs_s = _to_seconds(raw, p_unit)

    # 2) Wenn lfp_meta fehlt: relativ zur eigenen Puls-Zeitbasis (Start=0)
    if lfp_meta is None:
        return (pulses_abs_s - pulses_abs_s[0]).astype(float)

    # 3) LFP-Start in absolute Sekunden
    lfp_unit  = lfp_meta.get("unit", "s")
    lfp_t0    = float(lfp_meta.get("t0_raw", 0.0))
    lfp_t0_s  = lfp_t0 / 1e6 if lfp_unit == "us" else lfp_t0

    # Manche Pulse sind nicht absolut, sondern bereits relativ (um 0 herum).
    # Wenn alle Pulse weit außerhalb des LFP-Fensters liegen, nimm relative Basis.
    # (Hier kein Zugriff auf LFP-Zeitfenster, daher einfache Heuristik:)
    # Wenn min(pulses_abs_s) < -10 oder max(pulses_abs_s) < 1e4 -> vermutlich relative Sekunden.
    if (np.nanmin(pulses_abs_s) < -10.0) or (np.nanmax(pulses_abs_s) < 1e4):
        pulses_rel = pulses_abs_s - pulses_abs_s[0]
    else:
        # Absolute Timestamps: gegen LFP-Start referenzieren
        pulses_rel = pulses_abs_s - lfp_t0_s

    return pulses_rel.astype(float)




HEADER_BYTES = 16 * 1024  # Neuralynx headers are typically 16KB

def _read_header(f):
    hdr = f.read(HEADER_BYTES)
    return hdr

def read_nev_timestamps_and_ttl(nev_path: str):
    """
    Minimal reader for Neuralynx Events.nev:
    returns (ts_us, ttl_values, event_strings)

    IMPORTANT:
    Neuralynx NEV record layouts can vary slightly by version.
    This parser uses a very common layout where each record is 184 bytes and contains:
      - uint64 timestamp (µs)
      - uint16/ int16 TTL
      - 128-byte event string
    If your layout differs, we'll detect nonsense and raise a clear error.
    """
    if not os.path.exists(nev_path):
        raise FileNotFoundError(f"NEV not found: {nev_path}")

    with open(nev_path, "rb") as f:
        _ = _read_header(f)
        data = f.read()

    # common record size
    rec_size = 184
    if len(data) % rec_size != 0:
        # still try, but warn by raising a helpful error
        raise ValueError(
            f"NEV payload size ({len(data)}) not divisible by {rec_size}. "
            "Your NEV record layout may differ; we need to adjust the parser."
        )

    nrec = len(data) // rec_size
    ts = np.empty(nrec, dtype=np.uint64)
    ttl = np.empty(nrec, dtype=np.int32)
    estr = []

    # Heuristic offsets that match a common NEV record:
    # [ ... 8-byte timestamp ... ][ ... 2-byte TTL ... ][ ... 128-byte string ... ]
    # We'll parse with conservative offsets and sanity-check afterwards.
    for i in range(nrec):
        rec = data[i*rec_size:(i+1)*rec_size]

        # Timestamp often starts after 6 bytes (2+2+2) -> offset 6
        ts_us = struct.unpack_from("<Q", rec, 6)[0]

        # TTL often appears shortly after timestamp; common is offset 6+8+2 (=16) or nearby.
        # We'll try a couple candidates and pick the one that yields plausible TTL values.
        ttl_candidates = [
            struct.unpack_from("<h", rec, 16)[0],
            struct.unpack_from("<h", rec, 18)[0],
            struct.unpack_from("<H", rec, 16)[0],
            struct.unpack_from("<H", rec, 18)[0],
        ]

        # Event string is often last 128 bytes
        s = rec[-128:].split(b"\x00", 1)[0].decode("latin-1", errors="ignore")

        ts[i] = ts_us
        estr.append(s)

        # store temporary (we'll fix TTL below)
        ttl[i] = ttl_candidates[0]

    # --- choose best TTL candidate by re-parsing once with a better offset ---
    # Re-evaluate TTL using offsets 16 vs 18: pick the one that gives many small integers
    def score_ttl(offset, signed=True):
        vals = []
        for i in range(nrec):
            rec = data[i*rec_size:(i+1)*rec_size]
            fmt = "<h" if signed else "<H"
            vals.append(struct.unpack_from(fmt, rec, offset)[0])
        v = np.asarray(vals, dtype=np.int64)
        # "plausible" TTL: many values between 0..65535, and not all crazy large magnitude
        return np.mean((v >= -32768) & (v <= 65535)) - 0.001*np.mean(np.abs(v) > 1_000_000)

    # Try four combinations
    options = [
        (16, True), (18, True), (16, False), (18, False)
    ]
    best = max(options, key=lambda opt: score_ttl(*opt))
    best_off, best_signed = best

    vals = []
    fmt = "<h" if best_signed else "<H"
    for i in range(nrec):
        rec = data[i*rec_size:(i+1)*rec_size]
        vals.append(struct.unpack_from(fmt, rec, best_off)[0])
    ttl = np.asarray(vals, dtype=np.int64)

    # sanity check timestamps monotonic-ish
    if np.nanmedian(np.diff(ts.astype(np.float64))) <= 0:
        raise ValueError("NEV timestamps do not look monotonic. Parser offsets likely wrong.")

    return ts, ttl, estr


def ttl_to_on_off(ts_us: np.ndarray, ttl: np.ndarray, bit: int = 0):
    """
    Convert TTL words to rising/falling edges for a given bit index.
    Returns onset_us, offset_us.
    """
    ts_us = np.asarray(ts_us, dtype=np.uint64)
    ttl = np.asarray(ttl, dtype=np.int64)

    # bit mask
    m = 1 << int(bit)
    state = (ttl & m) != 0
    d = np.diff(state.astype(np.int8))

    on_idx = np.where(d == 1)[0] + 1
    off_idx = np.where(d == -1)[0] + 1

    onset = ts_us[on_idx]
    offset = ts_us[off_idx]

    # pair them safely
    if onset.size and offset.size:
        if offset[0] < onset[0]:
            offset = offset[1:]
        n = min(onset.size, offset.size)
        onset = onset[:n]
        offset = offset[:n]

    return onset, offset
