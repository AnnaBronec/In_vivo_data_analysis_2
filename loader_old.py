import os
import numpy as np
import pandas as pd

def _infer_time_seconds(t: np.ndarray):
    """Erkennt Sekunden vs. Mikrosekunden und normalisiert auf t0=0s."""
    t = np.asarray(t)
    if t.size < 2:
        return t * 0.0, "s", float(t[0]) if t.size else 0.0, 1.0
    dt = np.median(np.diff(t[: min(len(t), 1000)]))
    # sehr einfache, robuste Heuristik
    if dt > 1e2:     # große Schritte -> vermutlich µs
        unit, scale = "us", 1e6
    else:
        unit, scale = "s", 1.0
    t0 = float(t[0])
    return (t - t0) / scale, unit, t0, scale


def load_LFP_new(base_path: str, filename: str, max_channels: int | None = None):
    """
    Liest CSV mit Spalten ..._timestamps und ..._values.
    Gibt DataFrame im alten Format zurück:
      - 'timesamples' (Sekunden ab Aufnahmestart)
      - pri_0 .. pri_{N-1}
    Außerdem: channel_names (Originalnamen) und meta (t0/unit/scale).
    """
    filepath = os.path.join(base_path, filename)
    df = pd.read_csv(filepath)

    value_cols = [c for c in df.columns if c.endswith("_values")]
    ts_cols    = [c for c in df.columns if c.endswith("_timestamps")]
    if not value_cols or not ts_cols:
        raise ValueError("Erwarte *_values und *_timestamps Spalten.")

    if max_channels is not None:
        value_cols = value_cols[:max_channels]

    time_raw = df[ts_cols[0]].to_numpy()
    time_s, unit, t0_raw, scale = _infer_time_seconds(time_raw)

    out = pd.DataFrame({"timesamples": time_s})
    for i, col in enumerate(value_cols):
        out[f"pri_{i}"] = df[col].to_numpy()

    meta = {"t0_raw": t0_raw, "unit": unit, "scale": scale}
    return out, value_cols, meta


def load_lightpulses_simple(base_path: str, filename: str, lfp_meta: dict | None = None):
    """
    Liest LightPulses.csv mit EINER Spalte 'Light pulses'.
    Erkennt s vs. µs; gibt numpy-Array in Sekunden (relativ zum LFP-Start) zurück.
    """
    filepath = os.path.join(base_path, filename)
    pulses = pd.read_csv(filepath)

    if "Light pulses" not in pulses.columns:
        raise ValueError("In LightPulses.csv wird eine Spalte 'Light pulses' erwartet.")

    t = pulses["Light pulses"].to_numpy()

    # Einheit erkennen
    t_sec, unit, t0_raw, scale = _infer_time_seconds(t)

    # Falls beides absolute µs waren, auf LFP-Start referenzieren
    if lfp_meta is not None and unit == "us" and lfp_meta.get("unit") == "us":
        t_sec = (t - lfp_meta["t0_raw"]) / 1e6

    # (Bei Sekunden bleibt t_sec wie es ist.)
    return np.asarray(t_sec, dtype=float)

def read_lightpulses_csv(csv_path: str) -> np.ndarray:
    """Read a single-column light pulses CSV and return float seconds."""
    # Falls Trennzeichen unklar: sep=None + engine='python' erlaubt autodetect
    df = pd.read_csv(csv_path, sep=None, engine="python", comment="#")

    # Erlaubte Spaltennamen (wir unterstützen dein 'lightpulses' UND das alte 'Light pulses')
    candidates = ["lightpulses", "Light pulses", "Light_pulses", "LightPulses"]
    col = next((c for c in candidates if c in df.columns), None)

    # Falls kein Kandidat passt, nimm erste Spalte
    series = df.iloc[:, 0] if col is None else df[col]

    # In float konvertieren, unlesbare Werte droppen
    t = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    return t

def load_lightpulses_simple(base_path, light_csv_rel, lfp_meta=None):
    import os
    csv_path = os.path.join(base_path, light_csv_rel)
    t = read_lightpulses_csv(csv_path)
    return t

def _infer_time_seconds(t: np.ndarray):
    # t kommt jetzt als float-Array rein -> keine String-Diffs mehr
    dt = np.median(np.diff(t[: min(len(t), 1000)])) if len(t) > 1 else np.nan
    return t, "s", 0.0, 1.0