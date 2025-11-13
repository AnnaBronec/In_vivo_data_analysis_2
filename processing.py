
import os, math
import numpy as np
_np = np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

#Konstanten
DOWNSAMPLE_FACTOR = 50
HIGH_CUTOFF = 10
LOW_CUTOFF  = 2
ANALYSE_IN_AU = True
HTML_IN_uV    = True
DEFAULT_FS_XDAT = 32000.0   #ist das richtig?

_DEFAULT_SESSION = "/home/ananym/Code/In_vivo_data_analysis/Data/FOR ANNA IN VIVO/"
BASE_PATH   = globals().get("BASE_PATH", _DEFAULT_SESSION)


if "LFP_FILENAME" in globals():
    LFP_FILENAME = globals()["LFP_FILENAME"]
else:
    _base_tag = os.path.basename(os.path.normpath(BASE_PATH))
    LFP_FILENAME = f"{_base_tag}.csv"

SAVE_DIR = BASE_PATH
BASE_TAG = os.path.splitext(os.path.basename(LFP_FILENAME))[0]
os.makedirs(SAVE_DIR, exist_ok=True)

LOGFILE = os.path.join(SAVE_DIR, "runlog.txt")


def load_parts_to_array_streaming(
    base_path: str,
    ds_factor: int = 50,
    stim_cols = ("stim", "din_1", "din_2", "StartStop", "TTL", "DI0", "DI1"),
    dtype = np.float32,
):
    parts_dir = Path(base_path) / "_csv_parts"
    part_files = sorted(parts_dir.glob("*.part*.csv"))
    if not part_files:
        raise FileNotFoundError(f"Keine Parts unter {parts_dir} gefunden.")

    time_chunks = []
    data_chunks = []
    stim_cols_in_file = None
    chan_cols = None

    # Puls-Sammler (Listen → am Ende concat)
    p1_list, p2_list = [], []

    def _edges_from_series(t_vec, x_vec, rising_only=True, thr=None):
        x = pd.to_numeric(x_vec, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(x).any():
            return np.array([], dtype=float)
        if thr is None:
            lo, hi = np.nanpercentile(x, [10, 90])
            thr = (lo + hi) * 0.5
        b = (x > thr).astype(np.int8)
        if rising_only:
            idx = np.flatnonzero((b[1:] == 1) & (b[:-1] == 0)) + 1
        else:
            idx = np.flatnonzero(b[1:] != b[:-1]) + 1
        idx = idx[(idx >= 0) & (idx < t_vec.size)]
        return t_vec[idx].astype(float)

    for pf in part_files:
        df = pd.read_csv(pf, low_memory=False)

        # einmalig: stim-Spalten & Kanalspalten bestimmen
        if stim_cols_in_file is None:
            stim_cols_in_file = [c for c in stim_cols if c in df.columns]
            raw_chan_cols = [c for c in df.columns if c not in ("time", *stim_cols_in_file)]
            import re
            def _key_num(s):
                m = re.findall(r"\d+", s)
                return int(m[-1]) if m else 0
            chan_cols = sorted(raw_chan_cols, key=_key_num)

        keep_cols = ["time", *stim_cols_in_file, *chan_cols]
        df = df[keep_cols]

        if ds_factor > 1:
            df = df.iloc[::ds_factor, :].reset_index(drop=True)

        t_local = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)

        # Pulse sammeln
        if "din_1" in stim_cols_in_file:
            p1_list.append(_edges_from_series(t_local, df["din_1"]))
        if "din_2" in stim_cols_in_file:
            p2_list.append(_edges_from_series(t_local, df["din_2"]))
        if ("stim" in stim_cols_in_file) and ("din_1" not in stim_cols_in_file) and ("din_2" not in stim_cols_in_file):
            p1_list.append(_edges_from_series(t_local, df["stim"]))

        # Daten sammeln
        time_chunks.append(t_local)
        data_chunks.append(df[chan_cols].to_numpy(dtype=dtype))

        del df  # RAM frei

    time_s = np.concatenate(time_chunks, axis=0)
    data_all = np.concatenate(data_chunks, axis=0)     # (N, n_ch)
    LFP_array = data_all.T                              # (n_ch, N)

    # Pulse zusammenführen + entdoppeln
    pulse_times_1 = np.concatenate(p1_list, axis=0) if p1_list else np.array([], float)
    pulse_times_2 = np.concatenate(p2_list, axis=0) if p2_list else np.array([], float)
    if pulse_times_1.size:
        pulse_times_1 = np.unique(np.round(pulse_times_1, 9))
    if pulse_times_2.size:
        pulse_times_2 = np.unique(np.round(pulse_times_2, 9))

    return time_s, LFP_array, chan_cols, pulse_times_1, pulse_times_2

try:
    print(f"[INFO] pulses(from streaming): p1={len(pulse_times_1_full)}, "
          f"p2={len(pulse_times_2_full)}, "
          f"first/last p1: "
          f"{pulse_times_1_full[0] if len(pulse_times_1_full) else '—'} / "
          f"{pulse_times_1_full[-1] if len(pulse_times_1_full) else '—'}")
except NameError:
    print("[INFO] pulses(from streaming): (Variablen noch nicht definiert)")




def downsample_array_simple(
    ds_factor: int,
    time_s: np.ndarray,
    LFP_array: np.ndarray,
    pulse_times_1=None,
    pulse_times_2=None,
    snap_pulses=True,
):
    """
    Sehr einfache Array-Version vom Downsampling:
    - nimmt jeden ds_factor-ten Sample
    - wendet das auf time_s und alle Kanäle an
    - optional Pulsezeiten auf nächstgelegenen Sample einrasten
    """
    time_s = np.asarray(time_s, float)
    X = np.asarray(LFP_array)
    step = int(ds_factor) if ds_factor and ds_factor > 1 else 1

    time_ds = time_s[::step]
    X_ds = X[:, ::step]

    dt = float(time_ds[1] - time_ds[0]) if len(time_ds) > 1 else 1.0

    def _snap(pulses):
        if pulses is None or len(pulses) == 0:
            return pulses
        pulses = np.asarray(pulses, float)
        if not snap_pulses:
            return pulses
        idx = np.searchsorted(time_ds, pulses)
        idx = np.clip(idx, 0, len(time_ds)-1)
        return time_ds[idx]

    p1_ds = _snap(pulse_times_1)
    p2_ds = _snap(pulse_times_2)

    return time_ds, dt, X_ds, p1_ds, p2_ds




def _counts_to_uV(x, bits, vpp, gain):
    # x = integer/float "counts" (LSB), vpp = Volt p-p
    lsb_volt = float(vpp) / (2**bits)          # Volt pro LSB
    return (np.asarray(x, float) * lsb_volt / float(gain)) * 1e6  # -> µV

def _volts_to_uV(x):
    return np.asarray(x, float) * 1e6

def convert_df_to_uV(df, mode):
    df = df.copy()
    chan_cols_orig = [c for c in df.columns if c not in ("time","stim","din_1","din_2")]
    if mode == "uV":
        # bereits µV – nichts tun
        return df
    elif mode == "volts":
        for c in chan_cols_orig:
            df[c] = _volts_to_uV(pd.to_numeric(df[c], errors="coerce"))
        return df
    elif mode == "counts":
        for c in chan_cols_orig:
            g = PER_CH_GAIN.get(c, PREAMP_GAIN)
            df[c] = _counts_to_uV(pd.to_numeric(df[c], errors="coerce"), ADC_BITS, ADC_VPP, g)
        return df
    else:
        raise ValueError(f"Unbekannter CALIB_MODE: {mode}")

def _decimate_xy(x, Y, max_points=40000):
    """Reduziert Punktezahl, damit SVGs klein bleiben."""
    import numpy as np
    if max_points is None or len(x) <= max_points:
        return x, Y
    step = int(np.ceil(len(x) / max_points))
    return x[::step], Y[:, ::step]



def _ensure_main_channel(LFP_array, preferred_idx=10):
    """
    Liefert (main_channel, used_idx).
    Bevorzugt preferred_idx, sonst Kanal 0.
    Unabhängig von good_idx, damit früh nutzbar (z.B. fürs Spektrogramm).
    """
    num_ch = int(LFP_array.shape[0])
    if isinstance(preferred_idx, int) and 0 <= preferred_idx < num_ch:
        return LFP_array[preferred_idx, :], preferred_idx
    return LFP_array[0, :], 0


def _ensure_seconds(ts, time_ref, fs_xdat=DEFAULT_FS_XDAT):
    """
    Bringt ts (Pulszeiten) in die gleiche Einheit wie time_ref (Sekunden).
    Erkennt 'zu große' Werte heuristisch und teilt dann durch fs_xdat.
    """
    import numpy as np
    if ts is None: 
        return None
    ts = np.asarray(ts, float)
    if ts.size == 0 or time_ref is None or len(time_ref) == 0:
        return ts
    # Heuristik: Wenn Pulse deutlich außerhalb der time_s-Skala liegen -> in Samples
    tr_min, tr_max = float(time_ref[0]), float(time_ref[-1])
    if np.nanmax(ts) > 100.0 * max(1.0, tr_max):   # sehr konservativ
        return ts / float(fs_xdat)
    return ts


def _safe_crop_to_pulses(time_s, LFP_array, p1, p2, pad=0.5):
    import numpy as np
    t = np.asarray(time_s, float)
    if t.size == 0:
        print("[CROP] skip: empty time_s")
        return time_s, LFP_array, p1, p2

    tmin, tmax = float(t[0]), float(t[-1])

    def _clamp(ts):
        if ts is None: return None
        ts = np.asarray(ts, float)
        if ts.size == 0: return ts
        return ts[(ts >= tmin) & (ts <= tmax)]

    p1c = _clamp(p1)
    p2c = _clamp(p2)

    if (p1c is None or p1c.size == 0) and (p2c is None or p2c.size == 0):
        print("[CROP] no pulses in range -> no cropping")
        return time_s, LFP_array, p1, p2

    spans = []
    if p1c is not None and p1c.size: spans.append((float(np.min(p1c)), float(np.max(p1c))))
    if p2c is not None and p2c.size: spans.append((float(np.min(p2c)), float(np.max(p2c))))
    if not spans:
        print("[CROP] no valid spans -> no cropping")
        return time_s, LFP_array, p1, p2

    t0 = max(min(s[0] for s in spans) - pad, tmin)
    t1 = min(max(s[1] for s in spans) + pad, tmax)
    if not (t1 > t0):
        print(f"[CROP] invalid window {t0}..{t1} -> no cropping")
        return time_s, LFP_array, p1, p2

    i0 = int(np.searchsorted(t, t0, side="left"))
    i1 = int(np.searchsorted(t, t1, side="right"))
    i0 = max(0, min(i0, t.size))
    i1 = max(i0 + 1, min(i1, t.size))

    time_new = time_s[i0:i1]
    LFP_new  = LFP_array[:, i0:i1]

    def _keep_in(ts):
        if ts is None: return None
        ts = np.asarray(ts, float)
        if ts.size == 0: return ts
        return ts[(ts >= time_new[0]) & (ts <= time_new[-1])]

    p1_new = _keep_in(p1c)
    p2_new = _keep_in(p2c)

    print(f"[CROP] window {t0:.3f}–{t1:.3f} s -> time_s len={len(time_new)}, "
          f"LFP_array={LFP_new.shape}, p1={0 if p1_new is None else len(p1_new)}, "
          f"p2={0 if p2_new is None else len(p2_new)}")
    return time_new, LFP_new, p1_new, p2_new

def _empty_updict():
    import numpy as np
    ZI = np.array([], dtype=int); ZF = np.array([], dtype=float)
    return {
        "Spontaneous_UP": ZI, "Spontaneous_DOWN": ZI,
        "Pulse_triggered_UP": ZI, "Pulse_triggered_DOWN": ZI,
        "Pulse_associated_UP": ZI, "Pulse_associated_DOWN": ZI,
        "Spon_Peaks": ZF, "Trig_Peaks": ZF,
        "UP_start_i": ZI, "DOWN_start_i": ZI,
        "Total_power": None, "up_state_binary": None,
    }

def _clip_pairs(U, D, n):
    U = np.asarray(U, int); D = np.asarray(D, int)
    m = min(U.size, D.size)
    if m == 0: return U[:0], D[:0]
    U, D = U[:m], D[:m]
    mask = (U >= 0) & (D > U) & (D <= n)
    return U[mask], D[mask]


# --- Events/Pulse boundary-sicher machen ---
def _clip_events_to_bounds(pulse_times, time_s, pre_s, post_s):
    import numpy as np
    if pulse_times is None: 
        return np.array([], dtype=float)
    t = np.asarray(pulse_times, float)
    if t.size == 0 or len(time_s) == 0:
        return np.array([], dtype=float)
    lo = float(time_s[0]) + float(pre_s)
    hi = float(time_s[-1]) - float(post_s)
    if hi <= lo:
        return np.array([], dtype=float)
    return t[(t >= lo) & (t <= hi)]




def _upstate_amplitudes(signal, up_idx, down_idx):
    """
    Misst pro UP-Event die Amplitude (max - min) im Rohsignal.
    up_idx/down_idx: Sample-Indizes in 'signal' (wie aus classify_states).
    Rückgabe: np.ndarray [n_events] (float), NaN-frei gefiltert.
    """
    import numpy as np
    sig = np.asarray(signal, float)
    U = np.asarray(up_idx, dtype=int)
    D = np.asarray(down_idx, dtype=int)
    m = min(U.size, D.size)
    if m == 0:
        return np.array([], dtype=float)

    U, D = U[:m], D[:m]
    # chronologisch sortieren (optional)
    order = np.argsort(U)
    U, D = U[order], D[order]

    amps = []
    n = sig.size
    for u, d in zip(U, D):
        if not (0 <= u < n and 0 < d <= n and d > u):
            continue
        seg = sig[u:d]
        seg = seg[np.isfinite(seg)]
        if seg.size == 0:
            continue
        amps.append(float(np.nanmax(seg) - np.nanmin(seg)))
    return np.array(amps, dtype=float)



def _sem(x):
    import numpy as np
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return np.nanstd(x) / np.sqrt(max(1, x.size)) if x.size else np.nan

def _even_subsample(idx, k):
    idx = np.asarray(idx, int)
    if idx.size <= k: return idx
    pos = np.linspace(0, idx.size-1, k).round().astype(int)
    return idx[pos]



def _check_peak_indices(label, peaks, n):
    import numpy as np
    p = np.asarray(peaks, int)
    bad = np.sum((p < 0) | (p >= n))
    print(f"[DIAG] {label}: count={p.size}, out_of_bounds={bad} (n_time={n})")


