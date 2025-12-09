#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math
import numpy as np
_np = np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

from matplotlib.colors import SymLogNorm, TwoSlopeNorm
import gc
from scipy.ndimage import gaussian_filter

from matplotlib.colors import TwoSlopeNorm, SymLogNorm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from loader_old import load_LFP_new
from matplotlib import gridspec
from scipy.signal import welch
from matplotlib.colors import SymLogNorm
try:
    from preprocessing import downsampling_old as _ds_fun
except ImportError:
    from preprocessing import downsampling as _ds_fun
from preprocessing import filtering, get_main_channel, pre_post_condition
from TimeFreq_plot import Run_spectrogram
from state_detection import (
    classify_states, extract_upstate_windows,
    compare_spectra, _up_onsets, Generate_CSD_mean_from_onsets,
)
from pathlib import Path
from plotter import (
    plot_all_channels,
    plot_spont_up_mean,
    plot_upstate_duration_comparison,
    plot_upstate_amplitude_blocks_colored,
)

import sys, os
from datetime import datetime

from Exports import (
    export_interactive_lfp_html, 
    log,
     _nan_stats,
     _rms
)
from processing import (
    load_parts_to_array_streaming,
    downsample_array_simple, 
    _counts_to_uV, _volts_to_uV, 
    convert_df_to_uV, _decimate_xy, 
    _ensure_main_channel, _ensure_seconds, 
    _safe_crop_to_pulses,
    _empty_updict,
    _clip_pairs,
    _clip_events_to_bounds,
    _upstate_amplitudes,
    _sem,
    _even_subsample,
    _check_peak_indices,
    )
        
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


parts_dir = Path(BASE_PATH) / "_csv_parts"
is_merged_run = os.environ.get("BATCH_IS_MERGED_RUN", "0") == "1"
has_explicit_filename = "LFP_FILENAME" in globals()


if (not is_merged_run) and (not has_explicit_filename) and parts_dir.exists() and any(parts_dir.glob("*.part*.csv")):
    print(f"[INFO] Parts erkannt unter {parts_dir} -> streaming load")
    time_s, LFP_array, chan_cols, pulse_times_1_full, pulse_times_2_full = \
        load_parts_to_array_streaming(BASE_PATH, ds_factor=DOWNSAMPLE_FACTOR)
    lfp_meta = {"source": "parts-streaming"}

    # wichtig: KEIN riesiges DataFrame bauen
    LFP_df = None

    log(f"MODE: streaming={ (not is_merged_run) and (not has_explicit_filename) } parts_dir={parts_dir}")
   

else:
    from loader_old import load_LFP_new
    LFP_df, chan_cols, lfp_meta = load_LFP_new(BASE_PATH, LFP_FILENAME)
    time_s = pd.to_numeric(LFP_df["time"], errors="coerce").to_numpy(dtype=float)
    LFP_array = LFP_df[chan_cols].to_numpy(dtype=np.float32).T
    log(f"Streaming load OK: time={time_s.shape}, LFP_array={LFP_array.shape}, chans={len(chan_cols)}")

FROM_STREAM = (LFP_df is None)

if not FROM_STREAM:
    assert "time" in LFP_df.columns, "CSV braucht eine Spalte 'time'."



if LFP_df is not None:
    # alter Weg: wir haben ein DataFrame aus einer einzelnen CSV
    time_full = pd.to_numeric(LFP_df["time"], errors="coerce").to_numpy(dtype=float)

    chan_cols_raw = [c for c in LFP_df.columns if c not in ("time","stim","din_1","din_2")]
    ...
    LFP_df_ds = pd.DataFrame({"timesamples": time_full})
    for i, col in enumerate(chan_cols):
        LFP_df_ds[f"pri_{i}"] = pd.to_numeric(LFP_df[col], errors="coerce")
    NUM_CHANNELS = len(chan_cols)
else:
    # streaming-Weg: wir haben schon Array-Form
    time_full = time_s
    NUM_CHANNELS = LFP_array.shape[0]
    chan_cols = chan_cols  # die kamen aus load_parts_to_array_streaming
    # und wir können LFP_df_ds ganz weglassen, weil _ds_fun ja schon auf Array arbeitet

CALIB_MODE = "counts"   # "counts" | "volts" | "uV"
ADC_BITS   = 16         # bit
ADC_VPP    = 10.0       # Peak-to-Peak des ADC in Volt 
PREAMP_GAIN = 1000.0    # Gesamt-Gain vor dem ADC. Falls kanal-spezifisch, unten 'PER_CH_GAIN' nutzen.

# optional: kanal-spezifische Gains (Original-Spaltennamen nutzen, vor 'pri_*' Umbenennung)
PER_CH_GAIN = {
    # "CSC1_values": 2000.0,
    # "CSC2_values": 1000.0,
}

UNIT_LABEL = "µV/mm²"          
PSD_UNIT_LABEL = "µV²/Hz"


if ANALYSE_IN_AU:
    UNIT_LABEL = "a.u."
    PSD_UNIT_LABEL = "a.u.^2/Hz"

if FROM_STREAM:
    print(f"[INFO] pulses(from streaming): p1={len(pulse_times_1_full)}, p2={len(pulse_times_2_full)}")
else:
   #Pulse direkt aus dem DataFrame ziehen
    pulse_times_1_full = np.array([], dtype=float)
    pulse_times_2_full = np.array([], dtype=float)
    stim_like_cols = []

    time_full = pd.to_numeric(LFP_df["time"], errors="coerce").to_numpy(dtype=float)

    def _edges_from_col(col, rising_only=True, thr=None):
        x = pd.to_numeric(LFP_df[col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(x).any():
            return np.array([], dtype=float)
        if thr is None:
            lo, hi = np.nanpercentile(x, [10, 90])
            thr = (lo + hi) * 0.5
        b = (x > thr).astype(np.int8)
        idx = (np.flatnonzero((b[1:] == 1) & (b[:-1] == 0)) + 1) if rising_only \
              else (np.flatnonzero(b[1:] != b[:-1]) + 1)
        idx = idx[(idx >= 0) & (idx < time_full.size)]
        return time_full[idx].astype(float)

    def _is_quasi_binary(col):
        x = pd.to_numeric(LFP_df[col], errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 10:
            return False
        vals = np.unique(np.round(x, 3))
        if len(vals) <= 4:
            return True
        p01 = (np.isclose(x, 0).sum() + np.isclose(x, 1).sum()) / x.size
        return p01 >= 0.95

    # 1) Bevorzugte Spaltennamen durchgehen
    preferred = [c for c in ["din_1","din_2","stim","StartStop","TTL","DI0","DI1"] if c in LFP_df.columns]
    if "din_1" in preferred:
        t = _edges_from_col("din_1", rising_only=True, thr=None)
        if t.size: pulse_times_1_full, stim_like_cols = t, stim_like_cols+["din_1"]
    if "din_2" in preferred:
        t = _edges_from_col("din_2", rising_only=True, thr=None)
        if t.size: pulse_times_2_full, stim_like_cols = t, stim_like_cols+["din_2"]
    if pulse_times_1_full.size == 0 and "stim" in preferred:
        t = _edges_from_col("stim", rising_only=True, thr=None)
        if t.size: pulse_times_1_full, stim_like_cols = t, stim_like_cols+["stim"]
    # Falls StartStop/TTL nur eine Spur ist → als p1 nehmen
    if pulse_times_1_full.size == 0:
        for cand in ["StartStop","TTL","DI0","DI1"]:
            if cand in preferred:
                t = _edges_from_col(cand, rising_only=True, thr=None)
                if t.size:
                    pulse_times_1_full, stim_like_cols = t, stim_like_cols+[cand]
                    break

    # 2) Auto-Detect (wenn bisher nichts gefunden): nimm (nahezu) binäre Spalte mit meisten Flanken
    if pulse_times_1_full.size == 0 and pulse_times_2_full.size == 0:
        candidates = [c for c in LFP_df.columns if c not in ("time",)]
        bin_cols = [c for c in candidates if _is_quasi_binary(c)]
        best_col, best_count = None, -1
        for c in bin_cols:
            t = _edges_from_col(c, rising_only=True, thr=None)
            if t.size > best_count:
                best_col, best_count = c, t.size
        if best_col is not None and best_count > 0:
            pulse_times_1_full = _edges_from_col(best_col, rising_only=True, thr=None)
            stim_like_cols.append(best_col)
            print(f"[INFO] Auto-detected stim channel: {best_col} (rising edges: {len(pulse_times_1_full)})")

    print(f"[INFO] pulses(full): p1={len(pulse_times_1_full)}, p2={len(pulse_times_2_full)} "
          f"| from columns: {', '.join(stim_like_cols) if stim_like_cols else '—'}")

if not FROM_STREAM:
    chan_cols_raw = [c for c in LFP_df.columns if c not in ("time","stim","din_1","din_2")]
    # 2) Numerische Schlüssel aus Spaltennamen ziehen (z.B. "CSC10_values" -> 10, "8" -> 8)
    def _key_num(s):
        import re
        m = re.findall(r"\d+", s)
        return int(m[-1]) if m else 0

    # 3) Sortierte Reihenfolge (flach -> tief). Wenn du tief->flach willst: am Ende [::-1].
    order_idx = sorted(range(len(chan_cols_raw)), key=lambda i: _key_num(chan_cols_raw[i]))
    # optional: flippen, falls benötigt
    FLIP_DEPTH = False   # <- bei Bedarf True
    if FLIP_DEPTH:
        order_idx = order_idx[::-1]

    
    LFP_df_ds = pd.DataFrame({"timesamples": time_full})
    for i, col in enumerate(chan_cols):
        LFP_df_ds[f"pri_{i}"] = pd.to_numeric(LFP_df[col], errors="coerce")
    NUM_CHANNELS = len(chan_cols)
else:
    NUM_CHANNELS = LFP_array.shape[0]


if FROM_STREAM:
    # direktes Array-Downsampling, KEIN DataFrame
    time_s, dt, LFP_array, pulse_times_1, pulse_times_2 = downsample_array_simple(
        DOWNSAMPLE_FACTOR,
        time_s,
        LFP_array,
        pulse_times_1=pulse_times_1_full,
        pulse_times_2=pulse_times_2_full,
        snap_pulses=True,
    )
else:
    # alter Weg für nicht-gesplittete CSV
    time_s = pd.to_numeric(LFP_df["time"], errors="coerce").to_numpy(dtype=float)
    LFP_array = LFP_df[chan_cols].to_numpy(dtype=np.float32).T
    time_s, dt, LFP_array, pulse_times_1, pulse_times_2 = _ds_fun(
        DOWNSAMPLE_FACTOR, LFP_df_ds, NUM_CHANNELS,
        pulse_times_1=pulse_times_1_full,
        pulse_times_2=pulse_times_2_full,
        snap_pulses=True
    )

log(f"DS done: N={len(time_s)}, dt={dt}, shape={LFP_array.shape}, p1={0 if pulse_times_1 is None else len(pulse_times_1)}, p2={0 if pulse_times_2 is None else len(pulse_times_2)}")

if 'LFP_df' in globals() and LFP_df is not None:
    del LFP_df
if 'LFP_df_ds' in globals():
    del LFP_df_ds

gc.collect()


pulse_times_1_full = None if pulse_times_1 is None else np.array(pulse_times_1, float)
pulse_times_2_full = None if pulse_times_2 is None else np.array(pulse_times_2, float)


NUM_CHANNELS = LFP_array.shape[0]
good_idx = list(range(NUM_CHANNELS))  # Fallback: alle Kanäle
reasons = []                          # für Log-Ausgaben des Kanalfilters



if dt and (dt > 0) and ((1.0/dt) > 1e4 or dt > 1.0):  # sehr grob: "klein" => Sekunden, "groß" => Samples
    dt = dt / DEFAULT_FS_XDAT  # dt war in Samples

print("[CHECK] dt(s)=", dt, " median Δt from time_s=", float(np.median(np.diff(time_s))))


# ebenfalls auf Sekunden bringen:
if np.nanmax(time_s) > 1e6:
    time_s = time_s / DEFAULT_FS_XDAT

# Pulszeiten ebenfalls normalisieren (falls noch in Samples)
if (pulse_times_1 is not None) and (len(pulse_times_1) > 0) and (np.nanmax(pulse_times_1) > 1e6):
    pulse_times_1 = pulse_times_1 / DEFAULT_FS_XDAT
if (pulse_times_2 is not None) and (len(pulse_times_2) > 0) and (np.nanmax(pulse_times_2) > 1e6):
    pulse_times_2 = pulse_times_2 / DEFAULT_FS_XDAT

print(f"[DS][FIXED] dt={dt:.9f} s, Nyquist={0.5/dt:.3f} Hz, "
      f"time_s: {float(time_s[0]):.3f}->{float(time_s[-1]):.3f} s")


assert LFP_array.shape[0] == NUM_CHANNELS
assert LFP_array.shape[1] == len(time_s)
print(f"[DS] time {time_s[0]:.3f}->{time_s[-1]:.3f}s, N={len(time_s)}, dt={dt:.6f}s, "
      f"LFP_array={LFP_array.shape}, p1={len(pulse_times_1)}, p2={len(pulse_times_2)}")



# Kanalnamen ableiten 
ch_names_for_plot = [f"pri_{i}" for i in range(LFP_array.shape[0])]
svg_path = os.path.join(SAVE_DIR, f"{BASE_TAG}__all_channels_STACKED.svg")





# direkt nach _ds_fun(...)
pulse_times_1 = _ensure_seconds(pulse_times_1, time_s, DEFAULT_FS_XDAT)
pulse_times_2 = _ensure_seconds(pulse_times_2, time_s, DEFAULT_FS_XDAT)






# vor dem Crop:
if ((pulse_times_1 is None or len(pulse_times_1)==0) and
    (pulse_times_2 is None or len(pulse_times_2)==0)):
    print("[CROP] skip: no pulses -> keep full time range")
else:
    time_s, LFP_array, pulse_times_1, pulse_times_2 = _safe_crop_to_pulses(
        time_s, LFP_array, pulse_times_1, pulse_times_2, pad=0.5
    )








# Anwenden (benutze NUR die normalisierten pulse_times_1/2 – NICHT *_full):
time_s, LFP_array, pulse_times_1, pulse_times_2 = _safe_crop_to_pulses(
    time_s, LFP_array, pulse_times_1, pulse_times_2, pad=0.5
)

log(f"Crop done: time={time_s[0]:.3f}->{time_s[-1]:.3f}, shape={LFP_array.shape}, p1={len(pulse_times_1) if pulse_times_1 is not None else 0}, p2={len(pulse_times_2) if pulse_times_2 is not None else 0}")

# --- main_channel robust auswählen (nach evtl. Cropping) ---
main_channel, ch_idx_used = _ensure_main_channel(LFP_array, preferred_idx=10)

# --- Für HTML: Main-Channel in µV (ohne die Analyse anzurühren) ---
main_channel_uV = None
if HTML_IN_uV:
    # welches Gain für diesen physikalischen Kanal?
    orig_name = chan_cols[ch_idx_used] if (0 <= ch_idx_used < len(chan_cols)) else None
    gain_used = PER_CH_GAIN.get(orig_name, PREAMP_GAIN)

    if CALIB_MODE == "counts":
        main_channel_uV = _counts_to_uV(main_channel, ADC_BITS, ADC_VPP, gain_used)
    elif CALIB_MODE == "volts":
        main_channel_uV = _volts_to_uV(main_channel)
    elif CALIB_MODE == "uV":
        main_channel_uV = main_channel.copy()
    else:
        # Fallback: zeige eben a.u., falls unbekannter Modus
        main_channel_uV = main_channel.copy()




# --- XDAT-Erkennung (heuristisch) ---
def _is_xdat_format():
    """
    Liefert True, wenn die Kanalnamen stark nach XDAT/Intan aussehen,
    z.B. 'ch00', 'ch1', 'ch17' etc. oder wenn der Dateipfad 'xdat' enthält.
    Nutzt die *originalen* Spaltennamen vor dem pri_* Mapping.
    """
    import re
    # 1) Dateipfad-/Dateiname-Heuristik
    try:
        path_str = str(BASE_PATH).lower() + " " + str(LFP_FILENAME).lower()
        if "xdat" in path_str or path_str.endswith(".xdat"):
            return True
    except Exception:
        pass

    # 2) Kanalnamen-Heuristik (originale Namen bevorzugt)
    try:
        cols = chan_cols_raw  # aus dem Code weiter oben
    except NameError:
        # Fallback: aus dem DataFrame ableiten (ohne time/stim/din)
        cols = [c for c in LFP_df.columns if c not in ("time", "stim", "din_1", "din_2")]

    # Muster wie 'ch00', 'ch0', 'ch17', 'CH31' usw. oder reine Ziffern
    pat = re.compile(r"^(ch)?\d{1,3}$", re.IGNORECASE)
    hits = sum(1 for c in cols if pat.match(str(c).strip()))
    # Wenn ein deutlicher Teil der Kanäle so heißt, ist es sehr wahrscheinlich XDAT
    return (len(cols) >= 8 and hits / max(1, len(cols)) >= 0.6)


# ==== Feste Kanalwahl für .xdat: nur pri_1 .. pri_17 ====
if _is_xdat_format():
    fixed_idx = [i for i in range(1, 15) if i < LFP_array.shape[0]]  # pri_1..pri_17
    good_idx = fixed_idx[:]  # überschreibe Fallback
    print(f"[XDAT] GOOD_IDX override -> {good_idx} (n={len(good_idx)})")
else:
    # ===== Kanalqualitäts-Filter =====
    bad_idx, reasons = set(), []
    fs = 1.0 / dt

    def _is_quasi_binary_trace(x):
        x = np.asarray(x, float); x = x[np.isfinite(x)]
        if x.size < 10: return False
        vals = np.unique(np.round(x, 3))
        if len(vals) <= 4: return True
        p01 = (np.isclose(x,0).sum() + np.isclose(x,1).sum()) / x.size
        return p01 >= 0.95

    def _line_noise_ratio(x, fs):
        f, Pxx = welch(np.nan_to_num(x, nan=0.0), fs=fs, nperseg=min(len(x), 4096))
        def bp(f1,f2):
            m = (f>=f1) & (f<=f2)
            return float(np.trapz(Pxx[m], f[m])) if m.any() else 0.0
        total = bp(0.5, 120.0)
        line  = bp(49.0, 51.0)
        return line / (total + 1e-12)

    for i in range(NUM_CHANNELS):
        x = LFP_array[i]
        finite = np.isfinite(x)
        if finite.mean() < 0.95:
            bad_idx.add(i); reasons.append((i, "zu viele NaNs")); continue
        std = np.nanstd(x)
        if not np.isfinite(std) or std == 0:
            bad_idx.add(i); reasons.append((i, "konstant/0-Std")); continue
        if _is_quasi_binary_trace(x):
            bad_idx.add(i); reasons.append((i, "quasi-binär")); continue
        z = (x - np.nanmedian(x)) / (std if std else 1.0)
        if np.mean(np.abs(z) > 8) > 0.02:
            bad_idx.add(i); reasons.append((i, "Artefakte (>2% |z|>8)")); continue
        if _line_noise_ratio(x, fs) > 0.3:
            bad_idx.add(i); reasons.append((i, "50Hz-dominant")); continue

    good_idx = [j for j in range(NUM_CHANNELS) if j not in bad_idx]
    if len(good_idx) < 2:
        print("[CHAN-FILTER][WARN] zu wenige 'gute' Kanäle – benutze alle.")
        good_idx = list(range(NUM_CHANNELS))

LFP_array_good    = LFP_array[good_idx, :]
ch_names_good     = [f"pri_{j}" for j in good_idx]
NUM_CHANNELS_GOOD = len(good_idx)

# Tiefe aus den *behaltenen* Kanälen ableiten:
DZ_UM = 100.0
z_mm = (np.arange(NUM_CHANNELS_GOOD, dtype=float)) * (DZ_UM / 1000.0)
z_mm_csd = z_mm[1:-1]

if reasons:
    print("[CHAN-FILTER] excluded:", ", ".join([f"pri_{j}({r})" for j, r in reasons]))
print(f"[CHAN-FILTER] kept {NUM_CHANNELS_GOOD}/{NUM_CHANNELS} Kanäle:", ch_names_good[:10], ("..." if NUM_CHANNELS_GOOD>10 else ""))

log(f"Channel filter: kept={NUM_CHANNELS_GOOD}/{NUM_CHANNELS}, good_idx={good_idx}")


b_lp, a_lp, b_hp, a_hp = filtering(LOW_CUTOFF, HIGH_CUTOFF, dt)  # 2, 10



print(f"[INFO] NUM_CHANNELS={NUM_CHANNELS}, main_channel_len={len(main_channel)}")
pre, post, win_len, align_pre, align_post, align_len = pre_post_condition(dt)

FIXED_ALIGN_PRE_S  = 0.5   # 0.5 s vor Peak
FIXED_ALIGN_POST_S = 0.5   # 0.5 s nach Peak

align_pre  = int(round(FIXED_ALIGN_PRE_S  / dt))
align_post = int(round(FIXED_ALIGN_POST_S / dt))
align_len  = align_pre + align_post

align_pre_s  = FIXED_ALIGN_PRE_S
align_post_s = FIXED_ALIGN_POST_S

if len(time_s) >= 2 and LFP_array.shape[1] >= 2:
    Spect_dat = Run_spectrogram(main_channel, time_s)
else:
    raise RuntimeError("Spectrogram skipped: empty/too short segment after cropping.")



try:
    _save_all_channels_svg_from_array(
        time_s, LFP_array, [f"pri_{i}" for i in range(NUM_CHANNELS)],
        os.path.join(SAVE_DIR, f"{BASE_TAG}__all_channels_DS.svg"),
        max_points=40000
    )
except Exception as e:
    print("[ALL-CH][DS] skip:", e)


# --- Init vor Spektren/States, damit Namen garantiert existieren ---
freqs = spont_mean = pulse_mean = p_vals = None

try:
    res = compare_spectra(pulse_windows, spont_windows, dt, ignore_start_s=0.3)
    freqs, spont_mean, pulse_mean, p_vals = res
except Exception as e:
    print("[WARN] spectra compare skipped:", e)

pulse_times_1 = _clip_events_to_bounds(pulse_times_1, time_s, align_pre_s, align_post_s)
pulse_times_2 = _clip_events_to_bounds(pulse_times_2, time_s, align_pre_s, align_post_s)

log(f"Calling classify_states: len(time_s)={len(time_s)}, main_len={len(main_channel)}, dt={dt}, p1={len(pulse_times_1) if pulse_times_1 is not None else 0}, p2={len(pulse_times_2) if pulse_times_2 is not None else 0}")


# --- classify_states robust aufrufen ---
try:
    Up = classify_states(
        Spect_dat, time_s, pulse_times_1, pulse_times_2, dt,
        main_channel, LFP_array, b_lp, a_lp, b_hp, a_hp,
        align_pre, align_post, align_len
    )
except IndexError as e:
    log(f"classify_states FAILED: {e}")
    print(f"[WARN] classify_states skipped due to IndexError: {e}")
    Up = _empty_updict()


nT = LFP_array.shape[1]
for kU, kD in [
    ("Spontaneous_UP","Spontaneous_DOWN"),
    ("Pulse_triggered_UP","Pulse_triggered_DOWN"),
    ("Pulse_associated_UP","Pulse_associated_DOWN"),
]:
    Uc, Dc = _clip_pairs(Up.get(kU, []), Up.get(kD, []), nT)
    Up[kU], Up[kD] = Uc, Dc



def crop_up_intervals(UP, DOWN, dt, start_s=0.3, end_s=1.0):
    """
    Schneidet UP-Intervalle zu:
    neuer Bereich = [UP + start_s  ...  UP + end_s], aber niemals über DOWN hinaus.
    Gibt zwei Arrays zurück: cropped_UP, cropped_DOWN
    """
    UP = np.asarray(UP, int)
    DOWN = np.asarray(DOWN, int)
    m = min(len(UP), len(DOWN))

    if m == 0:
        return np.array([], int), np.array([], int)

    UP = UP[:m]
    DOWN = DOWN[:m]

    cropped_UP = []
    cropped_DOWN = []

    start_offset = int(round(start_s / dt))
    end_offset   = int(round(end_s / dt))

    for u, d in zip(UP, DOWN):
        new_u = u + start_offset
        new_d = min(u + end_offset, d)

        if new_d > new_u:   # sonst skip
            cropped_UP.append(new_u)
            cropped_DOWN.append(new_d)

    return np.array(cropped_UP, int), np.array(cropped_DOWN, int)

def refractory_overlay_ax(
    main_channel,
    time_s,
    UP_indices,
    DOWN_indices,
    dt,
    pre_s=0.5,       # wie weit vor UP-Ende anzeigen
    post_s=2.0,      # wie weit nach UP-Ende anzeigen
    ax=None,
    title="UP-offset-alignierte LFP-Segmente + nächster UP-Onset"
):
    """
    Visualisiert für eine Sequenz von UP/DOWN-Paaren:
    - alle LFP-Segmente relativ zum Ende eines UP (DOWN-Index) übereinander
    - 0 s = Ende des aktuellen UP
    - vertikale Linie bei 0 s (UP-Ende)
    - vertikale Linien für die Onsets des jeweils folgenden UP
    - markiert zusätzlich den mittleren Zeitpunkt des nächsten UP-Onsets
    """

    import numpy as np
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3.0))
    else:
        fig = ax.figure

    UP   = np.asarray(UP_indices,   int)
    DOWN = np.asarray(DOWN_indices, int)

    m = min(len(UP), len(DOWN))
    if m < 2:
        ax.text(0.5, 0.5, "zu wenige UP/DOWN-Paare\n(mind. 2 nötig)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    # Wir betrachten Paare: (DOWN[i] -> UP[i+1])
    pre_n  = int(round(pre_s  / dt))
    post_n = int(round(post_s / dt))

    segs = []
    rel_next_onsets = []

    for i in range(m - 1):
        idx_off = DOWN[i]       # Ende des aktuellen UP
        idx_next_on = UP[i+1]   # Beginn des nächsten UP

        if idx_off < 0 or idx_off >= len(time_s):
            continue
        if idx_next_on < 0 or idx_next_on >= len(time_s):
            continue

        t_off = time_s[idx_off]
        t_next = time_s[idx_next_on]
        rel_next = t_next - t_off   # Refraktärzeit (s)

        # Fenster um das UP-Ende
        s = idx_off - pre_n
        e = idx_off + post_n
        if s < 0 or e > len(main_channel):
            continue

        seg = main_channel[s:e]
        if len(seg) != (pre_n + post_n):
            continue

        segs.append(seg)
        rel_next_onsets.append(rel_next)

    if not segs:
        ax.text(0.5, 0.5, "keine gültigen Segmente für Refraktär-Overlay",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    segs = np.vstack(segs)
    rel_next_onsets = np.asarray(rel_next_onsets, float)
    n = segs.shape[0]

    t_rel = (np.arange(-pre_n, post_n) * dt)

    # Einzeltraces
    for i in range(n):
        ax.plot(t_rel, segs[i], alpha=0.10, lw=0.8)

    # Mitteltrace
    mean_trace = np.nanmean(segs, axis=0)
    ax.plot(t_rel, mean_trace, lw=2.0, label="Mean LFP")

    # Vertikale Linie beim UP-Ende
    ax.axvline(0.0, color="k", lw=1.0, ls="--", label="UP-Ende")

    # Vertikale Linien für nächste UP-Onsets (leicht transparent)
    valid_mask = np.isfinite(rel_next_onsets)
    rel_valid = rel_next_onsets[valid_mask]

    if rel_valid.size:
        for r in rel_valid:
            if (-pre_s <= r <= post_s):
                ax.axvline(r, color="red", alpha=0.18, lw=1.0)

        mean_ref = float(np.nanmean(rel_valid))
        if -pre_s <= mean_ref <= post_s:
            ax.axvline(mean_ref, color="red", lw=1.8, ls=":",
                       label=f"Mean nächster UP ({mean_ref*1000:.0f} ms)")

        # kleine Textbox mit n und Mean
        ax.text(
            0.02, 0.95,
            f"n = {rel_valid.size}\nMean = {mean_ref*1000:.1f} ms",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", fc="white", alpha=0.7)
        )
    else:
        ax.text(
            0.02, 0.95,
            f"n = 0\n(keine gültigen Refraktärzeiten)",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", fc="white", alpha=0.7)
        )

    ax.set_xlabel("Zeit relativ zum UP-Ende (s)")
    ax.set_ylabel(f"LFP ({UNIT_LABEL})")
    ax.set_title(title)

    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    return fig


def compute_refractory_period(UP_indices, DOWN_indices, time_s):
    """
    Berechnet die Refraktärzeit:
    Zeit vom Ende eines UP-Zustands (DOWN) bis zum Beginn des
    nächsten UP-Zustands (UP), in Sekunden.

    UP_indices, DOWN_indices: Arrays mit Indizes in time_s
    time_s: Vektor der Zeitstempel (Sekunden)
    """
    UP   = np.asarray(UP_indices,   int)
    DOWN = np.asarray(DOWN_indices, int)

    m = min(len(UP), len(DOWN))
    if m < 2:
        return np.array([], float)

    UP   = UP[:m]
    DOWN = DOWN[:m]

    # Sicherheit: nach Zeit sortieren
    order = np.argsort(time_s[UP])
    UP    = UP[order]
    DOWN  = DOWN[order]

    refrac = []
    for i in range(m - 1):
        t_off = time_s[DOWN[i]]      # Ende des aktuellen UP
        t_on  = time_s[UP[i+1]]      # Beginn des nächsten UP
        dt_ref = t_on - t_off
        if dt_ref >= 0:
            refrac.append(dt_ref)

    return np.array(refrac, float)



Spontaneous_UP        = Up.get("Spontaneous_UP",        np.array([], int))
Spontaneous_DOWN      = Up.get("Spontaneous_DOWN",      np.array([], int))
Pulse_triggered_UP    = Up.get("Pulse_triggered_UP",    np.array([], int))
Pulse_triggered_DOWN  = Up.get("Pulse_triggered_DOWN",  np.array([], int))
Pulse_associated_UP   = Up.get("Pulse_associated_UP",   np.array([], int))
Pulse_associated_DOWN = Up.get("Pulse_associated_DOWN", np.array([], int))
Spon_Peaks            = Up.get("Spon_Peaks",            np.array([], float))
Total_power           = Up.get("Total_power",           None)
up_state_binary       = Up.get("up_state_binary ", Up.get("up_state_binary", None))

print("[COUNTS] sponUP:", len(Spontaneous_UP), " trigUP:", len(Pulse_triggered_UP), " assocUP:", len(Pulse_associated_UP))
log(f"States: spon={len(Spontaneous_UP)}, trig={len(Pulse_triggered_UP)}, assoc={len(Pulse_associated_UP)}")


# --- Refraktärzeiten (Ende eines UP bis Beginn des nächsten UP) ---
refrac_spont = compute_refractory_period(
    Spontaneous_UP, Spontaneous_DOWN, time_s
)
refrac_trig = compute_refractory_period(
    Pulse_triggered_UP, Pulse_triggered_DOWN, time_s
)

print(f"[REFRAC] spont: n={len(refrac_spont)}, trig: n={len(refrac_trig)}")


# --- CSV: Refraktärzeiten exportieren ---
if len(refrac_spont) or len(refrac_trig):
    ref_data = []
    if len(refrac_spont):
        ref_data.append(pd.DataFrame({
            "group": "spontaneous",
            "refractory_s": refrac_spont,
            "refractory_ms": refrac_spont * 1000.0,
        }))
    if len(refrac_trig):
        ref_data.append(pd.DataFrame({
            "group": "triggered",
            "refractory_s": refrac_trig,
            "refractory_ms": refrac_trig * 1000.0,
        }))
    ref_df = pd.concat(ref_data, ignore_index=True)

    ref_csv_path = os.path.join(SAVE_DIR, f"{BASE_TAG}__refractory_periods.csv")
    ref_df.to_csv(ref_csv_path, index=False)
    print(f"[CSV] Refraktärzeiten geschrieben: {ref_csv_path}  (rows={len(ref_df)})")
else:
    print("[REFRAC] keine Refraktärzeiten (zu wenige UP/DOWN-Events)")

def pulse_to_up_latencies(pulse_times, up_indices, time_s, max_win_s=1.0):
    """
    Berechnet die Latenz zwischen Puls und Beginn des UP-Zustands.
    Für jeden UP-Index wird der *letzte Puls davor* gesucht (innerhalb von max_win_s),
    und die Differenz (UP_time - pulse_time) in Sekunden zurückgegeben.
    """
    if pulse_times is None or len(pulse_times) == 0 or len(up_indices) == 0:
        return np.array([], float)

    pulse_times = np.asarray(pulse_times, float)
    up_indices  = np.asarray(up_indices, int)

    # Zeitpunkte der UP-Onsets
    up_t = time_s[up_indices]
    latencies = []

    for t_up in up_t:
        # alle Pulse, die vor diesem UP liegen
        mask = pulse_times <= t_up
        if not mask.any():
            continue
        t_p = pulse_times[mask][-1]    # letzter Puls vor UP

        lat = t_up - t_p
        # Optionales Fenster: nur Pulse, die "in der Nähe" liegen
        if 0.0 <= lat <= max_win_s:
            latencies.append(lat)

    return np.array(latencies, float)


# --- Pulse→UP-Latenzen (Trigger-Pulse zu Beginn des UP-Zustands) ---
latencies_trig = pulse_to_up_latencies(
    pulse_times_1,          # oder pulse_times_2, je nach Setup
    Pulse_triggered_UP,
    time_s,
    max_win_s=1.0           # z.B. nur Pulse innerhalb von 1s berücksichtigen
)

if latencies_trig.size:
    lat_df = pd.DataFrame({
        "latency_s": latencies_trig,
        "latency_ms": latencies_trig * 1000.0,
    })
    lat_csv_path = os.path.join(SAVE_DIR, f"{BASE_TAG}__pulse_to_up_latency.csv")
    lat_df.to_csv(lat_csv_path, index=False)
    print(f"[CSV] Pulse→UP Latenzen geschrieben: {lat_csv_path}  (n={len(latencies_trig)})")
else:
    print("[INFO] keine Pulse→UP Latenzen gefunden (entweder keine Pulse oder keine Trigger-UPs)")


# --- NEU: gecroppte Intervalle (0.3–1.0 s ab UP-Start) ---
Spon_UP_crop, Spon_DOWN_crop = crop_up_intervals(
    Spontaneous_UP, Spontaneous_DOWN, dt, start_s=0.3, end_s=1.0
)
Trig_UP_crop, Trig_DOWN_crop = crop_up_intervals(
    Pulse_triggered_UP, Pulse_triggered_DOWN, dt, start_s=0.3, end_s=1.0
)

def upstate_amplitude_compare_ax(
    spont_amp, trig_amp,
    ax=None,
    title="UP Amplitude (max-min): Spontan vs. Getriggert"
):
    import numpy as np
    import matplotlib.pyplot as plt

    spont_amp = np.asarray(spont_amp, float)
    trig_amp  = np.asarray(trig_amp,  float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
    else:
        fig = ax.figure

    data, labels = [], []
    if spont_amp.size:
        data.append(spont_amp); labels.append("Spontan")
    if trig_amp.size:
        data.append(trig_amp); labels.append("Getriggert")

    if not data:
        ax.text(0.5, 0.5, "no UP amplitudes", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    ax.boxplot(data, labels=labels, whis=[5, 95], showfliers=False)
    ax.set_ylabel(f"Amplitude ({UNIT_LABEL})")
    ax.set_title(title)
    ax.grid(alpha=0.15, linestyle=":")
    return fig

def pulse_to_up_latency_hist_ax(latencies, ax=None, bins=30):
    import numpy as np
    import matplotlib.pyplot as plt

    latencies = np.asarray(latencies, float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3.0))
    else:
        fig = ax.figure

    if latencies.size == 0:
        ax.text(0.5, 0.5, "no Pulse→UP latencies", 
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    # in Millisekunden umrechnen
    lat_ms  = latencies * 1000.0
    mean_ms = float(np.nanmean(lat_ms))

    # Histogramm
    ax.hist(lat_ms, bins=bins, alpha=0.8, label="Einzel-Latenzen")

    # vertikale Linie beim Mittelwert
    ax.axvline(mean_ms, linestyle="--", linewidth=2,
               label=f"Mean = {mean_ms:.1f} ms")

    ax.set_xlabel("Pulse→UP Latenz (ms)")
    ax.set_ylabel("Anzahl")
    ax.set_title(f"Pulse→UP Latenzen (Mean = {mean_ms:.1f} ms)")

    # Textbox oben rechts mit Mean + n
    ax.text(
        0.98, 0.95,
        f"Mean = {mean_ms:.1f} ms\nn = {lat_ms.size}",
        transform=ax.transAxes,
        ha="right", va="top"
    )

    ax.legend()
    return fig




def upstate_duration_compare_ax(
    Trig_UP_crop, Trig_DOWN_crop,
    Spon_UP_crop, Spon_DOWN_crop,
    dt, ax=None
):


    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
    else:
        fig = ax.figure

    Trig_UP   = np.asarray(Trig_UP_crop,   int)
    Trig_DOWN = np.asarray(Trig_DOWN_crop, int)
    Spon_UP   = np.asarray(Spon_UP_crop,   int)
    Spon_DOWN = np.asarray(Spon_DOWN_crop, int)

    m_trig = min(len(Trig_UP), len(Trig_DOWN))
    m_spon = min(len(Spon_UP), len(Spon_DOWN))

    trig_dur = (Trig_DOWN[:m_trig] - Trig_UP[:m_trig]) * dt if m_trig > 0 else np.array([], float)
    spon_dur = (Spon_DOWN[:m_spon] - Spon_UP[:m_spon]) * dt if m_spon > 0 else np.array([], float)

    data, labels = [], []
    if spon_dur.size:
        data.append(spon_dur); labels.append("Spontan")
    if trig_dur.size:
        data.append(trig_dur); labels.append("Getriggert")

    if not data:
        ax.text(0.5, 0.5, "no UP durations", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    ax.boxplot(data, labels=labels, whis=[5, 95], showfliers=False)
    ax.set_ylabel("Dauer (s)")
    ax.set_title("UP-Dauern (cropped 0.3–1.0 s)")
    ax.grid(alpha=0.15, linestyle=":")
    return fig


def refractory_compare_ax(refrac_spont, refrac_trig, ax=None, title="Refraktärzeit bis zum nächsten UP"):
    import numpy as np
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
    else:
        fig = ax.figure

    refrac_spont = np.asarray(refrac_spont, float)
    refrac_trig  = np.asarray(refrac_trig,  float)

    data, labels = [], []
    if refrac_spont.size:
        data.append(refrac_spont)
        labels.append("Spontan")
    if refrac_trig.size:
        data.append(refrac_trig)
        labels.append("Getriggert")

    if not data:
        ax.text(0.5, 0.5, "keine Refraktärzeiten", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    ax.boxplot(
        data,
        labels=labels,
        whis=[5, 95],
        showfliers=False
    )
    ax.set_ylabel("Refraktärzeit bis nächster UP (s)")
    ax.set_title(title)
    ax.grid(alpha=0.15, linestyle=":")

    # kleine n-Angabe
    txt = []
    if refrac_spont.size:
        txt.append(f"Spontan: n={refrac_spont.size}")
    if refrac_trig.size:
        txt.append(f"Getriggert: n={refrac_trig.size}")
    ax.text(
        0.98, 0.95,
        "\n".join(txt),
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", alpha=0.7)
    )

    return fig

# --- Amplituden pro UP-Typ (max - min) berechnen + CSV ablegen ---
spont_amp = _upstate_amplitudes(main_channel, Spon_UP_crop, Spon_DOWN_crop)
trig_amp  = _upstate_amplitudes(main_channel, Trig_UP_crop, Trig_DOWN_crop)

amp_df = pd.DataFrame({
    "group": (["spontaneous"] * len(spont_amp)) + (["triggered"] * len(trig_amp)),
    "amplitude": np.concatenate([spont_amp, trig_amp]) if (len(spont_amp) or len(trig_amp)) else np.array([], float)
})
amp_csv_path = os.path.join(SAVE_DIR, f"{BASE_TAG}__upstate_amplitudes.csv")
amp_df.to_csv(amp_csv_path, index=False)
print(f"[CSV] UP-Amplituden geschrieben: {amp_csv_path}  (spont={len(spont_amp)}, trig={len(trig_amp)})")


# --- separate SVG mit dem Amplitudenvergleich ---
amp_svg_path = os.path.join(SAVE_DIR, f"{BASE_TAG}__upstate_amplitude_compare.svg")
fig_amp, ax_amp = plt.subplots(figsize=(6.5, 3.4))
upstate_amplitude_compare_ax(
    spont_amp, trig_amp, ax=ax_amp,
    title="UP Amplitude (max-min): Spontan vs. Getriggert"
)
fig_amp.tight_layout()
fig_amp.savefig(amp_svg_path, format="svg", bbox_inches="tight")
plt.close(fig_amp)
print("[SVG] amplitude compare:", amp_svg_path)
del fig_amp


# --- Interaktive HTML immer erzeugen (mit UP-Schattierung) ---
export_interactive_lfp_html(
    BASE_TAG, SAVE_DIR, time_s,
    main_channel_uV if (HTML_IN_uV and main_channel_uV is not None) else main_channel,
    pulse_times_1=pulse_times_1, pulse_times_2=pulse_times_2,
    up_spont=(Spontaneous_UP, Spontaneous_DOWN),
    up_trig=(Pulse_triggered_UP, Pulse_triggered_DOWN),
    up_assoc=(Pulse_associated_UP, Pulse_associated_DOWN),
    limit_to_last_pulse=False,
    title=f"{BASE_TAG} — Main LFP (interaktiv)",
    y_label=("LFP (µV)" if HTML_IN_uV else f"LFP ({UNIT_LABEL})")
)





# Extras für Plots
pulse_windows = extract_upstate_windows(Pulse_triggered_UP, main_channel[None, :], dt, window_s=1.0)
spont_windows = extract_upstate_windows(Spontaneous_UP, main_channel[None, :], dt, window_s=1.0)
freqs = spont_mean = pulse_mean = p_vals = None
try:
    freqs, spont_mean, pulse_mean, p_vals = compare_spectra(
        pulse_windows, spont_windows, dt, ignore_start_s=0.3
    )
except Exception as e:
    print("[WARN] spectra compare skipped:", e)


n_time = LFP_array_good.shape[1]

# --- Onsets aus UP/DOWN-Listen (zeitlich stabiler als Peaks) ---
Spon_Onsets = _up_onsets(Spontaneous_UP,       Spontaneous_DOWN)
Trig_Onsets = _up_onsets(Pulse_triggered_UP,   Pulse_triggered_DOWN)

# Gültigkeitsgrenzen
Spon_Onsets = Spon_Onsets[(Spon_Onsets >= 0) & (Spon_Onsets < LFP_array_good.shape[1])]
Trig_Onsets = Trig_Onsets[(Trig_Onsets >= 0) & (Trig_Onsets < LFP_array_good.shape[1])]

CSD_spont = CSD_trig = None

CSD_PRE_DESIRED  = 0.5   # 0.5 s vor Onset
CSD_POST_DESIRED = 0.5   # 0.5 s nach Onset

if NUM_CHANNELS_GOOD >= 7 and (Spon_Onsets.size >= 3 or Trig_Onsets.size >= 3):
    try:
        if Spon_Onsets.size >= 3:
            CSD_spont = Generate_CSD_mean_from_onsets(
                Spon_Onsets,
                LFP_array_good,
                dt,
                pre_s=CSD_PRE_DESIRED,
                post_s=CSD_POST_DESIRED,
                clip_to_down=Spontaneous_DOWN,   # optional
            )

        if Trig_Onsets.size >= 3:
            CSD_trig = Generate_CSD_mean_from_onsets(
                Trig_Onsets,
                LFP_array_good,
                dt,
                pre_s=CSD_PRE_DESIRED,
                post_s=CSD_POST_DESIRED,
                clip_to_down=Pulse_triggered_DOWN,
            )

        align_pre_s  = CSD_PRE_DESIRED
        align_post_s = CSD_POST_DESIRED

    except Exception as e:
        print("[WARN] CSD generation failed:", e)
        CSD_spont = CSD_trig = None
        align_pre_s  = CSD_PRE_DESIRED
        align_post_s = CSD_POST_DESIRED
else:
    print(f"[INFO] CSD skipped: channels={NUM_CHANNELS_GOOD}, "
          f"spon_onsets={Spon_Onsets.size}, trig_onsets={Trig_Onsets.size}")
    align_pre_s  = CSD_PRE_DESIRED
    align_post_s = CSD_POST_DESIRED


# CSD_spont = CSD_trig = None

# # dein Wunschfenster
# CSD_PRE_DESIRED  = 0.5   # 0.5 s vor Onset
# CSD_POST_DESIRED = 0.5   # 0.5 s nach Onset

# if NUM_CHANNELS_GOOD >= 7 and (Spon_Onsets.size >= 3 or Trig_Onsets.size >= 3):
#     try:
#         # 1) Fenster anhand der SPONT-Onsets bestimmen
#         CSD_spont, pre_eff_s, post_eff_s = Generate_CSD_mean_from_onsets(
#             Spon_Onsets,
#             LFP_array_good,
#             dt,
#             pre_s_desired=CSD_PRE_DESIRED,
#             post_s_desired=CSD_POST_DESIRED,
#             down_indices=Spontaneous_DOWN
#         )

#         # 2) Das gleiche (gekürzte) Fenster für TRIG verwenden,
#         #    damit Vergleich fair ist
#         if Trig_Onsets.size >= 3:
#             CSD_trig, _, _ = Generate_CSD_mean_from_onsets(
#                 Trig_Onsets,
#                 LFP_array_good,
#                 dt,
#                 pre_s_desired=pre_eff_s,    # schon gekürzt
#                 post_s_desired=post_eff_s,
#                 down_indices=Pulse_triggered_DOWN
#             )

#         # 3) Align-Variablen für Plot-Achsen
#         align_pre_s  = pre_eff_s
#         align_post_s = post_eff_s

#     except Exception as e:
#         print("[WARN] CSD generation failed:", e)
#         CSD_spont = CSD_trig = None
# else:
#     print(f"[INFO] CSD skipped: channels={NUM_CHANNELS_GOOD}, spon_onsets={Spon_Onsets.size}, trig_onsets={Trig_Onsets.size}")

# CSD_spont = CSD_trig = None

# # Zeitfenster für CSD relativ zum UP-Onset
# CSD_PRE_S  = 0.3   # 0.3 s vor Onset
# CSD_POST_S = 1.5   # 1.5 s nach Onset

# if NUM_CHANNELS_GOOD >= 7:
#     try:
#         if Spon_Onsets.size >= 3:
#             CSD_spont = Generate_CSD_mean_from_onsets(
#                 Spon_Onsets,
#                 LFP_array_good,
#                 dt,
#                 pre_s=CSD_PRE_S,
#                 post_s=CSD_POST_S,
#                 clip_to_down=Spontaneous_DOWN,   # optional; sonst None
#             )
#         if Trig_Onsets.size >= 3:
#             CSD_trig = Generate_CSD_mean_from_onsets(
#                 Trig_Onsets,
#                 LFP_array_good,
#                 dt,
#                 pre_s=CSD_PRE_S,
#                 post_s=CSD_POST_S,
#                 clip_to_down=Pulse_triggered_DOWN,  # optional
#             )
#     except Exception as e:
#         print("[WARN] CSD skipped:", e)
#         CSD_spont = CSD_trig = None
# else:
#     print(f"[INFO] CSD skipped: channels={NUM_CHANNELS_GOOD}, spon_onsets={Spon_Onsets.size}, trig_onsets={Trig_Onsets.size}")



# CSD_spont = CSD_trig = None
# if NUM_CHANNELS_GOOD >= 7 and (Spon_Onsets.size >= 3 or Trig_Onsets.size >= 3):
#     try:
#         if Spon_Onsets.size >= 3:
#             CSD_spont = Generate_CSD_mean(Spon_Onsets, LFP_array_good, dt)
#         if Trig_Onsets.size >= 3:
#             CSD_trig  = Generate_CSD_mean(Trig_Onsets,  LFP_array_good, dt)
#     except Exception as e:
#         print("[WARN] CSD skipped:", e)
# else:
#     print(f"[INFO] CSD skipped: channels={NUM_CHANNELS_GOOD}, spon_onsets={Spon_Onsets.size}, trig_onsets={Trig_Onsets.size}")


# USE_EQUAL_N_FOR_PLOT = True
# spon_all = np.asarray(Spon_Onsets, int)
# trig_all = np.asarray(Trig_Onsets,  int)
# m = int(min(spon_all.size, trig_all.size))

# if USE_EQUAL_N_FOR_PLOT and m >= 3:
#     spon_sel = np.sort(_even_subsample(spon_all, m))
#     trig_sel = np.sort(_even_subsample(trig_all, m))
#     try:
#         CSD_spont_eq = Generate_CSD_mean(spon_sel, LFP_array_good, dt)
#         CSD_trig_eq  = Generate_CSD_mean(trig_sel,  LFP_array_good, dt)
#         CSD_spont_plot = CSD_spont_eq if 'CSD_spont_eq' in locals() else CSD_spont
#         CSD_trig_plot  = CSD_trig_eq  if 'CSD_trig_eq'  in locals() else CSD_trig
# # …und diese in layout_rows an CSD_compare_side_by_side_ax übergeben.

#         print(f"[Equal-N] using m={m} events (spont:{spon_all.size}→{m}, trig:{trig_all.size}→{m})")
#     except Exception as e:
#         print("[WARN] Equal-N CSD failed:", e)
#         CSD_spont_plot = CSD_spont_eq if 'CSD_spont_eq' in locals() else CSD_spont
#         CSD_trig_plot  = CSD_trig_eq  if 'CSD_trig_eq'  in locals() else CSD_trig
# # …und diese in layout_rows an CSD_compare_side_by_side_ax übergeben.

# else:
#     CSD_spont_plot = CSD_spont_eq if 'CSD_spont_eq' in locals() else CSD_spont
#     CSD_trig_plot  = CSD_trig_eq  if 'CSD_trig_eq'  in locals() else CSD_trig
#     # …und diese in layout_rows an CSD_compare_side_by_side_ax übergeben.



# 0) Event-Zahlen + Index-Gültigkeit
_check_peak_indices("Spon_Peaks", Up.get("Spon_Peaks", []), LFP_array_good.shape[1])
_check_peak_indices("Trig_Peaks", Up.get("Trig_Peaks", []), LFP_array_good.shape[1])

# 1) CSD-Grundstats
_nan_stats("CSD_spont", CSD_spont)
_nan_stats("CSD_trig",  CSD_trig)
print(f"[DIAG] RMS CSD: spont={_rms(CSD_spont):.4g}, trig={_rms(CSD_trig):.4g}")

# # 2) Gleiche Event-Anzahl erzwingen (fairer Vergleich)
# Spon_Peaks_eq = Up.get("Spon_Peaks", np.array([], float))
# Trig_Peaks_eq = Up.get("Trig_Peaks", np.array([], float))
# m = min(len(Spon_Peaks_eq), len(Trig_Peaks_eq))
# if m >= 3:
#     Spon_Peaks_eq = np.asarray(Spon_Peaks_eq, int)[:m]
#     Trig_Peaks_eq  = np.asarray(Trig_Peaks_eq,  int)[:m]
#     try:
#         CSD_spont_eq = Generate_CSD_mean(Spon_Peaks_eq, LFP_array_good, dt)
#         CSD_trig_eq  = Generate_CSD_mean(Trig_Peaks_eq,  LFP_array_good, dt)
#         print(f"[DIAG] RMS CSD (equal N={m}): spont={_rms(CSD_spont_eq):.4g}, trig={_rms(CSD_trig_eq):.4g}")
#         _nan_stats("CSD_spont_eq", CSD_spont_eq)
#         _nan_stats("CSD_trig_eq",  CSD_trig_eq)
#     except Exception as e:
#         print("[DIAG][WARN] Equal-N CSD failed:", e)
# else:
#     print(f"[DIAG] Equal-N Skip: not enough events (spon={len(Spon_Peaks_eq)}, trig={len(Trig_Peaks_eq)})")

# 3) Prüfen, ob Cropping Spontan-Events stark reduziert
print(f"[DIAG] Cropped time_s: {time_s[0]:.3f}..{time_s[-1]:.3f} s, pulses p1={len(pulse_times_1)}, p2={len(pulse_times_2)}")
print(f"[DIAG] Counts: sponUP={len(Spontaneous_UP)}, trigUP={len(Pulse_triggered_UP)}, assocUP={len(Pulse_associated_UP)}")

# 4) Gleiche Zeitfenster/Alignment sicher? (pre/post)
print(f"[DIAG] align_pre={align_pre_s:.3f}s, align_post={align_post_s:.3f}s, dt={dt:.6f}s")

# ====== ENDE DIAGNOSE ======





# nach compare_spectra
if freqs is not None and spont_mean is not None:
    pd.DataFrame({"freq": freqs, "power": spont_mean}).to_csv(
        os.path.join(SAVE_DIR, "spectrum_spont.csv"), index=False)
if freqs is not None and pulse_mean is not None:
    pd.DataFrame({"freq": freqs, "power": pulse_mean}).to_csv(
        os.path.join(SAVE_DIR, "spectrum_trig.csv"), index=False)




# 0) Event-Zahlen + Index-Gültigkeit
_check_peak_indices("Spon_Peaks", Up.get("Spon_Peaks", []), LFP_array_good.shape[1])
_check_peak_indices("Trig_Peaks", Up.get("Trig_Peaks", []), LFP_array_good.shape[1])

# 1) CSD-Grundstats
_nan_stats("CSD_spont", CSD_spont)
_nan_stats("CSD_trig",  CSD_trig)
print(f"[DIAG] RMS CSD: spont={_rms(CSD_spont):.4g}, trig={_rms(CSD_trig):.4g}")

# # 2) Gleiche Event-Anzahl erzwingen (fairer Vergleich)
# Spon_Peaks_eq = Up.get("Spon_Peaks", np.array([], float))
# Trig_Peaks_eq = Up.get("Trig_Peaks", np.array([], float))
# m = min(len(Spon_Peaks_eq), len(Trig_Peaks_eq))
# if m >= 3:  # brauchbare Mindestzahl
#     Spon_Peaks_eq = np.asarray(Spon_Peaks_eq, int)[:m]
#     Trig_Peaks_eq = np.asarray(Trig_Peaks_eq, int)[:m]
#     try:
#         CSD_spont_eq = Generate_CSD_mean(Spon_Peaks_eq, LFP_array_good, dt)
#         CSD_trig_eq  = Generate_CSD_mean(Trig_Peaks_eq,  LFP_array_good, dt)
#         print(f"[DIAG] RMS CSD (equal N={m}): spont={_rms(CSD_spont_eq):.4g}, trig={_rms(CSD_trig_eq):.4g}")
#         _nan_stats("CSD_spont_eq", CSD_spont_eq)
#         _nan_stats("CSD_trig_eq",  CSD_trig_eq)
#     except Exception as e:
#         print("[DIAG][WARN] Equal-N CSD failed:", e)
# else:
#     print(f"[DIAG] Equal-N Skip: not enough events (spon={len(Spon_Peaks_eq)}, trig={len(Trig_Peaks_eq)})")

# 3) Prüfen, ob Cropping Spontan-Events stark reduziert
print(f"[DIAG] Cropped time_s: {time_s[0]:.3f}..{time_s[-1]:.3f} s, pulses p1={len(pulse_times_1)}, p2={len(pulse_times_2)}")
print(f"[DIAG] Counts: sponUP={len(Spontaneous_UP)}, trigUP={len(Pulse_triggered_UP)}, assocUP={len(Pulse_associated_UP)}")

print(f"[DIAG] align_pre={align_pre_s:.3f}s, align_post={align_post_s:.3f}s, dt={dt:.6f}s")



#wird nicht mehr aufgerufen! CHECK!
# Ax-fähige Mini-Plotter 
def lfp_overview_with_labels_ax(
    time_s,
    main_channel,
    Spontaneous_UP, Spontaneous_DOWN,
    Pulse_triggered_UP, Pulse_triggered_DOWN,
    Pulse_associated_UP, Pulse_associated_DOWN,
    pulse_times_1=None, pulse_times_2=None,
    Spon_Peaks=None, Trig_Peaks=None,
    ax=None,
    xlim=None
):


    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))
    else:
        fig = ax.figure

    # 1) Rohsignal
    ax.plot(time_s, main_channel, lw=0.8, label="LFP (main)")

    # y-Limits vom Signal holen und für Puls-Linien verwenden
    y0, y1 = ax.get_ylim()

    # 2) Helper zum sicheren Zeichnen
    def _shade_intervals(up_idx, down_idx, color, label):
        up_idx   = np.asarray(up_idx,   dtype=int)
        down_idx = np.asarray(down_idx, dtype=int)
        m = min(len(up_idx), len(down_idx))
        if m == 0: 
            return
        up_idx, down_idx = up_idx[:m], down_idx[:m]
        # sortiert nach Zeit
        order = np.argsort(time_s[up_idx])
        up_idx, down_idx = up_idx[order], down_idx[order]
        for i, j in zip(up_idx, down_idx):
            i = max(0, min(i, len(time_s)-1))
            j = max(0, min(j, len(time_s)-1))
            if j <= i:  # skip bad intervals
                continue
            ax.axvspan(time_s[i], time_s[j], color=color, alpha=0.22, lw=0, label=label)
            label = None  # nur einmal in Legende

    # 3) UP-Intervalle einfärben
    _shade_intervals(Spontaneous_UP,       Spontaneous_DOWN,       "#2ca02c", "UP spontaneous")
    _shade_intervals(Pulse_triggered_UP,   Pulse_triggered_DOWN,   "#ff7f0e", "UP triggered")
    _shade_intervals(Pulse_associated_UP,  Pulse_associated_DOWN,  "#1f77b4", "UP associated")

    # 4) Pulsezeiten, falls vorhanden, in aktuelle y-Limits zeichnen
    if pulse_times_1 is not None and len(pulse_times_1):
        t1 = np.asarray(pulse_times_1, float)
        # Bei extrem vielen Pulsen ausdünnen (max ~800 Linien)
        if t1.size > 800:
            step = int(np.ceil(t1.size / 800))
            t1 = t1[::step]
        ax.vlines(t1, y0, y1, lw=0.6, alpha=0.35, linestyles=":", label="Pulse 1", zorder=1)

    if pulse_times_2 is not None and len(pulse_times_2):
        t2 = np.asarray(pulse_times_2, float)
        if t2.size > 800:
            step = int(np.ceil(t2.size / 800))
            t2 = t2[::step]
        ax.vlines(t2, y0, y1, lw=0.6, alpha=0.35, linestyles="--", label="Pulse 2", zorder=1)

    # 5) optionale Peak-Marker
    if Spon_Peaks is not None and len(Spon_Peaks):
        sp = np.asarray(Spon_Peaks, dtype=int)
        sp = sp[(sp >= 0) & (sp < len(time_s))]
        ax.plot(time_s[sp], main_channel[sp], "o", ms=3, alpha=0.6, label="Spont peaks")
    if Trig_Peaks is not None and len(Trig_Peaks):
        tp = np.asarray(Trig_Peaks, dtype=int)
        tp = tp[(tp >= 0) & (tp < len(time_s))]
        ax.plot(time_s[tp], main_channel[tp], "x", ms=3, alpha=0.7, label="Trig peaks")

    # 6) Achsen/Kleinkram
    ax.set_xlabel("Zeit (s)")
    ax.set_ylabel(f"LFP ({UNIT_LABEL})")
    ax.set_title("Main channel with UP labels (spont / triggered / associated)")
    if xlim is not None:
        ax.set_xlim(*xlim)
    # Legende: doppelte Labels vermeiden
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    if uniq:
        ax.legend(uniq.values(), uniq.keys(), loc="upper right", fontsize=8, framealpha=0.9)

    return fig

print("[DEBUG] time range:", float(time_s[0]), "->", float(time_s[-1]))
if len(pulse_times_1):
    print("[DEBUG] p1 first/last:", float(pulse_times_1[0]), float(pulse_times_1[-1]), "count:", len(pulse_times_1))
if len(pulse_times_2):
    print("[DEBUG] p2 first/last:", float(pulse_times_2[0]), float(pulse_times_2[-1]), "count:", len(pulse_times_2))

def spont_up_mean_ax(main_channel, time_s, dt, Spon_Peaks, ax=None):
    import numpy as np
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8,3))
    else:
        fig = ax.figure

    half = int(0.5/dt)
    traces = []
    for pk in Spon_Peaks:
        if np.isnan(pk):
            continue
        pk = int(pk)
        s = pk - half
        e = pk + half
        if s >= 0 and e <= len(main_channel):
            traces.append(main_channel[s:e])

    if not traces:
        ax.text(0.5, 0.5, "no spontaneous peaks",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    tr = np.vstack(traces)
    n  = tr.shape[0]

    m  = np.nanmean(tr, 0)
    se = np.nanstd(tr, 0) / np.sqrt(n)
    t  = (np.arange(-half, half) * dt)

    ax.plot(t, m, lw=2)
    ax.fill_between(t, m-se, m+se, alpha=0.3)
    ax.axvline(0, ls="--", lw=1)

    # ⬅️ Hier wird n in den Plot geschrieben
    ax.text(
        0.98, 0.90,
        f"n = {n}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", fc="white", alpha=0.7)
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"LFP ({UNIT_LABEL})")
    ax.set_title("Spontaneous UP – mean ± SEM")

    return fig




def Total_power_plot_ax(Spect_dat, Total_power=None, ax=None, title="Gesamtleistung 0.1–150 Hz"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,3))
    else:
        fig = ax.figure
    try:
        t_vec = np.asarray(Spect_dat[1])
        if Total_power is None:
            Sxx_dB = np.asarray(Spect_dat[0])
            y = np.sum(10**(np.clip(Sxx_dB, -100, 100)/10.0), axis=0)
        else:
            y = np.asarray(Total_power)
        m = min(len(t_vec), len(y))
        if m == 0:
            raise ValueError("Empty power or t_vec")
        ax.plot(t_vec[:m], y[:m], lw=1.5)
        ax.set_xlabel("Zeit (s)")
        ax.set_ylabel(f"Power ({UNIT_LABEL})")
        ax.set_title(title)
    except Exception as e:
        ax.text(0.5, 0.5, f"Power-Plot fehlgeschlagen:\n{e}", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    return fig
def CSD_compare_side_by_side_ax(
    CSD_spont, CSD_trig, dt,
    *, z_mm=None,
    align_pre=0.5, align_post=0.5,
    cmap="Spectral_r",
    sat_pct=95,          # etwas konservativer
    interp="bilinear",
    contours=False,
    ax=None,
    title="CSD (Spon vs. Trig)",
    norm_mode="linear",  # default: linear für "paper look"
    linthresh_frac=0.03,
    flip_y=True,
    prefer_imshow=True,
    rasterize_csd=True,
    pcolor_shading="nearest",
    vmax_abs=None,
    **_unused
):
    from matplotlib.colors import TwoSlopeNorm, SymLogNorm

    # --- Robustheitschecks ---
    ok = lambda A: (isinstance(A, _np.ndarray) and A.ndim == 2 and A.size > 0)
    if not (ok(CSD_spont) and ok(CSD_trig)):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 3))
        else:
            fig = ax.figure
        ax.axis("off")
        ax.text(0.5, 0.5, "no CSD", ha="center", va="center", transform=ax.transAxes)
        return fig

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    else:
        fig = ax.figure

    # --- 1) In float umwandeln & 2D glätten (time × depth) ---
    CSD_sp = _np.asarray(CSD_spont, float)
    CSD_tr = _np.asarray(CSD_trig,  float)

    # sigma=(depth, time) – hier moderat, nicht zu stark
    CSD_sp_sm = gaussian_filter(CSD_sp, sigma=(2.0, 3.0))
    CSD_tr_sm = gaussian_filter(CSD_tr, sigma=(2.0, 3.0))

    # --- 2) Optional: Tiefe invertieren (superficial oben) ---
    if flip_y:
        CSD_sp_plot = CSD_sp_sm[::-1, :]
        CSD_tr_plot = CSD_tr_sm[::-1, :]
        if z_mm is not None:
            z_plot = _np.asarray(z_mm)[::-1]
        else:
            z_plot = None
    else:
        CSD_sp_plot = CSD_sp_sm
        CSD_tr_plot = CSD_tr_sm
        z_plot = _np.asarray(z_mm) if z_mm is not None else None

    # --- 3) Gemeinsame Skala (linear, zero-centered) ---
    stack = _np.concatenate([
        _np.abs(CSD_sp_plot).ravel(),
        _np.abs(CSD_tr_plot).ravel()
    ])
    stack = stack[_np.isfinite(stack)]
    if stack.size == 0:
        vmax = 1.0
    else:
        vmax = float(_np.nanpercentile(stack, sat_pct))
        if not _np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0

        if vmax_abs is not None:
            vmax = float(vmax_abs)

    if norm_mode == "linear":
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    else:
        # fallback auf symlog, falls du das explizit willst
        lt = max(float(linthresh_frac) * vmax, 1e-12)
        norm = SymLogNorm(linthresh=lt, vmin=-vmax, vmax=vmax)

    # --- 4) Zeitachse: centers & edges ---
    n_t = CSD_sp_plot.shape[1]
    t_centers = _np.linspace(-float(align_pre), float(align_post), n_t)

    def _edges_from_centers(c):
        c = _np.asarray(c, float)
        if c.size == 1:
            d = 0.5
            return _np.array([c[0]-d, c[0]+d], float)
        mid = 0.5 * (c[:-1] + c[1:])
        first = c[0] - (mid[0] - c[0])
        last  = c[-1] + (c[-1] - mid[-1])
        return _np.concatenate([[first], mid, [last]])

    t_edges = _edges_from_centers(t_centers)

    # --- 5) Subplots + Colorbar-Achse anlegen ---
    ax_left  = ax.inset_axes([0.00, 0.0, 0.45, 1.0])
    ax_right = ax.inset_axes([0.50, 0.0, 0.45, 1.0])
    cax      = ax.inset_axes([0.955, 0.1, 0.02, 0.8])
    ax.set_axis_off()

    artists = []

    # --- 6) Imshow vs pcolormesh ---
    use_imshow = (z_plot is None) or prefer_imshow

    if use_imshow:
        if z_plot is not None:
            z_edges = _edges_from_centers(z_plot)
            y0, y1 = float(z_edges[0]), float(z_edges[-1])
        else:
            y0, y1 = 0.0, float(CSD_sp_plot.shape[0] - 1)

        extent = [-float(align_pre), float(align_post), y0, y1]

        imL = ax_left.imshow(
            CSD_sp_plot,
            aspect="auto",
            origin="upper",
            extent=extent,
            cmap=cmap,
            norm=norm,
            interpolation=interp,
        )
        imR = ax_right.imshow(
            CSD_tr_plot,
            aspect="auto",
            origin="upper",
            extent=extent,
            cmap=cmap,
            norm=norm,
            interpolation=interp,
        )

        for a in (ax_left, ax_right):
            a.set_xlabel("Zeit (s)")
        ax_left.set_ylabel("Tiefe (mm)" if z_plot is not None else "Tiefe (arb.)")

        artists.extend([imL, imR])

    else:
        if z_plot is None:
            z_plot = _np.arange(CSD_sp_plot.shape[0], dtype=float)
        z_edges = _edges_from_centers(z_plot)

        imL = ax_left.pcolormesh(
            t_edges, z_edges, CSD_sp_plot,
            shading=pcolor_shading,
            cmap=cmap,
            norm=norm,
        )
        imR = ax_right.pcolormesh(
            t_edges, z_edges, CSD_tr_plot,
            shading=pcolor_shading,
            cmap=cmap,
            norm=norm,
        )

        for im in (imL, imR):
            try:
                im.set_antialiased(False)
                im.set_edgecolor("face")
                im.set_linewidth(0.0)
            except Exception:
                pass

        for a in (ax_left, ax_right):
            a.set_xlim(t_edges[0], t_edges[-1])
            a.set_ylim(z_edges[0], z_edges[-1])
            a.set_xlabel("Zeit (s)")
        ax_left.set_ylabel("Tiefe (mm)")

        artists.extend([imL, imR])

    # --- 7) Optional rasterisieren für hübsche PDFs ---
    if rasterize_csd:
        for im in artists:
            try:
                im.set_rasterized(True)
            except Exception:
                pass

    # --- 8) Titles & Colorbar ---
    ax_left.set_title("Spontaneous", fontsize=10)
    ax_right.set_title("Triggered", fontsize=10)
    cb = fig.colorbar(artists[-1], cax=cax)
    cb.set_label("CSD (a.u.)", rotation=90)

    ax_left.set_title(title, fontsize=11, pad=22)
    return fig



from scipy.ndimage import gaussian_filter  # falls oben noch nicht importiert

def CSD_single_panel_ax(
    CSD,
    dt,
    *,
    z_mm=None,
    align_pre=0.5,
    align_post=0.5,
    ax=None,
    title="CSD",
    cmap="Spectral_r",      # klassisches Blau-Rot
    sat_pct=95,
    smooth_sigma=(2.0, 3.0),   # (depth, time) Glättung
    flip_y=True
):
    """
    Paper-freundliche Darstellung eines einzelnen CSD:
    - glatte Darstellung
    - lineare, zero-centered Skala
    - klassisches Blau-Rot-Layout
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.figure

    if CSD is None or not isinstance(CSD, np.ndarray) or CSD.ndim != 2 or CSD.size == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "no CSD", ha="center", va="center", transform=ax.transAxes)
        return fig

    # 1) float + optional smoothing
    A = np.asarray(CSD, float)
    if smooth_sigma is not None:
        A = gaussian_filter(A, sigma=smooth_sigma)

    # 2) ggf. Tiefe invertieren (superficial oben)
    if flip_y:
        A = A[::-1, :]
        if z_mm is not None:
            z_plot = np.asarray(z_mm)[::-1]
        else:
            z_plot = None
    else:
        z_plot = np.asarray(z_mm) if z_mm is not None else None

    # 3) robuste Skala
    vals = np.abs(A[np.isfinite(A)])
    if vals.size == 0:
        vmax = 1.0
    else:
        vmax = float(np.nanpercentile(vals, sat_pct))
        if vmax <= 0 or not np.isfinite(vmax):
            vmax = 1.0
    vmin = -vmax
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    # 4) Zeitachse
    n_t = A.shape[1]
    t = np.linspace(-float(align_pre), float(align_post), n_t)

    # 5) extent für imshow
    if z_plot is not None:
        z0, z1 = float(z_plot[0]), float(z_plot[-1])
    else:
        z0, z1 = 0.0, float(A.shape[0] - 1)

    im = ax.imshow(
        A,
        aspect="auto",
        origin="upper",
        extent=[t[0], t[-1], z0, z1],
        cmap=cmap,
        norm=norm,
        interpolation="bilinear",   # weich, „paper“-Look
    )

    ax.set_xlabel("Zeit (s)")
    ax.set_ylabel("Tiefe (mm)" if z_plot is not None else "Tiefe (arb.)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="CSD (a.u.)")

    # optional: dünne vertikale Linie bei 0 s
    ax.axvline(0, color="k", lw=0.5, alpha=0.7)

    return fig



# def CSD_compare_side_by_side_ax(
#     CSD_spont, CSD_trig, dt,
#     *, z_mm=None,
#     align_pre=0.5, align_post=0.5,
#     cmap="turbo", sat_pct=98, interp="bilinear",
#     contours=False, ax=None, title="CSD (Spon vs. Trig)",
#     norm_mode="symlog",          # "symlog" | "linear"
#     linthresh_frac=0.03,         # Anteil von vmax für linearen Bereich
#     flip_y=True,                 # Kanal 0 oben?
#     # --- NEU: Anti-Patchiness/Export-Kontrolle ---
#     prefer_imshow=True,          # auch bei z_mm imshow benutzen (weiche Übergänge)
#     rasterize_csd=True,          # CSD-Artists für PDF/SVG rasterisieren
#     pcolor_shading="nearest",    # "nearest" reduziert Seams ggü. "auto"
#     **_unused
# ):
    
#     from matplotlib.colors import TwoSlopeNorm, SymLogNorm

#     def _edges_from_centers(c):
#         c = _np.asarray(c, float)
#         if c.size == 1:
#             d = 0.5
#             return _np.array([c[0]-d, c[0]+d], float)
#         mid = 0.5*(c[:-1]+c[1:])
#         first = c[0] - (mid[0]-c[0])
#         last  = c[-1] + (c[-1]-mid[-1])
#         return _np.concatenate([[first], mid, [last]])
    
    


#     def _robust_vmax(A, pct):
#         A = _np.asarray(A, float)
#         A = _np.abs(A[_np.isfinite(A)])
#         return float(_np.nanpercentile(A, pct)) if A.size else _np.nan

#     def _mask_t0(A, align_pre, align_post, ms0=300.0):
#         import numpy as np
#         A = np.array(A, float, copy=True)
#         n_t = A.shape[1]
#         t = np.linspace(-align_pre, align_post, n_t)
#         m = (t >= 0) & (t <= ms0/1000.0)
#         A[:, m] = np.nan
#         return A

#     CSD_sp_for_scale = _mask_t0(CSD_spont, align_pre, align_post, ms0=15)
#     CSD_tr_for_scale = _mask_t0(CSD_trig,  align_pre, align_post, ms0=15)
#     v_sp = _robust_vmax(CSD_sp_for_scale, sat_pct)
#     v_tr = _robust_vmax(CSD_tr_for_scale,  sat_pct)

#     ok = lambda A: (isinstance(A, _np.ndarray) and A.ndim == 2 and A.size > 0)
#     if not (ok(CSD_spont) and ok(CSD_trig)):
#         if ax is None: fig, ax = plt.subplots(figsize=(8,3))
#         else:          fig = ax.figure
#         ax.axis("off"); ax.text(0.5,0.5,"no CSD", ha="center", va="center", transform=ax.transAxes)
#         return fig

#     if ax is None: fig, ax = plt.subplots(figsize=(8,3))
#     else:          fig = ax.figure

#     # === Gemeinsame Skala ===
#     v_sp = _robust_vmax(CSD_spont, sat_pct)
#     v_tr = _robust_vmax(CSD_trig,  sat_pct)
#     vals = [v for v in (v_sp, v_tr) if _np.isfinite(v) and v > 0]
#     if len(vals) == 2: vmax_joint = float(_np.sqrt(vals[0]*vals[1]))
#     elif len(vals) == 1: vmax_joint = float(vals[0])
#     else: vmax_joint = 1.0
#     vmin_joint = -vmax_joint

#     # === Normierung ===
#     if norm_mode == "linear":
#         norm = TwoSlopeNorm(vmin=vmin_joint, vcenter=0.0, vmax=vmax_joint)
#     else:
#         lt = max(float(linthresh_frac) * vmax_joint, 1e-12)
#         norm = SymLogNorm(linthresh=lt, vmin=vmin_joint, vmax=vmax_joint)

#     # === Zeit-Achse ===
#     n_t = CSD_spont.shape[1]
#     t_centers = _np.linspace(-float(align_pre), float(align_post), n_t)
#     t_edges   = _edges_from_centers(t_centers)

#     ax_left  = ax.inset_axes([0.00, 0.0, 0.45, 1.0])
#     ax_right = ax.inset_axes([0.50, 0.0, 0.45, 1.0])
#     cax      = ax.inset_axes([0.955, 0.1, 0.02, 0.8])
#     ax.set_axis_off()

#     # === Darstellung ===

#     from scipy.ndimage import gaussian_filter

#     # 2D-Smoothing: (depth_sigma, time_sigma)
#     CSD_sp_sm  = gaussian_filter(CSD_spont, sigma=(1.2, 1.2))
#     CSD_tr_sm  = gaussian_filter(CSD_trig,  sigma=(1.2, 1.2))

#     use_imshow = (z_mm is None) or prefer_imshow
#     artists = []

#     if use_imshow:
#         # y-Extent aus z_mm (falls vorhanden) ableiten
#         if z_mm is not None:
#             z_edges = _edges_from_centers(_np.asarray(z_mm, float))
#             y0, y1 = float(z_edges[0]), float(z_edges[-1])
#         else:
#             y0, y1 = 0.0, float((CSD_spont.shape[0]-1))

#         extent = [-float(align_pre), float(align_post), y0, y1]
#         imL = ax_left.imshow(CSD_sp_sm, aspect="auto",
#                              origin="upper" if flip_y else "lower",
#                              extent=extent, cmap=cmap, norm=norm,
#                              interpolation=interp)
#         imR = ax_right.imshow(CSD_tr_sm,  aspect="auto",
#                               origin="upper" if flip_y else "lower",
#                               extent=extent, cmap=cmap, norm=norm,
#                               interpolation=interp)
#         for a in (ax_left, ax_right):
#             a.set_xlabel("Zeit (s)")
#             if flip_y:
#                 a.invert_yaxis()
#         ax_left.set_ylabel("Tiefe (mm)" if z_mm is not None else "Tiefe (arb.)")
#         artists.extend([imL, imR])

#     else:
#         # pcolormesh mit „seam“-freundlichen Settings
#         z_mm = _np.asarray(z_mm, float)
#         if z_mm.size != CSD_spont.shape[0]:
#             raise ValueError(f"z_mm hat {z_mm.size} Werte, CSD hat {CSD_spont.shape[0]} Zeilen.")
#         z_edges = _edges_from_centers(z_mm)

#         imL = ax_left.pcolormesh(t_edges, z_edges, CSD_spont,
#                                  shading=pcolor_shading, cmap=cmap, norm=norm)
#         imR = ax_right.pcolormesh(t_edges, z_edges, CSD_trig,
#                                   shading=pcolor_shading, cmap=cmap, norm=norm)

#         # Anti-aliasing/Kanten entschärfen
#         for im in (imL, imR):
#             try:
#                 im.set_antialiased(False)
#                 im.set_edgecolor('face')
#                 im.set_linewidth(0.0)
#             except Exception:
#                 pass

#         for a in (ax_left, ax_right):
#             a.set_xlim(t_edges[0], t_edges[-1])
#             a.set_ylim(z_edges[0], z_edges[-1])
#             a.set_xlabel("Zeit (s)")
#             if flip_y:
#                 a.invert_yaxis()
#         ax_left.set_ylabel("Tiefe (mm)")
#         artists.extend([imL, imR])

#     # Optional: Rasterisieren für saubere PDF/SVG
#     if rasterize_csd:
#         for im in artists:
#             try:
#                 im.set_rasterized(True)
#             except Exception:
#                 pass

#     # Titel/Colorbar
#     ax_left.set_title("Spontaneous", fontsize=10)
#     ax_right.set_title("Triggered", fontsize=10)
#     cb = fig.colorbar(artists[-1], cax=cax)
#     cb.set_label("CSD (a.u.)", rotation=90)
#     ax_left.set_title(title, fontsize=11, pad=22)
#     return fig


# def CSD_compare_side_by_side_ax(
#     CSD_spont, CSD_trig, dt,
#     *, dz_um=100.0, align_pre=0.5, align_post=0.5,
#     cmap="viridis", sat_pct=95, interp="bilinear",  
#     contours=False, n_contours=10,                
#     ax=None, title="CSD (Spon vs. Trig)"
# ):
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from matplotlib.colors import TwoSlopeNorm

#     if ax is None:
#         fig, ax = plt.subplots(figsize=(8, 3))
#     else:
#         fig = ax.figure

#     # Robustheit
#     if not (isinstance(CSD_spont, np.ndarray) and CSD_spont.ndim == 2 and
#             isinstance(CSD_trig,  np.ndarray) and CSD_trig.ndim  == 2):
#         ax.text(0.5, 0.5, "no CSD", ha="center", va="center", transform=ax.transAxes)
#         ax.set_axis_off()
#         return fig

#     # Gemeinsame Dynamik (robustes Clipping) + Zero Center
#     stack = np.concatenate([CSD_spont.ravel(), CSD_trig.ravel()])
#     stack = stack[np.isfinite(stack)]
#     if stack.size == 0:
#         ax.text(0.5, 0.5, "invalid CSD values", ha="center", va="center", transform=ax.transAxes)
#         ax.set_axis_off()
#         return fig
#     vmax = float(np.nanpercentile(np.abs(stack), sat_pct))
#     if vmax <= 0 or not np.isfinite(vmax):
#         vmax = 1.0
#     vmin = -vmax
#     norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

#     # Tiefe in mm, 0 mm oben
#     n_ch_sp, _ = CSD_spont.shape
#     n_ch_tr, _ = CSD_trig.shape
#     depth_mm_max_sp = (n_ch_sp - 1) * (dz_um / 1000.0)
#     depth_mm_max_tr = (n_ch_tr - 1) * (dz_um / 1000.0)

#     # Gemeinsame Zeitachse
#     t_min, t_max = -float(align_pre), float(align_post)

#     # Inset-Achsen + Colorbar
#     ax_left  = ax.inset_axes([0.00, 0.0, 0.45, 1.0])
#     ax_right = ax.inset_axes([0.50, 0.0, 0.45, 1.0])
#     cax      = ax.inset_axes([0.955, 0.1, 0.02, 0.8])
#     ax.set_axis_off()

#     # Spont
#     imL = ax_left.imshow(
#         CSD_spont, aspect="auto", origin="upper",
#         extent=[t_min, t_max, 0.0, depth_mm_max_sp],
#         cmap=cmap, norm=norm, interpolation=interp
#     )
#     ax_left.set_title("Spontaneous", fontsize=10)
#     ax_left.set_xlabel("Zeit (s)")
#     ax_left.set_ylabel("Tiefe (mm)")
#     ax_left.set_xlim(t_min, t_max)

#     # Trig
#     imR = ax_right.imshow(
#         CSD_trig, aspect="auto", origin="upper",
#         extent=[t_min, t_max, 0.0, depth_mm_max_tr],
#         cmap=cmap, norm=norm, interpolation=interp
#     )
#     ax_right.set_title("Triggered", fontsize=10)
#     ax_right.set_xlabel("Zeit (s)")
#     ax_right.set_yticks([])
#     ax_right.set_xlim(t_min, t_max)

#     # Optional: Konturen (für “Paper”-Haptik)
#     if contours:
#         try:
#             levels = np.linspace(vmin, vmax, n_contours)
#             ax_left.contour(CSD_spont, levels=levels, colors="k",
#                             linewidths=0.3, origin="upper",
#                             extent=[t_min, t_max, 0.0, depth_mm_max_sp])
#             ax_right.contour(CSD_trig, levels=levels, colors="k",
#                              linewidths=0.3, origin="upper",
#                              extent=[t_min, t_max, 0.0, depth_mm_max_tr])
#         except Exception:
#             pass

#     cb = fig.colorbar(imR, cax=cax)
#     cb.set_label("CSD (µV/mm²)")
    

#     # Gesamttitel
#     ax_left.set_title(title, fontsize=11, pad=22)
#     return fig

def _build_rollups(summary_path, out_name="upstate_summary_ALL.csv"):
    import os, glob
    import pandas as pd
    
    FIELDNAMES = [
        "Parent","Experiment","Dauer [s]","Samplingrate [Hz]","Kanäle",
        "Pulse count 1","Pulse count 2",
        "Upstates total","triggered","spon","associated",
        "Downstates total","UP/DOWN ratio",
        "Mean UP Dauer [s]","Mean UP Dauer Triggered [s]","Mean UP Dauer Spontaneous [s]",
        "Datum Analyse",
    ]
    print("[ROLLUP][DEBUG] summary_path =", summary_path)
    exp_dir       = os.path.dirname(summary_path)
    parent_dir    = os.path.dirname(exp_dir)
    for_david_dir = os.path.dirname(parent_dir)
    print("[ROLLUP][DEBUG] exp_dir =", exp_dir)
    print("[ROLLUP][DEBUG] parent_dir =", parent_dir)
    print("[ROLLUP][DEBUG] for_david_dir =", for_david_dir)

    files_parent = sorted(glob.glob(os.path.join(parent_dir, "*", "upstate_summary.csv")))
    print("[ROLLUP][DEBUG] files_parent =", files_parent)
    def _read_any(path):
        try:
            # liest Komma/Semikolon/Tabs automatisch
            df = pd.read_csv(path, sep=None, engine="python", dtype=str)
            for k in FIELDNAMES:
                if k not in df.columns:
                    df[k] = ""
            return df[FIELDNAMES]
        except Exception:
            return pd.DataFrame(columns=FIELDNAMES)

    def _write_semicolon(path, df):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, sep=";", index=False, encoding="utf-8")

    exp_dir       = os.path.dirname(summary_path)
    parent_dir    = os.path.dirname(exp_dir)        
    for_david_dir = os.path.dirname(parent_dir)     

    #Rollup pro Parent-Ordner 
    files_parent = sorted(glob.glob(os.path.join(parent_dir, "*", "upstate_summary.csv")))
    dfs = [_read_any(p) for p in files_parent]
    if dfs:
        r = (pd.concat(dfs, ignore_index=True)
               .drop_duplicates(subset=["Parent","Experiment"], keep="last"))
        out_parent = os.path.join(parent_dir, out_name)
        _write_semicolon(out_parent, r)
        print(f"[SUMMARY][ROLLUP Parent] {out_parent}  (Quellen: {len(files_parent)})")
    else:
        print("[SUMMARY][ROLLUP Parent] keine Quellen gefunden")

    # Rollup (alle Parents zusammen) 
    files_all = sorted(glob.glob(os.path.join(for_david_dir, "*", "*", "upstate_summary.csv")))
    dfs_all = [_read_any(p) for p in files_all]
    if dfs_all:
        r_all = (pd.concat(dfs_all, ignore_index=True)
                   .drop_duplicates(subset=["Parent","Experiment"], keep="last"))
        out_fd = os.path.join(for_david_dir, out_name)
        _write_semicolon(out_fd, r_all)
        print(f"[SUMMARY][ROLLUP For David] {out_fd}  (Quellen: {len(files_all)})")
    else:
        print("[SUMMARY][ROLLUP For David] keine Quellen gefunden")


#muss pot wieder rein 
# def save_all_channels_small_multiples_svg(
#     out_svg_path,
#     time_s,
#     X,                        # (n_ch, n_s)
#     ch_names=None,
#     width_in=20,  #0.6
#     height_per_channel=50,  # 0.35–0.5 inch pro Kanal
#     lw=1.2
# ):
#     X = np.asarray(X, float)
#     n_ch, _ = X.shape
#     if ch_names is None or len(ch_names) != n_ch:
#         ch_names = [f"ch{i:02d}" for i in range(n_ch)]

#     height_in = max(3.0, n_ch * height_per_channel)
#     fig, axes = plt.subplots(n_ch, 1, figsize=(width_in, height_in), sharex=True)
#     if n_ch == 1:
#         axes = [axes]

#     for i, ax in enumerate(axes):
#         ax.plot(time_s, X[i], lw=lw)
#         ax.set_ylabel(ch_names[i], rotation=0, ha="right", va="center", labelpad=10, fontsize=8)
#         ax.grid(alpha=0.15)

#     axes[-1].set_xlabel("Zeit (s)")
#     fig.suptitle(f"Alle Kanäle, n={n_ch}", y=0.995)
#     fig.tight_layout(rect=[0,0,1,0.985])
#     fig.savefig(out_svg_path, format="svg")
#     plt.close(fig)
#     del fig



# svg_path2 = os.path.join(SAVE_DIR, f"{BASE_TAG}__all_channels_.svg")
# save_all_channels_small_multiples_svg(
#     svg_path2,
#     time_s,
#     LFP_array,
#     ch_names=ch_names_for_plot,
#     height_per_channel=0.7,  # größer = mehr Platz je Kanal
#     lw=1.0
# )
# print("[SVG] all channels small-multiples:", svg_path2)



#muss vllt wieder rein 
# #  Gefilterte Variante (nach Kanal-Qualitätsfilter)
# svg_path2_good = os.path.join(SAVE_DIR, f"{BASE_TAG}__all_channels_GOOD.svg")
# save_all_channels_small_multiples_svg(
#     svg_path2_good, time_s, LFP_array_good, ch_names=ch_names_good,
#     height_per_channel=0.7, lw=1.0
# )
# print("[SVG] all channels small-multiples (filtered):", svg_path2_good)


# Helper
def _render_plotfunc_to_image(plot_func):
    before = set(plt.get_fignums())
    ret = plot_func()
    after  = set(plt.get_fignums())
    new_ids = sorted(after - before)
    figs = []
    if isinstance(ret, plt.Figure): figs = [ret]
    elif new_ids:                    figs = [plt.figure(n) for n in new_ids]
    elif plt.get_fignums():          figs = [plt.gcf()]
    if not figs:
        return [None]
    images = []
    for f in figs:
        try:
            f.canvas.draw()
            w, h = f.canvas.get_width_height()
            buf = np.frombuffer(f.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
            images.append(buf.copy())
        except Exception:
            images.append(None)
        finally:
            plt.close(f)
            del f
    return images

#Custom 1-Page Layout (1 groß, 2 klein, 1 groß) 
def export_onepage_custom(
    base_tag, save_dir,
    *,  # alles weitere nur per Keyword
    main_channel, time_s,
    Spontaneous_UP, Spontaneous_DOWN,
    Pulse_triggered_UP, Pulse_triggered_DOWN,
    Pulse_associated_UP, Pulse_associated_DOWN,
    pulse_times_1, pulse_times_2,
    dt, freqs, spont_mean, pulse_mean, p_vals,
    CSD_spont, CSD_trig, align_pre, align_post,
    dz_um=100.0, cmap="turbo", sat_pct=90, interp="bilinear", contours=True
):
    out_pdf = os.path.join(save_dir, f"{base_tag}_ALL_PLOTS_STACKED.pdf")
    with PdfPages(out_pdf) as pdf:
        # große Seite
        fig = plt.figure(figsize=(11.2, 12.0))
        gs  = fig.add_gridspec(nrows=3, ncols=2,
                               height_ratios=[1.2, 0.9, 1.3],
                               hspace=0.35, wspace=0.25)

        # TOP (span über 2 Spalten) – Main LFP + Labels
        ax_top = fig.add_subplot(gs[0, :])
        try:
            plot_up_classification_ax(
                main_channel, 
                Spontaneous_UP, Spontaneous_DOWN,
                Pulse_triggered_UP, Pulse_triggered_DOWN,
                Pulse_associated_UP, Pulse_associated_DOWN,
                time_s=time_s,
                pulse_times_1=pulse_times_1_full,
                pulse_times_2=pulse_times_2_full,
                ax=ax_top,
                title="Main channel with UP classification",
            )
        except Exception as e:
            ax_top.text(0.5, 0.5, f"Plot error (UP class):\n{e}", ha="center", va="center", transform=ax_top.transAxes)
            ax_top.set_axis_off()

        # MIDDLE LEFT – UP durations
        ax_midL = fig.add_subplot(gs[1, 0])
        try:
            upstate_duration_compare_ax(
                Pulse_triggered_UP, Pulse_triggered_DOWN,
                Spontaneous_UP,    Spontaneous_DOWN,
                dt, ax=ax_midL
            )
        except Exception as e:
            ax_midL.text(0.5, 0.5, f"Plot error (durations):\n{e}", ha="center", va="center", transform=ax_midL.transAxes)
            ax_midL.set_axis_off()

        # MIDDLE RIGHT – Power spectrum compare (f
        ax_midR = fig.add_subplot(gs[1, 1])
        try:
            if (freqs is not None) and (spont_mean is not None) and (pulse_mean is not None) and len(freqs):
                Power_spectrum_compare_ax(freqs, spont_mean, pulse_mean, p_vals=p_vals, ax=ax_midR)
            else:
                ax_midR.text(0.5, 0.5, "no spectra", ha="center", va="center", transform=ax_midR.transAxes)
                ax_midR.set_axis_off()
        except NameError:
            # falls die Funktion in Datei fehlt
            ax_midR.text(0.5, 0.5, "Power_spectrum_compare_ax() nicht definiert", ha="center", va="center", transform=ax_midR.transAxes)
            ax_midR.set_axis_off()
        except Exception as e:
            ax_midR.text(0.5, 0.5, f"Plot error (power):\n{e}", ha="center", va="center", transform=ax_midR.transAxes)
            ax_midR.set_axis_off()

        # BOTTOM (span über 2 Spalten) – CSD side-by-side
        ax_bottom = fig.add_subplot(gs[2, :])
        try:
            CSD_compare_side_by_side_ax(
                CSD_spont, CSD_trig, dt,
                z_mm=z_mm,
                align_pre=align_pre_s,
                align_post=align_post_s,
                cmap="Spectral",
                sat_pct=98,
                norm_mode="linear",
                linthresh_frac=0.02,
                ax=ax,
                title="CSD (Spont vs. Trig; UP-Onset = 0 s)"
            )
        except Exception as e:
            ax_bottom.text(0.5, 0.5, f"Plot error (CSD):\n{e}", ha="center", va="center", transform=ax_bottom.transAxes)
            ax_bottom.set_axis_off()

        fig.suptitle(base_tag, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        del fig

    print(f"[PDF] geschrieben: {out_pdf}")




def plot_up_classification_ax(
    main_channel, time_s,
    Spontaneous_UP, Spontaneous_DOWN,
    Pulse_triggered_UP, Pulse_triggered_DOWN,
    Pulse_associated_UP, Pulse_associated_DOWN,
    *,  # ab hier nur noch keyword-args
    pulse_times_1=None, pulse_times_2=None,
    Spon_Peaks=None, Trig_Peaks=None,
    ax=None, title="Main channel with UP classification"
):
  
   

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    else:
        fig = ax.figure
    # Limit: nur bis zum letzten Puls plotten
    if pulse_times_1 is not None and len(pulse_times_1):
        last_pulse_time = np.max(pulse_times_1)
    elif pulse_times_2 is not None and len(pulse_times_2):
        last_pulse_time = np.max(pulse_times_2)
    else:
        last_pulse_time = None


    

    # 1) LFP trace
    ax.plot(time_s, main_channel, lw=0.8, color="black", label="LFP (main)")
    ax.set_xlabel("Zeit (s)")
    ax.set_ylabel(f"LFP ({UNIT_LABEL})")

    # 2) Helper zum sicheren Schattieren
    def _shade(UP, DOWN, color, label):
        UP   = np.asarray(UP, dtype=int)
        DOWN = np.asarray(DOWN, dtype=int)
        m = min(len(UP), len(DOWN))
        if m == 0:
            return
        UP, DOWN = UP[:m], DOWN[:m]
        order = np.argsort(time_s[UP])
        UP, DOWN = UP[order], DOWN[order]
        for u, d in zip(UP, DOWN):
            if d > u and 0 <= u < len(time_s) and 0 < d <= len(time_s):
                ax.axvspan(time_s[u], time_s[d-1], color=color, alpha=0.22, lw=0, label=label)
                label = None  # nur einmal in Legende

    # 3) UP-Intervalle einfärben
    _shade(Spontaneous_UP,      Spontaneous_DOWN,      "green",  "UP spontaneous")
    _shade(Pulse_triggered_UP,  Pulse_triggered_DOWN,  "blue",   "UP triggered")
    _shade(Pulse_associated_UP, Pulse_associated_DOWN, "orange", "UP associated")

    # 4) y-Limits nach Schattierung holen
    y0, y1 = ax.get_ylim()

    # 5) Pulsezeiten als vlines (ohne die y-Limits zu verändern)
    def _vlines(ts, style, label):
        if ts is None or len(ts) == 0:
            return
        t = np.asarray(ts, float)
        if t.size > 800:  # bei sehr vielen Pulsen etwas ausdünnen
            step = int(np.ceil(t.size / 800))
            t = t[::step]
        ax.vlines(t, y0, y1, lw=0.6, color="red", alpha=0.35, linestyles=style, label=label, zorder=1)

    _vlines(pulse_times_1, ":",  "Pulse 1")
    _vlines(pulse_times_2, "--", "Pulse 2")

    # 6) optionale Peaks
    if Spon_Peaks is not None and len(Spon_Peaks):
        sp = np.asarray(Spon_Peaks, dtype=int)
        sp = sp[(sp >= 0) & (sp < len(time_s))]
        ax.plot(time_s[sp], main_channel[sp], "o", ms=3, alpha=0.6, label="Spont peaks")
    if Trig_Peaks is not None and len(Trig_Peaks):
        tp = np.asarray(Trig_Peaks, dtype=int)
        tp = tp[(tp >= 0) & (tp < len(time_s))]
        ax.plot(time_s[tp], main_channel[tp], "x", ms=3, alpha=0.7, label="Trig peaks")

    # 7) y-Limits fixieren (nicht von vlines beeinflussen lassen)
    ax.set_ylim(y0, y1)

    # 8) Legende ohne Duplikate
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    if uniq:
        ax.legend(uniq.values(), uniq.keys(), loc="upper right", fontsize=8, framealpha=0.9)

    ax.set_title(title)
   

    return fig


def _blank_ax(ax, msg=None):
    ax.axis("off")
    if msg:
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, alpha=0.4)

def Power_spectrum_compare_ax(freqs, spont_mean, pulse_mean, p_vals=None, alpha=0.05, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,3))
    else:
        fig = ax.figure
    if freqs is None or spont_mean is None or pulse_mean is None or len(freqs)==0:
        ax.text(0.5,0.5,"no spectra", ha="center", va="center", transform=ax.transAxes)
        return fig
    ax.plot(freqs, spont_mean, label="Spontan", lw=2)
    ax.plot(freqs, pulse_mean, label="Getriggert", lw=2)
    if p_vals is not None and np.size(p_vals)==np.size(freqs):
        sig = (p_vals < alpha)
        if np.any(sig):
            idx = np.where(sig)[0]
            # zusammenhängende Bereiche füllen
            start = idx[0]
            for i in range(1,len(idx)+1):
                if i==len(idx) or idx[i] != idx[i-1]+1:
                    ax.axvspan(freqs[start], freqs[idx[i-1]], alpha=0.12)
                    if i < len(idx): start = idx[i]
    ax.set_xlabel("Hz"); ax.set_ylabel(PSD_UNIT_LABEL) 
    ax.set_title("Power (Spontan vs. Getriggert)"); ax.legend()
    return fig



def _save_all_channels_svg_from_df(df, out_svg, *, max_points=20000):

    assert "time" in df.columns, "CSV braucht 'time'-Spalte"
    t = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
    chan_cols = [c for c in df.columns if c.lower() not in ("time", "stim", "din_1", "din_2")]

    if len(chan_cols) == 0:
        print("[ALL-CH] keine Kanalspalten gefunden"); return

    # Downsample/Decimate für schnelle Darstellung
    step = max(1, len(t) // max_points)
    t_ds = t[::step]
    t_ds = t_ds - (t_ds[0] if len(t_ds) else 0.0)

    # Figure-Höhe dynamisch
    h = max(3.0, 0.35 * len(chan_cols) + 1.0)
    fig, ax = plt.subplots(figsize=(11, h))

    offsets = []
    for i, col in enumerate(chan_cols):
        y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)[::step]
        if y.size != t_ds.size:
            # zur Sicherheit: auf gleiche Länge bringen
            m = min(y.size, t_ds.size)
            y = y[:m]; td = t_ds[:m]
        else:
            td = t_ds

        # Robust normalisieren (Offset-Stack)
        med = np.nanmedian(y)
        spread = np.nanpercentile(y, 95) - np.nanpercentile(y, 5)
        scale = spread if np.isfinite(spread) and spread > 0 else (np.nanstd(y) or 1.0)
        y_norm = (y - (med if np.isfinite(med) else 0.0)) / scale

        off = i * 2.5
        offsets.append(off)
        ax.plot(td, y_norm + off, lw=0.6)

    ax.set_xlabel("Zeit (s)")
    ax.set_yticks(offsets)
    ax.set_yticklabels(chan_cols, fontsize=8)
    ax.set_title("Alle Kanäle (gestapelt, robust skaliert)")
    ax.grid(True, alpha=0.15, linestyle=":")

    fig.tight_layout()
    fig.savefig(out_svg, format="svg")
    plt.close(fig)
    del fig
    print(f"[ALL-CH] SVG geschrieben: {out_svg}")


def _save_all_channels_svg_from_array(time_s, LFP_array, chan_labels, out_svg, *, max_points=20000):
    """
    Alternative, falls du schon das downsampled Array hast:
    LFP_array: shape (n_chan, n_time)
    """

    t = np.asarray(time_s, dtype=float)
    step = max(1, t.size // max_points)
    t_ds = t[::step]
    t_ds = t_ds - (t_ds[0] if t_ds.size else 0.0)

    n_ch, n_t = LFP_array.shape
    h = max(3.0, 0.35 * n_ch + 1.0)
    fig, ax = plt.subplots(figsize=(11, h))
    offsets = []

    for i in range(n_ch):
        y = np.asarray(LFP_array[i, :], dtype=float)[::step]
        y = y[:t_ds.size]
        med = np.nanmedian(y)
        spread = np.nanpercentile(y, 95) - np.nanpercentile(y, 5)
        scale = spread if np.isfinite(spread) and spread > 0 else (np.nanstd(y) or 1.0)
        y_norm = (y - (med if np.isfinite(med) else 0.0)) / scale
        off = i * 2.5
        offsets.append(off)
        ax.plot(t_ds, y_norm + off, lw=0.6)

    ax.set_xlabel("Zeit (s)")
    labels = chan_labels if chan_labels and len(chan_labels) == n_ch else [f"ch{i:02d}" for i in range(n_ch)]
    ax.set_yticks(offsets)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Alle Kanäle (gestapelt, robust skaliert) — downsampled")
    ax.grid(True, alpha=0.15, linestyle=":")

    fig.tight_layout()
    fig.savefig(out_svg, format="svg")
    plt.close(fig)
    del fig
    print(f"[ALL-CH] SVG geschrieben: {out_svg}")

def up_onset_mean_ax(main_channel, dt, onsets, ax=None, title="UPs – onset-aligned mean"):
    import numpy as np
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,3))
    else:
        fig = ax.figure

    onsets = np.asarray(onsets, int)
    if onsets.size == 0:
        ax.text(0.5, 0.5, "no onsets", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    half_win_s = 0.6
    half = int(round(half_win_s / dt))

    traces = []
    for o in onsets:
        s = o - half
        e = o + half
        if s < 0 or e > len(main_channel):
            continue
        seg = main_channel[s:e]
        if len(seg) == 2 * half:
            traces.append(seg)

    if not traces:
        ax.text(0.5, 0.5, "no valid segments",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    traces = np.vstack(traces)
    n = traces.shape[0]

    t = (np.arange(-half, half) * dt)
    m  = np.nanmean(traces, axis=0)
    se = np.nanstd(traces, axis=0) / np.sqrt(n)

    ax.plot(t, traces.T, alpha=0.07, lw=0.8)
    ax.plot(t, m, lw=2)
    ax.fill_between(t, m-se, m+se, alpha=0.25)

    ax.axvline(0, color="red", lw=1)
    ax.axhline(0, color="k", alpha=0.3, lw=0.8)

    ax.set_xlabel("Zeit relativ zum UP-Onset (s)")
    ax.set_ylabel(f"LFP ({UNIT_LABEL})")
    ax.set_title(title)

    # ⬅️ Textbox mit n
    ax.text(
        0.98, 0.90,
        f"n = {n}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", fc="white", alpha=0.7)
    )

    return fig

def pulse_triggered_up_overlay_ax(
    main_channel,
    time_s,
    pulse_times,
    Pulse_triggered_UP,
    dt,
    pre_s=0.2,         # wie weit vor dem Pulse anzeigen
    post_s=1.0,        # wie weit nach dem Pulse anzeigen
    max_win_s=1.0,     # max. erlaubte Pulse→UP-Latenz (Filter)
    ax=None,
    title="Pulse-alignierte Trigger-UPs (LFP overlay)"
):
    """
    Zeichnet für alle getriggerten UPs den Main-Channel LFP relativ zum Pulse:
    - 0 s = Pulse
    - alle Segmente übereinander
    - Mean-Trace
    - vertikale Linie bei 0 s (Pulse)
    - vertikale Linie beim mittleren UP-Onset relativ zum Pulse
    """

    import numpy as np
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3.0))
    else:
        fig = ax.figure

    if pulse_times is None or len(pulse_times) == 0 or len(Pulse_triggered_UP) == 0:
        ax.text(0.5, 0.5, "no pulses or triggered UPs",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    pulse_times = np.asarray(pulse_times, float)
    up_idx_arr  = np.asarray(Pulse_triggered_UP, int)

    pre_n  = int(round(pre_s  / dt))
    post_n = int(round(post_s / dt))

    segs = []
    up_rel_times = []

    for up_idx in up_idx_arr:
        if up_idx < 0 or up_idx >= len(time_s):
            continue

        t_up = time_s[up_idx]

        # letzter Pulse vor diesem UP
        mask = pulse_times <= t_up
        if not mask.any():
            continue
        t_p = pulse_times[mask][-1]

        lat = t_up - t_p
        if lat < 0 or lat > max_win_s:
            # UP ist zu weit vom Pulse weg -> als nicht "triggered" ignorieren
            continue

        # Index des Pulses im Zeitvektor
        ip = np.searchsorted(time_s, t_p)
        s = ip - pre_n
        e = ip + post_n
        if s < 0 or e > len(main_channel):
            continue

        seg = main_channel[s:e]
        if len(seg) != (pre_n + post_n):
            continue

        segs.append(seg)
        up_rel_times.append(lat)

    if not segs:
        ax.text(0.5, 0.5, "no valid pulse-aligned segments",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    segs = np.vstack(segs)                # (n_events, n_time)
    n    = segs.shape[0]
    t_rel = (np.arange(-pre_n, post_n) * dt)

    # alle Einzeltraces
    for i in range(n):
        ax.plot(t_rel, segs[i], alpha=0.12, lw=0.8)

    # Mittelwert-Trace
    mean_trace = np.nanmean(segs, axis=0)
    ax.plot(t_rel, mean_trace, lw=2.0, label="Mean LFP")

    # vertikale Linie beim Pulse (0 s)
    ax.axvline(0.0, color="k", lw=1.0, ls="--", label="Pulse")

    # mittlere UP-Latenz
    up_rel_times = np.asarray(up_rel_times, float)
    if up_rel_times.size:
        mean_lat = float(np.nanmean(up_rel_times))
        ax.axvline(mean_lat, color="red", lw=1.5, ls=":",
                   label=f"Mean UP onset ({mean_lat*1000:.0f} ms)")

    ax.set_xlabel("Zeit relativ zum Pulse (s)")
    ax.set_ylabel(f"LFP ({UNIT_LABEL})")
    ax.set_title(title)

    # n-Textbox
    ax.text(
        0.02, 0.95,
        f"n = {n}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", fc="white", alpha=0.7)
    )

    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    return fig



#  Layout definieren (Zeilen)
layout_rows = [
    # REIHE 1: Main channel (volle Breite)
    [lambda ax: plot_up_classification_ax(
        main_channel, time_s,
        Spontaneous_UP, Spontaneous_DOWN,
        Pulse_triggered_UP, Pulse_triggered_DOWN,
        Pulse_associated_UP, Pulse_associated_DOWN,
        pulse_times_1=pulse_times_1,
        pulse_times_2=pulse_times_2,
        ax=ax
    )],

    # REIHE 2: links Durations, rechts Power
    [lambda ax: upstate_duration_compare_ax(
        Trig_UP_crop, Trig_DOWN_crop,
        Spon_UP_crop, Spon_DOWN_crop, dt, ax=ax
    ),
     lambda ax: Power_spectrum_compare_ax(
        freqs, spont_mean, pulse_mean, p_vals=p_vals, ax=ax
    )],
    
 # REIHE 3: Amplituden-Vergleich (volle Breite)
    [lambda ax: upstate_amplitude_compare_ax(
        spont_amp, trig_amp, ax=ax, title="UP Amplitude (max-min): Spontan vs. Getriggert"
    )],

    # REIHE 4: CSD (volle Breite)
    [lambda ax: CSD_compare_side_by_side_ax(
        CSD_spont, CSD_trig, dt,
        z_mm=z_mm,
        align_pre=align_pre_s, align_post=align_post_s,
        # Paper-look:
        cmap="Spectral_r",          # <--- klassisches Diverging-Map
        sat_pct=95,             # <--- etwas weniger aggressiv
        norm_mode="linear",
        #vmax_abs=15,     # <--- einfache lineare Skala
        linthresh_frac=0.03,    # hier egal bei linear, kann bleiben
        ax=ax,
        title="CSD (Spont vs. Trig; UP-Onset = 0 s)"
    )],



      # REIHE 4 (NEU): Mittel-LFP um Onsets – Spont vs. Trig
    [lambda ax: up_onset_mean_ax(
        main_channel, dt, Spon_Onsets,
        ax=ax, title="Spont-UPs – onset-aligned mean"
    ),
     lambda ax: up_onset_mean_ax(
        main_channel, dt, Trig_Onsets,
        ax=ax, title="Trig-UPs – onset-aligned mean"
    )],

        # REIHE 4: CSD – zwei Panels nebeneinander (Spont / Trig)
    [lambda ax: CSD_single_panel_ax(
            CSD_spont, dt,
            z_mm=z_mm,
            align_pre=align_pre_s,
            align_post=align_post_s,
            ax=ax,
            title="CSD Spontaneous"
        ),
    lambda ax: CSD_single_panel_ax(
            CSD_trig, dt,
            z_mm=z_mm,
            align_pre=align_pre_s,
            align_post=align_post_s,
            ax=ax,
            title="CSD Triggered"
    
        )],
    # REIHE X: Pulse→UP Latenzen (Histogramm)
    [lambda ax: pulse_to_up_latency_hist_ax(latencies_trig, ax=ax)],

    [lambda ax: pulse_triggered_up_overlay_ax(
    main_channel,
    time_s,
    pulse_times_1,          # ggf. pulse_times_2
    Pulse_triggered_UP,
    dt,
    pre_s=0.2,
    post_s=1.0,
    max_win_s=1.0,
    ax=ax,
    title="Pulse-alignierte Trigger-UPs (LFP overlay)"
    )],

    [lambda ax: refractory_compare_ax(
        refrac_spont, refrac_trig, ax=ax,
        title="Refraktärzeit nach UP bis nächster UP"
    )],
    

        # REIHE: Refraktär-Overlay – offset-aligniert (Spontan / Getriggert)
    [lambda ax: refractory_overlay_ax(
        main_channel,
        time_s,
        Spontaneous_UP,
        Spontaneous_DOWN,
        dt,
        pre_s=0.5,
        post_s=2.0,
        ax=ax,
        title="Refraktär-Overlay – Spontane UPs"
    ),
     lambda ax: refractory_overlay_ax(
        main_channel,
        time_s,
        Pulse_triggered_UP,
        Pulse_triggered_DOWN,
        dt,
        pre_s=0.5,
        post_s=2.0,
        ax=ax,
        title="Refraktär-Overlay – Getriggerte UPs"
    )],


]


#print("CSD_trig mean ± std :", np.nanmean(CSD_trig), np.nanstd(CSD_trig))


def _write_summary_csv():
    import csv, io
    # Zielpfad
    summary_path = os.path.join(BASE_PATH, "upstate_summary.csv")

    # Delimiter erkennen (falls Datei existiert), sonst Standard = ';'
    delimiter = ';'
    if os.path.isfile(summary_path):
        with open(summary_path, "r", newline="", encoding="utf-8") as f:
            head = f.read(4096)
        try:
            dialect = csv.Sniffer().sniff(head, delimiters=[",",";","\t","|"])
            delimiter = dialect.delimiter
        except Exception:
            pass  # bleibt bei ';'

    # Feldnamen (Schema)
    FIELDNAMES = [
        "Parent","Experiment","Dauer [s]","Samplingrate [Hz]","Kanäle",
        "Pulse count 1","Pulse count 2",
        "Upstates total","triggered","spon","associated",
        "Downstates total","UP/DOWN ratio",
        "Mean UP Dauer [s]","Mean UP Dauer Triggered [s]","Mean UP Dauer Spontaneous [s]",
        "Datum Analyse",
    ]

    # Helfer: numpy/NaN -> plain
    def _py(v):
        import numpy as _np 
        try:
            if isinstance(v, (_np.floating, _np.float32, _np.float64)):
                f = float(v);  return "" if (f != f) else f  # NaN -> ""
            if isinstance(v, (_np.integer,)): return int(v)
        except Exception:
            pass
        if v is None: return ""
        if isinstance(v, float): return "" if (v != v) else round(v, 6)
        return v

    # aktuelle Zeile bauen
    experiment_name = os.path.basename(BASE_PATH)
    parent_folder   = os.path.basename(os.path.dirname(BASE_PATH))

    def _pairs(Up_states, time_vec):
        UP_i   = np.array(Up_states.get("UP_start_i",   []), dtype=int)
        DOWN_i = np.array(Up_states.get("DOWN_start_i", []), dtype=int)
        if DOWN_i.size == 0:
            sUP = np.array(Up_states.get("Spontaneous_UP_crop",       []), dtype=int)
            sDN = np.array(Up_states.get("Spontaneous_DOWN_crop",     []), dtype=int)
            tUP = np.array(Up_states.get("Pulse_triggered_UP_crop",   []), dtype=int)
            tDN = np.array(Up_states.get("Pulse_triggered_DOWN_crop", []), dtype=int)
            UP_i   = np.concatenate((tUP, sUP)) if (tUP.size or sUP.size) else np.array([], int)
            DOWN_i = np.concatenate((tDN, sDN)) if (tDN.size or sDN.size) else np.array([], int)
        m = min(len(UP_i), len(DOWN_i))
        UP_i, DOWN_i = UP_i[:m], DOWN_i[:m]
        if m>0:
            order = np.argsort(time_vec[UP_i])
            UP_i, DOWN_i = UP_i[order], DOWN_i[order]
        return UP_i, DOWN_i

    UP_start_i, DOWN_start_i = _pairs(Up, time_s)
    total_up   = len(Spontaneous_UP) + len(Pulse_triggered_UP) + len(Pulse_associated_UP)
    total_down = len(Spontaneous_DOWN) + len(Pulse_triggered_DOWN) + len(Pulse_associated_DOWN)

    def _mean_or_blank(arr):
        arr = np.asarray(arr)
        return "" if arr.size == 0 or not np.isfinite(arr).any() else float(np.nanmean(arr))

    row = {
        "Parent": parent_folder,
        "Experiment": experiment_name,
        "Dauer [s]": round(float(time_s[-1] - time_s[0]), 2) if len(time_s) else "",
        "Samplingrate [Hz]": round(1/dt, 2) if dt else "",
        "Kanäle": int(NUM_CHANNELS),
        "Pulse count 1": int(len(pulse_times_1) if pulse_times_1 is not None else 0),
        "Pulse count 2": int(len(pulse_times_2) if pulse_times_2 is not None else 0),
        "Upstates total": int(total_up),
        "triggered": int(len(Pulse_triggered_UP)),
        "spon": int(len(Spontaneous_UP)),
        "associated": int(len(Pulse_associated_UP)),
        "Downstates total": int(total_down),
        "UP/DOWN ratio": round(total_up / max(1, total_down), 3),
        "Mean UP Dauer [s]": _mean_or_blank((DOWN_start_i - UP_start_i) * dt) if len(UP_start_i) else "",
        "Mean UP Dauer Triggered [s]": _mean_or_blank((Pulse_triggered_DOWN - Pulse_triggered_UP) * dt) if len(Pulse_triggered_UP) else "",
        "Mean UP Dauer Spontaneous [s]": _mean_or_blank((Spontaneous_DOWN - Spontaneous_UP) * dt) if len(Spontaneous_UP) else "",
        "Datum Analyse": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    }

    # Debug: zeig die Zeile im Log
    print("[SUMMARY] target:", summary_path)
    print("[SUMMARY] delimiter:", repr(delimiter))
    print("[SUMMARY] row:", {k: _py(v) for k,v in row.items()})

    # vorhandene Zeilen laden & aufs Schema mappen
    rows = []
    if os.path.isfile(summary_path):
        with open(summary_path, "r", newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f, delimiter=delimiter)
            for r in rdr:
                rows.append({k: r.get(k, "") for k in FIELDNAMES})

    # updaten oder anhängen (Match: Parent+Experiment)
    updated = False
    for r in rows:
        if r.get("Experiment","") == experiment_name and r.get("Parent","") == parent_folder:
            for k in FIELDNAMES:
                r[k] = _py(row.get(k, r.get(k, "")))
            updated = True
            break
    if not updated:
        rows.append({k: _py(row.get(k, "")) for k in FIELDNAMES})

    # zurückschreiben mit erkanntem Delimiter
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter=delimiter)
        w.writeheader()
        for r in rows:
            # NaN endgültig leeren
            clean = {k: ("" if (isinstance(r[k], float) and r[k] != r[k]) else r[k]) for k in FIELDNAMES}
            w.writerow(clean)

    print(f"[SUMMARY] Tabelle aktualisiert: {summary_path}")
    # Rollups direkt hier bauen (Parent & For-David Ebene)
    try:
        _build_rollups(summary_path)
    except Exception as e:
        print("[SUMMARY][ROLLUP][ERROR]", e)


def export_with_layout(base_tag, save_dir, layout_rows, rows_per_page=4, also_save_each_svg=False):
    """
    layout_rows: Liste von Zeilen.
      - [callable]                -> 1 Plot, volle Breite (spannt 2 Spalten)
      - [callable, callable]      -> 2 Plots nebeneinander
    """
    _write_summary_csv()
    os.makedirs(save_dir, exist_ok=True)
    out_pdf = os.path.join(save_dir, f"{base_tag}_ALL_PLOTS_STACKED.pdf")

    def draw_into_ax(ax, spec):
        try:
            spec(ax)
        except Exception as e:
            ax.text(0.5, 0.5, f"Plot error:\n{e}", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()

    # Optional: einzelne SVGs
    if also_save_each_svg:
        k = 1
        for row in layout_rows:
            if len(row) == 1:
                fig, ax = plt.subplots(figsize=(10, 3.4))
                draw_into_ax(ax, row[0])
                fig.savefig(os.path.join(save_dir, f"{base_tag}_plot_{k:02d}.svg"),
                            format="svg", bbox_inches="tight")
                plt.close(fig); k += 1
                del fig
            elif len(row) == 2:
                for spec in row:
                    fig, ax = plt.subplots(figsize=(5, 3.4))
                    draw_into_ax(ax, spec)
                    fig.savefig(os.path.join(save_dir, f"{base_tag}_plot_{k:02d}.svg"),
                                format="svg", bbox_inches="tight")
                    plt.close(fig); k += 1
                    del fig

    # PDF Seiten
    with PdfPages(out_pdf) as pdf:
        for start in range(0, len(layout_rows), rows_per_page):
            chunk = layout_rows[start:start+rows_per_page]
            nrows = len(chunk)
            fig = plt.figure(figsize=(10.5, 3.6 * nrows))
            gs  = gridspec.GridSpec(nrows=nrows, ncols=2, figure=fig, wspace=0.25, hspace=0.5)

            for r, row in enumerate(chunk):
                if len(row) == 1:
                    ax = fig.add_subplot(gs[r, :])
                    draw_into_ax(ax, row[0])
                elif len(row) == 2:
                    axL = fig.add_subplot(gs[r, 0])
                    axR = fig.add_subplot(gs[r, 1])
                    draw_into_ax(axL, row[0])
                    draw_into_ax(axR, row[1])
                else:
                    ax = fig.add_subplot(gs[r, :])
                    ax.axis('off')
                    ax.text(0.5, 0.5, "Invalid layout row", ha="center", va="center", transform=ax.transAxes)

            fig.suptitle(base_tag, y=0.995)
            fig.tight_layout(rect=[0, 0, 1, 0.98])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            del fig


    print(f"[PDF] geschrieben: {out_pdf}")

if __name__ == "__main__":
    try:
        log("START")
        # ... dein Code ...
        log("FERTIG ohne Fehler")
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        log(f"EXCEPTION: {e}")
        with open(LOGFILE, "a", encoding="utf-8") as f:
            f.write("\n[EXCEPTION]\n")
            f.write(err)
        raise



log("Exporting layout PDF/SVG ...")

# Export aufrufen 
export_with_layout(
    BASE_TAG, SAVE_DIR, layout_rows,
    rows_per_page=3,          # 3 Zeilen -> alles auf eine Seite
    also_save_each_svg=True
)

log("Export finished")

print("[ORDER]", chan_cols)  # Originalnamen der LFP-Spalten
print("[GOOD_IDX]", good_idx)

print("[ORDER raw]", chan_cols_raw)
print("[ORDER sorted]", chan_cols)
print("[DEPTH flip?]", FLIP_DEPTH)

print("[CHAN-FILTER] kept:", good_idx)
print("[CHAN-FILTER] reasons:", reasons)

