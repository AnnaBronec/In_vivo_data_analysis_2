#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import numpy as _np 
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from scipy import stats
from scipy import signal
from matplotlib.colors import TwoSlopeNorm
import gc
from glob import glob
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from loader_old import load_LFP_new, read_nev_timestamps_and_ttl, ttl_to_on_off
from matplotlib import gridspec
from scipy.signal import welch
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
    CSD_compare_side_by_side_ax,
    _blank_ax,
    compute_refrac_from_spont_to_spon_and_trig,
)

from Exports import (
    export_interactive_lfp_html, 
    log,
     _nan_stats,
     _rms
)
from processing import (
    _counts_to_uV, _volts_to_uV, 
    convert_df_to_uV, _decimate_xy, 
    _ensure_main_channel,
    _ensure_seconds, 
    _safe_crop_to_pulses,
    _empty_updict,
    _clip_pairs,
    _clip_events_to_bounds,
    _upstate_amplitudes,
    _sem,
    _even_subsample,
    _check_peak_indices,
    crop_up_intervals,
    compute_refractory_period,
    compute_refractory_any_to_type,
    pulse_to_up_latencies,
    upstate_amplitude_compare_ax,
    pulse_to_up_latency_hist_ax,
    upstate_duration_compare_ax,
    refractory_compare_ax,
    CSD_single_panel_ax,
    _as_valid_idx,
    _build_rollups,
    )
        
#Konstanten
DOWNSAMPLE_FACTOR = 50
HIGH_CUTOFF = 10
LOW_CUTOFF  = 2
ANALYSE_IN_AU = True
HTML_IN_uV    = True
DEFAULT_FS_XDAT = 32000.0   #ist das richtig?
SPECTRA_BASELINE_MODE = None  # None | "pre_onset_db"
SPECTRA_USE_FDR = False
CSD_STYLE = "paper"  # "paper" | "diagnostic"
CSD_PAPER_CMAP = "RdBu_r"
CHANNEL_FILTER_MODE = "strict"  # "balanced" | "strict"
CLUSTER_N_PERM = int(os.environ.get("CLUSTER_N_PERM", "800"))
CLUSTER_ENABLE = os.environ.get("CLUSTER_ENABLE", "0") == "1"
AUTO_PULSE_EDGE_SHIFT = os.environ.get("AUTO_PULSE_EDGE_SHIFT", "0") == "1"
AUTO_CLEAR_TINY_OFFSETS = os.environ.get("AUTO_CLEAR_TINY_OFFSETS", "0") == "1"
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

# --- Default init (so we never get NameError) ---
pulse_times_1_html     = np.array([], dtype=float)
pulse_times_1_off_html = np.array([], dtype=float)
pulse_times_2_html     = np.array([], dtype=float)
pulse_times_2_off_html = np.array([], dtype=float)
chan_cols_raw = []
FLIP_DEPTH = False
DEBUG_MAIN_SAFE = os.environ.get("DEBUG_MAIN_SAFE", "0") == "1"


def debug_log(*args, **kwargs):
    if DEBUG_MAIN_SAFE:
        print(*args, **kwargs)


def edges_from_parts_csv(parts_dir, col, time_col="time", thr=None):
    """
    Liest col (z.B. 'stim', 'din_1', 'din_2') aus allen *_csv_parts/*.part*.csv
    und liefert (t_on, t_off) in Sekunden.
    Funktioniert auch, wenn die TTL-Spur 0/5V oder 0/1 ist.
    """


    # Part-Dateien sortiert
    files = sorted(glob.glob(os.path.join(str(parts_dir), "*.part*.csv")))
    if not files:
        return np.array([], float), np.array([], float)



    # --- time_col auto-detect
    if time_col is None:
        # nimm erste Datei, schau Header
        head = pd.read_csv(files[0], nrows=1)
        for cand in ["time", "timesamples", "timestamp", "t", "Time"]:
            if cand in head.columns:
                time_col = cand
                break
        if time_col is None:
            return np.array([], float), np.array([], float)
    # --- 1) Threshold bestimmen (wenn nicht gegeben)
    # Für quasi-binary TTL reicht min/max -> thr = (min+max)/2
    if thr is None:
        vmin = np.inf
        vmax = -np.inf
        for fp in files:
            try:
                df = pd.read_csv(fp, usecols=[time_col, col])
            except ValueError:
                # Spalte existiert in diesem Part nicht
                continue
            x = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
            x = x[np.isfinite(x)]
            if x.size == 0:
                continue
            vmin = min(vmin, float(np.min(x)))
            vmax = max(vmax, float(np.max(x)))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return np.array([], float), np.array([], float)
        thr = 0.5 * (vmin + vmax)

    # --- 2) Edges finden (mit Zustand über Part-Grenzen)
    t_on = []
    t_off = []
    prev_b = None

    for fp in files:
        try:
            df = pd.read_csv(fp, usecols=[time_col, col])
        except ValueError:
            continue

        tt = pd.to_numeric(df[time_col], errors="coerce").to_numpy(float)
        xx = pd.to_numeric(df[col], errors="coerce").to_numpy(float)

        m = np.isfinite(tt) & np.isfinite(xx)
        if not np.any(m):
            continue
        tt = tt[m]
        xx = xx[m]

        b = (xx > thr).astype(np.int8)

        # chunk-interne edges
        if b.size >= 2:
            rise = np.flatnonzero((b[1:] == 1) & (b[:-1] == 0)) + 1
            fall = np.flatnonzero((b[1:] == 0) & (b[:-1] == 1)) + 1
            if rise.size:
                t_on.extend(tt[rise].tolist())
            if fall.size:
                t_off.extend(tt[fall].tolist())

        # edge über Part-Grenze (prev_b -> erster b in diesem chunk)
        if prev_b is not None and b.size >= 1:
            if prev_b == 0 and b[0] == 1:
                t_on.append(float(tt[0]))
            elif prev_b == 1 and b[0] == 0:
                t_off.append(float(tt[0]))

        prev_b = int(b[-1]) if b.size else prev_b

    return np.asarray(t_on, float), np.asarray(t_off, float)

def edges_stateful(t_vec, x_vec, thr, prev_b=None):
    """
    Findet rising+falling edges in einem Chunk, aber behält Zustand über Chunkgrenzen.
    Returns: (t_on, t_off, prev_b_new)
    """

    t = np.asarray(t_vec, float)
    x = pd.to_numeric(x_vec, errors="coerce").to_numpy(float)

    m = np.isfinite(t) & np.isfinite(x)
    if not np.any(m):
        return np.array([], float), np.array([], float), prev_b

    t = t[m]
    x = x[m]
    b = (x > thr).astype(np.int8)

    on_idx  = np.flatnonzero((b[1:] == 1) & (b[:-1] == 0)) + 1
    off_idx = np.flatnonzero((b[1:] == 0) & (b[:-1] == 1)) + 1

    t_on  = t[on_idx].astype(float)  if on_idx.size  else np.array([], float)
    t_off = t[off_idx].astype(float) if off_idx.size else np.array([], float)

    # edge über Chunkgrenze
    if prev_b is not None and b.size:
        if prev_b == 0 and b[0] == 1:
            t_on = np.concatenate([[float(t[0])], t_on])
        elif prev_b == 1 and b[0] == 0:
            t_off = np.concatenate([[float(t[0])], t_off])

    prev_b_new = int(b[-1]) if b.size else prev_b
    return t_on, t_off, prev_b_new

def _edges_stateful_internal(t, x, thr, prev_b):
    b = (x > thr).astype(np.int8)

    on_idx  = np.flatnonzero((b[1:] == 1) & (b[:-1] == 0)) + 1
    off_idx = np.flatnonzero((b[1:] == 0) & (b[:-1] == 1)) + 1

    t_on  = t[on_idx].astype(float)
    t_off = t[off_idx].astype(float)

    # Flanke über Chunk-Grenze
    if prev_b is not None:
        if prev_b == 0 and b[0] == 1:
            t_on = np.insert(t_on, 0, t[0])
        elif prev_b == 1 and b[0] == 0:
            t_off = np.insert(t_off, 0, t[0])

    return t_on, t_off, int(b[-1])


def load_parts_to_array_streaming_with_ttl(
    base_path: str,
    ds_factor: int = 50,
    stim_cols=("stim", "din_1", "din_2", "StartStop", "TTL", "DI0", "DI1"),
    dtype=np.float32,
):
    """
    Streaming loader:
    - LFP wird downsampled (ds_factor)
    - TTL wird FULL-RES gelesen, edges stateful über part-Grenzen
    Returns:
      time_ds, LFP_array_ds, chan_cols,
      p1_on_full, p1_off_full, p2_on_full, p2_off_full,
      meta (dict mit stim_cols gefunden)
    """

    prev_b1 = None
    prev_b2 = None
    thr1 = None
    thr2 = None

    parts_dir = Path(base_path) / "_csv_parts"
    part_files = sorted(parts_dir.glob("*.part*.csv"))
    if not part_files:
        raise FileNotFoundError(f"Keine Parts unter {parts_dir} gefunden.")

    time_chunks = []
    data_chunks = []
    stim_cols_in_file = None
    chan_cols = None

    # TTL-sammler (full-res times)
    p1_on_list, p1_off_list = [], []
    p2_on_list, p2_off_list = [], []



    def _key_num(s):
        m = re.findall(r"\d+", str(s))
        return int(m[-1]) if m else 0

    for pf in part_files:
        df = pd.read_csv(pf, low_memory=False)

        # einmalig stim + channels bestimmen
        if stim_cols_in_file is None:
            stim_cols_in_file = [c for c in stim_cols if c in df.columns]
            raw_chan_cols = [c for c in df.columns if c not in ("time", *stim_cols_in_file)]
            chan_cols = sorted(raw_chan_cols, key=_key_num)

        # -------------------------
        # (A) TTL FULL-RES edges
        # -------------------------
        if "time" in df.columns:
            t_full = pd.to_numeric(df["time"], errors="coerce").to_numpy(float)
        else:
            # kein time -> skip
            t_full = None

        if t_full is not None and t_full.size:

            if "din_1" in stim_cols_in_file:
                x1 = df["din_1"]
                if thr1 is None:
                    xx = pd.to_numeric(x1, errors="coerce").to_numpy(float)
                    xx = xx[np.isfinite(xx)]
                    if xx.size:
                        lo, hi = np.nanpercentile(xx, [10, 90])
                        thr1 = 0.5 * (lo + hi)
                if thr1 is not None:
                    on, off, prev_b1 = edges_stateful(t_full, x1, thr1, prev_b1)
                    p1_on_list.append(on); p1_off_list.append(off)

            if "din_2" in stim_cols_in_file:
                x2 = df["din_2"]
                if thr2 is None:
                    xx = pd.to_numeric(x2, errors="coerce").to_numpy(float)
                    xx = xx[np.isfinite(xx)]
                    if xx.size:
                        lo, hi = np.nanpercentile(xx, [10, 90])
                        thr2 = 0.5 * (lo + hi)
                if thr2 is not None:
                    on, off, prev_b2 = edges_stateful(t_full, x2, thr2, prev_b2)
                    p2_on_list.append(on); p2_off_list.append(off)

            # Fallback: wenn weder din_1 noch din_2 existiert, nimm stim als p1
            if ("din_1" not in stim_cols_in_file) and ("din_2" not in stim_cols_in_file) and ("stim" in stim_cols_in_file):
                x1 = df["stim"]
                if thr1 is None:
                    xx = pd.to_numeric(x1, errors="coerce").to_numpy(float)
                    xx = xx[np.isfinite(xx)]
                    if xx.size:
                        lo, hi = np.nanpercentile(xx, [10, 90])
                        thr1 = 0.5 * (lo + hi)
                if thr1 is not None:
                    on, off, prev_b1 = edges_stateful(t_full, x1, thr1, prev_b1)
                    p1_on_list.append(on); p1_off_list.append(off)

        # -------------------------
        # (B) LFP DOWNSAMPLED
        # -------------------------
        keep_cols = ["time", *chan_cols]
        df_lfp = df[keep_cols]

        if ds_factor and ds_factor > 1:
            df_lfp = df_lfp.iloc[::int(ds_factor), :].reset_index(drop=True)

        t_ds = pd.to_numeric(df_lfp["time"], errors="coerce").to_numpy(float)
        time_chunks.append(t_ds)
        data_chunks.append(df_lfp[chan_cols].to_numpy(dtype=dtype))

        del df, df_lfp

    # concat LFP
    time_ds = np.concatenate(time_chunks, axis=0)
    data_all = np.concatenate(data_chunks, axis=0)      # (N, n_ch)
    LFP_array_ds = data_all.T                           # (n_ch, N)

    # concat pulses + uniq
    def _cat_uniq(lst):
        import numpy as np
        if not lst:
            return np.array([], float)
        x = np.concatenate(lst, axis=0)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.array([], float)
        return np.unique(np.round(x.astype(float), 9))

    p1_on_full  = _cat_uniq(p1_on_list)
    p1_off_full = _cat_uniq(p1_off_list)
    p2_on_full  = _cat_uniq(p2_on_list)
    p2_off_full = _cat_uniq(p2_off_list)

    meta = {"stim_cols_in_file": stim_cols_in_file, "parts_dir": str(parts_dir)}
    return time_ds, LFP_array_ds, chan_cols, p1_on_full, p1_off_full, p2_on_full, p2_off_full, meta



def snap_times_to_timebase(times, time_s):
    """
    Snap event times (in seconds) to nearest sample in time_s (monotonic).
    """
    import numpy as np
    if times is None:
        return None
    t = np.asarray(times, float)
    t = t[np.isfinite(t)]
    if t.size == 0:
        return np.array([], float)
    idx = np.searchsorted(time_s, t)
    idx = np.clip(idx, 1, len(time_s)-1)
    left = time_s[idx - 1]
    right = time_s[idx]
    take_left = (t - left) <= (right - t)
    idx = np.where(take_left, idx - 1, idx)
    return time_s[idx]


def _debug_event_snap_report(tag, raw_times, snapped_times, time_s, n_show=8):
    """
    Debug helper: compare raw event times to snapped timeline events.
    Emits compact stats + a few examples when DEBUG_MAIN_SAFE=1.
    """
    try:
        raw = np.asarray(raw_times, float) if raw_times is not None else np.array([], float)
        raw = raw[np.isfinite(raw)]
    except Exception:
        raw = np.array([], float)
    try:
        snp = np.asarray(snapped_times, float) if snapped_times is not None else np.array([], float)
        snp = snp[np.isfinite(snp)]
    except Exception:
        snp = np.array([], float)

    debug_log(f"[{tag}] raw_n={len(raw)} snapped_n={len(snp)} "
              f"time_range={float(time_s[0]):.3f}->{float(time_s[-1]):.3f}s")
    if raw.size == 0 or len(time_s) < 2:
        return

    exp = snap_times_to_timebase(raw, time_s)
    err_exp_ms = (exp - raw) * 1e3
    debug_log(f"[{tag}] raw->nearest-sample error ms | "
              f"median={float(np.median(np.abs(err_exp_ms))):.3f} "
              f"p90={float(np.percentile(np.abs(err_exp_ms), 90)):.3f} "
              f"max={float(np.max(np.abs(err_exp_ms))):.3f}")

    m = min(len(exp), len(snp))
    if m > 0:
        err_impl_ms = (snp[:m] - exp[:m]) * 1e3
        debug_log(f"[{tag}] impl-vs-expected snap error ms | "
                  f"median={float(np.median(np.abs(err_impl_ms))):.6f} "
                  f"max={float(np.max(np.abs(err_impl_ms))):.6f}")

        show = min(n_show, m)
        for i in range(show):
            debug_log(
                f"[{tag}] i={i:02d} raw={raw[i]:.6f}s expected={exp[i]:.6f}s "
                f"snapped={snp[i]:.6f}s raw->snap={(snp[i]-raw[i])*1e3:.3f}ms"
            )


parts_dir = Path(BASE_PATH) / "_csv_parts"
is_merged_run = os.environ.get("BATCH_IS_MERGED_RUN", "0") == "1"
has_explicit_filename = "LFP_FILENAME" in globals()

USE_STREAM = (
    (not is_merged_run) and (not has_explicit_filename)
    and parts_dir.exists()
    and any(parts_dir.glob("*.part*.csv"))
)

if USE_STREAM:
    print(f"[INFO] Parts erkannt unter {parts_dir} -> streaming load (LFP downsampled, TTL full-res)")
    time_s, LFP_array, chan_cols, p1_on_full, p1_off_full, p2_on_full, p2_off_full, lfp_meta = \
        load_parts_to_array_streaming_with_ttl(BASE_PATH, ds_factor=DOWNSAMPLE_FACTOR)

    LFP_df = None  # wichtig: signalisiert downstream "stream"
    print(f"[STREAM] pulses(full-res): p1_on={len(p1_on_full)} p1_off={len(p1_off_full)} "
          f"| p2_on={len(p2_on_full)} p2_off={len(p2_off_full)}")

else:
    LFP_df, chan_cols, lfp_meta = load_LFP_new(BASE_PATH, LFP_FILENAME)
    time_s = pd.to_numeric(LFP_df["time"], errors="coerce").to_numpy(dtype=float)
    LFP_array = LFP_df[chan_cols].to_numpy(dtype=np.float32).T
    print(f"[INFO] Non-stream load OK: time={time_s.shape}, LFP_array={LFP_array.shape}, chans={len(chan_cols)}")

FROM_STREAM = (LFP_df is None)

# EV wins: init pulses only if not already set
if "pulse_times_1_full" not in globals() or pulse_times_1_full is None:
    pulse_times_1_full = np.array([], dtype=float)
if "pulse_times_1_off_full" not in globals() or pulse_times_1_off_full is None:
    pulse_times_1_off_full = np.array([], dtype=float)
if "pulse_times_2_full" not in globals() or pulse_times_2_full is None:
    pulse_times_2_full = np.array([], dtype=float)
if "pulse_times_2_off_full" not in globals() or pulse_times_2_off_full is None:
    pulse_times_2_off_full = np.array([], dtype=float)

HAVE_NEV = False  # wird ggf. nach NEV-read True


def _try_load_nlx_events_from_raw(base_path, time_s):
    """
    Fallback for NCS sessions without Events.nev:
    read event timestamps (and if available durations) directly from Neuralynx raw files.
    Returns (on_s, off_s), both relative to the analysis time axis in seconds.
    """
    try:
        from neo.rawio import NeuralynxRawIO
    except Exception:
        return np.array([], float), np.array([], float)

    p = Path(base_path)
    if not p.exists():
        return np.array([], float), np.array([], float)

    try:
        skip_suffixes = {".nse", ".ntt", ".nst"}
        exclude_list = [f.name for f in p.iterdir()
                        if f.is_file() and f.suffix.lower() in skip_suffixes]
        rr = NeuralynxRawIO(
            dirname=str(p),
            exclude_filename=exclude_list,
            keep_original_times=False
        )
        rr.parse_header()
        try:
            ts, dur, _labels = rr.get_event_timestamps(block_index=0, seg_index=0)
        except TypeError:
            ts, dur, _labels = rr.get_event_timestamps()
        if ts is None or len(ts) == 0:
            return np.array([], float), np.array([], float)

        on = rr.rescale_event_timestamp(ts, dtype="float64")
        on = np.asarray(on, float)
        on = on[np.isfinite(on)]
        if on.size == 0:
            return np.array([], float), np.array([], float)

        # Try direct overlap with analysis timeline; if no overlap, anchor first event to t0.
        t0, t1 = float(time_s[0]), float(time_s[-1])
        if not ((np.nanmax(on) >= t0) and (np.nanmin(on) <= t1)):
            on = on - float(on[0]) + t0

        off = np.array([], float)
        if dur is not None and len(dur) == len(ts):
            try:
                dur_s = rr.rescale_event_timestamp(dur, dtype="float64")
            except Exception:
                dur_s = np.asarray(dur, float) / 1e6
            dur_s = np.asarray(dur_s, float)
            m = np.isfinite(dur_s) & (dur_s > 0)
            if np.any(m):
                on2 = on[:len(dur_s)][m]
                off = on2 + dur_s[m]
                on = on2

        on = on[(on >= t0) & (on <= t1)]
        if off.size:
            off = off[(off >= t0) & (off <= t1 + 1.0)]
        return np.asarray(on, float), np.asarray(off, float)
    except Exception:
        return np.array([], float), np.array([], float)




nev_candidates = sorted(Path(BASE_PATH).glob("*.nev")) + sorted(Path(BASE_PATH).glob("*.Nev")) + sorted(Path(BASE_PATH).glob("*.NEV"))
nev_path = None
for c in nev_candidates:
    if c.name.lower() == "events.nev":
        nev_path = str(c)
        break
if nev_path is None and nev_candidates:
    nev_path = str(nev_candidates[0])

if nev_path is not None and os.path.exists(nev_path):
    try:
        ts_us, ttl_words, _ = read_nev_timestamps_and_ttl(nev_path)

        # TTL bit automatisch wählen: möglichst viele gepaarte ON/OFF-Edges.
        cand = []
        max_pairs = 0
        for bit in range(16):
            _on, _off = ttl_to_on_off(ts_us, ttl_words, bit=bit)
            n_pair = min(len(_on), len(_off))
            max_pairs = max(max_pairs, n_pair)
            med_w_s = np.nan
            if n_pair > 0:
                w_s = (np.asarray(_off[:n_pair], float) - np.asarray(_on[:n_pair], float)) / 1e6
                w_s = w_s[np.isfinite(w_s) & (w_s > 0)]
                if w_s.size:
                    med_w_s = float(np.median(w_s))
            cand.append((bit, _on, _off, n_pair, med_w_s))

        best = None
        if cand:
            best = max(cand, key=lambda c: (c[3], len(c[1]), -c[0]))

        if best is None:
            on_us, off_us, best_bit = np.array([], float), np.array([], float), None
        else:
            best_bit, on_us, off_us, n_pair_best, med_w_best_s = best

        if best_bit is not None:
            if np.isfinite(med_w_best_s):
                print(f"[NEV] selected TTL bit={best_bit} on={len(on_us)} off={len(off_us)} "
                      f"median_width={med_w_best_s*1000:.2f} ms")
            else:
                print(f"[NEV] selected TTL bit={best_bit} on={len(on_us)} off={len(off_us)}")
        else:
            print("[NEV] selected TTL bit=None (no valid TTL edges)")
        debug_log("[NEV][bits] "
                  + " | ".join(
                      f"b{b}:pairs={npair},med_ms="
                      f"{(mw*1000.0 if np.isfinite(mw) else np.nan):.3f}"
                      for (b, _on, _off, npair, mw) in cand
                  ))

        t0_us = None
        if isinstance(lfp_meta, dict):
            for k in ["csc_t0_us", "t0_us", "first_ts_us", "start_ts_us", "first_timestamp_us"]:
                if k in lfp_meta and lfp_meta[k] is not None:
                    t0_us = int(lfp_meta[k])
                    break

        if t0_us is None:
            # Fallback: verankere NEV an der **vollen** LFP-Zeitbasis (vor Crop/DS),
            # nicht an time_s (bereits beschnitten). So vermeiden wir späte Pulse.
            try:
                t_ref = float(time_full[0]) if 'time_full' in locals() else float(time_s[0])
            except Exception:
                t_ref = float(time_s[0])
            t0_us = int(ts_us[0] - t_ref * 1e6)

        pulse_times_1_full     = (on_us  - t0_us) / 1e6
        pulse_times_1_off_full = (off_us - t0_us) / 1e6

        # optional: direkt auch intervals (full) bauen
        pulse_intervals_1_full = list(zip(pulse_times_1_full, pulse_times_1_off_full))

        print(f"[NEV] pulses loaded: on={len(pulse_times_1_full)} off={len(pulse_times_1_off_full)} "
              f"(t0_us={t0_us})")

    except Exception as e:
        print("[NEV][WARN] reading/parsing failed:", e)
else:
    print("[NEV] Events.nev not found -> fallback to CSV stim/din edges")
    if not FROM_STREAM:
        on_raw, off_raw = _try_load_nlx_events_from_raw(BASE_PATH, time_s)
        if on_raw.size:
            pulse_times_1_full = on_raw
            pulse_times_1_off_full = off_raw
            HAVE_NEV = True
            print(f"[NCS-EVENT] raw events loaded: on={len(on_raw)} off={len(off_raw)}")
            debug_log("[NCS-EVENT][RAW] first onsets (s):", np.asarray(on_raw[:10], float))
            if off_raw is not None and len(off_raw):
                debug_log("[NCS-EVENT][RAW] first offsets (s):", np.asarray(off_raw[:10], float))


if not FROM_STREAM:
    assert "time" in LFP_df.columns, "CSV braucht eine Spalte 'time'."



if LFP_df is not None:
    # alter Weg: wir haben ein DataFrame aus einer einzelnen CSV
    time_full = pd.to_numeric(LFP_df["time"], errors="coerce").to_numpy(dtype=float)

    chan_cols_raw = [c for c in LFP_df.columns if c not in ("time","stim","din_1","din_2")]
    
    LFP_df_ds = pd.DataFrame({"timesamples": time_full})
    for i, col in enumerate(chan_cols):
        LFP_df_ds[f"pri_{i}"] = pd.to_numeric(LFP_df[col], errors="coerce")
    NUM_CHANNELS = len(chan_cols)
else:
    time_full = time_s
    NUM_CHANNELS = LFP_array.shape[0]
    chan_cols = chan_cols  

CALIB_MODE = "counts"   # "counts" | "volts" | "uV"
ADC_BITS   = 16         # bit
ADC_VPP    = 10.0       # Peak-to-Peak des ADC in Volt 
PREAMP_GAIN = 1000.0    # Gesamt-Gain vor dem ADC. Falls kanal-spezifisch, unten 'PER_CH_GAIN' nutzen.


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
    # -------------------------------------------------------
    # CSV edge detect only if no event-derived pulses were loaded.
    # -------------------------------------------------------
    HAVE_NEV = (pulse_times_1_full is not None) and (len(pulse_times_1_full) > 0)
    if not HAVE_NEV:
        # Pulse direkt aus dem DataFrame ziehen (Fallback)
        pulse_times_1_full = np.array([], dtype=float)
        pulse_times_2_full = np.array([], dtype=float)
        pulse_times_1_off_full = np.array([], dtype=float)
        pulse_times_2_off_full = np.array([], dtype=float)

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

        def _rising_falling_from_col(col, thr=None):
            x = pd.to_numeric(LFP_df[col], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(x).any():
                return np.array([], float), np.array([], float)

            if thr is None:
                lo, hi = np.nanpercentile(x, [10, 90])
                thr = (lo + hi) * 0.5

            b = (x > thr).astype(np.int8)
            rising_idx  = np.flatnonzero((b[1:] == 1) & (b[:-1] == 0)) + 1
            falling_idx = np.flatnonzero((b[1:] == 0) & (b[:-1] == 1)) + 1

            rising_idx  = rising_idx[(rising_idx >= 0) & (rising_idx < time_full.size)]
            falling_idx = falling_idx[(falling_idx >= 0) & (falling_idx < time_full.size)]

            return time_full[rising_idx].astype(float), time_full[falling_idx].astype(float)

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

        preferred = [c for c in ["din_1","din_2","stim","StartStop","TTL","DI0","DI1"] if c in LFP_df.columns]

        if "din_1" in preferred:
            t_on, t_off = _rising_falling_from_col("din_1", thr=None)
            if t_on.size:
                pulse_times_1_full = t_on
                pulse_times_1_off_full = t_off
                stim_like_cols += ["din_1"]

        if "din_2" in preferred:
            t_on, t_off = _rising_falling_from_col("din_2", thr=None)
            if t_on.size:
                pulse_times_2_full = t_on
                pulse_times_2_off_full = t_off
                stim_like_cols += ["din_2"]

        if pulse_times_1_full.size == 0 and "stim" in preferred:
            t_on, t_off = _rising_falling_from_col("stim", thr=None)
            if t_on.size:
                pulse_times_1_full = t_on
                pulse_times_1_off_full = t_off
                stim_like_cols += ["stim"]

        if pulse_times_1_full.size == 0:
            for cand in ["StartStop","TTL","DI0","DI1"]:
                if cand in preferred:
                    t_on, t_off = _rising_falling_from_col(cand, thr=None)
                    if t_on.size:
                        pulse_times_1_full = t_on
                        pulse_times_1_off_full = t_off
                        stim_like_cols += [cand]
                        break

        if pulse_times_1_full.size == 0 and pulse_times_2_full.size == 0:
            candidates = [c for c in LFP_df.columns if c not in ("time",)]
            bin_cols = [c for c in candidates if _is_quasi_binary(c)]
            best_col, best_count = None, -1
            for c in bin_cols:
                t_on = _edges_from_col(c, rising_only=True, thr=None)
                if t_on.size > best_count:
                    best_col, best_count = c, t_on.size
            if best_col is not None and best_count > 0:
                t_on, t_off = _rising_falling_from_col(best_col, thr=None)
                pulse_times_1_full = t_on
                pulse_times_1_off_full = t_off
                stim_like_cols += [best_col]
                print(f"[INFO] Auto-detected stim channel: {best_col} (on={len(t_on)} off={len(t_off)})")

        print(f"[INFO] pulses(full): p1={len(pulse_times_1_full)}, p2={len(pulse_times_2_full)} "
              f"| from columns: {', '.join(stim_like_cols) if stim_like_cols else '—'}")

    else:
        # NEV already provided pulses -> DON'T overwrite
        print(f"[NEV] using NEV pulses: p1_on={len(pulse_times_1_full)} p1_off={len(pulse_times_1_off_full)}")


debug_log("[PULSE SOURCE CHECK] HAVE_NEV =", (pulse_times_1_full is not None and len(pulse_times_1_full) > 0),
          "| p1_on =", 0 if pulse_times_1_full is None else len(pulse_times_1_full),
          "| p1_off =", 0 if pulse_times_1_off_full is None else len(pulse_times_1_off_full))


# --- HARD GUARANTEE: if we detected a stim column, compute OFF too ---
if (pulse_times_1_full is not None and len(pulse_times_1_full) > 0):
    # try to infer which column produced p1
    stim_col_used = None
    if 'stim_like_cols' in locals() and len(stim_like_cols) > 0:
        stim_col_used = stim_like_cols[0]

    if stim_col_used is not None and ('LFP_df' in globals()) and (LFP_df is not None) and (stim_col_used in LFP_df.columns):
        t_on, t_off = _rising_falling_from_col(stim_col_used, thr=None)
        if (pulse_times_1_off_full is None) or (len(pulse_times_1_off_full) == 0):
            # Do not re-introduce pseudo widths for onset-only impulse stim tracks.
            ww = np.array([], float)
            if len(t_on) and len(t_off):
                m = min(len(t_on), len(t_off))
                ww = np.asarray(t_off[:m], float) - np.asarray(t_on[:m], float)
                ww = ww[np.isfinite(ww) & (ww > 0)]
            if ww.size and (float(np.median(ww)) <= 0.02):
                print(f"[FORCE-OFF] skip tiny-width OFF from '{stim_col_used}' (median={float(np.median(ww))*1000:.2f} ms)")
            else:
                pulse_times_1_off_full = t_off
                print(f"[FORCE-OFF] from '{stim_col_used}': on={len(t_on)} off={len(t_off)}")


if not FROM_STREAM:
    chan_cols_raw = [c for c in LFP_df.columns if c not in ("time","stim","din_1","din_2")]
    # 2) Numerische Schlüssel aus Spaltennamen ziehen (z.B. "CSC10_values" -> 10, "8" -> 8)
    def _key_num(s):
        import re
        m = re.findall(r"\d+", s)
        return int(m[-1]) if m else 0

    # 3) Sortierte Reihenfolge (flach -> tief). Wenn du tief->flach willst: am Ende [::-1].
    order_idx = sorted(range(len(chan_cols_raw)), key=lambda i: _key_num(chan_cols_raw[i]))
    FLIP_DEPTH = False   # <- bei Bedarf flippen
    if FLIP_DEPTH:
        order_idx = order_idx[::-1]

    chan_cols = [chan_cols_raw[i] for i in order_idx]

    LFP_df_ds = pd.DataFrame({"timesamples": time_full})
    for i, col in enumerate(chan_cols):
        LFP_df_ds[f"pri_{i}"] = pd.to_numeric(LFP_df[col], errors="coerce")
    NUM_CHANNELS = len(chan_cols)
else:
    NUM_CHANNELS = LFP_array.shape[0]


if FROM_STREAM:
    dt = float(np.median(np.diff(time_s))) if len(time_s) > 1 else 1.0

    pulse_times_1_html     = np.asarray(p1_on_full,  float)
    pulse_times_1_off_html = np.asarray(p1_off_full, float)
    pulse_times_2_html     = np.asarray(p2_on_full,  float)
    pulse_times_2_off_html = np.asarray(p2_off_full, float)

    pulse_times_1     = snap_times_to_timebase(pulse_times_1_html,     time_s)
    pulse_times_1_off = snap_times_to_timebase(pulse_times_1_off_html, time_s)
    pulse_times_2     = snap_times_to_timebase(pulse_times_2_html,     time_s)
    pulse_times_2_off = snap_times_to_timebase(pulse_times_2_off_html, time_s)


    print(f"[STREAM] after snap: p1_on={len(pulse_times_1)} p1_off={len(pulse_times_1_off)} "
          f"| p2_on={len(pulse_times_2)} p2_off={len(pulse_times_2_off)}")

else:
    # alter Weg: DataFrame downsampling (wie bei dir)
    time_full = pd.to_numeric(LFP_df["time"], errors="coerce").to_numpy(dtype=float)

    # LFP_df_ds bauen (wie du es schon machst)
    LFP_df_ds = pd.DataFrame({"timesamples": time_full})
    for i, col in enumerate(chan_cols):
        LFP_df_ds[f"pri_{i}"] = pd.to_numeric(LFP_df[col], errors="coerce")

    NUM_CHANNELS = len(chan_cols)

    time_s, dt, LFP_array, pulse_times_1, pulse_times_2 = _ds_fun(
        DOWNSAMPLE_FACTOR, LFP_df_ds, NUM_CHANNELS,
        pulse_times_1=pulse_times_1_full,
        pulse_times_2=pulse_times_2_full,
        snap_pulses=True
    )

    # OFF pulses: seconds normalize + snap
    pulse_times_1_off = snap_times_to_timebase(
        _ensure_seconds(pulse_times_1_off_full, time_s, DEFAULT_FS_XDAT), time_s
    )
    pulse_times_2_off = snap_times_to_timebase(
        _ensure_seconds(pulse_times_2_off_full, time_s, DEFAULT_FS_XDAT), time_s
    )

    # cleanup
    del LFP_df, LFP_df_ds

log(f"DS done: N={len(time_s)}, dt={dt}, shape={LFP_array.shape}, p1={0 if pulse_times_1 is None else len(pulse_times_1)}, p2={0 if pulse_times_2 is None else len(pulse_times_2)}")

if 'LFP_df' in globals() and LFP_df is not None:
    del LFP_df
if 'LFP_df_ds' in globals():
    del LFP_df_ds

gc.collect()


pulse_times_1_ds = None if pulse_times_1 is None else np.array(pulse_times_1, float)
pulse_times_2_ds = None if pulse_times_2 is None else np.array(pulse_times_2, float)

def _snap_event_times_to_timebase(event_times, time_s):
    if event_times is None:
        return None
    t = np.asarray(event_times, float)
    if t.size == 0:
        return t
    idx = np.searchsorted(time_s, t)
    idx = np.clip(idx, 1, len(time_s)-1)
    left = time_s[idx - 1]
    right = time_s[idx]
    take_left = (t - left) <= (right - t)
    idx = np.where(take_left, idx - 1, idx)
    return time_s[idx]


# Offsets -> Sekunden + snappen
pulse_times_1_off = _snap_event_times_to_timebase(
    _ensure_seconds(pulse_times_1_off_full, time_s, DEFAULT_FS_XDAT),
    time_s
)
pulse_times_2_off = _snap_event_times_to_timebase(
    _ensure_seconds(pulse_times_2_off_full, time_s, DEFAULT_FS_XDAT),
    time_s
)

if FROM_STREAM:
    print("[XDAT] after DS: p1_on", 0 if pulse_times_1 is None else len(pulse_times_1),
          "p1_off_full", 0 if pulse_times_1_off_full is None else len(pulse_times_1_off_full))


NUM_CHANNELS = LFP_array.shape[0]
good_idx = list(range(NUM_CHANNELS))  # Fallback: alle Kanäle
reasons = []                          # für Log-Ausgaben des Kanalfilters



if dt and (dt > 0) and ((1.0/dt) > 1e4 or dt > 1.0):  # sehr grob: "klein" => Sekunden, "groß" => Samples
    dt = dt / DEFAULT_FS_XDAT  # dt war in Samples

debug_log("[CHECK] dt(s)=", dt, " median Δt from time_s=", float(np.median(np.diff(time_s))))
if len(pulse_times_1_full) and len(pulse_times_1_off_full):
    widths = pulse_times_1_off_full[:min(len(pulse_times_1_full), len(pulse_times_1_off_full))] - \
             pulse_times_1_full[:min(len(pulse_times_1_full), len(pulse_times_1_off_full))]
    debug_log("[TTL][DBG] median width (s) =", float(np.median(widths[np.isfinite(widths)])))
    debug_log("[TTL][DBG] example widths (s) =", widths[:10])
    wv = widths[np.isfinite(widths) & (widths > 0)]
    if wv.size:
        print(f"[PULSE-WIDTH][full] median={float(np.median(wv))*1000:.2f} ms | p10/p90={float(np.percentile(wv,10))*1000:.2f}/{float(np.percentile(wv,90))*1000:.2f} ms")
        # Optional safeguard: only clear tiny offsets when explicitly enabled.
        if AUTO_CLEAR_TINY_OFFSETS and float(np.median(wv)) <= max(2.0 * float(dt), 0.015):
            pulse_times_1_off_full = np.array([], dtype=float)
            print("[PULSE-WIDTH] offsets cleared (onset-only events detected)")
        elif (not AUTO_CLEAR_TINY_OFFSETS) and float(np.median(wv)) <= max(2.0 * float(dt), 0.015):
            print("[PULSE-WIDTH] tiny offsets detected but kept (AUTO_CLEAR_TINY_OFFSETS=0)")


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





# safety normalize (nur falls irgendwas noch samples war)
pulse_times_1 = _ensure_seconds(pulse_times_1, time_s, DEFAULT_FS_XDAT)
pulse_times_2 = _ensure_seconds(pulse_times_2, time_s, DEFAULT_FS_XDAT)
pulse_times_1_off = _ensure_seconds(pulse_times_1_off, time_s, DEFAULT_FS_XDAT)
pulse_times_2_off = _ensure_seconds(pulse_times_2_off, time_s, DEFAULT_FS_XDAT)

if pulse_times_1_full is not None and len(pulse_times_1_full):
    _debug_event_snap_report("P1-ONSET-SNAP", pulse_times_1_full, pulse_times_1, time_s)
if pulse_times_1_off_full is not None and len(pulse_times_1_off_full):
    _debug_event_snap_report("P1-OFFSET-SNAP", pulse_times_1_off_full, pulse_times_1_off, time_s)

if ((pulse_times_1 is None or len(pulse_times_1)==0) and
    (pulse_times_2 is None or len(pulse_times_2)==0)):
    print("[CROP] skip: no pulses -> keep full time range")
else:
    time_s, LFP_array, pulse_times_1, pulse_times_2, pulse_times_1_off, pulse_times_2_off = _safe_crop_to_pulses(
        time_s, LFP_array,
        pulse_times_1, pulse_times_2,
        pulse_times_1_off, pulse_times_2_off,
        pad=5.0
    )

# clamp OFF inside crop
pulse_times_1_off = _clip_events_to_bounds(pulse_times_1_off, time_s, 0.0, 0.0)
pulse_times_2_off = _clip_events_to_bounds(pulse_times_2_off, time_s, 0.0, 0.0)
if (pulse_times_1 is not None and pulse_times_1_off is not None and
    len(pulse_times_1) and len(pulse_times_1_off)):
    m = min(len(pulse_times_1), len(pulse_times_1_off))
    ws = np.asarray(pulse_times_1_off[:m], float) - np.asarray(pulse_times_1[:m], float)
    ws = ws[np.isfinite(ws) & (ws > 0)]
    if ws.size:
        print(f"[PULSE-WIDTH][snapped] median={float(np.median(ws))*1000:.2f} ms | p10/p90={float(np.percentile(ws,10))*1000:.2f}/{float(np.percentile(ws,90))*1000:.2f} ms")





log(f"Crop done: time={time_s[0]:.3f}->{time_s[-1]:.3f}, shape={LFP_array.shape}, p1={len(pulse_times_1) if pulse_times_1 is not None else 0}, p2={len(pulse_times_2) if pulse_times_2 is not None else 0}")


# XDAT-Erkennung (heuristisch) 
def _is_xdat_format():
    """
    Liefert True, wenn die Kanalnamen stark nach XDAT/Intan aussehen,
    z.B. 'ch00', 'ch1', 'ch17' etc. oder wenn der Dateipfad 'xdat' enthält.
    Nutzt die *originalen* Spaltennamen vor dem pri_* Mapping.
    """
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


# ===== Kanalqualitäts-Filter =====
reasons = []
bad_idx = set()
fs = 1.0 / dt

# Kein positionsbasierter Zuschnitt: alle Kanäle in den Qualitätsfilter.
candidate_idx = list(range(NUM_CHANNELS))

mode = str(CHANNEL_FILTER_MODE).strip().lower()
if mode not in {"balanced", "strict"}:
    print(f"[CHAN-FILTER][WARN] unbekannter CHANNEL_FILTER_MODE='{CHANNEL_FILTER_MODE}' -> nutze 'balanced'")
    mode = "balanced"

cfg = {
    "balanced": {
        "min_finite_frac": 0.95,
        "artifact_z": 8.0,
        "artifact_frac": 0.02,
        "line_ratio_max": 0.30,
        "hf_ratio_max": 0.55,
        "std_rel_min": 0.12,
        "std_rel_max": 8.0,
        "corr_min": -0.05,
        "jump_ratio_max": 1.20,
    },
    "strict": {
        "min_finite_frac": 0.98,
        "artifact_z": 8.1,
        "artifact_frac": 0.022,
        "line_ratio_max": 0.28,
        "hf_ratio_max": 0.56,
        "std_rel_min": 0.13,
        "std_rel_max": 8.5,
        "corr_min": -0.10,
        "jump_ratio_max": 1.35,
    },
}[mode]
print(f"[CHAN-FILTER] mode={mode}")

def _is_quasi_binary_trace(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size < 10:
        return False
    vals = np.unique(np.round(x, 3))
    if len(vals) <= 4:
        return True
    p01 = (np.isclose(x, 0).sum() + np.isclose(x, 1).sum()) / x.size
    return p01 >= 0.95

def _line_noise_ratio(x, fs):
    f, Pxx = welch(np.nan_to_num(x, nan=0.0), fs=fs, nperseg=min(len(x), 4096))
    def bp(f1, f2):
        m = (f >= f1) & (f <= f2)
        return float(np.trapezoid(Pxx[m], f[m])) if m.any() else 0.0
    total = bp(0.5, 120.0)
    line = bp(49.0, 51.0)
    return line / (total + 1e-12)

def _hf_noise_ratio(x, fs):
    nyq = 0.5 * fs
    f_hi = min(120.0, 0.95 * nyq)
    if f_hi <= 5.0:
        return 0.0
    f, Pxx = welch(np.nan_to_num(x, nan=0.0), fs=fs, nperseg=min(len(x), 4096))
    def bp(f1, f2):
        m = (f >= f1) & (f <= f2)
        return float(np.trapezoid(Pxx[m], f[m])) if m.any() else 0.0
    total = bp(0.5, f_hi)
    hf_lo = min(80.0, 0.65 * f_hi)
    hf = bp(hf_lo, f_hi)
    return hf / (total + 1e-12)

def _corr_to_template(x, templ):
    m = np.isfinite(x) & np.isfinite(templ)
    if m.sum() < 100:
        return np.nan
    xx = x[m]
    tt = templ[m]
    sx = np.nanstd(xx)
    st = np.nanstd(tt)
    if not np.isfinite(sx) or not np.isfinite(st) or sx == 0 or st == 0:
        return np.nan
    return float(np.corrcoef(xx, tt)[0, 1])

def _jump_ratio(x):
    xx = np.asarray(x, float)
    m = np.isfinite(xx)
    xx = xx[m]
    if xx.size < 10:
        return np.nan
    s = float(np.nanstd(xx))
    if not np.isfinite(s) or s == 0:
        return np.nan
    dx = np.diff(xx)
    if dx.size == 0:
        return np.nan
    return float(np.nanmedian(np.abs(dx)) / (s + 1e-12))

std_list = []
for i in candidate_idx:
    xi = np.asarray(LFP_array[i], float)
    si = float(np.nanstd(xi))
    if np.isfinite(si) and si > 0:
        std_list.append(si)
std_med = float(np.nanmedian(std_list)) if std_list else np.nan
template = np.nanmedian(np.asarray(LFP_array[candidate_idx], float), axis=0)

for i in candidate_idx:
    x = LFP_array[i]
    finite = np.isfinite(x)
    finite_frac = float(finite.mean())
    if finite_frac < cfg["min_finite_frac"]:
        bad_idx.add(i); reasons.append((i, "zu viele NaNs")); continue
    std = np.nanstd(x)
    if not np.isfinite(std) or std == 0:
        bad_idx.add(i); reasons.append((i, "konstant/0-Std")); continue
    if np.isfinite(std_med) and std_med > 0:
        rel_std = float(std / std_med)
        if rel_std < cfg["std_rel_min"]:
            bad_idx.add(i); reasons.append((i, f"sehr niedriges Sigma(rel={rel_std:.2f})")); continue
        if rel_std > cfg["std_rel_max"]:
            bad_idx.add(i); reasons.append((i, f"sehr hohes Sigma(rel={rel_std:.2f})")); continue
    if _is_quasi_binary_trace(x):
        bad_idx.add(i); reasons.append((i, "quasi-binär")); continue
    z = (x - np.nanmedian(x)) / (std if std else 1.0)
    art_frac = float(np.mean(np.abs(z) > cfg["artifact_z"]))
    if art_frac > cfg["artifact_frac"]:
        bad_idx.add(i); reasons.append((i, f"Artefakte ({art_frac*100:.1f}% |z|>{cfg['artifact_z']:.1f})")); continue
    line_ratio = _line_noise_ratio(x, fs)
    if line_ratio > cfg["line_ratio_max"]:
        bad_idx.add(i); reasons.append((i, f"50Hz-dominant(r={line_ratio:.2f})")); continue
    hf_ratio = _hf_noise_ratio(x, fs)
    if hf_ratio > cfg["hf_ratio_max"]:
        bad_idx.add(i); reasons.append((i, f"HF-rauschig(r={hf_ratio:.2f})")); continue
    jr = _jump_ratio(x)
    if np.isfinite(jr) and jr > cfg["jump_ratio_max"]:
        bad_idx.add(i); reasons.append((i, f"zappelig(diff/std={jr:.2f})")); continue
    c = _corr_to_template(np.asarray(x, float), template)
    if np.isfinite(c) and c < cfg["corr_min"]:
        bad_idx.add(i); reasons.append((i, f"schwache Mehrkanal-Korrelation(r={c:.2f})")); continue

good_idx = [j for j in candidate_idx if j not in bad_idx]
if len(good_idx) < 2:
    print("[CHAN-FILTER][WARN] zu wenige 'gute' Kanäle im Kandidatenbereich – benutze Kandidaten ungefiltert.")
    good_idx = candidate_idx[:]
if len(good_idx) < 2:
    print("[CHAN-FILTER][WARN] fallback auf alle Kanäle.")
    good_idx = list(range(NUM_CHANNELS))

LFP_array_good    = LFP_array[good_idx, :]
ch_names_good     = [f"pri_{j}" for j in good_idx]
NUM_CHANNELS_GOOD = len(good_idx)

# Tiefe aus den *behaltenen* Kanälen ableiten:
DZ_UM = 100.0
z_mm = (np.arange(NUM_CHANNELS_GOOD, dtype=float)) * (DZ_UM / 1000.0)
z_mm_csd = z_mm.copy()

if reasons:
    print("[CHAN-FILTER] excluded:", ", ".join([f"pri_{j}({r})" for j, r in reasons]))
print(f"[CHAN-FILTER] kept {NUM_CHANNELS_GOOD}/{NUM_CHANNELS} Kanäle:", ch_names_good[:10], ("..." if NUM_CHANNELS_GOOD>10 else ""))

log(f"Channel filter: kept={NUM_CHANNELS_GOOD}/{NUM_CHANNELS}, good_idx={good_idx}")

# Main channel neu wählen: nur aus den gefilterten "good" Kanälen
main_channel_local, good_local_idx = _ensure_main_channel(
    LFP_array_good, preferred_idx=min(9, max(0, NUM_CHANNELS_GOOD - 1))
)
ch_idx_used = int(good_idx[int(good_local_idx)])  # globaler Kanalindex im Original-Array
main_channel = np.asarray(LFP_array[ch_idx_used], dtype=float)

# Für HTML: Main-Channel in µV (mit passendem Gain des globalen Kanals)
main_channel_uV = None
if HTML_IN_uV:
    orig_name = chan_cols[ch_idx_used] if (0 <= ch_idx_used < len(chan_cols)) else None
    gain_used = PER_CH_GAIN.get(orig_name, PREAMP_GAIN)
    if CALIB_MODE == "counts":
        main_channel_uV = _counts_to_uV(main_channel, ADC_BITS, ADC_VPP, gain_used)
    elif CALIB_MODE == "volts":
        main_channel_uV = _volts_to_uV(main_channel)
    elif CALIB_MODE == "uV":
        main_channel_uV = main_channel.copy()
    else:
        main_channel_uV = main_channel.copy()

print(f"[MAIN-CH] using filtered main channel: pri_{ch_idx_used}")


if HIGH_CUTOFF <= LOW_CUTOFF:
    raise ValueError(f"Invalid filter band: LOW_CUTOFF={LOW_CUTOFF} must be < HIGH_CUTOFF={HIGH_CUTOFF}")
b_lp, a_lp, b_hp, a_hp = filtering(HIGH_CUTOFF, LOW_CUTOFF, dt)  # Bandpass via LP(10 Hz) + HP(2 Hz)



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



S = np.asarray(Spect_dat[0])
t_feat = np.asarray(Spect_dat[1], float)
# dt auf Feature-Zeitachse sofort definieren (wird später gebraucht!)
if t_feat.size >= 2 and np.all(np.isfinite(t_feat)):
    dt_feat = float(np.median(np.diff(t_feat)))
else:
    dt_feat = float(dt)

debug_log("[CHECK] len(time_s) =", len(time_s))
debug_log("[CHECK] S.shape =", S.shape, "-> timebins =", S.shape[1])
debug_log("[CHECK] len(t_feat) =", len(t_feat))


t_feat = np.asarray(Spect_dat[1], float)
debug_log("[CHECK] len(t_feat) =", len(t_feat))
debug_log("[CHECK] time_s[0], time_s[-1] =", float(time_s[0]), float(time_s[-1]))
debug_log("[CHECK] t_feat[0], t_feat[-1] =", float(t_feat[0]), float(t_feat[-1]))
debug_log("[CHECK] median dt time_s =", float(np.median(np.diff(time_s))))
debug_log("[CHECK] median dt t_feat  =", float(np.median(np.diff(t_feat))))

# Vergleich der ersten paar Werte
debug_log("[CHECK] first 5 time_s:", time_s[:5])
debug_log("[CHECK] first 5 t_feat :", t_feat[:5])

# max absolute difference (wenn Längen kompatibel)
m = min(len(time_s), len(t_feat))
debug_log("[CHECK] max|time_s - t_feat| over first m:",
          float(np.nanmax(np.abs(time_s[:m] - t_feat[:m]))))




# --- Init vor Spektren/States, damit Namen garantiert existieren ---
freqs = spont_mean = pulse_mean = p_vals = None


def _count_pulse_hit_windows(pulse_t, up_idx, time_s, trig_win_s=0.35):
    p = np.asarray(pulse_t if pulse_t is not None else [], float)
    up_idx = np.asarray(up_idx if up_idx is not None else [], int)
    if p.size == 0 or up_idx.size == 0:
        return 0
    up_idx = up_idx[(up_idx >= 0) & (up_idx < len(time_s))]
    if up_idx.size == 0:
        return 0
    up_t = np.asarray(time_s[up_idx], float)
    up_t.sort()
    n = 0
    for pt in p:
        j = np.searchsorted(up_t, pt, side="left")
        if j < up_t.size and up_t[j] <= (pt + trig_win_s):
            n += 1
    return int(n)


def _shift_times(ts, shift_s):
    if ts is None:
        return None
    return np.asarray(ts, float) - float(shift_s)



pulse_times_1 = _clip_events_to_bounds(pulse_times_1, time_s, align_pre_s, align_post_s)
pulse_times_2 = _clip_events_to_bounds(pulse_times_2, time_s, align_pre_s, align_post_s)

pulse_times_1_off = _clip_events_to_bounds(pulse_times_1_off, time_s, align_pre_s, align_post_s)
pulse_times_2_off = _clip_events_to_bounds(pulse_times_2_off, time_s, align_pre_s, align_post_s)


log(f"Calling classify_states: len(time_s)={len(time_s)}, main_len={len(main_channel)}, dt={dt}, p1={len(pulse_times_1) if pulse_times_1 is not None else 0}, p2={len(pulse_times_2) if pulse_times_2 is not None else 0}")


try:
    Up = classify_states(
        Spect_dat, time_s, pulse_times_1, pulse_times_2, dt,
        main_channel, LFP_array, b_lp, a_lp, b_hp, a_hp,
        align_pre, align_post, align_len,
        pulse_times_1_off=pulse_times_1_off,
        pulse_times_2_off=pulse_times_2_off
    )

    # If pulse width collapses to ~1 sample and classification is mostly "associated",
    # check whether event markers likely represent a later edge and need a small back-shift.
    try:
        w_small = False
        if (pulse_times_1 is not None and pulse_times_1_off is not None and
            len(pulse_times_1) and len(pulse_times_1_off)):
            m_w = min(len(pulse_times_1), len(pulse_times_1_off))
            ww = np.asarray(pulse_times_1_off[:m_w], float) - np.asarray(pulse_times_1[:m_w], float)
            ww = ww[np.isfinite(ww) & (ww > 0)]
            if ww.size:
                med_w = float(np.median(ww))
                w_small = med_w <= max(2.0 * float(dt), 0.015)

        up_all_idx = np.asarray(Up.get("UP_start_i", []), int)
        n_assoc = int(len(np.asarray(Up.get("Pulse_associated_UP", []), int)))
        n_trig = int(len(np.asarray(Up.get("Pulse_triggered_UP", []), int)))
        edge_unknown = (pulse_times_1_full is not None and len(pulse_times_1_full) > 0 and
                        ((pulse_times_1_off_full is None) or (len(pulse_times_1_off_full) == 0)))

        if (AUTO_PULSE_EDGE_SHIFT and
            pulse_times_1 is not None and len(pulse_times_1) >= 5 and up_all_idx.size >= 5 and
            (w_small or edge_unknown) and n_assoc >= max(8, n_trig + 6)):
            base_trig = int(len(np.asarray(Up.get("Pulse_triggered_UP", []), int)))
            base_assoc = int(len(np.asarray(Up.get("Pulse_associated_UP", []), int)))
            base_score = float(base_trig) - 0.25 * float(base_assoc)
            best_shift = 0.0
            best_score = base_score
            best_up = Up
            for sh in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00):
                p_try = _clip_events_to_bounds(_shift_times(pulse_times_1, sh), time_s, align_pre_s, align_post_s)
                po_try = _clip_events_to_bounds(_shift_times(pulse_times_1_off, sh), time_s, align_pre_s, align_post_s)
                up_try = classify_states(
                    Spect_dat, time_s, p_try, pulse_times_2, dt,
                    main_channel, LFP_array, b_lp, a_lp, b_hp, a_hp,
                    align_pre, align_post, align_len,
                    pulse_times_1_off=po_try,
                    pulse_times_2_off=pulse_times_2_off
                )
                trig_try = int(len(np.asarray(up_try.get("Pulse_triggered_UP", []), int)))
                assoc_try = int(len(np.asarray(up_try.get("Pulse_associated_UP", []), int)))
                score_try = float(trig_try) - 0.25 * float(assoc_try)
                print(f"[PULSE-EDGE][test] shift={sh:.3f}s -> trig={trig_try}, assoc={assoc_try}, score={score_try:.2f}")
                if score_try > best_score:
                    best_score = score_try
                    best_shift = float(sh)
                    best_up = up_try

            if best_shift > 0 and best_score > base_score + 1.0:
                print(f"[PULSE-EDGE] auto-shift events earlier by {best_shift:.3f}s (score {base_score:.2f}->{best_score:.2f})")
                pulse_times_1 = _clip_events_to_bounds(_shift_times(pulse_times_1, best_shift), time_s, align_pre_s, align_post_s)
                pulse_times_1_off = _clip_events_to_bounds(_shift_times(pulse_times_1_off, best_shift), time_s, align_pre_s, align_post_s)
                pulse_times_1_full = _shift_times(pulse_times_1_full, best_shift)
                pulse_times_1_off_full = _shift_times(pulse_times_1_off_full, best_shift)
                pulse_times_1_html = _shift_times(pulse_times_1_html, best_shift)
                pulse_times_1_off_html = _shift_times(pulse_times_1_off_html, best_shift)
                Up = best_up

        # Always-on lightweight sanity check:
        # if many associated but almost no triggered, try one earlier shift by median pulse width.
        # This addresses sessions where event edges are systematically late.
        if (pulse_times_1 is not None and pulse_times_1_off is not None and
            len(pulse_times_1) >= 5 and len(pulse_times_1_off) >= 5):
            m_w = min(len(pulse_times_1), len(pulse_times_1_off))
            ww = np.asarray(pulse_times_1_off[:m_w], float) - np.asarray(pulse_times_1[:m_w], float)
            ww = ww[np.isfinite(ww) & (ww > 0)]
            if ww.size:
                med_w = float(np.median(ww))
                n_assoc_now = int(len(np.asarray(Up.get("Pulse_associated_UP", []), int)))
                n_trig_now = int(len(np.asarray(Up.get("Pulse_triggered_UP", []), int)))
                if (med_w >= 0.03 and med_w <= 2.0 and n_assoc_now >= max(8, n_trig_now + 6)):
                    sh = float(np.clip(med_w, 0.03, 0.8))
                    p_try = _clip_events_to_bounds(_shift_times(pulse_times_1, -sh), time_s, align_pre_s, align_post_s)
                    po_try = _clip_events_to_bounds(_shift_times(pulse_times_1_off, -sh), time_s, align_pre_s, align_post_s)
                    up_try = classify_states(
                        Spect_dat, time_s, p_try, pulse_times_2, dt,
                        main_channel, LFP_array, b_lp, a_lp, b_hp, a_hp,
                        align_pre, align_post, align_len,
                        pulse_times_1_off=po_try,
                        pulse_times_2_off=pulse_times_2_off
                    )
                    trig_try = int(len(np.asarray(up_try.get("Pulse_triggered_UP", []), int)))
                    assoc_try = int(len(np.asarray(up_try.get("Pulse_associated_UP", []), int)))
                    score_now = float(n_trig_now) - 0.25 * float(n_assoc_now)
                    score_try = float(trig_try) - 0.25 * float(assoc_try)
                    print(f"[PULSE-EDGE][sanity] shift=-{sh:.3f}s (med_width): trig={n_trig_now}->{trig_try}, assoc={n_assoc_now}->{assoc_try}, score={score_now:.2f}->{score_try:.2f}")
                    if score_try >= score_now + 2.0:
                        print(f"[PULSE-EDGE] apply sanity shift -{sh:.3f}s")
                        pulse_times_1 = p_try
                        pulse_times_1_off = po_try
                        pulse_times_1_full = _shift_times(pulse_times_1_full, -sh)
                        pulse_times_1_off_full = _shift_times(pulse_times_1_off_full, -sh)
                        pulse_times_1_html = _shift_times(pulse_times_1_html, -sh)
                        pulse_times_1_off_html = _shift_times(pulse_times_1_off_html, -sh)
                        Up = up_try
    except Exception as _e_shift:
        print(f"[PULSE-EDGE][WARN] auto-shift skipped: {_e_shift}")

    # --- ab hier: Peaks sauber machen (nach classify_states) ---
    Spontaneous_UP      = np.asarray(Up.get("Spontaneous_UP", []), dtype=int)
    Pulse_triggered_UP  = np.asarray(Up.get("Pulse_triggered_UP", []), dtype=int)

    Spon_Peaks_raw = Up.get("Spon_Peaks", None)
    Trig_Peaks_raw = Up.get("Trig_Peaks", None)

    n_sig = int(len(main_channel))  # oder: LFP_array.shape[1] / len(time_s)

    def _as_valid_idx(arr, n):
        if arr is None:
            return None
        a = np.asarray(arr)
        # wenn float -> sehr wahrscheinlich Sekundenwerte, dann NICHT als Index benutzen
        if np.issubdtype(a.dtype, np.floating):
            return None
        a = a.astype(int, copy=False)
        a = a[(a >= 0) & (a < n)]
        return a

    Spon_Peaks = _as_valid_idx(Spon_Peaks_raw, n_sig)
    Trig_Peaks = _as_valid_idx(Trig_Peaks_raw, n_sig)

    # Fallback: nimm Onsets
    if Spon_Peaks is None or Spon_Peaks.size == 0:
        Spon_Peaks = Spontaneous_UP.copy()
    if Trig_Peaks is None or Trig_Peaks.size == 0:
        Trig_Peaks = Pulse_triggered_UP.copy()

    # Wichtig: zurück ins Up dict schreiben, damit ALLES downstream konsistent ist
    Up["Spon_Peaks"] = Spon_Peaks
    Up["Trig_Peaks"] = Trig_Peaks

#except Exception as e:






    debug_log("[DEBUG] Up keys:", sorted(list(Up.keys()))[:50])
    debug_log("[DEBUG] aligned arrays:",
              type(Up.get("Trig_UP_peak_aligned_array")),
              getattr(Up.get("Trig_UP_peak_aligned_array"), "shape", None),
              type(Up.get("Spon_UP_peak_aligned_array")),
              getattr(Up.get("Spon_UP_peak_aligned_array"), "shape", None))
    debug_log("[DBG] peaks dtype:", Up["Spon_Peaks"].dtype, Up["Trig_Peaks"].dtype)
    debug_log("[DBG] peaks min/max:",
              (Up["Spon_Peaks"].min() if Up["Spon_Peaks"].size else None),
              (Up["Spon_Peaks"].max() if Up["Spon_Peaks"].size else None))
    debug_log("[DBG] n_sig:", n_sig, "align_len:", align_len)



except IndexError as e:
    log(f"classify_states FAILED: {e}")
    print(f"[WARN] classify_states skipped due to IndexError: {e}")
    Up = _empty_updict()

except Exception as e:
    print(f"[FATAL] classify_states crashed -> Up=_empty_updict() (continuing). Error: {e}")
    Up = _empty_updict()

# optional: falls classify_states explizit None zurückgibt
if Up is None:
    print("[FATAL] classify_states returned None -> Up=_empty_updict() (continuing)")
    Up = _empty_updict()

# --- Peak-aligned Arrays explizit aus Up holen ---
Trig_UP_peak_aligned_array = Up.get("Trig_UP_peak_aligned_array", None)
Spon_UP_peak_aligned_array = Up.get("Spon_UP_peak_aligned_array", None)
UP_Time = Up.get("UP_Time", None)


def compute_upstate_rate(time_s,
                          Spontaneous_UP, Pulse_triggered_UP, Pulse_associated_UP,
                          Spontaneous_DOWN=None, Pulse_triggered_DOWN=None, Pulse_associated_DOWN=None,
                          use_pairs_for_duration=False):
    """
    Berechnet UPstate-Frequenzen (Hz) für total + pro Typ.
    Default: Dauer = time_s[-1] - time_s[0] (nach Cropping).
    Optional: use_pairs_for_duration=True -> Dauer aus UP/DOWN-Paaren (meist nicht nötig).

    Returns dict:
      duration_s, total_up, rate_total_hz, rate_total_per_min,
      counts & rates pro Typ.
    """

    # --- Dauer bestimmen ---
    if time_s is None or len(time_s) < 2:
        duration_s = np.nan
    else:
        duration_s = float(time_s[-1] - time_s[0])

    def _count(x):
        return int(len(x)) if x is not None else 0

    n_sp  = _count(Spontaneous_UP)
    n_tr  = _count(Pulse_triggered_UP)
    n_as  = _count(Pulse_associated_UP)
    n_tot = n_sp + n_tr + n_as

    def _rate(n):
        if not np.isfinite(duration_s) or duration_s <= 0:
            return np.nan
        return float(n) / duration_s

    out = {
        "duration_s": duration_s,
        "total_up": n_tot,
        "spont_up": n_sp,
        "trig_up": n_tr,
        "assoc_up": n_as,

        "rate_total_hz": _rate(n_tot),
        "rate_spont_hz": _rate(n_sp),
        "rate_trig_hz": _rate(n_tr),
        "rate_assoc_hz": _rate(n_as),
    }
    out["rate_total_per_min"] = out["rate_total_hz"] * 60.0 if np.isfinite(out["rate_total_hz"]) else np.nan
    out["rate_spont_per_min"] = out["rate_spont_hz"] * 60.0 if np.isfinite(out["rate_spont_hz"]) else np.nan
    out["rate_trig_per_min"]  = out["rate_trig_hz"]  * 60.0 if np.isfinite(out["rate_trig_hz"])  else np.nan
    out["rate_assoc_per_min"] = out["rate_assoc_hz"] * 60.0 if np.isfinite(out["rate_assoc_hz"]) else np.nan
    
    return out


nT = LFP_array.shape[1]
for kU, kD in [
    ("Spontaneous_UP","Spontaneous_DOWN"),
    ("Pulse_triggered_UP","Pulse_triggered_DOWN"),
    ("Pulse_associated_UP","Pulse_associated_DOWN"),
]:
    Uc, Dc = _clip_pairs(Up.get(kU, []), Up.get(kD, []), nT)
    Up[kU], Up[kD] = Uc, Dc



Spontaneous_UP        = Up.get("Spontaneous_UP",        np.array([], int))
Spontaneous_DOWN      = Up.get("Spontaneous_DOWN",      np.array([], int))
Pulse_triggered_UP    = Up.get("Pulse_triggered_UP",    np.array([], int))
Pulse_triggered_DOWN  = Up.get("Pulse_triggered_DOWN",  np.array([], int))
Pulse_associated_UP   = Up.get("Pulse_associated_UP",   np.array([], int))
Pulse_associated_DOWN = Up.get("Pulse_associated_DOWN", np.array([], int))

Spon_Peaks_raw = Up.get("Spon_Peaks", None)
Trig_Peaks_raw = Up.get("Trig_Peaks", None)


Spon_Peaks = _as_valid_idx(Spon_Peaks_raw, LFP_array_good.shape[1])
Trig_Peaks = _as_valid_idx(Trig_Peaks_raw, LFP_array_good.shape[1])

if Spon_Peaks is None or Spon_Peaks.size == 0:
    Spon_Peaks = np.asarray(Spontaneous_UP, int)
if Trig_Peaks is None or Trig_Peaks.size == 0:
    Trig_Peaks = np.asarray(Pulse_triggered_UP, int)


# Total_power           = Up.get("Total_power", None)
# up_state_binary       = Up.get("up_state_binary ", Up.get("up_state_binary", None))


print("[COUNTS] sponUP:", len(Spontaneous_UP), " trigUP:", len(Pulse_triggered_UP), " assocUP:", len(Pulse_associated_UP))
log(f"States: spon={len(Spontaneous_UP)}, trig={len(Pulse_triggered_UP)}, assoc={len(Pulse_associated_UP)}")

# --- UP-State-Frequenz berechnen ---
up_rate = compute_upstate_rate(
    time_s,
    Spontaneous_UP,
    Pulse_triggered_UP,
    Pulse_associated_UP
)

print(
    f"[UP-RATE] duration={up_rate['duration_s']:.2f}s | "
    f"total={up_rate['total_up']} -> "
    f"{up_rate['rate_total_hz']:.4f} Hz "
    f"({up_rate['rate_total_per_min']:.2f}/min)"
)

print(
    f"[UP-RATE] spont={up_rate['spont_up']} "
    f"({up_rate['rate_spont_hz']:.4f} Hz), "
    f"trig={up_rate['trig_up']} "
    f"({up_rate['rate_trig_hz']:.4f} Hz), "
    f"assoc={up_rate['assoc_up']} "
    f"({up_rate['rate_assoc_hz']:.4f} Hz)"
)

log(
    f"UP_RATE duration_s={up_rate['duration_s']:.3f} "
    f"total={up_rate['total_up']} "
    f"rate_total_hz={up_rate['rate_total_hz']:.6f} "
    f"rate_spont_hz={up_rate['rate_spont_hz']:.6f} "
    f"rate_trig_hz={up_rate['rate_trig_hz']:.6f} "
    f"rate_assoc_hz={up_rate['rate_assoc_hz']:.6f}"
)

# Refraktärzeiten (Ende eines UP bis Beginn des nächsten UP) 
refrac_spont = compute_refractory_period(
    Spontaneous_UP, Spontaneous_DOWN, time_s
)
refrac_trig = compute_refractory_period(
    Pulse_triggered_UP, Pulse_triggered_DOWN, time_s
)

print(f"[REFRAC] spont: n={len(refrac_spont)}, trig: n={len(refrac_trig)}")


# Refraktärzeiten "Ende ANY-UP → nächster SPONT/TRIG-UP"
refrac_any_to_spont, refrac_any_to_trig = compute_refractory_any_to_type(
    Spontaneous_UP, Spontaneous_DOWN,
    Pulse_triggered_UP, Pulse_triggered_DOWN,
    Pulse_associated_UP, Pulse_associated_DOWN,
    time_s
)

print(f"[REFRAC-any] any→spont: n={len(refrac_any_to_spont)}, any→trig: n={len(refrac_any_to_trig)}")


# CSV: Refraktärzeiten exportieren 
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
    title="UP Amplitude (max-min, mean): Spontan vs. Getriggert"
)
fig_amp.tight_layout()
fig_amp.savefig(amp_svg_path, format="svg", bbox_inches="tight")
plt.close(fig_amp)
print("[SVG] amplitude compare:", amp_svg_path)
del fig_amp


def _pair_up_down_indices(up_idx, down_idx, n_time):
    up = np.asarray(up_idx, dtype=int)
    dn = np.asarray(down_idx, dtype=int)
    m = min(len(up), len(dn))
    if m == 0:
        return []
    up = up[:m]
    dn = dn[:m]
    out = []
    for u, d in zip(up, dn):
        if 0 <= u < n_time and 0 < d <= n_time and d > u:
            out.append((int(u), int(d)))
    return out


def detect_spindle_intervals_in_upstates(
    signal_1d,
    time_s,
    dt,
    up_pairs,
    *,
    f_lo=10.0,
    f_hi=15.0,
    thr_k_on=2.5,
    thr_k_off=1.5,
    min_dur_s=0.5,
    max_dur_s=2.0,
    max_gap_s=0.02,
    min_cycles=2,
    min_spindle_band_power_ratio=1.2,
    min_spindle_power_fraction=0.12,
    min_env_peak_quantile=0.50,
    min_env_peak_sigma=2.5,
    only_in_upstates=False,
    use_psd_check=False,
):
    x = np.asarray(signal_1d, float).reshape(-1)
    t = np.asarray(time_s, float).reshape(-1)
    if x.size < 10 or t.size != x.size or dt <= 0:
        return []

    fs = 1.0 / dt
    nyq = 0.5 * fs
    lo = max(0.5, float(f_lo))
    hi = min(float(f_hi), 0.95 * nyq)
    if lo >= hi:
        return []

    x0 = np.nan_to_num(x, nan=float(np.nanmedian(x)))
    try:
        b, a = signal.butter(3, [lo / nyq, hi / nyq], btype="bandpass")
        xb = signal.filtfilt(b, a, x0)
        env = np.abs(signal.hilbert(xb))
    except Exception:
        return []

    if only_in_upstates:
        up_mask = np.zeros_like(x, dtype=bool)
        for u, d in up_pairs:
            up_mask[u:d] = True
        if not np.any(up_mask):
            return []
    else:
        up_mask = np.ones_like(x, dtype=bool)

    e_up = env[up_mask]
    med = float(np.nanmedian(e_up))
    mad = float(np.nanmedian(np.abs(e_up - med)))
    robust_sigma = 1.4826 * mad
    sig = max(robust_sigma, 1e-12)
    thr_on = med + float(thr_k_on) * sig
    thr_off = med + float(thr_k_off) * sig
    if thr_off > thr_on:
        thr_off = thr_on
    env_peak_floor = max(
        float(np.nanquantile(e_up, float(min_env_peak_quantile))),
        med + float(min_env_peak_sigma) * sig,
    )

    # Hysteresis: start only above high threshold, end below lower threshold.
    segs = []
    in_seg = False
    s = 0
    for i in range(x.size):
        if not up_mask[i]:
            if in_seg:
                segs.append((s, i))
                in_seg = False
            continue
        if not in_seg:
            if env[i] >= thr_on:
                s = i
                in_seg = True
        else:
            if env[i] < thr_off:
                segs.append((s, i))
                in_seg = False
    if in_seg:
        segs.append((s, x.size))

    if not segs:
        return []

    max_gap = max(0, int(round(max_gap_s / dt)))
    if max_gap > 0 and len(segs) > 1:
        merged = [segs[0]]
        for s2, e2 in segs[1:]:
            s1, e1 = merged[-1]
            if (s2 - e1) <= max_gap:
                merged[-1] = (s1, e2)
            else:
                merged.append((s2, e2))
        segs = merged

    min_len = max(1, int(round(min_dur_s / dt)))
    max_len = max(min_len, int(round(max_dur_s / dt)))

    out = []
    n_after_duration = 0
    n_after_env = 0
    n_after_cycles = 0
    n_after_psd = 0
    min_peak_distance = max(1, int(round(fs / max(f_hi, 1.0) * 0.8)))
    for s, e in segs:
        L = e - s
        if L < min_len or L > max_len:
            continue
        n_after_duration += 1
        if float(np.nanmax(env[s:e])) < env_peak_floor:
            continue
        n_after_env += 1
        # Require a minimum number of oscillatory cycles in spindle band.
        peaks, _ = signal.find_peaks(xb[s:e], distance=min_peak_distance)
        if peaks.size < int(min_cycles):
            continue
        n_after_cycles += 1
        # Optional spectral sanity check against broadband/noisy fragments.
        psd_ok = True
        if use_psd_check:
            nper = min(256, L)
            if nper >= 32:
                freqs, pxx = signal.welch(x0[s:e], fs=fs, nperseg=nper)
                if pxx.size and np.any(np.isfinite(pxx)):
                    pxx = np.nan_to_num(pxx, nan=0.0, posinf=0.0, neginf=0.0)
                    f = freqs
                    p_sp = float(np.trapezoid(pxx[(f >= f_lo) & (f <= f_hi)], f[(f >= f_lo) & (f <= f_hi)]))
                    p_lo = float(np.trapezoid(pxx[(f >= 6.0) & (f < f_lo)], f[(f >= 6.0) & (f < f_lo)]))
                    p_hi = float(np.trapezoid(pxx[(f > f_hi) & (f <= 25.0)], f[(f > f_hi) & (f <= 25.0)]))
                    p_bg = max(p_lo, p_hi, 1e-12)
                    p_tot = float(np.trapezoid(pxx[(f >= 4.0) & (f <= 30.0)], f[(f >= 4.0) & (f <= 30.0)]))
                    if p_sp / p_bg < float(min_spindle_band_power_ratio):
                        psd_ok = False
                    if p_tot > 0 and (p_sp / p_tot) < float(min_spindle_power_fraction):
                        psd_ok = False
        if not psd_ok:
            continue
        n_after_psd += 1
        out.append((float(t[s]), float(t[e - 1])))
    if DEBUG_MAIN_SAFE:
        print(
            "[SPINDLE-DBG] segs=", len(segs),
            "after_dur=", n_after_duration,
            "after_env=", n_after_env,
            "after_cycles=", n_after_cycles,
            "after_psd=", n_after_psd,
            "accepted=", len(out),
        )
    return out


all_up_pairs = []
all_up_pairs += _pair_up_down_indices(Spontaneous_UP, Spontaneous_DOWN, len(time_s))
all_up_pairs += _pair_up_down_indices(Pulse_triggered_UP, Pulse_triggered_DOWN, len(time_s))
all_up_pairs += _pair_up_down_indices(Pulse_associated_UP, Pulse_associated_DOWN, len(time_s))
spindle_intervals_s = detect_spindle_intervals_in_upstates(
    main_channel, time_s, dt, all_up_pairs
)
print(f"[SPINDLE] 10-15 Hz intervals (global): n={len(spindle_intervals_s)}")

debug_log("[DBG before pair] p1_on/off:",
          0 if pulse_times_1 is None else len(pulse_times_1),
          0 if pulse_times_1_off is None else len(pulse_times_1_off))
debug_log("[DBG before pair] off head:",
          None if pulse_times_1_off is None else pulse_times_1_off[:5])


# --- HTML pulses must be in SECONDS and within plotted time range ---
pulse_times_1_html     = _ensure_seconds(pulse_times_1_full,     time_s, DEFAULT_FS_XDAT)
pulse_times_1_off_html = _ensure_seconds(pulse_times_1_off_full, time_s, DEFAULT_FS_XDAT)
pulse_times_2_html     = _ensure_seconds(pulse_times_2_full,     time_s, DEFAULT_FS_XDAT)
pulse_times_2_off_html = _ensure_seconds(pulse_times_2_off_full, time_s, DEFAULT_FS_XDAT)

# clip to current plot window (after cropping)
t0, t1 = float(time_s[0]), float(time_s[-1])
pulse_times_1_html     = pulse_times_1_html[(pulse_times_1_html >= t0) & (pulse_times_1_html <= t1)]
pulse_times_1_off_html = pulse_times_1_off_html[(pulse_times_1_off_html >= t0) & (pulse_times_1_off_html <= t1)]
pulse_times_2_html     = pulse_times_2_html[(pulse_times_2_html >= t0) & (pulse_times_2_html <= t1)]
pulse_times_2_off_html = pulse_times_2_off_html[(pulse_times_2_off_html >= t0) & (pulse_times_2_off_html <= t1)]



def _pair_on_off(t_on, t_off, max_width_s=10.0):
    if t_on is None or t_off is None:
        return []
    on = np.asarray(t_on, float); off = np.asarray(t_off, float)
    on = on[np.isfinite(on)]; off = off[np.isfinite(off)]
    if on.size == 0 or off.size == 0:
        return []
    on.sort(); off.sort()

    intervals = []
    j = 0
    for i in range(on.size):
        while j < off.size and off[j] <= on[i]:
            j += 1
        if j >= off.size:
            break
        width = off[j] - on[i]
        if 0 < width <= max_width_s:
            intervals.append((float(on[i]), float(off[j])))
            j += 1   # <-- entscheidend
    return intervals

ttl1_intervals = _pair_on_off(pulse_times_1_html, pulse_times_1_off_html, max_width_s=5.0)
ttl2_intervals = _pair_on_off(pulse_times_2_html, pulse_times_2_off_html, max_width_s=5.0)

debug_log("[HTML DBG] time range:", float(time_s[0]), "->", float(time_s[-1]))
debug_log("[HTML DBG] p1_on/off:", len(pulse_times_1_html), len(pulse_times_1_off_html))
if len(pulse_times_1_html) and len(pulse_times_1_off_html):
    m = min(len(pulse_times_1_html), len(pulse_times_1_off_html))
    w = pulse_times_1_off_html[:m] - pulse_times_1_html[:m]
    debug_log("[HTML DBG] width median (s):", float(np.nanmedian(w)))
debug_log("[HTML DBG] intervals:", len(ttl1_intervals))



print("[TTL] intervals:", len(ttl1_intervals), len(ttl2_intervals))
debug_log("[DBG] on/off counts:",
          "p1_on", 0 if pulse_times_1 is None else len(pulse_times_1),
          "p1_off", 0 if pulse_times_1_off is None else len(pulse_times_1_off),
          "| p2_on", 0 if pulse_times_2 is None else len(pulse_times_2),
          "p2_off", 0 if pulse_times_2_off is None else len(pulse_times_2_off))

if pulse_times_1 is not None and len(pulse_times_1):
    debug_log("[DBG] p1_on first/last:", float(pulse_times_1[0]), float(pulse_times_1[-1]))
if pulse_times_1_off is not None and len(pulse_times_1_off):
    debug_log("[DBG] p1_off first/last:", float(pulse_times_1_off[0]), float(pulse_times_1_off[-1]))
debug_log("[DBG] time_s range:", float(time_s[0]), "->", float(time_s[-1]))

debug_log("[DBG] on/off counts:",
          "p1_on", 0 if pulse_times_1 is None else len(pulse_times_1),
          "p1_off", 0 if pulse_times_1_off is None else len(pulse_times_1_off),
          "| p2_on", 0 if pulse_times_2 is None else len(pulse_times_2),
          "p2_off", 0 if pulse_times_2_off is None else len(pulse_times_2_off))

if pulse_times_1 is not None and len(pulse_times_1):
    debug_log("[DBG] p1_on first/last:", float(pulse_times_1[0]), float(pulse_times_1[-1]))
if pulse_times_1_off is not None and len(pulse_times_1_off):
    debug_log("[DBG] p1_off first/last:", float(pulse_times_1_off[0]), float(pulse_times_1_off[-1]))
debug_log("[DBG] time_s range:", float(time_s[0]), "->", float(time_s[-1]))

debug_log("[HTML-DBG] pulse_times_1 on/off:",
          0 if pulse_times_1 is None else len(pulse_times_1),
          0 if pulse_times_1_off is None else len(pulse_times_1_off))
debug_log("[HTML-DBG] ttl1_intervals:", len(ttl1_intervals), "example:", ttl1_intervals[:3])
debug_log("[HTML-CHECK] time range:", float(time_s[0]), "->", float(time_s[-1]))
debug_log("[HTML-CHECK] p1_on first/last:", pulse_times_1[:1], pulse_times_1[-1:])
debug_log("[HTML-CHECK] p1_off first/last:", pulse_times_1_off[:1], pulse_times_1_off[-1:])

# ---------- FINAL HTML PULSE CLIP (MUST MATCH time_s WINDOW) ----------
tmin = float(time_s[0])
tmax = float(time_s[-1])

def _clip_to_window(t, tmin, tmax):
    if t is None:
        return np.array([], float)
    t = np.asarray(t, float)
    t = t[np.isfinite(t)]
    if t.size == 0:
        return np.array([], float)
    return t[(t >= tmin) & (t <= tmax)]

pulse_times_1_html     = _clip_to_window(pulse_times_1_html,     tmin, tmax)
pulse_times_1_off_html = _clip_to_window(pulse_times_1_off_html, tmin, tmax)
pulse_times_2_html     = _clip_to_window(pulse_times_2_html,     tmin, tmax)
pulse_times_2_off_html = _clip_to_window(pulse_times_2_off_html, tmin, tmax)

# Intervals neu bauen (nach dem Clip!)
ttl1_intervals = _pair_on_off(pulse_times_1_html, pulse_times_1_off_html, max_width_s=5.0)
ttl2_intervals = _pair_on_off(pulse_times_2_html, pulse_times_2_off_html, max_width_s=5.0)

print("[HTML FINAL] p1_on/off:", len(pulse_times_1_html), len(pulse_times_1_off_html),
      "| ttl1_intervals:", len(ttl1_intervals))
print("[HTML FINAL] time window:", tmin, "->", tmax)

# If OFF edges are missing/unreliable, show only onset lines (no pulse-duration areas).
PULSE_ONSET_ONLY = False
if pulse_times_1_html is not None and len(pulse_times_1_html):
    if pulse_times_1_off_html is None or len(pulse_times_1_off_html) == 0:
        PULSE_ONSET_ONLY = True
    else:
        m_pw = min(len(pulse_times_1_html), len(pulse_times_1_off_html))
        if m_pw > 0:
            ww = np.asarray(pulse_times_1_off_html[:m_pw], float) - np.asarray(pulse_times_1_html[:m_pw], float)
            ww = ww[np.isfinite(ww) & (ww > 0)]
            if ww.size and float(np.median(ww)) <= max(2.0 * float(dt), 0.015):
                PULSE_ONSET_ONLY = True
if PULSE_ONSET_ONLY:
    ttl1_intervals = []
    ttl2_intervals = []
    print("[PULSE-PLOT] onset-only mode: pulse durations hidden in plots")






# Interaktive HTML (mit UP-Schattierung) 
export_interactive_lfp_html(
    BASE_TAG, SAVE_DIR, time_s,
    main_channel_uV if (HTML_IN_uV and main_channel_uV is not None) else main_channel,

    pulse_times_1=pulse_times_1_html,
    pulse_times_2=pulse_times_2_html,
    pulse_times_1_off=pulse_times_1_off_html,
    pulse_times_2_off=pulse_times_2_off_html,
    pulse_intervals_1=ttl1_intervals,
    pulse_intervals_2=ttl2_intervals,
    up_spont=(Spontaneous_UP, Spontaneous_DOWN),
    up_trig=(Pulse_triggered_UP, Pulse_triggered_DOWN),
    up_assoc=(Pulse_associated_UP, Pulse_associated_DOWN),
    spindle_intervals=spindle_intervals_s,
    limit_to_last_pulse=False,
    title=f"{BASE_TAG} — Main LFP (interaktiv)",
    y_label=("LFP (µV)" if HTML_IN_uV else f"LFP ({UNIT_LABEL})"),
    show_pulse_intervals=(not PULSE_ONSET_ONLY),
)


# Extras für Plots
pulse_windows = extract_upstate_windows(Pulse_triggered_UP, main_channel[None, :], dt, window_s=1.0)
spont_windows = extract_upstate_windows(Spontaneous_UP, main_channel[None, :], dt, window_s=1.0)
freqs = spont_mean = pulse_mean = p_vals = p_vals_fdr = None
spectra_meta = None
try:
    freqs, spont_mean, pulse_mean, p_vals, p_vals_fdr, spectra_meta = compare_spectra(
        pulse_windows,
        spont_windows,
        dt,
        ignore_start_s=0.3,
        baseline_mode=SPECTRA_BASELINE_MODE,
    )
except Exception as e:
    print("[WARN] spectra compare skipped:", e)



n_time = LFP_array_good.shape[1]

# Onsets aus UP/DOWN-Listen (zeitlich stabiler als Peaks) 
Spon_Onsets = _up_onsets(Spontaneous_UP,       Spontaneous_DOWN)
Trig_Onsets = _up_onsets(Pulse_triggered_UP,   Pulse_triggered_DOWN)

# Gültigkeitsgrenzen
Spon_Onsets = Spon_Onsets[(Spon_Onsets >= 0) & (Spon_Onsets < LFP_array_good.shape[1])]
Trig_Onsets = Trig_Onsets[(Trig_Onsets >= 0) & (Trig_Onsets < LFP_array_good.shape[1])]

CSD_spont = CSD_trig = None
CSD_DIFF = None
CSD_SEM_SPONT = CSD_SEM_TRIG = None
CSD_SEM_MED_SPONT = np.nan
CSD_SEM_MED_TRIG = np.nan
CSD_N_SPONT = CSD_N_TRIG = CSD_N_MATCH = 0
CSD_spont_latcorr = CSD_trig_latcorr = None
CSD_DIFF_LATCORR = None
CSD_SEM_MED_SPONT_LATCORR = np.nan
CSD_SEM_MED_TRIG_LATCORR = np.nan
CSD_N_SPONT_LATCORR = CSD_N_TRIG_LATCORR = CSD_N_MATCH_LATCORR = 0
CSD_TRIG_LAT_SHIFT_S = np.nan

CSD_PRE_DESIRED  = 0.5   # 0.5 s vor Onset
CSD_POST_DESIRED = 0.5   # 0.5 s nach Onset
CSD_BASELINE_PRE_S = 0.2
CSD_USE_CLIP_TO_DOWN = False
CSD_MIN_KEEP_FRAC = 0.8
ALIGN_TRIG_TO_PULSE_LATENCY = True
LATENCY_ALIGN_MAX_WIN_S = 1.0

if NUM_CHANNELS_GOOD >= 7 and (Spon_Onsets.size >= 3 or Trig_Onsets.size >= 3):
    try:
        CSD_sp_stack = None
        CSD_tr_stack = None
        CSD_tr_stack_latcorr = None
        Trig_Onsets_latcorr = None

        if ALIGN_TRIG_TO_PULSE_LATENCY and Trig_Onsets.size:
            lat_for_align = pulse_to_up_latencies(
                pulse_times_1,
                Pulse_triggered_UP,
                time_s,
                max_win_s=LATENCY_ALIGN_MAX_WIN_S,
            )
            lat_for_align = np.asarray(lat_for_align, float)
            lat_for_align = lat_for_align[np.isfinite(lat_for_align)]
            if lat_for_align.size:
                CSD_TRIG_LAT_SHIFT_S = float(np.nanmedian(lat_for_align))
                shift_n = int(round(CSD_TRIG_LAT_SHIFT_S / dt))
                Trig_Onsets_latcorr = Trig_Onsets + shift_n
                Trig_Onsets_latcorr = Trig_Onsets_latcorr[
                    (Trig_Onsets_latcorr >= 0) & (Trig_Onsets_latcorr < LFP_array_good.shape[1])
                ]
                print(
                    f"[CSD][LAT-ALIGN] shift={CSD_TRIG_LAT_SHIFT_S:.4f}s "
                    f"({shift_n} samples), n={len(Trig_Onsets_latcorr)}"
                )

        if Spon_Onsets.size >= 3:
            CSD_spont, CSD_sp_stack = Generate_CSD_mean_from_onsets(
                Spon_Onsets,
                LFP_array_good,
                dt,
                pre_s=CSD_PRE_DESIRED,
                post_s=CSD_POST_DESIRED,
                clip_to_down=(Spontaneous_DOWN if CSD_USE_CLIP_TO_DOWN else None),
                baseline_pre_s=CSD_BASELINE_PRE_S,
                min_keep_frac=CSD_MIN_KEEP_FRAC,
                return_stack=True,
            )

        if Trig_Onsets.size >= 3:
            CSD_trig, CSD_tr_stack = Generate_CSD_mean_from_onsets(
                Trig_Onsets,
                LFP_array_good,
                dt,
                pre_s=CSD_PRE_DESIRED,
                post_s=CSD_POST_DESIRED,
                clip_to_down=(Pulse_triggered_DOWN if CSD_USE_CLIP_TO_DOWN else None),
                baseline_pre_s=CSD_BASELINE_PRE_S,
                min_keep_frac=CSD_MIN_KEEP_FRAC,
                return_stack=True,
            )

        if Trig_Onsets_latcorr is not None and Trig_Onsets_latcorr.size >= 3:
            CSD_trig_latcorr, CSD_tr_stack_latcorr = Generate_CSD_mean_from_onsets(
                Trig_Onsets_latcorr,
                LFP_array_good,
                dt,
                pre_s=CSD_PRE_DESIRED,
                post_s=CSD_POST_DESIRED,
                clip_to_down=(Pulse_triggered_DOWN if CSD_USE_CLIP_TO_DOWN else None),
                baseline_pre_s=CSD_BASELINE_PRE_S,
                min_keep_frac=CSD_MIN_KEEP_FRAC,
                return_stack=True,
            )

        # n-matched Vergleich (robuster bei ungleichen Trialzahlen)
        def _csd_pair_summary(sp_stack, tr_stack):
            n_sp = int(sp_stack.shape[0])
            n_tr = int(tr_stack.shape[0])
            n_match = int(min(n_sp, n_tr))
            if n_match <= 0:
                return None
            tmin = int(min(sp_stack.shape[2], tr_stack.shape[2]))
            sp_stack = sp_stack[:, :, :tmin]
            tr_stack = tr_stack[:, :, :tmin]
            rng = np.random.default_rng(0)
            idx_sp = rng.choice(n_sp, size=n_match, replace=False)
            idx_tr = rng.choice(n_tr, size=n_match, replace=False)
            sp_m = sp_stack[idx_sp]
            tr_m = tr_stack[idx_tr]
            sem_sp = np.nanstd(sp_m, axis=0) / np.sqrt(max(1, n_match))
            sem_tr = np.nanstd(tr_m, axis=0) / np.sqrt(max(1, n_match))
            return {
                "sp_mean": np.nanmean(sp_m, axis=0),
                "tr_mean": np.nanmean(tr_m, axis=0),
                "diff": np.nanmean(tr_m, axis=0) - np.nanmean(sp_m, axis=0),
                "sem_sp_med": float(np.nanmedian(np.abs(sem_sp))),
                "sem_tr_med": float(np.nanmedian(np.abs(sem_tr))),
                "n_sp": n_sp,
                "n_tr": n_tr,
                "n_match": n_match,
            }

        if (CSD_sp_stack is not None) and (CSD_tr_stack is not None):
            out_raw = _csd_pair_summary(CSD_sp_stack, CSD_tr_stack)
            if out_raw is not None:
                CSD_spont = out_raw["sp_mean"]
                CSD_trig = out_raw["tr_mean"]
                CSD_DIFF = out_raw["diff"]
                CSD_SEM_MED_SPONT = out_raw["sem_sp_med"]
                CSD_SEM_MED_TRIG = out_raw["sem_tr_med"]
                CSD_N_SPONT = out_raw["n_sp"]
                CSD_N_TRIG = out_raw["n_tr"]
                CSD_N_MATCH = out_raw["n_match"]

        if (CSD_sp_stack is not None) and (CSD_tr_stack_latcorr is not None):
            out_corr = _csd_pair_summary(CSD_sp_stack, CSD_tr_stack_latcorr)
            if out_corr is not None:
                CSD_spont_latcorr = out_corr["sp_mean"]
                CSD_trig_latcorr = out_corr["tr_mean"]
                CSD_DIFF_LATCORR = out_corr["diff"]
                CSD_SEM_MED_SPONT_LATCORR = out_corr["sem_sp_med"]
                CSD_SEM_MED_TRIG_LATCORR = out_corr["sem_tr_med"]
                CSD_N_SPONT_LATCORR = out_corr["n_sp"]
                CSD_N_TRIG_LATCORR = out_corr["n_tr"]
                CSD_N_MATCH_LATCORR = out_corr["n_match"]

        align_pre_s  = CSD_PRE_DESIRED
        align_post_s = CSD_POST_DESIRED

    except Exception as e:
        print("[WARN] CSD generation failed:", e)
        CSD_spont = CSD_trig = None
        CSD_DIFF = None
        CSD_spont_latcorr = CSD_trig_latcorr = None
        CSD_DIFF_LATCORR = None
        align_pre_s  = CSD_PRE_DESIRED
        align_post_s = CSD_POST_DESIRED
else:
    print(f"[INFO] CSD skipped: channels={NUM_CHANNELS_GOOD}, "
          f"spon_onsets={Spon_Onsets.size}, trig_onsets={Trig_Onsets.size}")
    align_pre_s  = CSD_PRE_DESIRED
    align_post_s = CSD_POST_DESIRED

_check_peak_indices("Spon_Peaks", Spon_Peaks, LFP_array_good.shape[1])
_check_peak_indices("Trig_Peaks", Trig_Peaks, LFP_array_good.shape[1])


# 1) CSD-Grundstats
_nan_stats("CSD_spont", CSD_spont)
_nan_stats("CSD_trig",  CSD_trig)
debug_log(f"[DIAG] RMS CSD: spont={_rms(CSD_spont):.4g}, trig={_rms(CSD_trig):.4g}")


# 3) Prüfen, ob Cropping Spontan-Events stark reduziert
debug_log(f"[DIAG] Cropped time_s: {time_s[0]:.3f}..{time_s[-1]:.3f} s, pulses p1={len(pulse_times_1)}, p2={len(pulse_times_2)}")
debug_log(f"[DIAG] Counts: sponUP={len(Spontaneous_UP)}, trigUP={len(Pulse_triggered_UP)}, assocUP={len(Pulse_associated_UP)}")

# 4) Gleiche Zeitfenster/Alignment sicher? (pre/post)
debug_log(f"[DIAG] align_pre={align_pre_s:.3f}s, align_post={align_post_s:.3f}s, dt={dt:.6f}s")


# nach compare_spectra
if freqs is not None and spont_mean is not None:
    pd.DataFrame({"freq": freqs, "power": spont_mean}).to_csv(
        os.path.join(SAVE_DIR, "spectrum_spont.csv"), index=False)
if freqs is not None and pulse_mean is not None:
    pd.DataFrame({"freq": freqs, "power": pulse_mean}).to_csv(
        os.path.join(SAVE_DIR, "spectrum_trig.csv"), index=False)


# 1) CSD-Grundstats
_nan_stats("CSD_spont", CSD_spont)
_nan_stats("CSD_trig",  CSD_trig)
debug_log(f"[DIAG] RMS CSD: spont={_rms(CSD_spont):.4g}, trig={_rms(CSD_trig):.4g}")



# 3) Prüfen, ob Cropping Spontan-Events stark reduziert
debug_log(f"[DIAG] Cropped time_s: {time_s[0]:.3f}..{time_s[-1]:.3f} s, pulses p1={len(pulse_times_1)}, p2={len(pulse_times_2)}")
debug_log(f"[DIAG] Counts: sponUP={len(Spontaneous_UP)}, trigUP={len(Pulse_triggered_UP)}, assocUP={len(Pulse_associated_UP)}")

debug_log(f"[DIAG] align_pre={align_pre_s:.3f}s, align_post={align_post_s:.3f}s, dt={dt:.6f}s")


debug_log("[DEBUG] time range:", float(time_s[0]), "->", float(time_s[-1]))
if len(pulse_times_1):
    debug_log("[DEBUG] p1 first/last:", float(pulse_times_1[0]), float(pulse_times_1[-1]), "count:", len(pulse_times_1))
if len(pulse_times_2):
    debug_log("[DEBUG] p2 first/last:", float(pulse_times_2[0]), float(pulse_times_2[-1]), "count:", len(pulse_times_2))



def plot_up_classification_ax(
    main_channel, time_s,
    Spontaneous_UP, Spontaneous_DOWN,
    Pulse_triggered_UP, Pulse_triggered_DOWN,
    Pulse_associated_UP, Pulse_associated_DOWN,
    *,  # ab hier nur noch keyword-args
    pulse_times_1=None, pulse_times_2=None,
    spindle_intervals=None,
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
    if spindle_intervals:
        first = True
        for t0, t1 in spindle_intervals:
            if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
                continue
                ax.axvspan(float(t0), float(t1), color="#8a2be2", alpha=0.30, lw=0,
                       label=("Spindles 10-15 Hz" if first else None))
            first = False

    # 4) y-Limits nach Schattierung holen
    y0, y1 = ax.get_ylim()

    # 5) Pulsezeiten als vlines (ohne die y-Limits zu verändern)

    def _vlines(ts, style, label, color="red"):
        if ts is None or len(ts) == 0:
            return
        t = np.asarray(ts, float)
        if t.size > 800:  # bei sehr vielen Pulsen etwas ausdünnen
            step = int(np.ceil(t.size / 800))
            t = t[::step]
        ax.vlines(
            t, y0, y1,
            lw=0.9,
            color=color,
            alpha=0.35,
            linestyles=style,
            label=label,
            zorder=1
        )

    _vlines(pulse_times_1, ":", "Pulse 1", color="red")
    _vlines(pulse_times_2, ":", "Pulse 2", color="red")


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


def csd_delta_raw_vs_latcorr_ax(
    csd_diff_raw,
    csd_diff_latcorr,
    dt,
    *,
    z_mm=None,
    align_pre=0.5,
    align_post=0.5,
    ax=None,
    title="ΔCSD (raw vs latency-corrected)",
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3.4))
    else:
        fig = ax.figure

    A = None if csd_diff_raw is None else np.asarray(csd_diff_raw, float)
    B = None if csd_diff_latcorr is None else np.asarray(csd_diff_latcorr, float)
    if A is None or B is None or A.ndim != 2 or B.ndim != 2 or A.size == 0 or B.size == 0:
        _blank_ax(ax, "no raw-vs-corrected CSD delta")
        return fig

    tmin = int(min(A.shape[1], B.shape[1]))
    cmin = int(min(A.shape[0], B.shape[0]))
    A = A[:cmin, :tmin]
    B = B[:cmin, :tmin]
    D = B - A

    vmax = float(np.nanpercentile(np.abs(D[np.isfinite(D)]), 95)) if np.isfinite(D).any() else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    t = np.linspace(-float(align_pre), float(align_post), tmin)
    if z_mm is not None:
        z = np.asarray(z_mm, float)[:cmin]
        extent = [float(t[0]), float(t[-1]), float(z[0]), float(z[-1])]
        ylab = "Tiefe (mm)"
    else:
        extent = [float(t[0]), float(t[-1]), 0.0, float(cmin - 1)]
        ylab = "Tiefe (arb.)"

    im = ax.imshow(
        D, aspect="auto", origin="upper",
        extent=extent, cmap="Spectral_r", norm=norm, interpolation="bilinear"
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label("ΔCSD corrected-raw (a.u.)")
    ax.set_xlabel("Zeit (s)")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.axvline(0.0, color="k", lw=0.8, alpha=0.4, ls="--")
    return fig


# def _blank_ax(ax, msg=None):
#     ax.axis("off")
#     if msg:
#         ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, alpha=0.4)




def Power_spectrum_compare_ax(freqs, spont_mean, pulse_mean, p_vals=None, p_vals_fdr=None, meta=None, alpha=0.05, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,3))
    else:
        fig = ax.figure
    if freqs is None or spont_mean is None or pulse_mean is None or len(freqs)==0:
        ax.text(0.5,0.5,"no spectra", ha="center", va="center", transform=ax.transAxes)
        return fig
    f = np.asarray(freqs, float).ravel()
    y_sp = np.asarray(spont_mean, float).ravel()
    y_tr = np.asarray(pulse_mean, float).ravel()
    m = min(f.size, y_sp.size, y_tr.size)
    if m == 0:
        ax.text(0.5, 0.5, "no spectra", ha="center", va="center", transform=ax.transAxes)
        return fig
    f = f[:m]
    y_sp = y_sp[:m]
    y_tr = y_tr[:m]

    # Optisch bis 150 Hz verlängern (falls Spektrum vorher endet).
    if np.isfinite(f[-1]) and f[-1] < 150:
        f_plot = np.append(f, 150.0)
        y_sp_plot = np.append(y_sp, y_sp[-1])
        y_tr_plot = np.append(y_tr, y_tr[-1])
    else:
        f_plot, y_sp_plot, y_tr_plot = f, y_sp, y_tr

    ax.plot(f_plot, y_sp_plot, label="Spontan (mean)", lw=2)
    ax.plot(f_plot, y_tr_plot, label="Getriggert (mean)", lw=2)

    # Sanity: Median-Kurven
    if isinstance(meta, dict):
        sp_med = np.asarray(meta.get("spont_median", []), float).ravel()
        tr_med = np.asarray(meta.get("trig_median", []), float).ravel()
        if sp_med.size and tr_med.size:
            m2 = min(m, sp_med.size, tr_med.size)
            f2 = f[:m2]
            spm = sp_med[:m2]
            trm = tr_med[:m2]
            if np.isfinite(f2[-1]) and f2[-1] < 150:
                f2p = np.append(f2, 150.0)
                spm = np.append(spm, spm[-1])
                trm = np.append(trm, trm[-1])
            else:
                f2p = f2
            ax.plot(f2p, spm, lw=1.2, ls="--", alpha=0.9, label="Spontan (median)")
            ax.plot(f2p, trm, lw=1.2, ls="--", alpha=0.9, label="Getriggert (median)")

        # Sanity: n-gematchte Mittelwerte
        sp_mm = np.asarray(meta.get("spont_mean_matched", []), float).ravel()
        tr_mm = np.asarray(meta.get("trig_mean_matched", []), float).ravel()
        if sp_mm.size and tr_mm.size:
            m3 = min(m, sp_mm.size, tr_mm.size)
            f3 = f[:m3]
            sp3 = sp_mm[:m3]
            tr3 = tr_mm[:m3]
            if np.isfinite(f3[-1]) and f3[-1] < 150:
                f3p = np.append(f3, 150.0)
                sp3 = np.append(sp3, sp3[-1])
                tr3 = np.append(tr3, tr3[-1])
            else:
                f3p = f3
            ax.plot(f3p, sp3, lw=1.0, ls=":", alpha=0.9, label="Spontan (matched mean)")
            ax.plot(f3p, tr3, lw=1.0, ls=":", alpha=0.9, label="Getriggert (matched mean)")
    if SPECTRA_USE_FDR and (p_vals_fdr is not None and np.size(p_vals_fdr) == np.size(freqs)):
        p_sig = p_vals_fdr
    else:
        p_sig = p_vals
    if p_sig is not None and np.size(p_sig) == np.size(f):
        sig = (p_sig < alpha)
        if np.any(sig):
            idx = np.where(sig)[0]
            # zusammenhängende Bereiche füllen
            start = idx[0]
            for i in range(1,len(idx)+1):
                if i==len(idx) or idx[i] != idx[i-1]+1:
                    ax.axvspan(f[start], f[idx[i-1]], alpha=0.12)
                    if i < len(idx): start = idx[i]
    ax.set_xlim(0, 150)
    ax.set_xlabel("Hz")
    if SPECTRA_BASELINE_MODE == "pre_onset_db":
        ax.set_ylabel("ΔPower vs. pre-onset (dB)")
        ax.set_title("Power (Spontan vs. Getriggert, pre-onset normalized)")
    else:
        ax.set_ylabel(PSD_UNIT_LABEL)
        ax.set_title("Power (Spontan vs. Getriggert)")

    if isinstance(meta, dict):
        n_sp = int(meta.get("n_spont", 0))
        n_tr = int(meta.get("n_trig", 0))
        n_m = int(meta.get("n_match", 0))
        ax.text(
            0.98, 0.95,
            f"n_sp={n_sp}, n_tr={n_tr}\nmatched n={n_m}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.75)
        )
    ax.legend()
    return fig


def _save_all_channels_svg_from_array(time_s, LFP_array, chan_labels, out_svg, *, max_points=20000, title=None):
    """
    Alternative, falls schon das downsampled Array hast:
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
    if title is None:
        title = "Alle Kanäle (gestapelt, robust skaliert) — downsampled"
    ax.set_title(title)
    ax.grid(True, alpha=0.15, linestyle=":")

    fig.tight_layout()
    fig.savefig(out_svg, format="svg")
    plt.close(fig)
    del fig
    print(f"[ALL-CH] SVG geschrieben: {out_svg}")


try:
    _save_all_channels_svg_from_array(
        time_s, LFP_array_good, ch_names_good,
        os.path.join(SAVE_DIR, f"{BASE_TAG}__all_channels_GOOD.svg"),
        max_points=40000,
        title=f"Gefilterte Kanäle (n={NUM_CHANNELS_GOOD}/{NUM_CHANNELS})"
    )
except Exception as e:
    print("[ALL-CH][DS] skip:", e)

try:
    excluded_idx = [i for i in range(NUM_CHANNELS) if i not in set(good_idx)]
    if excluded_idx:
        _save_all_channels_svg_from_array(
            time_s,
            LFP_array[excluded_idx, :],
            [f"pri_{i}" for i in excluded_idx],
            os.path.join(SAVE_DIR, f"{BASE_TAG}__all_channels_EXCLUDED.svg"),
            max_points=40000
        )
    else:
        print("[ALL-CH][EXCLUDED] keine ausgeschlossenen Kanäle")
except Exception as e:
    print("[ALL-CH][EXCLUDED] skip:", e)


def up_onset_mean_ax(main_channel, dt, onsets, ax=None, title="UPs – onset-aligned mean"):

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
    ax.axvline(0.0, color="red", lw=1.0, ls="--", label="Pulse")

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



def refractory_from_spont_to_type_overlay_ax(
    main_channel,
    time_s,
    Spontaneous_UP, Spontaneous_DOWN,
    Pulse_triggered_UP, Pulse_triggered_DOWN,
    Pulse_associated_UP, Pulse_associated_DOWN,
    dt,
    target_type="spont",   # "spont" oder "trig"
    pre_s=0.5,
    post_s=2.0,
    ax=None,
    title=None
):
    """
    Overlay:
      0 s = Ende eines *spontanen* UP
      danach: LFP bis zum *ersten* UP-Event, ABER:

        - wenn erstes Event 'associated' ist -> SPONT-Off wird ignoriert
        - wenn erstes Event 'spont' ist UND target_type="spont" -> Event wird geplottet
        - wenn erstes Event 'trig'  ist UND target_type="trig"  -> Event wird geplottet
    """

    import numpy as np
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3.0))
    else:
        fig = ax.figure

    Spontaneous_UP      = np.asarray(Spontaneous_UP,      int)
    Spontaneous_DOWN    = np.asarray(Spontaneous_DOWN,    int)
    Pulse_triggered_UP  = np.asarray(Pulse_triggered_UP,  int)
    Pulse_associated_UP = np.asarray(Pulse_associated_UP, int)

    m_spon = min(len(Spontaneous_UP), len(Spontaneous_DOWN))
    if m_spon == 0:
        ax.text(0.5, 0.5, "no spontaneous UPs",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    Spontaneous_UP   = Spontaneous_UP[:m_spon]
    Spontaneous_DOWN = Spontaneous_DOWN[:m_spon]

    # Onset-Zeiten
    times_spon  = time_s[Spontaneous_UP]      if len(Spontaneous_UP)      else np.array([], float)
    times_trig  = time_s[Pulse_triggered_UP]  if len(Pulse_triggered_UP)  else np.array([], float)
    times_assoc = time_s[Pulse_associated_UP] if len(Pulse_associated_UP) else np.array([], float)

    # dazu passende Onset-Indizes
    idx_spon  = Spontaneous_UP
    idx_trig  = Pulse_triggered_UP
    idx_assoc = Pulse_associated_UP

    if times_spon.size == 0 and times_trig.size == 0 and times_assoc.size == 0:
        ax.text(0.5, 0.5, "no UP events",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    # 0 = spont, 1 = trig, 2 = assoc
    all_times   = np.concatenate([times_spon,         times_trig,         times_assoc])
    all_labels  = np.concatenate([
        np.zeros_like(times_spon,  dtype=int),
        np.ones_like(times_trig,   dtype=int),
        np.full_like(times_assoc,  2,         dtype=int)
    ])
    all_indices = np.concatenate([idx_spon,           idx_trig,           idx_assoc])

    order       = np.argsort(all_times)
    all_times   = all_times[order]
    all_labels  = all_labels[order]
    all_indices = all_indices[order]

    # welches Label ist target?
    if   target_type == "spont":
        desired_label = 0
    elif target_type == "trig":
        desired_label = 1
    else:
        desired_label = None  # theoretisch könnte man anderes unterstützen

    pre_n  = int(round(pre_s  / dt))
    post_n = int(round(post_s / dt))

    segs         = []
    refrac_times = []

    for down_idx in Spontaneous_DOWN:
        t_off = time_s[down_idx]

        # erster UP nach dem Off
        j = np.searchsorted(all_times, t_off + 1e-9)
        if j >= all_times.size:
            continue

        lab = all_labels[j]

        # assoc zuerst → diese SPONT-Off komplett ignorieren
        if lab == 2:
            continue

        # passt nicht zum gewünschten Zieltyp → ignorieren
        if desired_label is not None and lab != desired_label:
            continue

        dt_ref = all_times[j] - t_off
        if dt_ref < 0:
            continue

        # Segment um das SPONT-Off herum (0 s = Ende SPONT-UP)
        s = down_idx - pre_n
        e = down_idx + post_n
        if s < 0 or e > len(main_channel):
            continue

        seg = main_channel[s:e]
        if len(seg) != (pre_n + post_n):
            continue

        segs.append(seg)
        refrac_times.append(dt_ref)

    if not segs:
        ax.text(0.5, 0.5, "no valid segments",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    segs         = np.vstack(segs)
    refrac_times = np.asarray(refrac_times, float)
    n            = segs.shape[0]
    t_rel        = (np.arange(-pre_n, post_n) * dt)

    # Einzeltraces
    for i in range(n):
        ax.plot(t_rel, segs[i], alpha=0.12, lw=0.8)

    # Mitteltrace
    mean_trace = np.nanmean(segs, axis=0)
    ax.plot(t_rel, mean_trace, lw=2.0, label="Mean LFP")

    # vertikale Linie beim SPONT-UP-Off
    ax.axvline(0.0, color="k", lw=1.0, ls="--", label="SPONT-UP offset")

    # mittlere Refraktärzeit
    mean_ref = float(np.nanmean(refrac_times))
    ax.axvline(mean_ref, color="red", lw=1.5, ls=":",
               label=f"Mean next {target_type} UP ({mean_ref*1000:.0f} ms)")

    ax.set_xlabel("Zeit relativ zum Ende des spontanen UP (s)")
    ax.set_ylabel(f"LFP ({UNIT_LABEL})")

    if title is None:
        if target_type == "spont":
            title = "SPONT offset → nächste SPONT-UPs (Overlay)"
        elif target_type == "trig":
            title = "SPONT offset → nächste TRIG-UPs (Overlay)"
        else:
            title = f"SPONT offset → nächste {target_type}-UPs (Overlay)"

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


def refractory_from_spont_single_folder_ax(
    refrac_spon2spon, refrac_spon2trig,
    folder_name="Folder",
    ax=None,
    title="Refraktärzeit (SPONT-Off) – spon→spon vs. spon→trig"
):
    import numpy as np
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4))
    else:
        fig = ax.figure

    data   = [
        np.asarray(refrac_spon2spon, float),
        np.asarray(refrac_spon2trig, float)
    ]
    labels = ["spon → spon", "spon → trig"]
    x_pos  = [0, 1]

    for i, arr in enumerate(data):
        if arr.size == 0:
            continue

        # 🔴 Jeder Punkt = EINE Refraktärzeit dieser Kategorie
        jitter = (np.random.rand(arr.size) - 0.5) * 0.15
        ax.scatter(
            x_pos[i] + jitter,
            arr,
            alpha=0.6,
            s=20,
            color="black"
        )

        # rote Mittelwertlinie pro Kategorie
        mean_val = float(np.nanmean(arr))
        ax.hlines(
            mean_val,
            x_pos[i] - 0.25,
            x_pos[i] + 0.25,
            color="red",
            linewidth=2
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Refraktärzeit (s)")
    ax.set_title(f"{title}\n{folder_name}")
    ax.grid(alpha=0.2, linestyle=":")
    return fig


def spontaneous_up_full_overlay_normtime_ax(
    main_channel,
    time_s,
    Spontaneous_UP,
    Spontaneous_DOWN,
    *,
    n_points=300,          # Auflösung der normierten Zeitachse
    ax=None,
    title="Spontaneous UPs (Onset→Offset) overlay, time-normalized"
):

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.8, 3.2))
    else:
        fig = ax.figure

    U = np.asarray(Spontaneous_UP, int)
    D = np.asarray(Spontaneous_DOWN, int)
    m = min(len(U), len(D))
    if m == 0:
        ax.text(0.5, 0.5, "no spontaneous UPs", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    U, D = U[:m], D[:m]
    # nach Zeit sortieren
    order = np.argsort(time_s[U])
    U, D = U[order], D[order]

    X = []
    for u, d in zip(U, D):
        if d <= u + 3:
            continue
        if u < 0 or d > len(main_channel):
            continue
        seg = np.asarray(main_channel[u:d], float)
        if seg.size < 5 or not np.isfinite(seg).any():
            continue

        # auf normierte Zeit 0..1 resamplen
        x_old = np.linspace(0.0, 1.0, seg.size)
        x_new = np.linspace(0.0, 1.0, n_points)
        seg_rs = np.interp(x_new, x_old, np.nan_to_num(seg, nan=np.nanmedian(seg)))
        X.append(seg_rs)

    if not X:
        ax.text(0.5, 0.5, "no valid UP segments", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    X = np.vstack(X)  # (n_events, n_points)
    n = X.shape[0]
    t = np.linspace(0.0, 100.0, n_points)  # Prozent der UP-Dauer

    # Einzeltraces
    for i in range(n):
        ax.plot(t, X[i], alpha=0.10, lw=0.8)

    # Mean ± SEM
    mtrace = np.nanmean(X, axis=0)
    sem    = np.nanstd(X, axis=0) / np.sqrt(max(1, n))
    ax.plot(t, mtrace, lw=2.2, label="Mean")
    ax.fill_between(t, mtrace-sem, mtrace+sem, alpha=0.20, label="SEM")

    ax.set_xlabel("Zeit innerhalb UP (% von Onset→Offset)")
    ax.set_ylabel(f"LFP ({UNIT_LABEL})")
    ax.set_title(title)

    ax.text(
        0.02, 0.95, f"n = {n}",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=10, bbox=dict(boxstyle="round", fc="white", alpha=0.7)
    )

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    if uniq:
        ax.legend(uniq.values(), uniq.keys(), loc="upper right", fontsize=8, framealpha=0.9)

    return fig

def _rms(x):
    x = np.asarray(x, float)
    return float(np.sqrt(np.nanmean(x*x))) if x.size else np.nan

def _extract_resampled_up_segments(
    main_channel, time_s, U, D,
    *, n_points=300, min_len_s=0.05, baseline_frac=0.1, do_rms_norm=True
):
    """
    Extrahiert UP-Segmente onset->offset, resampelt jedes Segment auf n_points (0..1),
    baseline-subtract (Median der ersten baseline_frac) und optional RMS-normalisiert.
    Returns: X (n_events, n_points), meta dict
    """

    U = np.asarray(U, int)
    D = np.asarray(D, int)
    m = min(len(U), len(D))
    if m == 0:
        return np.empty((0, n_points), float), {"kept": 0, "skipped": 0}

    U, D = U[:m], D[:m]
    order = np.argsort(time_s[U])
    U, D = U[order], D[order]

    X = []
    skipped = 0
    for u, d in zip(U, D):
        if u < 0 or d > len(main_channel) or d <= u + 3:
            skipped += 1
            continue

        seg = np.asarray(main_channel[u:d], float)
        if seg.size < 5:
            skipped += 1
            continue

        dur_s = float(time_s[d-1] - time_s[u]) if (0 <= d-1 < len(time_s)) else 0.0
        if dur_s < float(min_len_s):
            skipped += 1
            continue

        # baseline subtract: Median der ersten baseline_frac
        nb = max(1, int(round(baseline_frac * seg.size)))
        base = float(np.nanmedian(seg[:nb])) if np.isfinite(seg[:nb]).any() else 0.0
        seg = seg - base

        # resample auf 0..1
        x_old = np.linspace(0.0, 1.0, seg.size)
        x_new = np.linspace(0.0, 1.0, n_points)
        seg_rs = np.interp(x_new, x_old, np.nan_to_num(seg, nan=np.nanmedian(seg)))

        # RMS norm (Form statt Amplitude)
        if do_rms_norm:
            r = _rms(seg_rs)
            if not np.isfinite(r) or r <= 1e-12:
                skipped += 1
                continue
            seg_rs = seg_rs / r

        X.append(seg_rs)

    if not X:
        return np.empty((0, n_points), float), {"kept": 0, "skipped": skipped}

    X = np.vstack(X)
    return X, {"kept": int(X.shape[0]), "skipped": int(skipped)}

def fit_pca_from_spont(X_spont, *, n_components=3):
    """
    PCA via SVD (ohne sklearn).
    Input: X_spont shape (n, T)
    Returns dict mit mean, components (k,T), scores_spont (n,k), cov_inv (k,k)
    """
    import numpy as np

    X = np.asarray(X_spont, float)
    n, T = X.shape
    if n < 3:
        raise ValueError(f"Need >=3 spontaneous events for PCA, got n={n}")

    mu = np.nanmean(X, axis=0)
    Xc = X - mu

    # SVD: Xc = U S Vt
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = int(min(n_components, Vt.shape[0]))
    comps = Vt[:k, :]             # (k, T)
    scores = Xc @ comps.T         # (n, k)

    # Cov im Score-space + Regularisierung
    cov = np.cov(scores, rowvar=False)
    if cov.ndim == 0:  # k=1 edge
        cov = np.array([[float(cov)]], float)
    reg = 1e-6 * np.trace(cov) if np.isfinite(np.trace(cov)) else 1e-6
    cov_reg = cov + reg * np.eye(cov.shape[0])
    cov_inv = np.linalg.inv(cov_reg)

    return {
        "mean": mu,
        "components": comps,
        "scores_spont": scores,
        "cov_inv": cov_inv,
    }

def pca_project_and_similarity(X, pca_fit):
    """
    projiziert X (n,T) in PCA und berechnet:
    - pc_scores (n,k)
    - pc1_z (z-score relativ zu spont pc1)
    - mahal (Mahalanobis distance im k-space)
    """
 

    X = np.asarray(X, float)
    mu = pca_fit["mean"]
    comps = pca_fit["components"]
    scores_sp = np.asarray(pca_fit["scores_spont"], float)
    cov_inv = np.asarray(pca_fit["cov_inv"], float)

    Xc = X - mu
    sc = Xc @ comps.T  # (n,k)

    # PC1 z-score relativ zu spont
    pc1_sp = scores_sp[:, 0]
    pc1_mu = float(np.nanmean(pc1_sp))
    pc1_sd = float(np.nanstd(pc1_sp)) if np.isfinite(np.nanstd(pc1_sp)) and np.nanstd(pc1_sp) > 1e-12 else 1.0
    pc1_z = (sc[:, 0] - pc1_mu) / pc1_sd

    # Mahalanobis (im k-space), zentriert um spont-mean im score-space
    sc_mu = np.nanmean(scores_sp, axis=0)
    d = sc - sc_mu
    # mahal^2 = d @ cov_inv @ d^T (rowwise)
    mahal2 = np.einsum("ni,ij,nj->n", d, cov_inv, d)
    mahal = np.sqrt(np.maximum(mahal2, 0.0))

    return sc, pc1_z, mahal

# ---------- Plotter (ax-friendly) ----------

def pca_template_pc1_ax(
    pca_fit_joint, *, ax=None, title="Spontaneous PCA Template (PC1)"
):

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3.0))
    else:
        fig = ax.figure

    pc1 = np.asarray(pca_fit_joint["components"][0], float)
    t = np.linspace(0.0, 100.0, pc1.size)

    ax.plot(t, pc1, lw=2.2, label="PC1")
    ax.axhline(0, lw=0.8, alpha=0.4)
    ax.set_xlabel("Zeit innerhalb UP (% Onset→Offset)")
    ax.set_ylabel("Amplitude (normiert)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    return fig

def pca_similarity_stats_ax(corr_sp, corr_tr, p_val=None, ax=None,
                            title="UP similarity to spontaneous PC1"):
    import numpy as np
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))
    else:
        fig = ax.figure

    data = [corr_sp, corr_tr]
    labels = ["Spontaneous", "Triggered"]

    # Violin + Punkte
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_alpha(0.4)

    rng = np.random.default_rng(0)
    for i, arr in enumerate(data, start=1):
        jitter = (rng.random(len(arr)) - 0.5) * 0.15
        ax.scatter(np.full(len(arr), i) + jitter, arr,
                   s=18, alpha=0.7, color="black")

        # Mittelwert
        m = np.nanmean(arr)
        ax.hlines(m, i - 0.25, i + 0.25, color="red", lw=2)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels)
    ax.set_ylabel("Correlation to PC1_spont")
    ax.set_title(title)
    ax.grid(alpha=0.2, linestyle=":")

    # Statistik-Textbox
    if p_val is not None:
        ax.text(
            0.98, 0.95,
            f"p = {p_val:.2e}\n"
            f"n_sp = {len(corr_sp)}\n"
            f"n_tr = {len(corr_tr)}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.7)
        )

    return fig


def pca_similarity_scores_ax(
    scores_spont, pc1z_trig, mahal_trig,
    *, ax=None, title="Triggered UP similarity vs. spontaneous PCA"
):


    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
    else:
        fig = ax.figure

    pc1_sp = np.asarray(scores_spont[:, 0], float)
    # z-score der spont PC1 (damit gleiche Skala)
    sp_mu = float(np.nanmean(pc1_sp))
    sp_sd = float(np.nanstd(pc1_sp)) if np.nanstd(pc1_sp) > 1e-12 else 1.0
    pc1z_sp = (pc1_sp - sp_mu) / sp_sd

    pc1z_trig = np.asarray(pc1z_trig, float)
    mahal_trig = np.asarray(mahal_trig, float)

    # Histogramm der spont-Verteilung (z)
    ax.hist(pc1z_sp, bins=20, alpha=0.35, label="Spont PC1 z")

    # Trigger als Punkte auf derselben Achse (jitter in y über counts ist nervig -> wir setzen sie auf 0-Linie)
    y0 = np.zeros_like(pc1z_trig)
    ax.scatter(pc1z_trig, y0, s=18, alpha=0.8, label="Trig PC1 z")

    ax.axvline(0.0, lw=1.0, ls="--", alpha=0.6)
    ax.set_xlabel("PC1 z-score (relativ zu spont)")
    ax.set_ylabel("Spont histogram / Trig markers")
    ax.set_title(title)

    # kleine Summary box
    def _nanmean(x): 
        x = np.asarray(x, float)
        return float(np.nanmean(x)) if x.size else np.nan

    ax.text(
        0.98, 0.95,
        f"trig n={pc1z_trig.size}\n"
        f"|z| mean={_nanmean(np.abs(pc1z_trig)):.2f}\n"
        f"mahal mean={_nanmean(mahal_trig):.2f}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.7)
    )

    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    return fig



# --- PCA Template aus spontanen UPs (onset->offset, time-normalized) ---
X_spont, meta_sp = _extract_resampled_up_segments(
    main_channel, time_s,
    Spontaneous_UP, Spontaneous_DOWN,
    n_points=300, baseline_frac=0.1, do_rms_norm=True
)

X_trig, meta_tr = _extract_resampled_up_segments(
    main_channel, time_s,
    Pulse_triggered_UP, Pulse_triggered_DOWN,
    n_points=300, baseline_frac=0.1, do_rms_norm=True
)


try:
    pca_fit_sp
except NameError:
    pca_fit_sp = None


# --- PCA im gemeinsamen Raum: X_joint = spont + trig ---
pca_fit_joint = None
pc_scores_sp = pc1z_sp = mahal_sp = None
pc_scores_tr = pc1z_trig = mahal_trig = None

pc1_ref = None

# --- PCA im gemeinsamen Raum: X_joint = spont + trig ---
pca_fit_joint = None
pc_scores_sp = pc1z_sp = mahal_sp = None
pc_scores_tr = pc1z_trig = mahal_trig = None

pc1_ref = None

# pc1_ref aus spont-PCA holen, falls vorhanden
if (pca_fit_sp is not None
    and isinstance(pca_fit_sp, dict)
    and ("components" in pca_fit_sp)
    and (pca_fit_sp["components"] is not None)):
    
    comps = np.asarray(pca_fit_sp["components"], float)
    if comps.ndim >= 2 and comps.shape[0] >= 1 and comps.shape[1] > 0:
        pc1_ref = comps[0].copy()
    else:
        print("[PCA] pc1_ref skipped: components empty/invalid shape:", getattr(comps, "shape", None))
else:
    print("[PCA] pc1_ref skipped: pca_fit_sp is missing/None or missing 'components'")

# Vorzeichen fixieren NUR wenn pc1_ref gültig ist
if pc1_ref is None:
    print("[PCA] pc1_ref is None -> skip sign fix")
else:
    pc1_ref = np.asarray(pc1_ref, float)
    if pc1_ref.size == 0 or np.all(np.isnan(pc1_ref)):
        print("[PCA] pc1_ref empty/all-NaN -> skip sign fix")
    else:
        if np.nanmean(pc1_ref) < 0:
            pc1_ref = -pc1_ref

if X_spont.shape[0] >= 3 and X_trig.shape[0] >= 3:
    X_joint = np.vstack([X_spont, X_trig])

    # PCA fit im gemeinsamen Raum
    pca_fit_joint = fit_pca_from_spont(X_joint, n_components=3)

    # Projektionen beider Gruppen in denselben Raum
    pc_scores_sp, pc1z_sp, mahal_sp     = pca_project_and_similarity(X_spont, pca_fit_joint)
    pc_scores_tr, pc1z_trig, mahal_trig = pca_project_and_similarity(X_trig,  pca_fit_joint)

    # CSV export: beide Gruppen
    import pandas as pd, os, numpy as np
    df_sp = pd.DataFrame({
        "event_type": ["spont"] * len(pc1z_sp),
        "pc1_score": pc_scores_sp[:, 0],
        "pc1_z": pc1z_sp,
        "mahal_k3": mahal_sp,
    })
    df_tr = pd.DataFrame({
        "event_type": ["triggered"] * len(pc1z_trig),
        "pc1_score": pc_scores_tr[:, 0],
        "pc1_z": pc1z_trig,
        "mahal_k3": mahal_trig,
    })
    pca_csv = os.path.join(SAVE_DIR, f"{BASE_TAG}__pca_similarity_joint.csv")
    pd.concat([df_sp, df_tr], ignore_index=True).to_csv(pca_csv, index=False)
    print("[CSV] PCA joint:", pca_csv, f"(sp={len(df_sp)}, trig={len(df_tr)})")
else:
    print(f"[PCA JOINT] skipped: need >=3 spont AND >=3 trig "
          f"(sp={X_spont.shape[0]}, trig={X_trig.shape[0]}) meta_sp={meta_sp} meta_tr={meta_tr}")

pca_fit_sp = None
if X_spont.shape[0] >= 3:
    pca_fit_sp = fit_pca_from_spont(X_spont, n_components=3)

# Trigger-PCA separat (für Template/Overlay-Plots in layout_rows)
pca_fit_tr = None
if X_trig.shape[0] >= 3:
    pca_fit_tr = fit_pca_from_spont(X_trig, n_components=3)

# Falls pc1_ref oben noch nicht verfügbar war, jetzt aus spont-PCA ziehen.
if pc1_ref is None and pca_fit_sp is not None:
    comps_sp = np.asarray(pca_fit_sp.get("components", []), float)
    if comps_sp.ndim >= 2 and comps_sp.shape[0] >= 1 and comps_sp.shape[1] > 0:
        pc1_ref = comps_sp[0].copy()
        if np.isfinite(pc1_ref).any() and np.nanmean(pc1_ref) < 0:
            pc1_ref = -pc1_ref


def corr_to_template(X, template):
    template = (template - template.mean()) / template.std()
    out = []
    for x in X:
        xz = (x - x.mean()) / x.std()
        out.append(np.corrcoef(xz, template)[0, 1])
    return np.asarray(out)

if pc1_ref is None:
    print("[PCA] corr_to_template skipped: pc1_ref is None (likely NO-UP or PCA skipped)")
    corr_sp = None
else:
    corr_sp = corr_to_template(X_spont, pc1_ref)

if pc1_ref is None:
    print("[PCA] corr_to_template(trig) skipped: pc1_ref is None")
    corr_tr = None
else:
    corr_tr = corr_to_template(X_trig, pc1_ref)



def mahal_compare_ax(mahal_sp, mahal_trig, ax=None, title="Mahalanobis distance to spontaneous PCA-space"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
    else:
        fig = ax.figure

    ms = np.asarray(mahal_sp, float)
    mt = np.asarray(mahal_trig, float)

    data = []
    labels = []

    if ms.size:
        data.append(ms); labels.append("Spont")
    if mt.size:
        data.append(mt); labels.append("Trig")

    if not data:
        ax.text(0.5, 0.5, "no mahal distances", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    # Boxplot
    ax.boxplot(data, labels=labels, whis=[5, 95], showfliers=False)

    # Punkte drüber (jitter)
    rng = np.random.default_rng(0)
    for i, arr in enumerate(data, start=1):
        jitter = (rng.random(arr.size) - 0.5) * 0.18
        ax.scatter(np.full(arr.size, i) + jitter, arr, s=18, alpha=0.6, color="black")

        # Mittelwertlinie
        m = float(np.nanmean(arr)) if arr.size else np.nan
        if np.isfinite(m):
            ax.hlines(m, i-0.25, i+0.25, color="red", lw=2)

    ax.set_ylabel("Mahalanobis distance (k PCs)")
    ax.set_title(title)
    ax.grid(alpha=0.15, linestyle=":")

    # kleine summary
    def _s(arr):
        arr = np.asarray(arr, float)
        if arr.size == 0: return "n=0"
        return f"n={arr.size}, mean={np.nanmean(arr):.2f}, med={np.nanmedian(arr):.2f}"
    ax.text(
        0.98, 0.95,
        f"Spont: {_s(ms)}\nTrig:  {_s(mt)}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.7)
    )
    return fig

def _find_clusters(mask_bool):
    """Return list of (start, end) inclusive clusters where mask_bool is True."""
    clusters = []
    in_cl = False
    start = 0
    for i, v in enumerate(mask_bool):
        if v and not in_cl:
            in_cl = True
            start = i
        elif not v and in_cl:
            clusters.append((start, i - 1))
            in_cl = False
    if in_cl:
        clusters.append((start, len(mask_bool) - 1))
    return clusters



def ranked_trigger_overlays_ax(
    X_spont, X_trig, mahal_trig,
    ax=None,
    top_n=5,
    title="Triggered UPs ranked by similarity (Mahalanobis)"
):
    """
    X_spont, X_trig: resampled (n, T), baseline-subtracted, RMS-normalized (wie in deinem PCA-Preprocessing)
    mahal_trig: (n_trig,) Mahalanobis distances (kleiner = ähnlicher)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
    else:
        fig = ax.figure

    Xs = np.asarray(X_spont, float)
    Xt = np.asarray(X_trig, float)
    md = np.asarray(mahal_trig, float)

    if Xs.size == 0 or Xt.size == 0 or md.size == 0:
        ax.text(0.5, 0.5, "no data for ranked overlays", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    T = Xt.shape[1]
    t = np.linspace(0.0, 100.0, T)

    # spont: dünn grau als Referenz + median als robuste "reference"
    for i in range(min(Xs.shape[0], 80)):  # nicht zu voll
        ax.plot(t, Xs[i], lw=0.8, alpha=0.06, color="black")
    ref = np.nanmedian(Xs, axis=0)
    ax.plot(t, ref, lw=2.2, color="black", alpha=0.9, label="Spont median (ref)")

    # Trigger sortieren
    order = np.argsort(md)
    top_idx = order[:min(top_n, len(order))]
    bot_idx = order[-min(top_n, len(order)):] if len(order) > top_n else np.array([], int)

    # Top (ähnlich): grün-ish -> ich lasse die Farbe default schwarz, aber mit linestyle unterscheiden
    # (Du hast oft keine festen Farben – hier mache ich nur linestyle/alpha unterschiedlich.)
    for j in top_idx:
        ax.plot(t, Xt[j], lw=1.6, alpha=0.85, linestyle="-",
                label=f"Top (md={md[j]:.2f})" if j == top_idx[0] else None)

    # Bottom (unähnlich): gestrichelt
    for j in bot_idx:
        ax.plot(t, Xt[j], lw=1.6, alpha=0.85, linestyle="--",
                label=f"Bottom (md={md[j]:.2f})" if j == bot_idx[0] else None)

    ax.axhline(0, lw=0.8, alpha=0.3)
    ax.set_xlabel("Zeit innerhalb UP (% Onset→Offset)")
    ax.set_ylabel("Amplitude (normiert)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.12, linestyle=":")
    return fig

def _tstats_time(x, y):
    """Welch t-test per time bin, ignoring NaNs. Returns t, p (n_time,)"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n_time = x.shape[1]
    tvals = np.full(n_time, np.nan)
    pvals = np.full(n_time, np.nan)

    for t in range(n_time):
        xt = x[:, t]
        yt = y[:, t]
        xt = xt[~np.isnan(xt)]
        yt = yt[~np.isnan(yt)]
        if len(xt) >= 2 and len(yt) >= 2:
            tt, pp = stats.ttest_ind(xt, yt, equal_var=False)
            tvals[t] = tt
            pvals[t] = pp
    return tvals, pvals


def _is_valid_2d(a, min_trials=3, min_time=2):
    return (
        a is not None
        and isinstance(a, np.ndarray)
        and a.ndim == 2
        and a.shape[0] >= min_trials
        and a.shape[1] >= min_time
    )

X_ = globals().get("X", None)
Y_ = globals().get("Y", None)

if (not _is_valid_2d(X_)) or (not _is_valid_2d(Y_)):
    print(f"[CLUSTER] skipped: X={None if X_ is None else X_.shape} "
          f"Y={None if Y_ is None else Y_.shape}")
    t_obs, p_point, clusters = None, None, None
else:
    t_obs, p_point, clusters = permutation_cluster_test_time(X_, Y_, seed=1)

def permutation_cluster_test_time(
    x, y,
    n_perm=2000,
    alpha=0.05,
    tail="two-sided",
    seed=0,
):
    """
    Cluster permutation test across time.

    Inputs:
      x: array-like, shape (n_trials_x, n_time)
      y: array-like, shape (n_trials_y, n_time)

    Returns:
      t_obs (n_time,), p_pointwise (n_time,),
      clusters: list of dicts {start, end, mass, p_cluster}

    If inputs are missing/invalid/too small, returns (None, None, None).
    """

    # -------- guard: never crash on missing/invalid data --------
    if x is None or y is None:
        return None, None, None

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Must be 2D
    if x.ndim != 2 or y.ndim != 2:
        return None, None, None

    # Must have enough trials and timepoints
    if x.shape[0] < 3 or y.shape[0] < 3 or x.shape[1] < 2 or y.shape[1] < 2:
        return None, None, None

    # Must match time dimension
    if x.shape[1] != y.shape[1]:
        return None, None, None

    # Must have finite variance somewhere (avoid degenerate t-stats)
    # (optional but helpful in noisy pipelines)
    if (np.nanstd(x, axis=0).sum() == 0) or (np.nanstd(y, axis=0).sum() == 0):
        return None, None, None

    rng = np.random.default_rng(seed)

    # -------- observed stats --------
    t_obs, p_obs = _tstats_time(x, y)

    # If stats returned weird shapes, bail safely
    if t_obs is None or p_obs is None:
        return None, None, None
    t_obs = np.asarray(t_obs, dtype=float).ravel()
    p_obs = np.asarray(p_obs, dtype=float).ravel()
    if t_obs.size != x.shape[1] or p_obs.size != x.shape[1]:
        return None, None, None

    # pointwise threshold mask
    sig_mask = p_obs < alpha
    clusters_obs = _find_clusters(sig_mask)

    # -------- cluster mass statistic --------
    def cluster_mass(tvals, start, end):
        seg = tvals[start:end + 1]
        if tail == "two-sided":
            return np.nansum(np.abs(seg))
        elif tail == "greater":
            return np.nansum(seg)
        elif tail == "less":
            return np.nansum(-seg)
        else:
            raise ValueError("tail must be 'two-sided', 'greater', or 'less'")

    obs_masses = [cluster_mass(t_obs, s, e) for s, e in clusters_obs]

    # -------- permutation null of max cluster mass --------
    all_data = np.vstack([x, y])
    n_x = x.shape[0]
    n_total = all_data.shape[0]

    max_masses = np.zeros(int(n_perm), dtype=float)
    for p in range(int(n_perm)):
        perm_idx = rng.permutation(n_total)
        xp = all_data[perm_idx[:n_x], :]
        yp = all_data[perm_idx[n_x:], :]

        t_p, p_p = _tstats_time(xp, yp)
        if t_p is None or p_p is None:
            max_masses[p] = 0.0
            continue

        t_p = np.asarray(t_p, dtype=float).ravel()
        p_p = np.asarray(p_p, dtype=float).ravel()
        if t_p.size != x.shape[1] or p_p.size != x.shape[1]:
            max_masses[p] = 0.0
            continue

        mask_p = p_p < alpha
        cl_p = _find_clusters(mask_p)
        if len(cl_p) == 0:
            max_masses[p] = 0.0
        else:
            masses = [cluster_mass(t_p, s, e) for s, e in cl_p]
            max_masses[p] = np.nanmax(masses) if len(masses) else 0.0

    # -------- cluster p-values --------
    clusters = []
    for (s, e), mass in zip(clusters_obs, obs_masses):
        p_cluster = (np.sum(max_masses >= mass) + 1) / (int(n_perm) + 1)  # conservative
        clusters.append({
            "start": int(s),
            "end": int(e),
            "mass": float(mass),
            "p_cluster": float(p_cluster),
        })

    return t_obs, p_obs, clusters




def pca_pc1_overlay_corr_diff_ax(pca_fit_sp, pca_fit_tr, ax=None,
                                title="PC1: Spont vs Triggered (overlay / corr / diff)"):


    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
    else:
        fig = ax.figure

    pc1_sp = np.asarray(pca_fit_sp["components"][0], float)
    pc1_tr = np.asarray(pca_fit_tr["components"][0], float)

    # gleiche Länge erzwingen (nur falls mal n_points abweicht)
    m = min(pc1_sp.size, pc1_tr.size)
    pc1_sp = pc1_sp[:m]
    pc1_tr = pc1_tr[:m]

    # Vorzeichen angleichen (PCs sind bis ± definiert)
    if np.dot(pc1_sp, pc1_tr) < 0:
        pc1_tr = -pc1_tr

    # normieren (optional, macht Overlay besser interpretierbar)
    def _znorm(x):
        x = np.asarray(x, float)
        s = float(np.nanstd(x))
        if not np.isfinite(s) or s < 1e-12:
            return x
        return (x - float(np.nanmean(x))) / s

    sp = _znorm(pc1_sp)
    tr = _znorm(pc1_tr)

    t = np.linspace(0.0, 100.0, m)

    ax.plot(t, sp, lw=2.2, label="PC1 spont (z)")
    ax.plot(t, tr, lw=2.2, ls="--", label="PC1 trig (z)")
    ax.plot(t, (tr - sp), lw=1.6, alpha=0.85, label="Diff (trig - spont)")

    ax.axhline(0, lw=0.8, alpha=0.35)
    ax.set_xlabel("Zeit innerhalb UP (% Onset→Offset)")
    ax.set_ylabel("Amplitude (z)")
    ax.set_title(title)

    # Korrelation (Pearson) anzeigen
    r = np.corrcoef(sp, tr)[0, 1] if (np.isfinite(sp).all() and np.isfinite(tr).all()) else np.nan
    ax.text(
        0.98, 0.95, f"corr(PC1) r = {r:.2f}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=10, bbox=dict(boxstyle="round", fc="white", alpha=0.7)
    )

    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.12, linestyle=":")
    return fig

# --- Robust guard: skip stats if correlations missing/empty ---
corr_sp_arr = None if corr_sp is None else np.asarray(corr_sp, float)
corr_tr_arr = None if corr_tr is None else np.asarray(corr_tr, float)

if (corr_sp_arr is None or corr_tr_arr is None or
    corr_sp_arr.size < 2 or corr_tr_arr.size < 2 or
    (not np.isfinite(corr_sp_arr).any()) or (not np.isfinite(corr_tr_arr).any())):
    print("[PCA] Mann-Whitney skipped: corr_sp/corr_tr missing or too few valid samples")
    u, p = np.nan, np.nan
else:
    u, p = stats.mannwhitneyu(corr_sp_arr, corr_tr_arr, alternative="two-sided")

print(f"Mann–Whitney U: U={u:.1f}, p={p:.3e}")

def cliffs_delta(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return (np.sum(a[:, None] > b[None, :]) -
            np.sum(a[:, None] < b[None, :])) / (len(a) * len(b))

# --- Robust guard: skip Cliff's delta if correlations missing/empty ---
corr_sp_arr = None if corr_sp is None else np.asarray(corr_sp, float)
corr_tr_arr = None if corr_tr is None else np.asarray(corr_tr, float)

if (corr_sp_arr is None or corr_tr_arr is None or
    corr_sp_arr.ndim == 0 or corr_tr_arr.ndim == 0 or
    corr_sp_arr.size < 2 or corr_tr_arr.size < 2 or
    (not np.isfinite(corr_sp_arr).any()) or (not np.isfinite(corr_tr_arr).any())):
    print("[PCA] Cliff's delta skipped: corr_sp/corr_tr missing or too few valid samples")
    delta = np.nan
else:
    delta = cliffs_delta(corr_sp_arr, corr_tr_arr)

print(f"Cliff's delta = {delta:.2f}")

try:
    from mne.stats import permutation_cluster_test
    HAVE_MNE = True
except ModuleNotFoundError:
    HAVE_MNE = False



def _nanmean(a, axis=0):
    return np.nanmean(a, axis=axis)

def _nanvar(a, axis=0):
    return np.nanvar(a, axis=axis, ddof=1)

def cohens_d_time(x, y):
    """
    x, y: arrays (n_trials, n_time)
    returns d(t) shape (n_time,)
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    mx, my = np.nanmean(x, axis=0), np.nanmean(y, axis=0)
    vx, vy = _nanvar(x, axis=0), _nanvar(y, axis=0)
    nx = np.sum(~np.isnan(x), axis=0)
    ny = np.sum(~np.isnan(y), axis=0)

    # pooled std per timepoint
    denom = (nx + ny - 2)
    pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / np.maximum(denom, 1))
    d = (mx - my) / np.where(pooled == 0, np.nan, pooled)
    return d


def sliding_window_corr(mean_a, mean_b, win_s, dt):
    """Correlation of two mean traces in a sliding window."""
    mean_a = np.asarray(mean_a, float)
    mean_b = np.asarray(mean_b, float)
    win = int(round(win_s / dt))
    win = max(win, 3)
    r = np.full_like(mean_a, np.nan, dtype=float)
    for i in range(0, len(mean_a) - win + 1):
        a = mean_a[i:i+win]
        b = mean_b[i:i+win]
        if np.all(np.isnan(a)) or np.all(np.isnan(b)):
            continue
        # drop NaNs pairwise
        m = ~np.isnan(a) & ~np.isnan(b)
        if np.sum(m) >= 3:
            r[i + win//2] = np.corrcoef(a[m], b[m])[0, 1]
    return r

def compare_triggered_vs_spontaneous(
    trig_aligned,
    spon_aligned,
    dt,
    up_time=None,          # optional time axis (n_time,)
    n_perm=2000,
    do_cluster=True,
    alpha=0.05,
    tail="two-sided",
    seed=0,
    corr_win_s=0.05,       # 50 ms default
):
    """
    Drop-in comparison:
      - Cohen's d(t)
      - pointwise t/p
      - cluster permutation p-values
      - sliding-window correlation of mean traces

    Returns dict with all results.
    """

    def _empty_compare():
        return {
            "X": None, "Y": None,
            "time": None,
            "mean_trig": None,
            "mean_spon": None,
            "cohens_d": None,
            "t_obs": None,
            "p_pointwise": None,
            "sliding_corr": None,
            "clusters": [],
            "n_trials": {"trig": 0, "spon": 0},
        }

    def _coerce_trials_2d(a):
        if a is None:
            return np.empty((0, 0), dtype=float)
        arr = np.asarray(a, dtype=float)
        if arr.ndim == 0:
            return np.empty((0, 0), dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        elif arr.ndim > 2:
            # Keep first axis as trials, flatten trailing dims into time.
            arr = arr.reshape(arr.shape[0], -1)
        if arr.size == 0:
            return np.empty((0, 0), dtype=float)
        return arr

    X = _coerce_trials_2d(trig_aligned)
    Y = _coerce_trials_2d(spon_aligned)

    # safety: remove all-NaN trials
    if X.ndim != 2 or Y.ndim != 2:
        return _empty_compare()
    if X.shape[1] == 0 or Y.shape[1] == 0:
        return _empty_compare()
    X = X[~np.all(np.isnan(X), axis=1)]
    Y = Y[~np.all(np.isnan(Y), axis=1)]
    if X.shape[0] < 3 or Y.shape[0] < 3:
        return _empty_compare()
    if X.shape[1] != Y.shape[1]:
        n_min = min(X.shape[1], Y.shape[1])
        if n_min <= 0:
            return _empty_compare()
        X = X[:, :n_min]
        Y = Y[:, :n_min]

    n_time = X.shape[1]
    if up_time is None:
        up_time = np.arange(n_time) * dt

    mean_trig = np.nanmean(X, axis=0)
    mean_spon = np.nanmean(Y, axis=0)

    d_t = cohens_d_time(X, Y)
    if do_cluster and int(n_perm) > 0:
        print(f"[CLUSTER] start permutation test (n_perm={int(n_perm)})")
        t_obs, p_point, clusters = permutation_cluster_test_time(
            X, Y, n_perm=n_perm, alpha=alpha, tail=tail, seed=seed
        )
        print("[CLUSTER] done")
    else:
        print("[CLUSTER] skipped by config")
        t_obs, p_point, clusters = None, None, []
    r_t = sliding_window_corr(mean_trig, mean_spon, win_s=corr_win_s, dt=dt)

    return {
        "time": up_time,
        "mean_trig": mean_trig,
        "mean_spon": mean_spon,
        "cohens_d": d_t,
        "t_obs": t_obs,
        "p_pointwise": p_point,
        "clusters": clusters,
        "sliding_corr": r_t,
        "params": {"n_perm": n_perm, "alpha": alpha, "tail": tail, "seed": seed, "corr_win_s": corr_win_s},
        "n_trials": {"trig": int(X.shape[0]), "spon": int(Y.shape[0])},
    }

debug_log("[DBG] Trig aligned:", None if Trig_UP_peak_aligned_array is None else Trig_UP_peak_aligned_array.shape,
          "nan%=", np.isnan(Trig_UP_peak_aligned_array).mean() if Trig_UP_peak_aligned_array is not None else "na")
debug_log("[DBG] Spon aligned:", None if Spon_UP_peak_aligned_array is None else Spon_UP_peak_aligned_array.shape,
          "nan%=", np.isnan(Spon_UP_peak_aligned_array).mean() if Spon_UP_peak_aligned_array is not None else "na")

if Trig_UP_peak_aligned_array is not None:
    _tmp_tr = np.asarray(Trig_UP_peak_aligned_array, float)
    if _tmp_tr.ndim == 2:
        debug_log("[DBG] Trig all-NaN rows:", np.sum(np.all(np.isnan(_tmp_tr), axis=1)))
if Spon_UP_peak_aligned_array is not None:
    _tmp_sp = np.asarray(Spon_UP_peak_aligned_array, float)
    if _tmp_sp.ndim == 2:
        debug_log("[DBG] Spon all-NaN rows:", np.sum(np.all(np.isnan(_tmp_sp), axis=1)))


res = compare_triggered_vs_spontaneous(
    Trig_UP_peak_aligned_array,
    Spon_UP_peak_aligned_array,
    dt=dt,
    up_time=UP_Time,
    n_perm=CLUSTER_N_PERM,
    do_cluster=CLUSTER_ENABLE,
    alpha=0.05,
    seed=1
) or {}

n_trials = res.get("n_trials", {})
clusters = res.get("clusters") or []

print("n trials:", n_trials)

sig = [(c.get("start"), c.get("end"), c.get("p_cluster"))
       for c in clusters
       if isinstance(c, dict) and c.get("p_cluster", 1.0) < 0.05]

print("significant clusters (p<0.05):", sig if sig else "[CLUSTER] none/empty")



def _nansem(a, axis=0):
    a = np.asarray(a, float)
    n = np.sum(~np.isnan(a), axis=axis)
    return np.nanstd(a, axis=axis, ddof=1) / np.sqrt(np.maximum(n, 1))



def upstate_similarity_timecourse_ax(
    res,
    trig_aligned,
    spon_aligned,
    ax,
    alpha=0.05,
    title="Triggered vs Spontaneous UP states (peak-aligned)",
):
    """
    Single-axis plot:
    - Mean ± SEM (Triggered vs Spontaneous)
    - Significant cluster bars (p < alpha)
    """

    t = res.get("time") if isinstance(res, dict) else None
    mean_trig = res.get("mean_trig") if isinstance(res, dict) else None
    mean_spon = res.get("mean_spon") if isinstance(res, dict) else None
    clusters = res.get("clusters", []) if isinstance(res, dict) else []

    def _coerce_trials_2d(a):
        if a is None:
            return np.empty((0, 0), float)
        arr = np.asarray(a, float)
        if arr.ndim == 0:
            return np.empty((0, 0), float)
        if arr.ndim == 1:
            arr = arr[None, :]
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        return arr

    X = _coerce_trials_2d(trig_aligned)
    Y = _coerce_trials_2d(spon_aligned)
    if X.size:
        X = X[~np.all(np.isnan(X), axis=1)]
    if Y.size:
        Y = Y[~np.all(np.isnan(Y), axis=1)]

    n = 0
    if t is not None:
        t = np.asarray(t, float).ravel()
        n = max(n, t.size)
    if mean_trig is not None:
        mean_trig = np.asarray(mean_trig, float).ravel()
        n = max(n, mean_trig.size)
    if mean_spon is not None:
        mean_spon = np.asarray(mean_spon, float).ravel()
        n = max(n, mean_spon.size)
    if X.ndim == 2 and X.shape[1] > 0:
        n = max(n, X.shape[1])
    if Y.ndim == 2 and Y.shape[1] > 0:
        n = max(n, Y.shape[1])

    if n == 0:
        _blank_ax(ax, "no similarity timecourse")
        return ax.figure

    def _fit_len_1d(v):
        if v is None:
            return None
        a = np.asarray(v, float).ravel()
        if a.size == n:
            return a
        if a.size > n:
            return a[:n]
        return np.pad(a, (0, n - a.size), constant_values=np.nan)

    def _fit_len_2d(a):
        if a.size == 0:
            return np.empty((0, n), float)
        if a.shape[1] == n:
            return a
        if a.shape[1] > n:
            return a[:, :n]
        return np.pad(a, ((0, 0), (0, n - a.shape[1])), constant_values=np.nan)

    X = _fit_len_2d(X)
    Y = _fit_len_2d(Y)
    t = _fit_len_1d(t)
    mean_trig = _fit_len_1d(mean_trig)
    mean_spon = _fit_len_1d(mean_spon)

    if t is None:
        dt_plot = float(globals().get("dt", 1.0))
        t = (np.arange(n, dtype=float) - (n // 2)) * dt_plot

    if mean_trig is None:
        mean_trig = np.nanmean(X, axis=0) if X.size else np.full(n, np.nan)
    if mean_spon is None:
        mean_spon = np.nanmean(Y, axis=0) if Y.size else np.full(n, np.nan)

    def nansem(a, axis=0):
        n = np.sum(~np.isnan(a), axis=axis)
        return np.nanstd(a, axis=axis, ddof=1) / np.sqrt(np.maximum(n, 1))

    sem_trig = nansem(X, axis=0) if X.size else np.zeros_like(mean_trig)
    sem_spon = nansem(Y, axis=0) if Y.size else np.zeros_like(mean_spon)

    # --- Mean ± SEM ---
    ax.plot(t, mean_trig, label=f"Triggered (n={X.shape[0]})")
    ax.fill_between(t, mean_trig - sem_trig, mean_trig + sem_trig, alpha=0.25)

    ax.plot(t, mean_spon, label=f"Spontaneous (n={Y.shape[0]})")
    ax.fill_between(t, mean_spon - sem_spon, mean_spon + sem_spon, alpha=0.25)

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_ylabel("LFP (a.u.)")
    ax.set_title(title)
    ax.legend(frameon=False)

    # --- Cluster significance bars ---
    with np.errstate(invalid="ignore"):
        y_max = np.nanmax([mean_trig + sem_trig, mean_spon + sem_spon])
        y_min = np.nanmin([mean_trig - sem_trig, mean_spon - sem_spon])
    if not np.isfinite(y_max) or not np.isfinite(y_min):
        y_max, y_min = 1.0, -1.0
    y_bar = y_max + 0.08 * (y_max - y_min + 1e-12)

    for c in clusters:
        if not isinstance(c, dict):
            continue
        if c.get("p_cluster", 1.0) < alpha:
            s, e = c.get("start"), c.get("end")
            if s is None or e is None:
                continue
            s = int(max(0, min(n - 1, s)))
            e = int(max(0, min(n - 1, e)))
            if e >= s:
                ax.plot([t[s], t[e]], [y_bar, y_bar], linewidth=4)

    ax.set_ylim(top=y_bar + 0.12 * (y_bar - y_min))
    ax.set_xlabel("Time (s)")

def plot_upstate_rates_bar(rates, out_path, title="UP-state rate (total)"):
    """
    Balkenplot für rate_total_per_min und rate_total_hz.
    Speichert als SVG/PNG je nach Dateiendung in out_path.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    labels = ["total [/min]", "total [Hz]"]
    values = [
        float(rates.get("rate_total_per_min", np.nan)),
        float(rates.get("rate_total_hz", np.nan)),
    ]

    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=200)
    x = np.arange(len(labels))
    ax.bar(x, values)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("value")
    ax.set_title(title)

    # optional: Zahlen über die Balken
    for xi, v in zip(x, values):
        if np.isfinite(v):
            ax.text(xi, v, f"{v:.3g}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def upstate_rate_bar_ax(rates, ax=None, title="UP-state rate (total)"):
    """
    Balkendiagramm: total rate [/min] und total rate [Hz]
    erwartet keys: 'rate_total_per_min' und 'rate_total_hz'
    """
    import numpy as np

    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 3))

    labels = ["total [/min]", "total [Hz]"]
    vals = [
        float(rates.get("rate_total_per_min", np.nan)),
        float(rates.get("rate_total_hz", np.nan)),
    ]

    x = np.arange(len(labels))
    ax.bar(x, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("value")
    ax.set_title(title)

    # Werte über die Balken schreiben
    for xi, v in zip(x, vals):
        if np.isfinite(v):
            ax.text(xi, v, f"{v:.3g}", ha="center", va="bottom", fontsize=9)

    return ax

# Refraktärzeiten ab SPONT-UP-Off getrennt nach nächstem UP-Typ 
debug_log("[CHECK FINAL PULSES]",
          "p1_on", 0 if pulse_times_1 is None else len(pulse_times_1),
          "p1_off", 0 if pulse_times_1_off is None else len(pulse_times_1_off),
          "| p2_on", 0 if pulse_times_2 is None else len(pulse_times_2),
          "p2_off", 0 if pulse_times_2_off is None else len(pulse_times_2_off))
debug_log("[CHECK time range]", float(time_s[0]), "->", float(time_s[-1]))
if pulse_times_1 is not None and len(pulse_times_1):
    debug_log("[CHECK p1 first/last]", float(pulse_times_1[0]), float(pulse_times_1[-1]))
if pulse_times_1_off is not None and len(pulse_times_1_off):
    debug_log("[CHECK p1_off first/last]", float(pulse_times_1_off[0]), float(pulse_times_1_off[-1]))
debug_log("[CHECK intervals] ttl1", len(ttl1_intervals), "ttl2", len(ttl2_intervals))



refrac_spon2spon, refrac_spon2trig = compute_refrac_from_spont_to_spon_and_trig(
    Spontaneous_UP,
    Spontaneous_DOWN,
    Pulse_triggered_UP,
    Pulse_associated_UP,
    time_s
)


rates = compute_upstate_rate(
    time_s,
    Spontaneous_UP,
    Pulse_triggered_UP,
    Pulse_associated_UP
)



print(f"[REFRAC SPONT] spon→spon: n={len(refrac_spon2spon)}, "
      f"spon→trig: n={len(refrac_spon2trig)}")
# #  Layout definieren (Zeilen)
# layout_rows = [
#     # REIHE 1: Main channel (volle Breite)
#     [lambda ax: plot_up_classification_ax(
#         main_channel, time_s,
#         Spontaneous_UP, Spontaneous_DOWN,
#         Pulse_triggered_UP, Pulse_triggered_DOWN,
#         Pulse_associated_UP, Pulse_associated_DOWN,
#         pulse_times_1=pulse_times_1,
#         pulse_times_2=pulse_times_2,
#         ax=ax
#     )],

#     # REIHE 2: links Durations, rechts Power
#     [lambda ax: upstate_duration_compare_ax(
#         Trig_UP_crop, Trig_DOWN_crop,
#         Spon_UP_crop, Spon_DOWN_crop, dt, ax=ax
#     ),
#      lambda ax: Power_spectrum_compare_ax(
#         freqs, spont_mean, pulse_mean, p_vals=p_vals, ax=ax
#     )],
    
#  # REIHE 3: Amplituden-Vergleich (volle Breite)
#     [lambda ax: upstate_amplitude_compare_ax(
#         spont_amp, trig_amp, ax=ax, title="UP Amplitude (max-min): Spontan vs. Getriggert"
#     )],

#     # REIHE 4: CSD (volle Breite)
#     [lambda ax: CSD_compare_side_by_side_ax(
#         CSD_spont, CSD_trig, dt,
#         z_mm=z_mm,
#         align_pre=align_pre_s, align_post=align_post_s,
#         # Paper-look:
#         cmap="Spectral_r",          # <--- klassisches Diverging-Map
#         sat_pct=95,             # <--- etwas weniger aggressiv
#         norm_mode="linear",
#         #vmax_abs=15,     # <--- einfache lineare Skala
#         linthresh_frac=0.03,    # hier egal bei linear, kann bleiben
#         ax=ax,
#         title="CSD (Spont vs. Trig; UP-Onset = 0 s)"
#     )],



#       # REIHE 4 (NEU): Mittel-LFP um Onsets – Spont vs. Trig
#     [lambda ax: up_onset_mean_ax(
#         main_channel, dt, Spon_Onsets,
#         ax=ax, title="Spont-UPs – onset-aligned mean"
#     ),
#      lambda ax: up_onset_mean_ax(
#         main_channel, dt, Trig_Onsets,
#         ax=ax, title="Trig-UPs – onset-aligned mean"
#     )],

#         # REIHE 4: CSD – zwei Panels nebeneinander (Spont / Trig)
#     [lambda ax: CSD_single_panel_ax(
#             CSD_spont, dt,
#             z_mm=z_mm,
#             align_pre=align_pre_s,
#             align_post=align_post_s,
#             ax=ax,
#             title="CSD Spontaneous"
#         ),
#     lambda ax: CSD_single_panel_ax(
#             CSD_trig, dt,
#             z_mm=z_mm,
#             align_pre=align_pre_s,
#             align_post=align_post_s,
#             ax=ax,
#             title="CSD Triggered"
    
#         )],
#     # REIHE X: Pulse→UP Latenzen (Histogramm)
#     [lambda ax: pulse_to_up_latency_hist_ax(latencies_trig, ax=ax)],

#     [lambda ax: pulse_triggered_up_overlay_ax(
#     main_channel,
#     time_s,
#     pulse_times_1,          # ggf. pulse_times_2
#     Pulse_triggered_UP,
#     dt,
#     pre_s=0.2,
#     post_s=1.0,
#     max_win_s=1.0,
#     ax=ax,
#     title="Pulse-alignierte Trigger-UPs (LFP overlay)"
#     )],

#     [lambda ax: refractory_compare_ax(
#         refrac_spont, refrac_trig, ax=ax,
#         title="Refraktärzeit nach UP bis nächster UP"
#     )],
    

#     [lambda ax: refractory_from_spont_to_type_overlay_ax(
#         main_channel,
#         time_s,
#         Spontaneous_UP, Spontaneous_DOWN,
#         Pulse_triggered_UP, Pulse_triggered_DOWN,
#         Pulse_associated_UP, Pulse_associated_DOWN,
#         dt,
#         target_type="spont",
#         pre_s=10.0,
#         post_s=30.0,
#         ax=ax,
#         title="SPONT offset → nächste SPONT-UPs"
#     )],
#     [lambda ax: refractory_from_spont_to_type_overlay_ax(
#         main_channel,
#         time_s,
#         Spontaneous_UP, Spontaneous_DOWN,
#         Pulse_triggered_UP, Pulse_triggered_DOWN,
#         Pulse_associated_UP, Pulse_associated_DOWN,
#         dt,
#         target_type="trig",
#         pre_s=10.0,
#         post_s=30.0,
#         ax=ax,
#         title="SPONT offset → nächste TRIG-UPs"
#     )],

#     [lambda ax: refractory_from_spont_single_folder_ax(
#         refrac_spon2spon, refrac_spon2trig,
#         folder_name=BASE_TAG,
#         ax=ax
#     )],
#     [lambda ax: spontaneous_up_full_overlay_normtime_ax(
#     main_channel, time_s,
#     Spontaneous_UP, Spontaneous_DOWN,
#     n_points=300,
#     ax=ax,
#     title="Spontaneous UPs (full Onset→Offset), time-normalized overlay"
# )],

# [
#   lambda ax: (
#       _blank_ax(ax, "PCA skipped") if (pca_fit_joint is None) else
#       pca_template_pc1_ax(pca_fit_joint, ax=ax, title="Spontaneous PCA Template (PC1)")
#   ),
#   lambda ax: (
#       _blank_ax(ax, "PCA skipped / no trig") if (pca_fit_joint is None or pc1z_trig is None) else
#       pca_similarity_scores_ax(
#           pca_fit_joint["scores_spont"],
#           pc1z_trig,
#           mahal_trig,
#           ax=ax,
#           title="Triggered similarity vs. spontaneous PCA"
#       )
#   )
# ],

# # REIHE NEU: Mahalanobis-Distanz + Top/Bottom Trigger Overlays
# [
#   lambda ax: (
#       _blank_ax(ax, "PCA skipped") if (pca_fit_joint is None or mahal_trig is None) else
#       mahal_compare_ax(mahal_sp, mahal_trig, ax=ax,
#                        title="Mahalanobis distance to spontaneous PCA space")
#   ),
#   lambda ax: (
#       _blank_ax(ax, "PCA skipped / no trig") if (pca_fit_joint is None or mahal_trig is None or X_trig.shape[0]==0) else
#       ranked_trigger_overlays_ax(X_spont, X_trig, mahal_trig, ax=ax, top_n=5,
#                                  title="Triggered UPs: most vs least similar (Mahalanobis)")
#   )
# ],


# # PCA Template + Similarity Scores
# [
#   lambda ax: (
#       _blank_ax(ax, "PCA skipped") if (pca_fit_joint is None) else
#       pca_template_pc1_ax(pca_fit_joint, ax=ax, title="JOINT PCA Template (PC1) (spont+trig)")
#   ),
#   lambda ax: (
#       _blank_ax(ax, "PCA skipped / no trig") if (pca_fit_joint is None or pc1z_trig is None) else
#       pca_similarity_scores_ax(
#           pca_fit_joint["scores_spont"],
#           pc1z_trig,
#           mahal_trig,
#           ax=ax,
#           title="Triggered similarity in JOINT PCA space"
#       )
#   )
# ],

# # Mahalanobis + Ranked overlays
# [
#   lambda ax: (
#       _blank_ax(ax, "PCA skipped") if (pca_fit_joint is None or mahal_trig is None) else
#       mahal_compare_ax(mahal_sp, mahal_trig, ax=ax,
#                        title="Mahalanobis distance in JOINT PCA space")
#   ),
#   lambda ax: (
#       _blank_ax(ax, "PCA skipped / no trig") if (pca_fit_joint is None or mahal_trig is None or X_trig.shape[0]==0) else
#       ranked_trigger_overlays_ax(X_spont, X_trig, mahal_trig, ax=ax, top_n=5,
#                                  title="Triggered UPs ranked (JOINT PCA space)")
#   )
# ],

# # REIHE: Separate PCA – Templates & Overlay
# [
#   lambda ax: (
#       _blank_ax(ax, "PCA spont skipped") if (pca_fit_sp is None) else
#       pca_template_pc1_ax(pca_fit_sp, ax=ax, title="Spontaneous PCA Template (PC1)")
#   ),
#   lambda ax: (
#       _blank_ax(ax, "PCA trig skipped") if (pca_fit_tr is None) else
#       pca_template_pc1_ax(pca_fit_tr, ax=ax, title="Triggered PCA Template (PC1)")
#   )
# ],
# [
#   lambda ax: (
#       _blank_ax(ax, "Need both PCAs") if (pca_fit_sp is None or pca_fit_tr is None) else
#       pca_pc1_overlay_corr_diff_ax(pca_fit_sp, pca_fit_tr, ax=ax,
#                                    title="PC1 overlay + corr + diff (spont vs trig)")
#   )
# ],

# [
#     lambda ax: (
#         _blank_ax(ax, "no PCA stats")
#         if (corr_sp is None or corr_tr is None)
#         else pca_similarity_stats_ax(
#             corr_sp,
#             corr_tr,
#             p_val=p,
#             ax=ax,
#             title="Similarity to spontaneous UP template (PC1)"
#         )
#     )
# ],

#     [lambda ax: upstate_similarity_timecourse_ax(
#         res,
#         Trig_UP_peak_aligned_array,
#         Spon_UP_peak_aligned_array,
#         ax=ax,
#         alpha=0.05,
#         title="Triggered vs Spontaneous UP states (peak-aligned)"
#     )],

#     [lambda ax: CSD_compare_side_by_side_ax(
#         CSD_spont, CSD_trig, dt,
#         z_mm=z_mm,
#         align_pre=align_pre_s, align_post=align_post_s,
#         cmap="Spectral_r",
#         sat_pct=95,
#         norm_mode="linear",
#         linthresh_frac=0.03,
#         ax=ax,
#         title="CSD (Spont vs. Trig; UP-Onset = 0 s)"
#     )],
#         # REIHE NEU: UP-rate Balkendiagramm (volle Breite)
#     [lambda ax: upstate_rate_bar_ax(
#         rates,
#         ax=ax,
#         title="UP rate (total): [/min] vs [Hz]"
#     )],

# ]


layout_rows = [

    # ========================================================
    # REIHE 1: Main channel (volle Breite) – UP-Klassifikation + Pulse
    # ========================================================
    [lambda ax: plot_up_classification_ax(
        main_channel, time_s,
        Spontaneous_UP, Spontaneous_DOWN,
        Pulse_triggered_UP, Pulse_triggered_DOWN,
        Pulse_associated_UP, Pulse_associated_DOWN,
        pulse_times_1=pulse_times_1,
        pulse_times_2=pulse_times_2,
        spindle_intervals=spindle_intervals_s,
        ax=ax
    )],

    # ========================================================
    # REIHE 2: UP-Dauern (links) + Power Spectrum (rechts)
    # ========================================================
    [lambda ax: upstate_duration_compare_ax(
        Pulse_triggered_UP, Pulse_triggered_DOWN,
        Spontaneous_UP, Spontaneous_DOWN, dt, ax=ax
    ),
     lambda ax: Power_spectrum_compare_ax(
        freqs, spont_mean, pulse_mean, p_vals=p_vals, p_vals_fdr=p_vals_fdr, meta=spectra_meta, ax=ax
    )],

    # ========================================================
    # REIHE 3: UP-Amplitudenvergleich (volle Breite)
    # ========================================================
    [lambda ax: upstate_amplitude_compare_ax(
        spont_amp, trig_amp, ax=ax,
        title="UP Amplitude (max-min, mean): Spontan vs. Getriggert"
    )],

    # ========================================================
    # REIHE 4: CSD Vergleich (Spont vs Trig; side-by-side) – (volle Breite)
    # (Einmal reicht; die spätere Wiederholung wurde entfernt.)
    # ========================================================
    [lambda ax: CSD_compare_side_by_side_ax(
        CSD_spont, CSD_trig, dt,
        z_mm=z_mm_csd,
        align_pre=align_pre_s, align_post=align_post_s,
        cmap=(CSD_PAPER_CMAP if CSD_STYLE == "paper" else "Spectral_r"),
        sat_pct=(98 if CSD_STYLE == "paper" else 95),
        norm_mode="linear",
        linthresh_frac=0.03,
        CSD_diff=CSD_DIFF,
        show_diff=(CSD_STYLE == "diagnostic"),
        interp=("bicubic" if CSD_STYLE == "paper" else "bilinear"),
        prefer_imshow=True,
        n_spont=(None if CSD_STYLE == "paper" else CSD_N_SPONT),
        n_trig=(None if CSD_STYLE == "paper" else CSD_N_TRIG),
        n_match=(None if CSD_STYLE == "paper" else CSD_N_MATCH),
        sem_spont_med=(None if CSD_STYLE == "paper" else CSD_SEM_MED_SPONT),
        sem_trig_med=(None if CSD_STYLE == "paper" else CSD_SEM_MED_TRIG),
        ax=ax,
        title="CSD (Spont vs. Trig; UP-Onset = 0 s)"
    )],

    # ========================================================
    # REIHE 4b: CSD Vergleich (latency-corrected Trig) – (volle Breite)
    # ========================================================
    [lambda ax: (
        _blank_ax(ax, "no latency-corrected CSD")
        if (CSD_spont_latcorr is None or CSD_trig_latcorr is None)
        else CSD_compare_side_by_side_ax(
            CSD_spont_latcorr, CSD_trig_latcorr, dt,
            z_mm=z_mm_csd,
            align_pre=align_pre_s, align_post=align_post_s,
            cmap=(CSD_PAPER_CMAP if CSD_STYLE == "paper" else "Spectral_r"),
            sat_pct=(98 if CSD_STYLE == "paper" else 95),
            norm_mode="linear",
            linthresh_frac=0.03,
            CSD_diff=CSD_DIFF_LATCORR,
            show_diff=(CSD_STYLE == "diagnostic"),
            interp=("bicubic" if CSD_STYLE == "paper" else "bilinear"),
            prefer_imshow=True,
            n_spont=(None if CSD_STYLE == "paper" else CSD_N_SPONT_LATCORR),
            n_trig=(None if CSD_STYLE == "paper" else CSD_N_TRIG_LATCORR),
            n_match=(None if CSD_STYLE == "paper" else CSD_N_MATCH_LATCORR),
            sem_spont_med=(None if CSD_STYLE == "paper" else CSD_SEM_MED_SPONT_LATCORR),
            sem_trig_med=(None if CSD_STYLE == "paper" else CSD_SEM_MED_TRIG_LATCORR),
            ax=ax,
            title=(
                "CSD (Trig latency-corrected; "
                f"shift={0.0 if not np.isfinite(CSD_TRIG_LAT_SHIFT_S) else CSD_TRIG_LAT_SHIFT_S:.3f}s)"
            )
        )
    )],

    # ========================================================
    # REIHE 4c: ΔCSD (raw vs latency-corrected) – (volle Breite)
    # ========================================================
    [lambda ax: (
        _blank_ax(ax, "delta panel hidden in paper mode")
        if CSD_STYLE == "paper"
        else csd_delta_raw_vs_latcorr_ax(
            CSD_DIFF,
            CSD_DIFF_LATCORR,
            dt,
            z_mm=z_mm_csd,
            align_pre=align_pre_s,
            align_post=align_post_s,
            ax=ax,
            title="ΔCSD (Trig-Spont): corrected minus raw"
        )
    )],

    # ========================================================
    # REIHE 5: Mittel-LFP um Onsets – Spont (links) vs Trig (rechts)
    # ========================================================
    [lambda ax: up_onset_mean_ax(
        main_channel, dt, Spon_Onsets,
        ax=ax, title="Spont-UPs – onset-aligned mean"
    ),
     lambda ax: up_onset_mean_ax(
        main_channel, dt, Trig_Onsets,
        ax=ax, title="Trig-UPs – onset-aligned mean"
    )],

    # ========================================================
    # REIHE 6: Pulse → UP Latenzen (Histogramm) – (volle Breite)
    # ========================================================
    [lambda ax: pulse_to_up_latency_hist_ax(latencies_trig, ax=ax)],

    # ========================================================
    # REIHE 7: Pulse-alignierte Trigger-UPs (Overlay) – (volle Breite)
    # ========================================================
    [lambda ax: pulse_triggered_up_overlay_ax(
        main_channel,
        time_s,
        pulse_times_1,
        Pulse_triggered_UP,
        dt,
        pre_s=0.2,
        post_s=1.0,
        max_win_s=1.0,
        ax=ax,
        title="Pulse-alignierte Trigger-UPs (LFP overlay)"
    )],

    # ========================================================
    # REIHE 8: Refraktärzeit Vergleich (Spont vs Trig) – (volle Breite)
    # ========================================================
    [lambda ax: refractory_compare_ax(
        refrac_spont, refrac_trig, ax=ax,
        title="Refraktärzeit nach UP bis nächster UP"
    )],

    # ========================================================
    # REIHE 9: Refraktär-Overlay: SPONT offset → nächste SPONT-UPs – (volle Breite)
    # ========================================================
    [lambda ax: refractory_from_spont_to_type_overlay_ax(
        main_channel,
        time_s,
        Spontaneous_UP, Spontaneous_DOWN,
        Pulse_triggered_UP, Pulse_triggered_DOWN,
        Pulse_associated_UP, Pulse_associated_DOWN,
        dt,
        target_type="spont",
        pre_s=10.0,
        post_s=30.0,
        ax=ax,
        title="SPONT offset → nächste SPONT-UPs"
    )],

    # ========================================================
    # REIHE 10: Refraktär-Summary pro Folder – (volle Breite)
    # (Das zweite Overlay „…→ nächste TRIG-UPs“ hab ich rausgekürzt, weil sehr ähnlich.)
    # ========================================================
    [lambda ax: refractory_from_spont_single_folder_ax(
        refrac_spon2spon, refrac_spon2trig,
        folder_name=BASE_TAG,
        ax=ax
    )],

    # ========================================================
    # REIHE 11: Spontaneous UPs – time-normalized overlay – (volle Breite)
    # ========================================================
    [lambda ax: spontaneous_up_full_overlay_normtime_ax(
        main_channel, time_s,
        Spontaneous_UP, Spontaneous_DOWN,
        n_points=300,
        ax=ax,
        title="Spontaneous UPs (full Onset→Offset), time-normalized overlay"
    )],

    # ========================================================
    # REIHE 12: PCA (JOINT) – Template + Similarity (2 Panels)
    # (Nur EIN konsistenter JOINT-Block; die doppelte Variante wurde entfernt.)
    # ========================================================
    [
      lambda ax: (
          _blank_ax(ax, "PCA skipped") if (pca_fit_joint is None) else
          pca_template_pc1_ax(pca_fit_joint, ax=ax, title="JOINT PCA Template (PC1) (spont+trig)")
      ),
      lambda ax: (
          _blank_ax(ax, "PCA skipped / no trig") if (pca_fit_joint is None or pc1z_trig is None) else
          pca_similarity_scores_ax(
              pca_fit_joint["scores_spont"],
              pc1z_trig,
              mahal_trig,
              ax=ax,
              title="Triggered similarity in JOINT PCA space"
          )
      )
    ],

    # ========================================================
    # REIHE 13: PCA (JOINT) – Mahalanobis + Ranked Overlays (2 Panels)
    # ========================================================
    [
      lambda ax: (
          _blank_ax(ax, "PCA skipped") if (pca_fit_joint is None or mahal_trig is None) else
          mahal_compare_ax(mahal_sp, mahal_trig, ax=ax,
                           title="Mahalanobis distance in JOINT PCA space")
      ),
      lambda ax: (
          _blank_ax(ax, "PCA skipped / no trig")
          if (pca_fit_joint is None or mahal_trig is None or X_trig.shape[0] == 0) else
          ranked_trigger_overlays_ax(X_spont, X_trig, mahal_trig, ax=ax, top_n=5,
                                     title="Triggered UPs ranked (JOINT PCA space)")
      )
    ],

    # ========================================================
    # REIHE 14: Separate PCA – Spont vs Trig Templates (2 Panels)
    # ========================================================
    [
      lambda ax: (
          _blank_ax(ax, "PCA spont skipped") if (pca_fit_sp is None) else
          pca_template_pc1_ax(pca_fit_sp, ax=ax, title="Spontaneous PCA Template (PC1)")
      ),
      lambda ax: (
          _blank_ax(ax, "PCA trig skipped") if (pca_fit_tr is None) else
          pca_template_pc1_ax(pca_fit_tr, ax=ax, title="Triggered PCA Template (PC1)")
      )
    ],

    # ========================================================
    # REIHE 15: Separate PCA – PC1 Overlay + corr + diff – (volle Breite)
    # ========================================================
    [
      lambda ax: (
          _blank_ax(ax, "Need both PCAs") if (pca_fit_sp is None or pca_fit_tr is None) else
          pca_pc1_overlay_corr_diff_ax(pca_fit_sp, pca_fit_tr, ax=ax,
                                       title="PC1 overlay + corr + diff (spont vs trig)")
      )
    ],

    # ========================================================
    # REIHE 16: PCA Similarity Stats – (volle Breite)
    # ========================================================
    [
      lambda ax: (
          _blank_ax(ax, "no PCA stats")
          if (corr_sp is None or corr_tr is None)
          else pca_similarity_stats_ax(
              corr_sp,
              corr_tr,
              p_val=p,
              ax=ax,
              title="Similarity to spontaneous UP template (PC1)"
          )
      )
    ],

    # ========================================================
    # REIHE 17: Similarity Timecourse (peak-aligned) – (volle Breite)
    # ========================================================
    [lambda ax: upstate_similarity_timecourse_ax(
        res,
        Trig_UP_peak_aligned_array,
        Spon_UP_peak_aligned_array,
        ax=ax,
        alpha=0.05,
        title="Triggered vs Spontaneous UP states (peak-aligned)"
    )],

    # ========================================================
    # REIHE 18: UP-rate Balkendiagramm – (volle Breite)
    # ========================================================
    [lambda ax: upstate_rate_bar_ax(
        rates,
        ax=ax,
        title="UP rate (total): [/min] vs [Hz]"
    )],
]


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

    rates = compute_upstate_rate(
        time_s,
        Spontaneous_UP,
        Pulse_triggered_UP,
        Pulse_associated_UP
    )

    # Feldnamen (Schema)
    FIELDNAMES = [
        "Parent","Experiment","Dauer [s]","Samplingrate [Hz]","Kanäle",
        "Pulse count 1","Pulse count 2",
        "Upstates total","triggered","spon","associated",
        "Downstates total","UP/DOWN ratio",
        "Mean UP Dauer [s]","Mean UP Dauer Triggered [s]","Mean UP Dauer Spontaneous [s]",
        "Datum Analyse",
        "UP rate total [Hz]",
        "UP rate total [/min]",
    ]

    # Helfer: numpy/NaN -> plain
    def _py(v):
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
        "UP rate total [Hz]": rates["rate_total_hz"],
        "UP rate total [/min]": rates["rate_total_per_min"],
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
    
    
rates = compute_upstate_rate(
        time_s,
        Spontaneous_UP,
        Pulse_triggered_UP,
        Pulse_associated_UP
    )


rate_bar_svg = os.path.join(SAVE_DIR, f"{BASE_TAG}__upstate_rate_bar.svg")
plot_upstate_rates_bar(rates, rate_bar_svg)


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


def main():
    log("START")
    log("Exporting layout PDF/SVG ...")

    # Export aufrufen
    export_with_layout(
        BASE_TAG, SAVE_DIR, layout_rows,
        rows_per_page=3,          # 3 Zeilen -> alles auf eine Seite
        also_save_each_svg=True
    )

    log("Export finished")

    if DEBUG_MAIN_SAFE:
        print("[ORDER]", chan_cols)  # Originalnamen der LFP-Spalten
        print("[GOOD_IDX]", good_idx)
        print("[ORDER raw]", chan_cols_raw)
        print("[ORDER sorted]", chan_cols)
        print("[DEPTH flip?]", FLIP_DEPTH)
        print("[CHAN-FILTER] kept:", good_idx)
        print("[CHAN-FILTER] reasons:", reasons)

    log("FERTIG ohne Fehler")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        log(f"EXCEPTION: {e}")
        log(err)
        raise
