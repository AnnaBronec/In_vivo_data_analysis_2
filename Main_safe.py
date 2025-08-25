#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# --- Deine Module ---
from loader_old import load_LFP_new
try:
    from preprocessing import downsampling_old as _ds_fun
except ImportError:
    from preprocessing import downsampling as _ds_fun
from preprocessing import filtering, get_main_channel, pre_post_condition
from TimeFreq_plot import Run_spectrogram
from state_detection import (
    classify_states, Generate_CSD_mean, extract_upstate_windows,
    compare_spectra, plot_contrast_heatmap,
    average_amplitude_in_upstates, plot_CSD_comparison
)
from plotter import plot_all_channels, plot_spont_up_mean, plot_upstate_amplitude_blocks_colored

# ======================
# PARAMS
# ======================
DOWNSAMPLE_FACTOR = 50
HIGH_CUTOFF = 10
LOW_CUTOFF  = 2

BASE_PATH    = "/home/ananym/Code/In_vivo_data_analysis/Data/FOR ANNA IN VIVO/DRD cross/2017-8-9_13-52-30onePulse200msX20per15s"
LFP_FILENAME = "2017-8-9_13-52-30onePulse200msX20per15s.csv"

# ======================
# LOAD LFP
# ======================
LFP_df, ch_names, lfp_meta = load_LFP_new(BASE_PATH, LFP_FILENAME)
assert "time" in LFP_df.columns, "CSV braucht eine Spalte 'time'."
print("[INFO] CSV rows:", len(LFP_df),
      "time range:", float(LFP_df["time"].iloc[0]), "->", float(LFP_df["time"].iloc[-1]))

# ======================
# PULSES (Prio: din_1/din_2, sonst stim)
# ======================
time_full = LFP_df["time"].to_numpy(float)

pulse_times_1_full = np.array([], dtype=float)
pulse_times_2_full = np.array([], dtype=float)

if "din_1" in LFP_df.columns:
    din1 = pd.to_numeric(LFP_df["din_1"], errors="coerce").fillna(0).to_numpy()
    r1 = np.flatnonzero((din1[1:] == 1) & (din1[:-1] == 0)) + 1
    pulse_times_1_full = time_full[r1]
if "din_2" in LFP_df.columns:
    din2 = pd.to_numeric(LFP_df["din_2"], errors="coerce").fillna(0).to_numpy()
    # falls dein Code 2 als "high" nutzt, passend ändern:
    r2 = np.flatnonzero((din2[1:] == 1) & (din2[:-1] == 0)) + 1
    pulse_times_2_full = time_full[r2]

if pulse_times_1_full.size == 0 and "stim" in LFP_df.columns:
    stim = pd.to_numeric(LFP_df["stim"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    rising = np.flatnonzero((stim[1:] > 0) & (stim[:-1] == 0)) + 1
    pulse_times_1_full = time_full[rising]

print(f"[INFO] pulses(full): p1={len(pulse_times_1_full)}, p2={len(pulse_times_2_full)}")

# ======================
# CHANNELS -> pri_* (für Downsampling-Kompat.)
# ======================
chan_cols = [c for c in LFP_df.columns if c not in ("time", "stim", "din_1", "din_2")]
assert len(chan_cols) > 0, "Keine Kanalspalten gefunden."
LFP_df_ds = pd.DataFrame({"timesamples": time_full})
for i, col in enumerate(chan_cols):
    LFP_df_ds[f"pri_{i}"] = pd.to_numeric(LFP_df[col], errors="coerce")
NUM_CHANNELS = len(chan_cols)

# ======================
# DOWNSAMPLING (einheitlich für Zeit/Signale/Pulse)
# ======================
time_s, dt, LFP_array, pulse_times_1, pulse_times_2 = _ds_fun(
    DOWNSAMPLE_FACTOR,
    LFP_df_ds,
    NUM_CHANNELS,
    pulse_times_1=pulse_times_1_full,
    pulse_times_2=pulse_times_2_full,
    snap_pulses=False
)
assert LFP_array.shape[0] == NUM_CHANNELS, f"LFP_array hat {LFP_array.shape[0]} Kanäle, erwartet {NUM_CHANNELS}"
assert LFP_array.shape[1] == len(time_s), "Zeit- und Signal-Länge passen nicht zusammen."
print(f"[DS] time {time_s[0]:.3f}->{time_s[-1]:.3f}s, N={len(time_s)}, dt={dt:.6f}s, "
      f"LFP_array={LFP_array.shape}, p1={len(pulse_times_1)}, p2={len(pulse_times_2)}")

# ======================
# PREPROCESS + SPECTROGRAM
# ======================
b_lp, a_lp, b_hp, a_hp = filtering(HIGH_CUTOFF, LOW_CUTOFF, dt)
main_channel = get_main_channel(DOWNSAMPLE_FACTOR, LFP_df_ds, NUM_CHANNELS)
pre, post, win_len, align_pre, align_post, align_len = pre_post_condition(dt)
Spect_dat = Run_spectrogram(main_channel, time_s)

# ======================
# STATE DETECTION
# ======================
Up_states = classify_states(
    Spect_dat, time_s, pulse_times_1, pulse_times_2,
    dt, main_channel, LFP_array,
    b_lp, a_lp, b_hp, a_hp,
    align_pre, align_post, align_len
)

Spontaneous_UP        = Up_states.get("Spontaneous_UP", np.array([], int))
Spontaneous_DOWN      = Up_states.get("Spontaneous_DOWN", np.array([], int))
Pulse_triggered_UP    = Up_states.get("Pulse_triggered_UP", np.array([], int))
Pulse_triggered_DOWN  = Up_states.get("Pulse_triggered_DOWN", np.array([], int))
Pulse_associated_UP   = Up_states.get("Pulse_associated_UP", np.array([], int))
Pulse_associated_DOWN = Up_states.get("Pulse_associated_DOWN", np.array([], int))
Spon_Peaks            = Up_states.get("Spon_Peaks", np.array([], int))
UP_start_i            = Up_states.get("UP_start_i", np.array([], int))
Total_power           = Up_states.get("Total_power", None)
up_state_binary       = Up_states.get("up_state_binary ", Up_states.get("up_state_binary", None))

print("[COUNTS] sponUP:", len(Spontaneous_UP),
      " trigUP:", len(Pulse_triggered_UP),
      " assocUP:", len(Pulse_associated_UP))

# ======================
# SPECTRA / WINDOWS (mismatch-sicher)
# ======================
pulse_windows = extract_upstate_windows(Pulse_triggered_UP, LFP_array, dt, window_s=1.0)
spont_windows = extract_upstate_windows(Spontaneous_UP,    LFP_array, dt, window_s=1.0)
if len(pulse_windows) and len(spont_windows) and (len(pulse_windows) != len(spont_windows)):
    try:
        # Mittelwerte vergleichen; Heatmap mit gleich vielen Trials
        freqs, spont_mean, pulse_mean, p_vals = compare_spectra(pulse_windows, spont_windows, dt, ignore_start_s=0.3)
        m = min(len(pulse_windows), len(spont_windows))
        plot_contrast_heatmap(pulse_windows[:m], spont_windows[:m], dt)
    except Exception as e:
        print("[WARN] spectra/heatmap mismatch handled:", e)
else:
    freqs, spont_mean, pulse_mean, p_vals = compare_spectra(pulse_windows, spont_windows, dt, ignore_start_s=0.3)
    try:
        plot_contrast_heatmap(pulse_windows, spont_windows, dt)
    except Exception as e:
        print("[WARN] heatmap skipped:", e)

# ======================
# PLOTS/ANALYSEN (Beispiele)
# ======================
plot_all_channels(NUM_CHANNELS, time_s, LFP_array)

plot_spont_up_mean(
    main_channel, time_s, dt, Spon_Peaks, up_state_binary,
    pulse_times_1, pulse_times_2,
    Pulse_triggered_UP, Pulse_triggered_DOWN,
    Spontaneous_UP, Spontaneous_DOWN
)

# Amplituden-Beispiel (geordnet nach Zeit)
all_ups  = np.concatenate((Pulse_triggered_UP, Spontaneous_UP))
all_down = np.concatenate((Pulse_triggered_DOWN, Spontaneous_DOWN))
if all_ups.size and all_down.size:
    idx = np.argsort(time_s[all_ups])
    all_ups_sorted  = all_ups[idx]
    all_down_sorted = all_down[idx]
    average_amplitude_in_upstates(
        main_channel=main_channel,
        time_s=time_s,
        UP_start_i=all_ups_sorted,
        DOWN_start_i=all_down_sorted,
        start_idx=3, end_idx=min(7, len(all_ups_sorted))
    )

# CSD (guarded)
try:
    Trig_Peaks = Up_states.get("Trig_Peaks", np.array([], int))
    CSD_spont  = Generate_CSD_mean(Spon_Peaks, LFP_array, dt)
    CSD_trig   = Generate_CSD_mean(Trig_Peaks, LFP_array, dt)
    if (CSD_spont is not None and CSD_trig is not None and
        getattr(CSD_spont, "ndim", 0) == 2 and getattr(CSD_trig, "ndim", 0) == 2):
        plot_CSD_comparison(CSD_spont, CSD_trig, dt)
except Exception as e:
    print("[WARN] CSD skipped:", e)
