#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Eigene Module
from preprocessing import downsampling_old, filtering, get_main_channel, pre_post_condition
from loader_old import load_LFP_new  # nur noch der LFP-Loader!
# (load_lightpulses_simple NICHT mehr verwenden)

# =========================
# User-Parameter
# =========================
DOWNSAMPLE_FACTOR = 50
HIGH_CUTOFF = 10
LOW_CUTOFF  = 2
BASE_PATH = "/home/ananym/Code/In_vivo_data_analysis/Data/FOR ANNA IN VIVO/DRD cross/2017-8-9_13-52-30onePulse200msX20per15s"
LFP_FILENAME = "2017-8-9_13-52-30onePulse200msX20per15s.csv"

# Optional: bestimmte Kanäle wählen (z.B. ["CSC1","CSC3"])
# Wenn nicht definiert, werden automatisch alle Kanäle genommen.
# INCLUDED_CHANNELS = ["CSC1", "CSC2"]

# =========================
# Laden: LFP (neues Layout)
# =========================
LFP_df, ch_names, lfp_meta = load_LFP_new(BASE_PATH, LFP_FILENAME)
print("[INFO] LFP geladen – Spalten:", LFP_df.columns.tolist())

# ================
# Pulse aus 'stim'
# ================
time_full = LFP_df["time"].to_numpy(dtype=float)

if "stim" in LFP_df.columns:
    stim = pd.to_numeric(LFP_df["stim"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    rising_idx = np.flatnonzero((stim[1:] > 0) & (stim[:-1] == 0)) + 1
    pulse_times_1_full = time_full[rising_idx]
else:
    pulse_times_1_full = np.array([], dtype=float)

pulse_times_2_full = np.array([], dtype=float)  # kein zweiter Pulskanal

print(f"[INFO] Pulse (full): {len(pulse_times_1_full)} – Range:",
      (float(pulse_times_1_full.min()), float(pulse_times_1_full.max())) if len(pulse_times_1_full) else "n/a")

# =========================
# Kanäle auswählen
# =========================
def _norm(ch: str) -> str:
    ch = ch.strip()
    if ch.lower().startswith("csc"):
        try:
            return f"CSC{int(ch[3:])}"  # 'CSC01' -> 'CSC1'
        except:
            return ch.upper()
    return ch.upper()

channel_cols = [c for c in LFP_df.columns if c not in ("time", "stim")]
if not channel_cols:
    raise RuntimeError("Keine Kanäle gefunden (Spalten außer 'time'/'stim').")

try:
    desired_raw = list(INCLUDED_CHANNELS)  # optional
except NameError:
    desired_raw = []

if desired_raw:
    avail_norm = {_norm(c): c for c in channel_cols}
    selected = [avail_norm[_norm(d)] for d in desired_raw if _norm(d) in avail_norm]
    if not selected:
        selected = channel_cols  # Fallback: alle
else:
    selected = channel_cols

LFP_numeric_full = LFP_df[selected].apply(pd.to_numeric, errors="coerce")
LFP_array_full = LFP_numeric_full.to_numpy().T  # (channels x samples)
NUM_CHANNELS = LFP_array_full.shape[0]

print(f"[INFO] Channels selected ({NUM_CHANNELS}): {selected}")
print(f"[INFO] LFP_array_full shape: {LFP_array_full.shape}  (channels x samples)")



# ==========================================
# Downsampling – einheitlich für alles
# (downsampling_old erwartet 'timesamples')
# ==========================================
LFP_df_ds = pd.DataFrame({"timesamples": LFP_df["time"].to_numpy(dtype=float)})

print("CSV rows:", len(LFP_df))
print("time start/end:", float(LFP_df["time"].iloc[0]), float(LFP_df["time"].iloc[-1]))


# Kanäle in pri_0..pri_{N-1} umbenennen
for i, col in enumerate(selected):
    LFP_df_ds[f"pri_{i}"] = pd.to_numeric(LFP_df[col], errors="coerce")

NUM_CHANNELS = len(selected)  # jetzt passt es zur Anzahl pri_*

# Pulse aus stim (volle Auflösung) hast du schon als pulse_times_1_full / _2_full
time, dt, LFP_array, pulse_times_1, pulse_times_2 = downsampling_old(
    DOWNSAMPLE_FACTOR,
    LFP_df_ds,
    NUM_CHANNELS,
    pulse_times_1=pulse_times_1_full,
    pulse_times_2=pulse_times_2_full,
    snap_pulses=False
)

print(f"[DS] time: {time.shape}, LFP_array: {LFP_array.shape}, "
      f"pulses: {len(pulse_times_1)} in [{time.min():.3f}, {time.max():.3f}] s")

# =========================
# Quick-Plot (Kanal 0)
# =========================
def plot_one_channel_with_pulses(time, signal, pulses1=None, pulses2=None,
                                 title="Channel + Stimuli", y_label="LFP (a.u.)"):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, signal, lw=0.9, color="black", label="LFP")
    ymin, ymax = float(np.nanmin(signal)), float(np.nanmax(signal))
    if np.isfinite(ymin) and np.isfinite(ymax) and ymin != ymax:
        pad = 0.05 * (ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)
    ymin, ymax = ax.get_ylim()
    if pulses1 is not None and len(pulses1):
        ax.vlines(pulses1, ymin=ymin, ymax=ymax, linewidth=1.0, alpha=0.7,
                  linestyles="--", color="red", label="Stimuli 1")
    if pulses2 is not None and len(pulses2):
        ax.vlines(pulses2, ymin=ymin, ymax=ymax, linewidth=1.0, alpha=0.7,
                  linestyles=":", color="blue", label="Stimuli 2")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 1:
        ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()

plot_one_channel_with_pulses(
    time,
    LFP_array[0, :],
    pulses1=pulse_times_1,
    pulses2=pulse_times_2,
    title=f"{selected[0]} + Stimuli (downsampled)"
)

# ==========================================
# Ab hier: deine bestehende Pipeline
# (b_lp/a_lp, main_channel, Spectrogram, States, CSD ...)
# ==========================================
from preprocessing import filtering, get_main_channel, pre_post_condition
from TimeFreq_plot import Run_spectrogram
from state_detection_old import (
    classify_states, Generate_CSD_mean, extract_upstate_windows,
    compute_spectra, compare_spectra, plot_contrast_heatmap,
    average_amplitude_in_upstates, compute_peak_aligned_segments,
    plot_CSD_comparison
)

b_lp, a_lp, b_hp, a_hp = filtering(HIGH_CUTOFF, LOW_CUTOFF, dt)
main_channel = get_main_channel(DOWNSAMPLE_FACTOR, LFP_df_ds, NUM_CHANNELS)
pre, post, win_len, align_pre, align_post, align_len = pre_post_condition(dt)

Spect_dat = Run_spectrogram(main_channel, time)

Up_states = classify_states(
    Spect_dat, time, pulse_times_1, pulse_times_2,
    dt, main_channel, LFP_array,
    b_lp, a_lp, b_hp, a_hp,
    align_pre, align_post, align_len
)

# ... (dein Rest wie gehabt)


# Auspacken
Spontaneous_UP        = Up_states["Spontaneous_UP"]
Spontaneous_DOWN      = Up_states["Spontaneous_DOWN"]
Pulse_triggered_UP    = Up_states["Pulse_triggered_UP"]
Pulse_triggered_DOWN  = Up_states["Pulse_triggered_DOWN"]
Pulse_associated_UP   = Up_states["Pulse_associated_UP"]
Pulse_associated_DOWN = Up_states["Pulse_associated_DOWN"]
Spon_UP_array         = Up_states["Spon_UP_array"]
Spon_UP_peak_alligned_array = Up_states["Spon_UP_peak_alligned_array"]
Spon_Peaks            = Up_states["Spon_Peaks"]
UP_Time               = Up_states["UP_Time"]
Total_power           = Up_states["Total_power"]
UP_start_i            = Up_states["UP_start_i"]
up_state_binary       = Up_states["up_state_binary "]

# Triggered_Peaks robust bereitstellen
Triggered_Peaks = Up_states.get("Trig_Peaks", None)
if Triggered_Peaks is None or (isinstance(Triggered_Peaks, np.ndarray) and Triggered_Peaks.size == 0):
    pulsed_UP = Up_states.get("Pulse_triggered_UP", np.array([], dtype=int))
    if pulsed_UP is None:
        pulsed_UP = np.array([], dtype=int)
    if pulsed_UP.size > 0:
        Trig_Peaks, _ = compute_peak_aligned_segments(
            pulsed_UP, LFP_array, dt,
            b_lp, a_lp, b_hp, a_hp,
            align_pre, align_post, align_len
        )
        Triggered_Peaks = np.asarray(Trig_Peaks, dtype=float)
    else:
        Triggered_Peaks = np.array([], dtype=float)

# Spektren (optional)
pulse_windows = extract_upstate_windows(Pulse_triggered_UP, LFP_array, dt, window_s=1.0)
spont_windows = extract_upstate_windows(Spontaneous_UP,    LFP_array, dt, window_s=1.0)
freqs, spont_mean, pulse_mean, p_vals = compare_spectra(pulse_windows, spont_windows, dt)
plot_contrast_heatmap(pulse_windows, spont_windows, dt)
compare_spectra(pulse_windows, spont_windows, dt, ignore_start_s=0.3)

# Amplituden-Analyse (optional)
all_ups          = np.concatenate((Pulse_triggered_UP, Spontaneous_UP))
all_times        = time[all_ups]
sort_idx         = np.argsort(all_times)
all_ups_sorted   = all_ups[sort_idx]
all_downs_sorted = np.concatenate((Pulse_triggered_DOWN, Spontaneous_DOWN))[sort_idx]
if len(all_ups_sorted) == 0:
    print("⛔ Keine UP-Zustände gefunden – überspringe Amplitudenanalyse.")
else:
    average_amplitude_in_upstates(
        main_channel=main_channel,
        time=time,
        UP_start_i=all_ups_sorted,
        DOWN_start_i=all_downs_sorted,
        start_idx=1,
        end_idx=len(all_ups_sorted)
    )

# CSD (robust „guarded“)
CSD_spont = Generate_CSD_mean(Spon_Peaks,      LFP_array, dt)
CSD_trig  = Generate_CSD_mean(Triggered_Peaks, LFP_array, dt)

if (CSD_spont is None or CSD_trig is None or
    getattr(CSD_spont, "size", 0) == 0 or getattr(CSD_trig, "size", 0) == 0 or
    any(np.asarray(x).ndim != 2 for x in [CSD_spont, CSD_trig])):
    print("⛔ CSD-Plot übersprungen (leer/None/Shape).")
else:
    plot_CSD_comparison(CSD_spont, CSD_trig, dt)

# Abschluss-Info
print("Pulse count (final):", len(pulse_times_1))
if len(pulse_times_1):
    print("Pulse range (final s):", float(np.min(pulse_times_1)), "->", float(np.max(pulse_times_1)))





