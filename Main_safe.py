#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============ IMPORTS ============
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  # oder "Qt5Agg", wenn Tk nicht installiert ist
import matplotlib.pyplot as plt
plt.ion()

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
    compare_spectra, plot_contrast_heatmap, plot_CSD_comparison
)
from plotter import (
    plot_all_channels,
    plot_spont_up_mean,
    plot_upstate_amplitude_blocks_colored,
    plot_upstate_duration_comparison   # optional am Ende
)

# ============ PARAMS ============
DOWNSAMPLE_FACTOR = 50
HIGH_CUTOFF = 10
LOW_CUTOFF  = 2

BASE_PATH    = "/home/ananym/Code/In_vivo_data_analysis/Data/FOR ANNA IN VIVO/DRD cross/2017-8-9_13-52-30onePulse200msX20per15s"
LFP_FILENAME = "2017-8-9_13-52-30onePulse200msX20per15s.csv"

# Alle Plots in denselben Ordner wie die Daten speichern
SAVE_DIR = BASE_PATH
BASE_TAG = os.path.basename(os.path.normpath(BASE_PATH))


# ============ SAVE HELPERS ============
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def run_and_save(plot_func, hint, *args, dpi=200, close=True, **kwargs):
    """
    Führt plot_func aus, erkennt neue Figures ODER nutzt das zurückgegebene fig,
    speichert sie nach SAVE_DIR als '<BASE_TAG>_<hint>[_NN].svg' und schließt sie (close=True).
    """
    before = set(plt.get_fignums())
    fig = plot_func(*args, **kwargs)  # Plotfunktion sollte eine neue Figure erzeugen und returnen
    after  = set(plt.get_fignums())
    new_nums = sorted(after - before)

    if isinstance(fig, plt.Figure):
        figs_to_save = [fig]
    elif new_nums:
        figs_to_save = [plt.figure(n) for n in new_nums]
    else:
        figs_to_save = [plt.gcf()]  # Fallback

    _ensure_dir(SAVE_DIR)
    multi = len(figs_to_save) > 1
    for j, f in enumerate(figs_to_save, 1):
        try:
            f.tight_layout()
        except Exception:
            pass
        suffix = f"_{j:02d}" if multi else ""
        out_path = os.path.join(SAVE_DIR, f"{BASE_TAG}_{hint}{suffix}.svg")
        f.savefig(out_path, format="svg", dpi=dpi, bbox_inches="tight")
        print(f"[SAVED] {out_path}")
    if close:
        for f in figs_to_save:
            plt.close(f)
    return fig


# ============ LOAD LFP ============
LFP_df, ch_names, lfp_meta = load_LFP_new(BASE_PATH, LFP_FILENAME)
assert "time" in LFP_df.columns, "CSV braucht eine Spalte 'time'."
print("[INFO] CSV rows:", len(LFP_df),
      "time range:", float(LFP_df["time"].iloc[0]), "->", float(LFP_df["time"].iloc[-1]))

# ============ PULSES ============
time_full = LFP_df["time"].to_numpy(float)

pulse_times_1_full = np.array([], dtype=float)
pulse_times_2_full = np.array([], dtype=float)

if "din_1" in LFP_df.columns:
    din1 = pd.to_numeric(LFP_df["din_1"], errors="coerce").fillna(0).to_numpy()
    r1 = np.flatnonzero((din1[1:] == 1) & (din1[:-1] == 0)) + 1
    pulse_times_1_full = time_full[r1]
if "din_2" in LFP_df.columns:
    din2 = pd.to_numeric(LFP_df["din_2"], errors="coerce").fillna(0).to_numpy()
    r2 = np.flatnonzero((din2[1:] == 1) & (din2[:-1] == 0)) + 1
    pulse_times_2_full = time_full[r2]

if pulse_times_1_full.size == 0 and "stim" in LFP_df.columns:
    stim = pd.to_numeric(LFP_df["stim"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    rising = np.flatnonzero((stim[1:] > 0) & (stim[:-1] == 0)) + 1
    pulse_times_1_full = time_full[rising]

print(f"[INFO] pulses(full): p1={len(pulse_times_1_full)}, p2={len(pulse_times_2_full)}")


# ============ CHANNELS -> pri_* ============
chan_cols = [c for c in LFP_df.columns if c not in ("time", "stim", "din_1", "din_2")]
assert len(chan_cols) > 0, "Keine Kanalspalten gefunden."
LFP_df_ds = pd.DataFrame({"timesamples": time_full})
for i, col in enumerate(chan_cols):
    LFP_df_ds[f"pri_{i}"] = pd.to_numeric(LFP_df[col], errors="coerce")
NUM_CHANNELS = len(chan_cols)

# ============ DOWNSAMPLING ============
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

# ============ PREPROCESS + SPECTROGRAM ============
b_lp, a_lp, b_hp, a_hp = filtering(HIGH_CUTOFF, LOW_CUTOFF, dt)
main_channel = get_main_channel(DOWNSAMPLE_FACTOR, LFP_df_ds, NUM_CHANNELS)
pre, post, win_len, align_pre, align_post, align_len = pre_post_condition(dt)
Spect_dat = Run_spectrogram(main_channel, time_s)

# ============ STATE DETECTION ============
Up_states = classify_states(
    Spect_dat, time_s, pulse_times_1, pulse_times_2,
    dt, main_channel, LFP_array,
    b_lp, a_lp, b_hp, a_hp,
    align_pre, align_post, align_len
)

# --- Alles NÖTIGE aus Up_states lesen ---
Spontaneous_UP        = Up_states.get("Spontaneous_UP",        np.array([], int))
Spontaneous_DOWN      = Up_states.get("Spontaneous_DOWN",      np.array([], int))
Pulse_triggered_UP    = Up_states.get("Pulse_triggered_UP",    np.array([], int))
Pulse_triggered_DOWN  = Up_states.get("Pulse_triggered_DOWN",  np.array([], int))
Pulse_associated_UP   = Up_states.get("Pulse_associated_UP",   np.array([], int))
Pulse_associated_DOWN = Up_states.get("Pulse_associated_DOWN", np.array([], int))
Spon_Peaks            = Up_states.get("Spon_Peaks",            np.array([], int))
Total_power           = Up_states.get("Total_power",           None)
up_state_binary       = Up_states.get("up_state_binary ", Up_states.get("up_state_binary", None))

print("[COUNTS] sponUP:", len(Spontaneous_UP),
      " trigUP:", len(Pulse_triggered_UP),
      " assocUP:", len(Pulse_associated_UP))

# ============ HELFER: UP/DOWN-Paare konsistent bauen ============
def build_up_down_pairs(Up_states, time_vec):
    UP_i   = np.array(Up_states.get("UP_start_i",   []), dtype=int)
    DOWN_i = np.array(Up_states.get("DOWN_start_i", []), dtype=int)

    if DOWN_i.size == 0:
        sUP   = np.array(Up_states.get("Spontaneous_UP",       []), dtype=int)
        sDN   = np.array(Up_states.get("Spontaneous_DOWN",     []), dtype=int)
        tUP   = np.array(Up_states.get("Pulse_triggered_UP",   []), dtype=int)
        tDN   = np.array(Up_states.get("Pulse_triggered_DOWN", []), dtype=int)
        UP_i   = np.concatenate((tUP, sUP)) if (tUP.size or sUP.size) else np.array([], int)
        DOWN_i = np.concatenate((tDN, sDN)) if (tDN.size or sDN.size) else np.array([], int)

    m = min(len(UP_i), len(DOWN_i))
    UP_i, DOWN_i = UP_i[:m], DOWN_i[:m]
    if m > 0:
        order = np.argsort(time_vec[UP_i])
        UP_i, DOWN_i = UP_i[order], DOWN_i[order]
    return UP_i, DOWN_i

UP_start_i, DOWN_start_i = build_up_down_pairs(Up_states, time_s)
print("[PAIRING] UP/DOWN Paare:", len(UP_start_i))

# ============ SPECTRA / WINDOWS ============
pulse_windows = extract_upstate_windows(Pulse_triggered_UP, LFP_array, dt, window_s=1.0)
spont_windows = extract_upstate_windows(Spontaneous_UP,    LFP_array, dt, window_s=1.0)

freqs = spont_mean = pulse_mean = p_vals = None
try:
    if len(pulse_windows) and len(spont_windows) and (len(pulse_windows) != len(spont_windows)):
        freqs, spont_mean, pulse_mean, p_vals = compare_spectra(pulse_windows, spont_windows, dt, ignore_start_s=0.3)
        m = min(len(pulse_windows), len(spont_windows))
        run_and_save(plot_contrast_heatmap, "contrast_heatmap", pulse_windows[:m], spont_windows[:m], dt)
    else:
        freqs, spont_mean, pulse_mean, p_vals = compare_spectra(pulse_windows, spont_windows, dt, ignore_start_s=0.3)
        run_and_save(plot_contrast_heatmap, "contrast_heatmap", pulse_windows,    spont_windows,    dt)
except Exception as e:
    print("[WARN] heatmap skipped:", e)

# ============ LOKALE PLOTFUNKTIONS (eigene Figures + return fig) ============
def Total_power_plot(Spect_dat):
    fig, ax = plt.subplots()
    ax.plot(Spect_dat[1], np.sum(Spect_dat[0], axis=0))
    ax.set_title("Gesamtleistung im 1–10 Hz Bereich")
    ax.set_xlabel("Zeit (s)")
    ax.set_ylabel("Power (summiert)")
    fig.tight_layout()
    return fig



def plot_power_spectrum_comparison(freqs, spont_mean, pulse_mean, p_vals=None, alpha=0.05, title=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freqs, spont_mean, label="Spontan",    linewidth=2)
    ax.plot(freqs, pulse_mean, label="Getriggert", linewidth=2)
    if p_vals is not None:
        sig = (p_vals < alpha)
        if np.any(sig):
            idx = np.where(sig)[0]
            # zusammenhängende Segmente markieren
            starts = [idx[0]]
            for i in range(1, len(idx)):
                if idx[i] != idx[i-1] + 1:
                    starts.append(idx[i])
            ends = []
            for s in starts[1:] + [None]:
                ends.append(idx[np.where(idx < s)[0][-1]] if s is not None else idx[-1])
            for s, e in zip(starts, ends):
                ax.axvspan(freqs[s], freqs[e], alpha=0.12)
    ax.set_xlabel("Frequenz (Hz)")
    ax.set_ylabel("Power (a.u.)")
    ax.set_title(title or "Power (Spontan vs. Getriggert)")
    ax.legend()
    fig.tight_layout()
    return fig

# ============ CSD (einmal, sauber) ============
try:
    Trig_Peaks = Up_states.get("Trig_Peaks", np.array([], int))
    CSD_spont  = Generate_CSD_mean(Spon_Peaks, LFP_array, dt)
    CSD_trig   = Generate_CSD_mean(Trig_Peaks,  LFP_array, dt)
    if (CSD_spont is not None and CSD_trig is not None and
        getattr(CSD_spont, "ndim", 0) == 2 and getattr(CSD_trig, "ndim", 0) == 2):
        run_and_save(plot_CSD_comparison, "CSD_spont_vs_trig", CSD_spont, CSD_trig, dt)
except Exception as e:
    print("[WARN] CSD skipped:", e)

# ============ RUN & SAVE – genau einmal pro Plot ============
run_and_save(plot_all_channels, "all_channels", NUM_CHANNELS, time_s, LFP_array)

run_and_save(
    plot_spont_up_mean, "spont_up_mean",
    main_channel, time_s, dt, Spon_Peaks, up_state_binary,
    pulse_times_1, pulse_times_2,
    Pulse_triggered_UP, Pulse_triggered_DOWN,
    Spontaneous_UP, Spontaneous_DOWN
)
run_and_save(
    plot_upstate_duration_comparison, "upstate_duration_compare",
    Pulse_triggered_UP, Pulse_triggered_DOWN,
    Spontaneous_UP, Spontaneous_DOWN, dt
)



# Power-Zeitreihe + Marks
run_and_save(Total_power_plot, "total_power", Spect_dat)


# Frequenz-Achse Vergleich (falls compare_spectra geliefert hat)
if (freqs is not None) and (spont_mean is not None) and (pulse_mean is not None):
    run_and_save(
        plot_power_spectrum_comparison, "power_vs_freq_compare",
        freqs, spont_mean, pulse_mean, p_vals=p_vals, alpha=0.05,
        title="Power (Spontan vs. Getriggert) – Frequenzachse"
    )

# Optional: Amplituden-Blöcke (wenn du sie brauchst)
index_blocks = [(1,10), (11,20), (21,30), (31,40)]  # an deine Daten anpassen
run_and_save(
    plot_upstate_amplitude_blocks_colored, "up_blocks_colored",
    main_channel, UP_start_i, DOWN_start_i, index_blocks, LFP_FILENAME
)

print("[DONE] Alle Plot-Funktionen mit Auto-Save in:", SAVE_DIR)




# ======================
# ERGEBNIS-TABELLE SCHREIBEN
# ======================
import csv

# ======================
# ERGEBNIS-TABELLE SCHREIBEN (mit Überschreiben falls Experiment schon vorhanden)
# ======================
import csv

experiment_name = os.path.basename(BASE_PATH)

# Übergeordneter Ordner (z.B. "DRD cross")
parent_folder = os.path.basename(os.path.dirname(BASE_PATH))

# Counts bestimmen
total_up = len(Spontaneous_UP) + len(Pulse_triggered_UP) + len(Pulse_associated_UP)
row = {
    "Parent": parent_folder, 
    "Experiment": experiment_name,
    "Dauer [s]": round(float(time_s[-1] - time_s[0]), 2),
    "Samplingrate [Hz]": round(1/dt, 2),
    "Kanäle": NUM_CHANNELS,
    "Pulse count 1": len(pulse_times_1),
    "Pulse count 2": len(pulse_times_2),
    "Upstates total": total_up,
    "triggered": len(Pulse_triggered_UP),
    "spon": len(Spontaneous_UP),
    "associated": len(Pulse_associated_UP),
    "Downstates total": len(Spontaneous_DOWN) + len(Pulse_triggered_DOWN) + len(Pulse_associated_DOWN),
    "UP/DOWN ratio": round(total_up / max(1, (len(Spontaneous_DOWN)+len(Pulse_triggered_DOWN)+len(Pulse_associated_DOWN))), 3),
    "Mean UP Dauer [s]": np.mean((DOWN_start_i - UP_start_i) * dt) if len(UP_start_i) else np.nan,
    "Datum Analyse": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
}


summary_path = os.path.join(os.path.dirname(BASE_PATH), "upstate_summary.csv")

rows = []
# Falls Datei existiert → einlesen
if os.path.isfile(summary_path):
    with open(summary_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

# Liste aktualisieren: falls Experiment schon existiert, überschreiben
found = False
for r in rows:
    if r["Experiment"] == experiment_name:
        r.update(row)
        found = True
        break
if not found:
    rows.append(row)

# CSV neu schreiben (immer komplett)
with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=row.keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"[SUMMARY] Tabelle aktualisiert: {summary_path}")
