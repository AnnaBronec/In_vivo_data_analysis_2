import csv
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # oder "Qt5Agg", wenn Tk nicht installiert ist
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tempfile import TemporaryFile
import pandas as pd
from scipy import signal
#from scipy.optimize import curve_fit
from TimeFreq_plot import Run_spectrogram
from CSD import CSD_calc
import seaborn as sns
from scipy.signal import find_peaks
import scipy.stats as stats
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import downsampling, filtering
from state_detection import classify_states
from loader import load_lightpulses

def plot_all_channels(num_channels: int, time_s, LFP_array):
    fig, axs = plt.subplots(num_channels, 1, figsize=(12, 2 * num_channels), sharex=True)
    for i in range(num_channels):
        axs[i].plot(time_s, LFP_array[i])
        axs[i].set_ylabel(f"pri_{i}")
        axs[i].grid(True)
    axs[-1].set_xlabel("Zeit (s)")
    fig.suptitle("LFP-Kanäle über Zeit")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig

def plot_spont_up_mean(main_channel, time_s, dt, Spon_Peaks, up_state_binary,
                       pulse_times_1, pulse_times_2,
                       Pulse_triggered_UP, Pulse_triggered_DOWN,
                       Spontaneous_UP, Spontaneous_DOWN):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time_s, main_channel, label='LFP', color='black')

    added_labels = set()
    for up, down in zip(Pulse_triggered_UP, Pulse_triggered_DOWN):
        up, down = int(up), int(down)
        if down < len(time_s):
            label = 'getriggert' if 'getriggert' not in added_labels else None
            ax.axvspan(time_s[up], time_s[down], color='red', alpha=0.3, label=label)
            added_labels.add('getriggert')

    for up, down in zip(Spontaneous_UP, Spontaneous_DOWN):
        up, down = int(up), int(down)
        if down < len(time_s):
            label = 'spontan' if 'spontan' not in added_labels else None
            ax.axvspan(time_s[up], time_s[down], color='lightblue', alpha=0.3, label=label)
            added_labels.add('spontan')

    vis1 = pulse_times_1[(pulse_times_1 >= time_s[0]) & (pulse_times_1 <= time_s[-1])]
    for i, p in enumerate(vis1):
        ax.axvline(p, color='red', linestyle='--', linewidth=0.8, label='Licht 1' if i == 0 else None)

    vis2 = pulse_times_2[(pulse_times_2 >= time_s[0]) & (pulse_times_2 <= time_s[-1])]
    for i, p in enumerate(vis2):
        ax.axvline(p, color='blue', linestyle=':', linewidth=0.8, label='Licht 2' if i == 0 else None)

    all_ups = np.concatenate((Pulse_triggered_UP, Spontaneous_UP))
    all_times_sorted = time_s[np.sort(all_ups)]
    y0, y1 = ax.get_ylim()
    for i, t in enumerate(all_times_sorted):
        ax.annotate(str(i + 1), xy=(t, y0), xytext=(t, y0 - 0.05 * (y1 - y0)),
                    ha='center', va='top', fontsize=7, rotation=90, color='black')

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys())
    ax.set_xlabel("Zeit (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("UP-States: spontan (blau) vs. getriggert (rot)")
    fig.tight_layout()
    return fig

def Total_power_plot(Spect_dat):
    fig, ax = plt.subplots()
    ax.plot(Spect_dat[1], np.sum(Spect_dat[0], axis=0))
    ax.set_title("Gesamtleistung im 1–10 Hz Bereich")
    ax.set_xlabel("Zeit (s)")
    ax.set_ylabel("Power (summiert)")
    fig.tight_layout()
    return fig



def plot_upstate_amplitudes(main_channel, UP_start_i, DOWN_start_i, start_idx, end_idx):
    assert 1 <= start_idx <= end_idx <= len(UP_start_i), "Ungültiger Indexbereich."
    indices = list(range(start_idx, end_idx + 1))
    amplitudes = []
    for i in range(start_idx - 1, end_idx):
        up = UP_start_i[i]; down = DOWN_start_i[i]
        if down > up and down < len(main_channel):
            segment = main_channel[up:down]
            amplitudes.append(np.mean(np.abs(segment)))
        else:
            amplitudes.append(np.nan)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(indices, amplitudes, color='mediumseagreen')
    ax.set_xlabel("UP-State Index"); ax.set_ylabel("Ø Amplitude")
    ax.set_title(f"Mittlere Amplituden der UP-Zustände {start_idx}–{end_idx}")
    ax.set_xticks(indices); ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    return fig

def plot_upstate_amplitude_mean(main_channel, UP_start_i, DOWN_start_i, start_idx, end_idx):
    assert 1 <= start_idx <= end_idx <= len(UP_start_i), "Ungültiger Indexbereich."
    amplitudes = []
    for i in range(start_idx - 1, end_idx):
        up = UP_start_i[i]; down = DOWN_start_i[i]
        if down > up and down < len(main_channel):
            amplitudes.append(np.mean(np.abs(main_channel[up:down])))
    fig = None
    if amplitudes:
        avg_amp = np.mean(amplitudes)
        fig, ax = plt.subplots(figsize=(4, 5))
        ax.bar([f"UPs {start_idx}-{end_idx}"], [avg_amp], color='mediumslateblue')
        ax.set_ylabel("Ø Amplitude"); ax.set_title("Durchschnittliche Amplitude")
        fig.tight_layout()
        print(f"✅ Durchschnittliche Amplitude von UPs {start_idx}–{end_idx}: {avg_amp:.4f}")
    else:
        print("❌ Keine gültigen UP-Zustände gefunden.")
    return fig

def plot_upstate_amplitude_blocks_colored(main_channel, UP_start_i, DOWN_start_i, index_blocks, filename):
    block_labels, block_means, block_colors, block_legend_labels = [], [], [], []
    for idx, (start_idx, end_idx) in enumerate(index_blocks):
        assert 1 <= start_idx <= end_idx <= len(UP_start_i), f"Ungültiger Bereich: {start_idx}-{end_idx}"
        amps = []
        for i in range(start_idx - 1, end_idx):
            up = UP_start_i[i]; down = DOWN_start_i[i]
            if down > up and down < len(main_channel):
                amps.append(np.mean(np.abs(main_channel[up:down])))
        mean_amp = np.mean(amps) if amps else np.nan
        block_means.append(mean_amp)
        block_labels.append(f"{start_idx}-{end_idx}")
        if idx % 2 == 0:
            block_colors.append("orange"); block_legend_labels.append("orange light")
        else:
            block_colors.append("violet"); block_legend_labels.append("violet light")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(block_labels, block_means, color=block_colors)
    seen = set()
    for bar, label in zip(bars, block_legend_labels):
        if label not in seen:
            bar.set_label(label); seen.add(label)
    ax.set_xlabel("UP-Bereich (Index)")
    ax.set_ylabel("Ø Amplitude")
    ax.set_title("Vergleich mittlerer Amplituden: violet vs. orange light")
    if filename: fig.suptitle(f"Datei: {filename}", fontsize=10, y=0.98)
    ax.legend(); ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    return fig

def plot_upstate_duration_comparison(Pulse_triggered_UP, Pulse_triggered_DOWN,
                                     Spontaneous_UP, Spontaneous_DOWN, dt):
    trig_durations = (Pulse_triggered_DOWN - Pulse_triggered_UP) * dt if len(Pulse_triggered_UP) else []
    spon_durations = (Spontaneous_DOWN - Spontaneous_UP) * dt if len(Spontaneous_UP) else []
    mean_trig = float(np.mean(trig_durations)) if len(trig_durations) else np.nan
    mean_spon = float(np.mean(spon_durations)) if len(spon_durations) else np.nan

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["Triggered", "Spontaneous"], [mean_trig, mean_spon], color=["red", "blue"], alpha=0.7)
    ax.set_ylabel("Mean UP Dauer (s)")
    ax.set_title("Durchschnittliche UP-Dauer")
    for i, val in enumerate([mean_trig, mean_spon]):
        if not np.isnan(val):
            ax.text(i, val, f"{val:.2f}s", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig

