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

def plot_all_channels(num_channels: int, time_s:int, LFP_array):
    fig, axs = plt.subplots(num_channels, 1, figsize=(12, 2 * num_channels), sharex=True)
    for i in range(num_channels):
        axs[i].plot(time_s, LFP_array[i])
        axs[i].set_ylabel(f"pri_{i}")
        axs[i].grid(True)
    axs[-1].set_xlabel("Zeit (s)")
    plt.suptitle("Plot 1 – LFP-Kanäle über Zeit (getrennte Darstellung)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def plot_spont_up_mean(main_channel, time_s, dt, Spon_Peaks, up_state_binary,
                       pulse_times_1, pulse_times_2,
                       Pulse_triggered_UP, Pulse_triggered_DOWN,
                       Spontaneous_UP, Spontaneous_DOWN):
    plt.figure(figsize=(12, 5))
    plt.plot(time_s, main_channel, label='LFP', color='black')

    added_labels = set()

    # 🔷 Triggered UPs: rot
    for up, down in zip(Pulse_triggered_UP, Pulse_triggered_DOWN):
        up, down = int(up), int(down)
        if down < len(time_s):
            label = 'getriggert' if 'getriggert' not in added_labels else None
            plt.axvspan(time_s[up], time_s[down], color='red', alpha=0.3, label=label)
            added_labels.add('getriggert')

    # 🔵 Spontane UPs: blau
    for up, down in zip(Spontaneous_UP, Spontaneous_DOWN):
        up, down = int(up), int(down)
        if down < len(time_s):
            label = 'spontan' if 'spontan' not in added_labels else None
            plt.axvspan(time_s[up], time_s[down], color='lightblue', alpha=0.3, label=label)
            added_labels.add('spontan')

    # 🔴 Pulse Typ 1 (rot gestrichelt)
    visible_pulses_1 = pulse_times_1[(pulse_times_1 >= time_s[0]) & (pulse_times_1 <= time_s[-1])]
    for i, p in enumerate(visible_pulses_1):
        plt.axvline(p, color='red', linestyle='--', linewidth=0.8, label='Licht 1' if i == 0 else None)

    # 🔵 Pulse Typ 2 (blau gepunktet)
    visible_pulses_2 = pulse_times_2[(pulse_times_2 >= time_s[0]) & (pulse_times_2 <= time_s[-1])]
    for i, p in enumerate(visible_pulses_2):
        plt.axvline(p, color='blue', linestyle=':', linewidth=0.8, label='Licht 2' if i == 0 else None)

    # 🧠 Alle UPs zusammenführen und nummerieren
    all_ups = np.concatenate((Pulse_triggered_UP, Spontaneous_UP))
    all_times = time_s[all_ups]
    sort_idx = np.argsort(all_times)
    all_ups_sorted = all_ups[sort_idx]
    all_times_sorted = time_s[all_ups_sorted]

    for i, t in enumerate(all_times_sorted):
        plt.annotate(str(i + 1), xy=(t, plt.ylim()[0]), xytext=(t, plt.ylim()[0] - 0.05 * np.ptp(main_channel)),
                     ha='center', va='top', fontsize=7, rotation=90, color='black')

       # 📋 Liste der UP-State-Zeitpunkte ausgeben (nummeriert)
    print("\n🧾 Liste der nummerierten UP-Zustände:")
    for i, t in enumerate(all_times_sorted):
        print(f"{i + 1}.  {t:.3f} s")

    # 📦 Format
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys())

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("UP-States: spontaneous (blau) vs. triggered (red)xxxxx")
    plt.tight_layout()
    plt.show()

    return all_ups_sorted 



def Total_power_plot(Spect_dat):
    plt.plot(Spect_dat[1], np.sum(Spect_dat[0], axis=0))
    plt.title("Gesamtleistung im 1–10Hz Bereich")
    plt.xlabel("Zeit (s)")
    plt.ylabel("Power (summiert)")
    plt.show()


def tests(Spect_dat, time_s, Total_power, UP_start_i):
    plt.plot(Spect_dat[1], np.sum(Spect_dat[0], axis=0))
    plt.scatter(time_s[UP_start_i.astype(int)], Total_power[UP_start_i.astype(int)], color="red", label="Detected UP")

    plt.show()


def plot_upstate_amplitudes(main_channel, UP_start_i, DOWN_start_i, start_idx, end_idx):
    """
    Plottet die mittlere Amplitude jedes UP-Zustands im Bereich [start_idx, end_idx] (1-basiert).
    """

    assert 1 <= start_idx <= end_idx <= len(UP_start_i), "Ungültiger Indexbereich."

    indices = list(range(start_idx, end_idx + 1))
    amplitudes = []

    for i in range(start_idx - 1, end_idx):
        up = UP_start_i[i]
        down = DOWN_start_i[i]

        if down > up and down < len(main_channel):
            segment = main_channel[up:down]
            amp = np.mean(np.abs(segment))
            amplitudes.append(amp)
        else:
            amplitudes.append(np.nan)

    plt.figure(figsize=(10, 4))
    plt.bar(indices, amplitudes, color='mediumseagreen')
    plt.xlabel("UP-State Index")
    plt.ylabel("Ø Amplitude")
    plt.title(f"Mittlere Amplituden der UP-Zustände {start_idx}–{end_idx}")
    plt.xticks(indices)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    return amplitudes


def plot_upstate_amplitude_mean(main_channel, UP_start_i, DOWN_start_i, start_idx, end_idx):
    """
    Plottet einen einzelnen Balken: den Durchschnitt der mittleren Amplituden der UP-Zustände [start_idx, end_idx].
    """
    assert 1 <= start_idx <= end_idx <= len(UP_start_i), "Ungültiger Indexbereich."

    amplitudes = []

    for i in range(start_idx - 1, end_idx):
        up = UP_start_i[i]
        down = DOWN_start_i[i]

        if down > up and down < len(main_channel):
            segment = main_channel[up:down]
            amp = np.mean(np.abs(segment))
            amplitudes.append(amp)

    if amplitudes:
        avg_amp = np.mean(amplitudes)
        plt.figure(figsize=(4, 5))
        plt.bar([f"UPs {start_idx}-{end_idx}"], [avg_amp], color='mediumslateblue')
        plt.ylabel("Ø Amplitude")
        plt.title("Durchschnittliche Amplitude")
        plt.tight_layout()
        plt.show()
        print(f"✅ Durchschnittliche Amplitude von UPs {start_idx}–{end_idx}: {avg_amp:.4f}")
    else:
        print("❌ Keine gültigen UP-Zustände gefunden.")

def plot_upstate_amplitude_blocks_colored(main_channel, UP_start_i, DOWN_start_i, index_blocks, filename):
    """
    Plottet die durchschnittlichen Amplituden für UP-Blocks mit spezifischen Farben und Labels.
    Block 1 & 3 = "violet light", Block 2 & 4 = "orange light"
    """

    block_labels = []
    block_means = []
    block_colors = []
    block_legend_labels = []

    for idx, (start_idx, end_idx) in enumerate(index_blocks):
        assert 1 <= start_idx <= end_idx <= len(UP_start_i), f"Ungültiger Bereich: {start_idx}-{end_idx}"

        amplitudes = []
        for i in range(start_idx - 1, end_idx):
            up = UP_start_i[i]
            down = DOWN_start_i[i]

            if down > up and down < len(main_channel):
                segment = main_channel[up:down]
                amp = np.mean(np.abs(segment))
                amplitudes.append(amp)

        mean_amp = np.mean(amplitudes) if amplitudes else np.nan
        block_means.append(mean_amp)

        # Beschriftung
        label_range = f"{start_idx}-{end_idx}"
        block_labels.append(label_range)

        # Farbe & Gruppenname
        if idx % 2 == 0:
            block_colors.append("orange")
            block_legend_labels.append("orange light")
        else:
            block_colors.append("violet")
            block_legend_labels.append("violet light")

    # Plot
    plt.figure(figsize=(8, 5))
    bars = plt.bar(block_labels, block_means, color=block_colors)

    # Nur einmal pro Farbe ins Legende aufnehmen
    seen = set()
    for bar, label in zip(bars, block_legend_labels):
        if label not in seen:
            bar.set_label(label)
            seen.add(label)

    plt.xlabel("UP-Bereich (Index)")
    plt.ylabel("Ø Amplitude")
    plt.title("Vergleich mittlerer Amplituden: violet vs. orange light")
    if filename:
        plt.suptitle(f"Datei: {filename}", fontsize=10, y=0.98)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

    return dict(zip(block_labels, block_means))

