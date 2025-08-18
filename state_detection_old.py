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
from TimeFreq_plot import Compute_spectrogram, Run_spectrogram
from CSD import CSD_calc
import seaborn as sns
from scipy.signal import find_peaks
import scipy.stats as stats
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loader import load_LFP, load_lightpulses
from preprocessing import downsampling_old, filtering, get_main_channel, pre_post_condition
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_ind
from scipy.signal import welch



def compute_peak_aligned_segments(up_indices, LFP_array, dt, b_lp, a_lp, b_hp, a_hp, align_pre, align_post, align_len):
    """
    Extrahiert peak-aligned LFP-Segmente für UP-Zustände.
    Gibt ein Array (Trials × Zeit) und eine Liste globaler Peak-Indizes zurück.
    """
    peak_segments = np.full((len(up_indices), align_len), np.nan)
    peak_indices = []

    for i, up in enumerate(up_indices):
        start_idx = up - int(0.75 / dt)
        end_idx = up + int(2 / dt)
        slice_data = LFP_array[0, start_idx:end_idx]

        if len(slice_data) < int(2.75 / dt):
            padded = np.full((int(2.75 / dt),), np.nan)
            padded[:len(slice_data)] = slice_data
            current_data = padded
        else:
            current_data = slice_data[:int(2.75 / dt)]

        V_filt = signal.filtfilt(b_lp, a_lp, current_data)
        V_filt = signal.filtfilt(b_hp, a_hp, V_filt)

        peaks, _ = signal.find_peaks(
            V_filt[int(0.25 / dt):int(1.25 / dt)],
            height=np.round(np.std(V_filt), 3),
            distance=150
        )
        peaks += int(0.25 / dt)

        if len(peaks) > 0:
            peak_global = peaks[0] - int(0.75 / dt) + up
            peak_indices.append(peak_global)
            align_start = peaks[0] - align_pre
            align_end = peaks[0] + align_post
            if align_start >= 0 and align_end <= len(current_data):
                peak_segments[i] = current_data[align_start:align_end]

    return np.array(peak_indices, dtype=float), peak_segments



def classify_states(
    Spect_dat,
    time_s,
    pulse_times_1=None,
    pulse_times_2=None,
    dt=None,
    main_channel=None,
    LFP_array=None,
    b_lp=None, a_lp=None, b_hp=None, a_hp=None,
    align_pre=None, align_post=None, align_len=None,
    **kwargs
):
    # Defaults/Robustheit
    pulse_times_1 = np.asarray(pulse_times_1 if pulse_times_1 is not None else [], dtype=float)
    pulse_times_2 = np.asarray(pulse_times_2 if pulse_times_2 is not None else [], dtype=float)

    # ... ab hier verwendest du NUR diese lokalen pulse_times_* ...
    # (deine bestehende Logik unverändert weiterführen)
    # return {...}

    # Frequenzachse holen
    freqs = Spect_dat[2]  # z. B. array([0.01, 0.5, 1.0, ..., 10])
    Spect_dB = np.clip(Spect_dat[0], -100, 100)

    # Frequenzmaske für z. B. 0.5–4 Hz
    delta_mask = (freqs >= 0.1) & (freqs <= 150.0)

    # Nur Delta-Power extrahieren
    linear_power_delta = 10 ** (Spect_dB[delta_mask, :] / 10)
    Total_power = np.sum(linear_power_delta, axis=0)

    
    print("Total_power stats:", np.min(Total_power), np.max(Total_power), np.isnan(Total_power).sum())
  
    Total_power_smooth = gaussian_filter1d(Total_power, sigma=2)  # glättet über ca. 5 Zeitschritte
    
    plt.plot(Total_power)
    plt.title("Total Power (linear)")
    plt.xlabel("Time")
    plt.ylabel("Power")

    # Thresholding
    threshold = np.percentile(Total_power, 65)  # obere 25 % als UP-Zustand 
    up_state_binary = Total_power > threshold  # Binaeres Array: True (1), wenn ueber Schwelle (potenzieller UP-Zustand), sonst False (0)
    up_state_binary = Total_power_smooth > threshold

    
    # Optional: Schließen kurzer False-Lücken innerhalb von UP-Zuständen (z. B. < 200 ms)
    min_gap = int(0.2 / dt)  # z. B. 200 ms Lücke
    binary = up_state_binary.astype(int)

    # Finde Indizes, wo 0er-Stücke ("Gaps") kürzer als min_gap sind
    from itertools import groupby

    gap_starts = []
    gap_lengths = []

    i = 0
    while i < len(binary):
        if binary[i] == 0:
            start = i
            while i < len(binary) and binary[i] == 0:
                i += 1
            length = i - start
            if length < min_gap:
                gap_starts.append(start)
                gap_lengths.append(length)
        else:
            i += 1

    # Fülle kurze Gaps mit 1ern
    for start, length in zip(gap_starts, gap_lengths):
        binary[start:start+length] = 1

    # Überschreibe up_state_binary
    up_state_binary = binary.astype(bool)

    # Transitions
    up_transitions = np.where(np.diff(up_state_binary.astype(int)) == 1)[0]  # Indizes, an denen uebergang von DOWN zu UP stattfindet
    down_transitions = np.where(np.diff(up_state_binary.astype(int)) == -1)[0]  # Indizes, an denen uebergang von UP zu DOWN stattfindet

    # Index-Grenzen korrigieren, falls uebergaenge unvollstaendig
    if down_transitions[0] < up_transitions[0]:  # Wenn erster DOWN-uebergang vor erstem UP liegt (nicht gueltig)
        down_transitions = down_transitions[1:]  # ersten DOWN-Index entfernen
    if up_transitions.shape[0] > down_transitions.shape[0]:  # Falls mehr UPs als DOWNs erkannt wurden
        up_transitions = up_transitions[:-1]  # letztes UP-Event entfernen, da es kein passendes DOWN gibt

    # Neue Arrays nur mit validen UPs
    filtered_UP = []  # Liste fuer gueltige UP-Beginne
    filtered_DOWN = []  # Liste fuer gueltige DOWN-Beginne

    for u, d in zip(up_transitions, down_transitions):  # Schleife ueber gepaarte UP- und DOWN-Zeiten
        duration = time_s[d] - time_s[u]  # Dauer des Zustands in Sekunden berechnen
        print(duration)
        if duration >= 0.3:  # Nur Zustaende >= 0.5 Sekunden Dauer zulassen HIER OFT VERÄNDERN! TIM
            filtered_UP.append(u)  # gueltigen UP-Index speichern
            filtered_DOWN.append(d)  # gueltigen DOWN-Index speichern
        

    UP_start_i = np.array(filtered_UP)  # gueltige UP-Startindizes als NumPy-Array
    DOWN_start_i = np.array(filtered_DOWN)  # gueltige DOWN-Startindizes als NumPy-Array

    # 3. Build pulse time array
    #Pulse_times_array = []  # Initialisiere leere Liste fuer Puls-Zeitpunkte
    # 3. Kombiniere beide Pulstypen
    Pulse_times_array = np.sort(np.concatenate([pulse_times_1, pulse_times_2]))



    # 4. Detect pulse-triggered UP states
    Pulse_triggered_array = [[1, 0]]
    for i in range(len(UP_start_i) - 1):
        t_up = time_s[UP_start_i[i]]
        triggered = np.where((Pulse_times_array >= t_up - 0.25) & (Pulse_times_array <= t_up + 0.25))[0] #200ms ? ASK TIM


        if len(triggered) > 0:
            Pulse_triggered_array = np.concatenate((Pulse_triggered_array, [[UP_start_i[i], DOWN_start_i[i]]]), axis=0)

    # 5. Detect pulse-associated UP states
    Pulse_associated_array = [[1, 0]]
    for j in range(len(UP_start_i) - 1):
        t_up, t_down = time_s[UP_start_i[j]], time_s[DOWN_start_i[j]]
        associated = np.where((Pulse_times_array >= t_up) & (Pulse_times_array <= t_down))[0]
        if len(associated) > 0:
            Pulse_associated_array = np.concatenate((Pulse_associated_array, [[UP_start_i[j], DOWN_start_i[j]]]), axis=0)

    # 6. Remove overlaps
    Pulse_triggered_array = np.array(Pulse_triggered_array)
    Pulse_associated_array = np.array(Pulse_associated_array)

    Pulse_triggered_UP = Pulse_triggered_array[1:, 0]
    Pulse_triggered_DOWN = Pulse_triggered_array[1:, 1]
    Pulse_associated_up = Pulse_associated_array[1:, 0]

    Pulse_associated_down = Pulse_associated_array[1:, 1]

    Pulse_associated_UP = [k for k, x in enumerate(~np.isin(Pulse_associated_up, Pulse_triggered_UP)) if x]
    Pulse_associated_DOWN = [l for l, x in enumerate(~np.isin(Pulse_associated_down, Pulse_triggered_DOWN)) if x]

    Pulse_associated_UP = Pulse_associated_up[Pulse_associated_UP]
    Pulse_associated_DOWN = Pulse_associated_down[Pulse_associated_DOWN]

    Pulse_coincident_UP = np.append(Pulse_triggered_UP, Pulse_associated_UP)
    Pulse_coincident_DOWN = np.append(Pulse_triggered_DOWN, Pulse_associated_DOWN)

    # 7. Identify spontaneous UP states
    Spontaneous_UP = UP_start_i[~np.isin(UP_start_i, Pulse_coincident_UP)]
    Spontaneous_DOWN = DOWN_start_i[~np.isin(DOWN_start_i, Pulse_coincident_DOWN)]
    Spontaneous_UP = Spontaneous_UP[:-1]
    Spontaneous_DOWN = Spontaneous_DOWN[:len(Spontaneous_UP)]

    # 8. Compute durations
    if len(Spontaneous_UP) > 1 and len(Pulse_triggered_UP) > 1:
        Duration_Pulse_Triggered = time_s[Pulse_triggered_DOWN] - time_s[Pulse_triggered_UP]
        Duration_Spontaneous = time_s[Spontaneous_DOWN] - time_s[Spontaneous_UP]
        Duration_Pulse_Associated = time_s[Pulse_associated_DOWN] - time_s[Pulse_associated_UP]
        Dur_stat, Dur_p = stats.ttest_ind(Duration_Spontaneous, Duration_Pulse_Triggered)
    else:
        Duration_Pulse_Triggered = np.array([])
        Duration_Spontaneous = np.array([])
        Duration_Pulse_Associated = np.array([])
        Dur_stat, Dur_p = np.nan, np.nan

    Ctrl_UP = np.zeros(len(Pulse_triggered_UP), dtype=int)
    Ctrl_DOWN = np.zeros(len(Pulse_triggered_DOWN), dtype=int)
    for i in range(len(Ctrl_UP)):
        Ctrl_UP[i] = random.randint(0, len(time_s) - int(4 / dt))
        Ctrl_DOWN[i] = Ctrl_UP[i] + int(Pulse_triggered_DOWN[i] - Pulse_triggered_UP[i])

    # 9. Extract spontaneous UP state segments and peaks
    Spon_UP_array = np.zeros((len(Spontaneous_UP), align_len))
    Spon_UP_peak_alligned_array = np.full((len(Spontaneous_UP), align_len), np.nan)
    Spon_Peaks = np.full((len(Spontaneous_UP),), np.nan)
    UP_Time = np.arange(Spon_UP_peak_alligned_array.shape[1]) * dt - align_pre * dt
    Spon_Peaks = []

    for i_Spon in range(len(Spontaneous_UP)):
        start_idx = Spontaneous_UP[i_Spon] - int(0.75 / dt)
        end_idx = Spontaneous_UP[i_Spon] + int(2 / dt)
        slice_data = LFP_array[0, start_idx:end_idx]

        if len(slice_data) < int(2.75 / dt):
            padded = np.full((int(2.75 / dt),), np.nan)
            padded[:len(slice_data)] = slice_data
            current_data = padded
        else:
            current_data = slice_data[:int(2.75 / dt)]

        V_filt = signal.filtfilt(b_lp, a_lp, current_data)
        V_filt = signal.filtfilt(b_hp, a_hp, V_filt)

        peaks, _ = find_peaks(
            V_filt[int(0.25 / dt):int(1.25 / dt)],
            height=np.round(np.std(V_filt), 3),
            distance=150
        )
        peaks += int(0.25 / dt)

        if len(peaks) > 0:
            peak_global = peaks[0] - int(0.75 / dt) + Spontaneous_UP[i_Spon]
            Spon_Peaks.append(peak_global)

            align_start = peaks[0] - align_pre
            align_end = peaks[0] + align_post
            if align_start >= 0 and align_end <= len(current_data):
                Spon_UP_peak_alligned_array[i_Spon] = current_data[align_start:align_end]
            else:
                Spon_UP_peak_alligned_array[i_Spon] = np.full((align_len,), np.nan)
        else:
            Spon_Peaks.append(np.nan)
            Spon_UP_peak_alligned_array[i_Spon] = np.full((align_len,), np.nan)

    Spon_Peaks = np.array(Spon_Peaks, dtype=float)
    Trig_Peaks, Trig_UP_peak_aligned_array = compute_peak_aligned_segments(
        Pulse_triggered_UP, LFP_array, dt, b_lp, a_lp, b_hp, a_hp, align_pre, align_post, align_len
    )

    # 10. Extract pulse-triggered UP state peaks
    Trig_Peaks = []
    Trig_UP_peak_alligned_array = np.full((len(Pulse_triggered_UP), align_len), np.nan)

    for i_Trig in range(len(Pulse_triggered_UP)):
        start_idx = Pulse_triggered_UP[i_Trig] - int(0.75 / dt)
        end_idx = Pulse_triggered_UP[i_Trig] + int(2 / dt)
        slice_data = LFP_array[0, start_idx:end_idx]

        if len(slice_data) < int(2.75 / dt):
            padded = np.full((int(2.75 / dt),), np.nan)
            padded[:len(slice_data)] = slice_data
            current_data = padded
        else:
            current_data = slice_data[:int(2.75 / dt)]

        V_filt = signal.filtfilt(b_lp, a_lp, current_data)
        V_filt = signal.filtfilt(b_hp, a_hp, V_filt)

        peaks, _ = find_peaks(
            V_filt[int(0.25 / dt):int(1.25 / dt)],
            height=np.round(np.std(V_filt), 3),
            distance=150
        )
        peaks += int(0.25 / dt)

        if len(peaks) > 0:
            peak_global = peaks[0] - int(0.75 / dt) + Pulse_triggered_UP[i_Trig]
            Trig_Peaks.append(peak_global)

            align_start = peaks[0] - align_pre
            align_end = peaks[0] + align_post
            if align_start >= 0 and align_end <= len(current_data):
                Trig_UP_peak_alligned_array[i_Trig] = current_data[align_start:align_end]
            else:
                Trig_UP_peak_alligned_array[i_Trig] = np.full((align_len,), np.nan)
        else:
            Trig_Peaks.append(np.nan)
            Trig_UP_peak_alligned_array[i_Trig] = np.full((align_len,), np.nan)

    Trig_Peaks = np.array(Trig_Peaks, dtype=float)



    return {
        "Pulse_triggered_UP": Pulse_triggered_UP,
        "Pulse_triggered_DOWN": Pulse_triggered_DOWN,
        "Pulse_associated_UP": Pulse_associated_UP,
        "Pulse_associated_DOWN": Pulse_associated_DOWN,
        "Spontaneous_UP": Spontaneous_UP,
        "Spontaneous_DOWN": Spontaneous_DOWN,
        "Duration_Pulse_Triggered": Duration_Pulse_Triggered,
        "Duration_Pulse_Associated": Duration_Pulse_Associated,
        "Duration_Spontaneous": Duration_Spontaneous,
        "Dur_stat": Dur_stat,
        "Dur_p": Dur_p,
        "Ctrl_UP": Ctrl_UP,
        "Ctrl_DOWN": Ctrl_DOWN,
        "Spon_Peaks": Spon_Peaks,
        "Spon_UP_peak_alligned_array": Spon_UP_peak_alligned_array,
        "UP_Time": UP_Time,
        "Spon_UP_array": Spon_UP_array,
        "Total_power": Total_power,
        "UP_start_i": UP_start_i,
        "DOWN_start_i": DOWN_start_i,
        "up_state_binary ": up_state_binary, 
        "Trig_Peaks": Trig_Peaks,
        "Trig_UP_peak_alligned_array": Trig_UP_peak_aligned_array


    }

    


def Generate_CSD_mean(peaks_list, signal_array, dt):
    all_csd = []
    for i, peak in enumerate(peaks_list):
        if not np.isnan(peak):
            start = int(peak - int(0.5 / dt))
            end = int(peak + int(0.5 / dt))
            if start >= 1 and end <= signal_array.shape[1] - 1:
                segment = signal_array[:, start:end]
                CSD_out = CSD_calc(segment, dt)
                all_csd.append(CSD_out)

    if len(all_csd) > 0:
        return np.nanmean(np.stack(all_csd), axis=0)
    else:
        print("❌ Kein CSD berechnet – Rückgabe: Dummy mit NaNs")
        return np.full((signal_array.shape[0], int(1 / dt)), np.nan)



def plot_CSD_comparison(csd_spont, csd_triggered, dt):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

     # Zeitachse
    n_timepoints = csd_spont.shape[1]
    t = np.linspace(-0.5, 0.5, n_timepoints)

    # Gemeinsame Farbsättigung berechnen
    v_abs = np.nanmax(np.abs([csd_spont, csd_triggered]))

    im1 = axs[0].imshow(csd_spont, aspect='auto', origin='lower',
                        extent=[t[0], t[-1], 0, csd_spont.shape[0]],
                        cmap='seismic', vmin=-v_abs, vmax=v_abs)
    axs[0].set_title("Spontaneous UP CSD")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Channel")
    fig.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(csd_triggered, aspect='auto', origin='lower',
                        extent=[t[0], t[-1], 0, csd_triggered.shape[0]],
                        cmap='seismic', vmin=-v_abs, vmax=v_abs)
    axs[1].set_title("Pulse-triggered UP CSD")
    axs[1].set_xlabel("Time (s)")
    #cbar = fig.colorbar(im1, ax=axs, shrink=0.8, location='right', label='CSD (a.u.)')

    plt.suptitle("Vergleich: Spontane vs. Getriggerte CSDs", fontsize=14)
    plt.tight_layout()
    plt.show()


def Spectrum(V_win):
	LFP_win = V_win - np.mean(V_win)
	# Windowing if you want
	w = np.hanning(len(V_win))
	LFP_win = w * LFP_win
	# Calculate power spectrum for window
	Fs = srate
	N = len(LFP_win)
	xdft = np.fft.fft(LFP_win)
	xdft = xdft[0:int((N / 2) + 1)]
	psdx = (1 / (Fs * N)) * np.abs(xdft) ** 2
	freq = np.arange(0, (Fs / 2) + Fs / N, Fs / N)
	Pow = psdx
	Pow = np.zeros((501, 1))
	for j in range(0, 200):
		Pow[j] = psdx[j]
	return Pow, freq


def extract_upstate_windows(UP_indices, LFP_array, dt, window_s):
    samples = int(window_s / dt)
    up_windows = []
    for idx in UP_indices:
        start = idx - samples // 2
        end = idx + samples // 2
        if start >= 0 and end <= LFP_array.shape[1]:
            window = LFP_array[0, start:end]
            up_windows.append(window)
    return up_windows

# state_detection_old.py



def compute_spectra(windows, dt, ignore_start_s: float = 0.0, nperseg: int | None = None):
    """
    windows: iterable aus 1D-Arrays (ein Kanal je Fenster)
    Rückgabe: (spectra: (n_windows, n_freqs), freqs: (n_freqs,))
              Bei keinen gültigen Fenstern → (array shape (0,0), array shape (0,))
    """
    fs = 1.0 / dt
    spectra = []
    freqs = None

    if windows is None:
        windows = []

    for w in windows:
        w = np.asarray(w)
        if w.ndim > 1:
            w = w.squeeze()

        # Anfangsbereich optional abschneiden
        if ignore_start_s and ignore_start_s > 0:
            cut = int(round(ignore_start_s / dt))
            if cut >= w.size:
                continue
            w = w[cut:]

        # Zu kurze Fenster überspringen
        if w.size < 4:
            continue

        seg = min(nperseg or 256, w.size)
        f, Pxx = welch(w, fs=fs, nperseg=seg)

        if freqs is None:
            freqs = f
        spectra.append(Pxx)

    if freqs is None:
        # Keine gültigen Fenster
        return np.empty((0, 0), dtype=float), np.empty((0,), dtype=float)

    return np.vstack(spectra) if len(spectra) else np.empty((0, freqs.size), dtype=float), freqs





def compare_spectra(pulse_windows, spont_windows, dt, ignore_start_s: float = 0.0):
    print(f"🚨 compare_spectra wurde aufgerufen mit ignore_start_s = {ignore_start_s}")
    cut_samples = int(round(ignore_start_s / dt))
    print(f"⏩ compute_spectra schneidet {ignore_start_s} Sekunden = {cut_samples} Samples ab")

    pulse_spec, freqs_p = compute_spectra(pulse_windows, dt, ignore_start_s)
    spont_spec, freqs_s = compute_spectra(spont_windows, dt, ignore_start_s)

    # Falls nichts Brauchbares da ist, sauber zurückkehren
    if freqs_p.size == 0 or freqs_s.size == 0:
        print("⚠️ Keine gültigen Fenster (nach Cut/Filter) – breche Vergleich ab.")
        return np.empty((0,), dtype=float), np.empty((0,), dtype=float), np.empty((0,), dtype=float), np.empty((0,), dtype=float)

    # Frequenzachsen abgleichen (zur Sicherheit)
    if not np.array_equal(freqs_p, freqs_s):
        print("ℹ️ Frequenzachsen unterscheiden sich – schneide auf Schnittmenge zu.")
        common = np.intersect1d(freqs_p, freqs_s)
        if common.size == 0:
            print("⚠️ Keine gemeinsame Frequenzachse.")
            return np.empty((0,), dtype=float), np.empty((0,), dtype=float), np.empty((0,), dtype=float), np.empty((0,), dtype=float)
        # Indizes zuordnen
        idx_p = np.nonzero(np.in1d(freqs_p, common))[0]
        idx_s = np.nonzero(np.in1d(freqs_s, common))[0]
        pulse_spec = pulse_spec[:, idx_p] if pulse_spec.size else pulse_spec
        spont_spec = spont_spec[:, idx_s] if spont_spec.size else spont_spec
        freqs = common
    else:
        freqs = freqs_p

    # Mittelwerte bilden (leere Arrays handhaben)
    pulse_mean = pulse_spec.mean(axis=0) if pulse_spec.size else np.empty((freqs.size,), dtype=float)
    spont_mean = spont_spec.mean(axis=0) if spont_spec.size else np.empty((freqs.size,), dtype=float)

    # Einfaches p-Value-Array (hier Platzhalter, falls du später Stats machst)
    try:
        import scipy.stats as stats
        if pulse_spec.shape[0] > 1 and spont_spec.shape[0] > 1:
            p_vals = np.array([stats.ttest_ind(pulse_spec[:, i], spont_spec[:, i], equal_var=False, nan_policy="omit").pvalue
                               for i in range(freqs.size)])
        else:
            p_vals = np.full(freqs.size, np.nan)
    except Exception:
        p_vals = np.full(freqs.size, np.nan)

    return freqs, spont_mean, pulse_mean, p_vals


def plot_contrast_heatmap(pulse_windows, spont_windows, dt, ignore_start_s: float = 0.0):
    pulse_spec, f_p = compute_spectra(pulse_windows, dt, ignore_start_s)
    spont_spec, f_s = compute_spectra(spont_windows, dt, ignore_start_s)

    # Frühzeitiger Ausstieg, falls leer
    if f_p.size == 0 or f_s.size == 0 or pulse_spec.size == 0 or spont_spec.size == 0:
        print("⚠️ plot_contrast_heatmap: keine gültigen Spektren – skip.")
        return

    # Frequenzen angleichen (Schnittmenge)
    if not np.array_equal(f_p, f_s):
        common = np.intersect1d(f_p, f_s)
        if common.size == 0:
            print("⚠️ plot_contrast_heatmap: keine gemeinsame Frequenzachse – skip.")
            return
        idx_p = np.nonzero(np.in1d(f_p, common))[0]
        idx_s = np.nonzero(np.in1d(f_s, common))[0]
        pulse_spec = pulse_spec[:, idx_p]
        spont_spec = spont_spec[:, idx_s]
        freqs = common
    else:
        freqs = f_p

    # Mittelwerte trimmen (falls deine Originalfunktion das macht)
    pulse_trim = pulse_spec
    spont_trim = spont_spec

    # Sicherheit: gleiche Form
    if pulse_trim.shape[1] != spont_trim.shape[1]:
        print("⚠️ plot_contrast_heatmap: Form-Unterschied – skip.")
        return

    contrast = pulse_trim - spont_trim

    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(contrast, aspect='auto', origin='lower',
               extent=[freqs[0], freqs[-1], 0, contrast.shape[0]])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Window #")
    plt.title("Pulse − Spont Contrast")
    plt.colorbar(label="Power diff")
    plt.tight_layout()
    plt.show()




def average_amplitude_in_upstates(main_channel, time_s, UP_start_i, DOWN_start_i, start_idx, end_idx):
    """
    Berechnet die mittlere Amplitude (mean(abs)) der UP-Zustände im Bereich [start_idx, end_idx] (1-basiert).
    """

    assert 1 <= start_idx <= end_idx <= len(UP_start_i), "Ungültiger Indexbereich."

    selected_amplitudes = []

    for i in range(start_idx - 1, end_idx):
        up = UP_start_i[i]
        down = DOWN_start_i[i]

        if down > up and down < len(main_channel):
            segment = main_channel[up:down]
            amp = np.mean(np.abs(segment))  # Oder np.ptp(segment) für peak-to-peak
            selected_amplitudes.append(amp)

    if selected_amplitudes:
        avg_amp = np.mean(selected_amplitudes)
        print(f"✅ Durchschnittliche Amplitude der UP-Zustände {start_idx}–{end_idx}: {avg_amp:.4f}")
        print("📋 Einzelwerte:")
        for i, a in enumerate(selected_amplitudes, start=start_idx):
            print(f"{i}. {a:.4f}")
    else:
        print("❌ Keine gültigen UP-Zustände im angegebenen Bereich gefunden.")




def plot_LFP_average_around_peaks(peaks, LFP_array, dt, window_s=1.0, channel_idx=0, label="Spontaneous"):
    half_window = int((window_s / 2) / dt)
    traces = []

    for peak in peaks:
        if not np.isnan(peak):
            peak = int(peak)
            start = peak - half_window
            end = peak + half_window
            if start >= 0 and end <= LFP_array.shape[1]:
                trace = LFP_array[channel_idx, start:end]
                traces.append(trace)

    if len(traces) == 0:
        print(f"❌ Keine gültigen Trials für {label}")
        return

    traces = np.array(traces)
    mean_trace = np.nanmean(traces, axis=0)
    sem_trace = np.nanstd(traces, axis=0) / np.sqrt(traces.shape[0])
    t = np.arange(-half_window, half_window) * dt

    plt.plot(t, mean_trace, label=label)
    plt.fill_between(t, mean_trace - sem_trace, mean_trace + sem_trace, alpha=0.3)
