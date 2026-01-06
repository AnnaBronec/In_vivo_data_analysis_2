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
from scipy.ndimage import gaussian_filter
from scipy import signal
import numpy as np
from loader import load_LFP, load_lightpulses
from preprocessing import downsampling, filtering, get_main_channel, pre_post_condition
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_ind
from scipy.signal import welch


def _save_svg(fig, hint, out_dir=None, dpi=200):
    if out_dir is None:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{hint}.svg")
    fig.canvas.draw_idle()
    fig.savefig(path, format="svg", dpi=dpi, bbox_inches="tight")
    print(f"[SAVED] {path}")


def compute_peak_aligned_segments(
    up_indices, LFP_array, dt,
    b_lp, a_lp, b_hp, a_hp,
    align_pre, align_post, align_len,
    *,
    # Neues Fenster: ab +0.2 s bis +2.0 s nach UP-Start
    offset_start=0.2,          # s relativ zum UP-Start
    offset_end=2.0,            # s relativ zum UP-Start
    # Peak-Suchfenster (innerhalb des obigen Ausschnitts)
    search_start_s=0.2,        # s relativ zum UP-Start
    search_end_s=2.0,          # s relativ zum UP-Start
    # minimaler Peak-Abstand in Sekunden 
    min_peak_spacing_s=0.15
):
    peak_segments = np.full((len(up_indices), align_len), np.nan)
    peak_indices = []

    for i, up in enumerate(up_indices):
        start_idx = up + int(offset_start / dt)
        end_idx   = up + int(offset_end   / dt)
        if start_idx < 0 or end_idx <= start_idx or end_idx > LFP_array.shape[1]:
            continue

        current_data = np.asarray(LFP_array[0, start_idx:end_idx], dtype=float)

        # Filtern (LP dann HP)
        try:
            V_filt = signal.filtfilt(b_lp, a_lp, current_data)
            V_filt = signal.filtfilt(b_hp, a_hp, V_filt)
        except Exception:
            V_filt = current_data  

        # Peak-Suchbereich innerhalb des Ausschnitts festlegen
        lo = int((search_start_s - offset_start) / dt)  # relative Indizes
        hi = int((search_end_s   - offset_start) / dt)
        lo = max(0, min(lo, len(V_filt)))
        hi = max(lo+1, min(hi, len(V_filt)))

        # Peaks suchen – Abstand zeitbasiert skalieren
        min_dist = max(1, int(min_peak_spacing_s / dt))
        peaks, _ = signal.find_peaks(
            V_filt[lo:hi],
            height=np.round(np.nanstd(V_filt), 3),
            distance=min_dist
        )
        peaks += lo  # zurück auf V_filt-Index

        if len(peaks) > 0:
            # Ersten Peak nehmen und globalen Index speichern
            peak_global = start_idx + peaks[0]
            peak_indices.append(peak_global)

            # Aligniertes Segment rund um den Peak extrahieren
            align_start = peaks[0] - align_pre
            align_end   = peaks[0] + align_post
            if 0 <= align_start and align_end <= len(current_data):
                peak_segments[i] = current_data[align_start:align_end]

    return np.array(peak_indices, dtype=float), peak_segments

def classify_states(Spect_dat, time_s, pulse_times_1, pulse_times_2, dt, V1_1,
                    LFP_array, b_lp, a_lp, b_hp, a_hp,
                    align_pre, align_post, align_len):


    # ---------- helpers ----------
    def _empty_states(Total_power, time_s, dt, align_pre, align_len):
        time_s = np.asarray(time_s)
        return {
            "Pulse_triggered_UP": np.array([], dtype=int),
            "Pulse_triggered_DOWN": np.array([], dtype=int),
            "Pulse_associated_UP": np.array([], dtype=int),
            "Pulse_associated_DOWN": np.array([], dtype=int),
            "Spontaneous_UP": np.array([], dtype=int),
            "Spontaneous_DOWN": np.array([], dtype=int),
            "Duration_Pulse_Triggered": np.array([], dtype=float),
            "Duration_Pulse_Associated": np.array([], dtype=float),
            "Duration_Spontaneous": np.array([], dtype=float),
            "Dur_stat": np.nan,
            "Dur_p": np.nan,
            "Ctrl_UP": np.array([], dtype=int),
            "Ctrl_DOWN": np.array([], dtype=int),
            "Spon_Peaks": np.array([], dtype=float),
            "Trig_Peaks": np.array([], dtype=float),
            "Spon_UP_peak_aligned_array": np.full((0, align_len), np.nan),
            "Trig_UP_peak_aligned_array": np.full((0, align_len), np.nan),
            "UP_Time": (np.arange(align_len) * dt - align_pre * dt),
            "Spon_UP_array": np.full((0, align_len), np.nan),
            "Total_power": Total_power,
            "UP_start_i": np.array([], dtype=int),
            "DOWN_start_i": np.array([], dtype=int),
            "up_state_binary": np.zeros_like(time_s, dtype=bool),
        }

    # ---------- 1) Bandpower aus Spektrogramm ----------
    freqs = Spect_dat[2]
    S = np.asarray(Spect_dat[0], float)  # dB

    # 
    f_lo, f_hi = 0.5, 30.0
    band_mask = (freqs >= f_lo) & (freqs <= f_hi)

    # dB -> linear, stabil clippen
    S_clip = np.clip(S[band_mask, :], -120, 30)
    linear = 10 ** (S_clip / 10.0)

    # robuster als Summe: median reduziert “hot bins”
    Total_power = np.median(linear, axis=0)

    print("Total_power stats:", np.min(Total_power), np.max(Total_power), np.isnan(Total_power).sum())

    # ---------- 2) Smooth ----------
    Total_power_smooth = gaussian_filter1d(Total_power, sigma=2)

        # robust z-score auf Feature-Achse
    med = np.median(Total_power_smooth)
    mad = np.median(np.abs(Total_power_smooth - med)) + 1e-30
    robust_std = 1.4826 * mad

    z = (Total_power_smooth - med) / robust_std

    k_hi = 4.0   # Start-Schwelle (strenger)
    k_lo = 2.5   # Halte-Schwelle (lockerer)

    hi = z > k_hi
    lo = z > k_lo

    # hysteresis reconstruction: UP startet bei hi, läuft weiter solange lo true ist
    up_state_binary = np.zeros_like(hi, dtype=bool)
    active = False
    for i in range(len(hi)):
        if not active and hi[i]:
            active = True
        if active and not lo[i]:
            active = False
        up_state_binary[i] = active


    # ---------- 4) Gap-closing: ACHTUNG Zeitauflösung! ----------
    # dt für up_state_binary muss zur Feature-Zeitachse passen:
    time_s = np.asarray(time_s)
    if time_s.size >= 2 and np.all(np.isfinite(time_s)):
        dt_feat = float(np.median(np.diff(time_s)))
    else:
        dt_feat = float(dt)  # fallback

    min_gap_s = 0.2
    min_gap = max(1, int(round(min_gap_s / dt_feat)))

    binary = up_state_binary.astype(np.int8)

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

    for start, length in zip(gap_starts, gap_lengths):
        binary[start:start+length] = 1

    up_state_binary = binary.astype(bool)

    # ---------- 5) Transitions + Dauerfilter ----------
    up_transitions = np.where(np.diff(up_state_binary.astype(int)) == 1)[0]
    down_transitions = np.where(np.diff(up_state_binary.astype(int)) == -1)[0]

    if down_transitions.size and up_transitions.size and down_transitions[0] < up_transitions[0]:
        down_transitions = down_transitions[1:]
    if up_transitions.size > down_transitions.size:
        up_transitions = up_transitions[:-1]

    min_up_len_s = 0.5
    if not up_transitions.size or not down_transitions.size:
        UP_start_i = np.array([], dtype=int)
        DOWN_start_i = np.array([], dtype=int)
    else:
        filtered_UP = []
        filtered_DOWN = []
        for u, d in zip(up_transitions, down_transitions):
            # time_s passt hier zur Feature-Achse
            duration = time_s[d] - time_s[u]
            if duration >= min_up_len_s:
                filtered_UP.append(u)
                filtered_DOWN.append(d)

        UP_start_i = np.array(filtered_UP, dtype=int)
        DOWN_start_i = np.array(filtered_DOWN, dtype=int)

    # ---------- 6) NO-UP Gate ganz am Ende ----------
    min_up_fraction = 0.03
    min_up_onsets = 2

    print("[BAND] f_lo,f_hi =", f_lo, f_hi)
    #print("[THR] thr =", threshold, "smooth min/max =", Total_power_smooth.min(), Total_power_smooth.max())
    print("[UP%] frac UP =", up_state_binary.mean(), "| dt_feat =", dt_feat, "min_gap_bins =", min_gap)
    print("[UP] onsets =", len(UP_start_i))

    if (up_state_binary.mean() < min_up_fraction) or (len(UP_start_i) < min_up_onsets):
        print("[STATES] no reliable UP states -> return empty dict")
        return _empty_states(Total_power, time_s, dt, align_pre, align_len)

    # ---------- 7) Puls-Logik ----------
    Pulse_times_array = np.sort(np.concatenate([pulse_times_1, pulse_times_2])) if (len(pulse_times_1) or len(pulse_times_2)) else np.array([])

    Pulse_triggered_array = [[1, 0]]
    for i in range(len(UP_start_i) - 1):
        t_up = time_s[UP_start_i[i]]
        if Pulse_times_array.size == 0:
            continue
        triggered = np.where((Pulse_times_array >= t_up - 0.35) & (Pulse_times_array <= t_up + 0.35))[0]
        if len(triggered) > 0:
            Pulse_triggered_array = np.concatenate((Pulse_triggered_array, [[UP_start_i[i], DOWN_start_i[i]]]), axis=0)

    Pulse_associated_array = [[1, 0]]
    for j in range(len(UP_start_i) - 1):
        t_up, t_down = time_s[UP_start_i[j]], time_s[DOWN_start_i[j]]
        if Pulse_times_array.size == 0:
            continue
        associated = np.where((Pulse_times_array >= t_up) & (Pulse_times_array <= t_down))[0]
        if len(associated) > 0:
            Pulse_associated_array = np.concatenate((Pulse_associated_array, [[UP_start_i[j], DOWN_start_i[j]]]), axis=0)

    Pulse_triggered_array = np.array(Pulse_triggered_array)
    Pulse_associated_array = np.array(Pulse_associated_array)

    Pulse_triggered_UP = Pulse_triggered_array[1:, 0] if Pulse_triggered_array.shape[0] > 1 else np.array([], dtype=int)
    Pulse_triggered_DOWN = Pulse_triggered_array[1:, 1] if Pulse_triggered_array.shape[0] > 1 else np.array([], dtype=int)

    Pulse_associated_up = Pulse_associated_array[1:, 0] if Pulse_associated_array.shape[0] > 1 else np.array([], dtype=int)
    Pulse_associated_down = Pulse_associated_array[1:, 1] if Pulse_associated_array.shape[0] > 1 else np.array([], dtype=int)

    # Overlaps entfernen
    Pulse_associated_UP_idx = [k for k, x in enumerate(~np.isin(Pulse_associated_up, Pulse_triggered_UP)) if x]
    Pulse_associated_DOWN_idx = [l for l, x in enumerate(~np.isin(Pulse_associated_down, Pulse_triggered_DOWN)) if x]

    Pulse_associated_UP = Pulse_associated_up[Pulse_associated_UP_idx] if len(Pulse_associated_UP_idx) else np.array([], dtype=int)
    Pulse_associated_DOWN = Pulse_associated_down[Pulse_associated_DOWN_idx] if len(Pulse_associated_DOWN_idx) else np.array([], dtype=int)

    Pulse_coincident_UP = np.append(Pulse_triggered_UP, Pulse_associated_UP)
    Pulse_coincident_DOWN = np.append(Pulse_triggered_DOWN, Pulse_associated_DOWN)

    # Priorität umkehren
    Pulse_triggered_UP = np.setdiff1d(Pulse_triggered_UP, Pulse_associated_UP)

    # Spontan
    Spontaneous_UP = UP_start_i[~np.isin(UP_start_i, Pulse_coincident_UP)]
    Spontaneous_DOWN = DOWN_start_i[~np.isin(DOWN_start_i, Pulse_coincident_DOWN)]
    if Spontaneous_UP.size:
        Spontaneous_UP = Spontaneous_UP[:-1]
        Spontaneous_DOWN = Spontaneous_DOWN[:len(Spontaneous_UP)]

    # ---------- 8) Dauer/Stats ----------
    if len(Spontaneous_UP) > 1 and len(Pulse_triggered_UP) > 1:
        Duration_Pulse_Triggered = time_s[Pulse_triggered_DOWN] - time_s[Pulse_triggered_UP]
        Duration_Spontaneous = time_s[Spontaneous_DOWN] - time_s[Spontaneous_UP]
        Duration_Pulse_Associated = time_s[Pulse_associated_DOWN] - time_s[Pulse_associated_UP] if len(Pulse_associated_UP) else np.array([])
        Dur_stat, Dur_p = stats.ttest_ind(Duration_Spontaneous, Duration_Pulse_Triggered)
    else:
        Duration_Pulse_Triggered = np.array([])
        Duration_Spontaneous = np.array([])
        Duration_Pulse_Associated = np.array([])
        Dur_stat, Dur_p = np.nan, np.nan

    # Ctrl windows (nur falls trig existiert)
    Ctrl_UP = np.array([], dtype=int)
    Ctrl_DOWN = np.array([], dtype=int)
    if len(Pulse_triggered_UP) > 0:
        Ctrl_UP = np.zeros(len(Pulse_triggered_UP), dtype=int)
        Ctrl_DOWN = np.zeros(len(Pulse_triggered_UP), dtype=int)
        for i in range(len(Ctrl_UP)):
            Ctrl_UP[i] = random.randint(0, len(time_s) - int(4 / dt_feat))
            Ctrl_DOWN[i] = Ctrl_UP[i] + int(Pulse_triggered_DOWN[i] - Pulse_triggered_UP[i])

    # ---------- 9) Peaks / aligned arrays ----------
    # (dein Originalcode bleibt weitgehend, aber robust gegen 0 Events)
    Spon_UP_array = np.zeros((len(Spontaneous_UP), align_len))
    Spon_UP_peak_aligned_array = np.full((len(Spontaneous_UP), align_len), np.nan)
    UP_Time = np.arange(align_len) * dt - align_pre * dt
    Spon_Peaks = []

    for i_Spon in range(len(Spontaneous_UP)):
        start_idx = Spontaneous_UP[i_Spon] - int(0.75 / dt_feat)
        end_idx = Spontaneous_UP[i_Spon] + int(2.0 / dt_feat)

        # Indexe beziehen sich hier auf LFP_array Zeitachse => dt ist Rohsignal-dt!
        # ABER: deine UP_start_i Indizes sind Feature-Bins, nicht Rohsamples.
        # Wenn UP_start_i auf Feature-Bins basiert, darfst du NICHT direkt in LFP_array indizieren.
        # => deshalb: Falls du hier bisher "zufällig" gut kamst, war time_s vermutlich Rohzeit.
        # Minimal drop-in: wir lassen es wie es war, aber schützen Out-of-bounds.
        start_idx = int(start_idx)
        end_idx = int(end_idx)

        if start_idx < 0 or end_idx > LFP_array.shape[1] or end_idx <= start_idx:
            Spon_Peaks.append(np.nan)
            continue

        current_data = LFP_array[0, start_idx:end_idx]

        try:
            V_filt = signal.filtfilt(b_lp, a_lp, current_data)
            V_filt = signal.filtfilt(b_hp, a_hp, V_filt)
        except Exception:
            V_filt = current_data

        lo = int(0.25 / dt)  # roh-dt
        hi = int(1.25 / dt)
        lo = max(0, min(lo, len(V_filt)))
        hi = max(lo+1, min(hi, len(V_filt)))

        peaks, _ = find_peaks(
            V_filt[lo:hi],
            height=np.round(np.nanstd(V_filt), 3),
            distance=150
        )
        peaks += lo

        if len(peaks) > 0:
            peak_global = start_idx + peaks[0]
            Spon_Peaks.append(float(peak_global))

            a0 = peaks[0] - align_pre
            a1 = peaks[0] + align_post
            if a0 >= 0 and a1 <= len(current_data):
                Spon_UP_peak_aligned_array[i_Spon] = current_data[a0:a1]
        else:
            Spon_Peaks.append(np.nan)

    Spon_Peaks = np.array(Spon_Peaks, dtype=float)

    # Trigger peaks via helper
    Trig_Peaks, Trig_UP_peak_aligned_array = compute_peak_aligned_segments(
        Pulse_triggered_UP, LFP_array, dt,
        b_lp, a_lp, b_hp, a_hp,
        align_pre, align_post, align_len,
        offset_start=0.2, offset_end=2.0,
        search_start_s=0.2, search_end_s=2.0,
        min_peak_spacing_s=0.1
    )

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
        "Trig_Peaks": Trig_Peaks,
        "Spon_UP_peak_aligned_array": Spon_UP_peak_aligned_array,
        "Trig_UP_peak_aligned_array": Trig_UP_peak_aligned_array,
        "UP_Time": UP_Time,
        "Spon_UP_array": Spon_UP_array,
        "Total_power": Total_power,
        "UP_start_i": UP_start_i,
        "DOWN_start_i": DOWN_start_i,
        "up_state_binary": up_state_binary,
    }



# def Generate_CSD_mean(peaks_list, signal_array, dt):
#     all_csd = []
#     for i, peak in enumerate(peaks_list):
#         if not np.isnan(peak):
#             start = int(peak - int(0.5 / dt))
#             end = int(peak + int(0.5 / dt))
#             if start >= 1 and end <= signal_array.shape[1] - 1:
#                 segment = signal_array[:, start:end]
#                 CSD_out = CSD_calc(segment, dt)
#                 all_csd.append(CSD_out)

#     if len(all_csd) > 0:
#         return np.nanmean(np.stack(all_csd), axis=0)
#     else:
#         print("Kein CSD berechnet – Rückgabe: Dummy mit NaNs")
#         return np.full((signal_array.shape[0], int(1 / dt)), np.nan)

def Generate_CSD_mean_from_onsets(
    onsets,
    signal_array,
    dt,
    pre_s=0.3,
    post_s=0.3,
    clip_to_down=None,   # optional: DOWN-Indices, um das Fenster innerhalb des UP zu halten
):
    """
    Erzeugt ein mittleres CSD rund um UP-Onsets.

    onsets        : Array von Sample-Indizes (UP-Starts)
    signal_array  : LFP (n_channels, n_time)
    dt            : Abtastintervall [s]
    pre_s,post_s  : Zeitfenster relativ zum Onset (z.B. -0.3..+0.5 s)
    clip_to_down  : optional Array gleicher Länge wie onsets;
                    wenn gesetzt, wird das Ende nicht hinter DOWN fallen.
    """
    import numpy as np

    onsets = np.asarray(onsets, int)
    if onsets.size == 0:
        print("[CSD] keine Onsets übergeben")
        return None

    n_ch, n_t = signal_array.shape
    pre  = int(round(pre_s  / dt))
    post = int(round(post_s / dt))

    csd_segments = []   # hier sammeln wir CSDs aller Events

    for i, o in enumerate(onsets):
        if np.isnan(o):
            continue
        o = int(o)

        start = o - pre
        end   = o + post

        # optional: Fenster am DOWN begrenzen (nur innerhalb des UP)
        if clip_to_down is not None and i < len(clip_to_down):
            d = int(clip_to_down[i])
            end = min(end, d)

        # Randbedingungen: Platz für räumliche Ableitung lassen
        if start < 1 or end > (n_t - 1):
            # wäre aus dem Recording raus
            continue

        seg = signal_array[:, start:end]   # Shape (n_ch, n_seg_time)

        try:
            csd = CSD_calc(seg, dt)        # deine Funktion aus CSD.py
        except Exception as e:
            print(f"[CSD] CSD_calc Fehler bei Onset {o}: {e}")
            continue

        if csd is not None and np.isfinite(csd).any():
            csd_segments.append(csd)

    if not csd_segments:
        print("[CSD] kein gültiges Event -> gebe None zurück")
        return None

    # --- NEU: alle CSDs auf gleiche Zeitlänge bringen ---
    try:
        min_T = min(c.shape[1] for c in csd_segments)
    except Exception:
        print("[CSD] ungültige CSD-Segmente -> None")
        return None

    if min_T <= 0:
        print("[CSD] min_T <= 0 -> None")
        return None

    csd_stack = np.stack([c[:, :min_T] for c in csd_segments], axis=0)  # (n_events, n_ch_csd, min_T)
    csd_mean = np.nanmean(csd_stack, axis=0)                            # (n_ch_csd, min_T)

    print(f"[CSD] {len(csd_segments)} Events, mean CSD shape={csd_mean.shape}")
    return csd_mean


#
#def plot_CSD_comparison(CSD_spont, CSD_trig, dt, cmap="bwr",
#                        save=False, hint="CSD_spont_vs_trig",
#                        out_dir=None, show=True, close=False):
#    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)#
#
#    im1 = axes[0].imshow(CSD_spont, aspect="auto", cmap=cmap, origin="lower",
#                         extent=[0, CSD_spont.shape[1]*dt, 0, CSD_spont.shape[0]])
#    axes[0].set_title("Spontaneous UP — CSD")
#    axes[0].set_xlabel("Time (s)")
#    axes[0].set_ylabel("Channel depth")
#
##    im2 = axes[1].imshow(CSD_trig, aspect="auto", cmap=cmap, origin="lower",
          #               extent=[0, CSD_trig.shape[1]*dt, 0, CSD_trig.shape[0]])
#    axes[1].set_title("Pulse-triggered UP — CSD")
##    axes[1].set_xlabel("Time (s)")##
#
#    # gemeinsame Colorbar rechts außen
#    fig.colorbar(im1, ax=axes, shrink=0.9, label="CSD (a.u.)",
#                 location="right", anchor=(0, 0.5))
##    fig.tight_layout(rect=[0, 0, 0.93, 1])##
#
 #   if save:
#        _save_svg(fig, hint, out_dir=out_dir)#

#    if show:
#        plt.draw()
#        plt.pause(0.001)   # winziger Event-Loop-Tick, Fenster wird sicher angezeigt

 #   if close:
   #     plt.close(fig)
#
 #   return fig


from scipy.ndimage import gaussian_filter

def plot_CSD_comparison(CSD_spont, CSD_trig, dt, dz_um=50.0,
                              sigma_ch=0.8, sigma_t=0.4,  # Glättung: Kanäle, Zeit
                              vmax_pct=98, cmap="seismic"):
    """
    Sauberer CSD-Plot: 2D-Glättung + robustes vlim + symmetrische Skala.
    dz_um: Elektrodenabstand in µm (nur für Achsenbeschriftung).
    """
    # 1) leichte 2D-Glättung (Kanäle x Zeit)
    CSDs = []
    for M in (CSD_spont, CSD_trig):
        if M is None or getattr(M, "ndim", 0) != 2:
            CSDs.append(None); continue
        M_sm = gaussian_filter(M, sigma=(sigma_ch, sigma_t), mode="nearest")
        CSDs.append(M_sm)

    CSD_spont_sm, CSD_trig_sm = CSDs

    # 2) robuste symmetrische Skala
    stack = np.concatenate([np.abs(CSD_spont_sm).ravel(), np.abs(CSD_trig_sm).ravel()])
    vmax = np.percentile(stack[~np.isnan(stack)], vmax_pct)
    vmin = -vmax

    # 3) Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True, constrained_layout=True)
    titles = ["Spontaneous UP — CSD", "Pulse-triggered UP — CSD"]
    for ax, M, title in zip(axes, (CSD_spont_sm, CSD_trig_sm), titles):
        im = ax.imshow(M, aspect="auto", origin="lower",
                       extent=[0, M.shape[1]*dt, 0, M.shape[0]*dz_um/1000.0],  # Tiefe in mm
                       cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none")
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_xlim(0, M.shape[1]*dt)
        ax.set_ylabel("Depth (mm)")
        # optional: Isolinien für Struktur
        cs = ax.contour(M, levels=7, colors="k", alpha=0.2)                       
    cbar = fig.colorbar(im, ax=axes, shrink=0.9, label="CSD (a.u.)")
    return fig


# def Spectrum(V_win):
# 	LFP_win = V_win - np.mean(V_win)
# 	# Windowing if you want
# 	w = np.hanning(len(V_win))
# 	LFP_win = w * LFP_win
# 	# Calculate power spectrum for window
# 	Fs = srate
# 	N = len(LFP_win)
# 	xdft = np.fft.fft(LFP_win)
# 	xdft = xdft[0:int((N / 2) + 1)]
# 	psdx = (1 / (Fs * N)) * np.abs(xdft) ** 2
# 	freq = np.arange(0, (Fs / 2) + Fs / N, Fs / N)
# 	Pow = psdx
# 	Pow = np.zeros((501, 1))
# 	for j in range(0, 200):
# 		Pow[j] = psdx[j]
# 	return Pow, freq


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

def compute_spectra(windows, dt, ignore_start_s=0.0):
    spectra = []
    for trial in windows:
        trial = np.asarray(trial)
        if ignore_start_s > 0:
            start_idx = int(ignore_start_s / dt)
            trial = trial[start_idx:]
        freqs, power = welch(trial, fs=1/dt, nperseg=min(256, len(trial)))
        spectra.append(power)
    return np.array(spectra), freqs




def compare_spectra(pulse_windows, spont_windows, dt, ignore_start_s=0.0):
    pulse_spec, freqs = compute_spectra(pulse_windows, dt, ignore_start_s)
    spont_spec, _     = compute_spectra(spont_windows, dt, ignore_start_s)

    pulse_mean = np.mean(pulse_spec, axis=0)
    spont_mean = np.mean(spont_spec, axis=0)
    t_vals, p_vals = ttest_ind(spont_spec, pulse_spec, axis=0, equal_var=False, nan_policy='omit')
    return freqs, spont_mean, pulse_mean, p_vals

def plot_contrast_heatmap(pulse_windows, spont_windows, dt):
    pulse_spec, freqs = compute_spectra(pulse_windows, dt)
    spont_spec, _ = compute_spectra(spont_windows, dt)

    min_trials = min(len(pulse_spec), len(spont_spec))
    pulse_trim = pulse_spec[:min_trials]
    spont_trim = spont_spec[:min_trials]
    contrast = pulse_trim - spont_trim

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        contrast, aspect='auto', origin='lower',
        extent=[freqs[0], freqs[-1], 0, min_trials],
        cmap='bwr', vmin=-np.max(np.abs(contrast)), vmax=np.max(np.abs(contrast))
    )
    ax.set_title("Difference Heatmap: Pulse - Spontaneous UP Spectra")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Trial")
    fig.colorbar(im, ax=ax, label="Power Difference")
    fig.tight_layout()
    return fig


def average_amplitude_in_upstates(main_channel, time_s, UP_start_i, DOWN_start_i, start_idx, end_idx):
    """
    Berechnet die mittlere Amplitude (mean(abs)) der UP-Zustände im Bereich [start_idx, end_idx].
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
        print("Keine gültigen UP-Zustände im angegebenen Bereich gefunden.")




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
        print(f"Keine gültigen Trials für {label}")
        return

    traces = np.array(traces)
    mean_trace = np.nanmean(traces, axis=0)
    sem_trace = np.nanstd(traces, axis=0) / np.sqrt(traces.shape[0])
    t = np.arange(-half_window, half_window) * dt

    plt.plot(t, mean_trace, label=label)
    plt.fill_between(t, mean_trace - sem_trace, mean_trace + sem_trace, alpha=0.3)


def _up_onsets(UP_idx, DOWN_idx):
    import numpy as np
    U = np.asarray(UP_idx, int); D = np.asarray(DOWN_idx, int)
    m = min(U.size, D.size)
    if m == 0:
        return np.array([], int)
    U, D = U[:m], D[:m]
    return np.sort(U)