import csv
import os
import numpy as np
import matplotlib
try:
    if os.environ.get("DISPLAY"):
        matplotlib.use("TkAgg")
    else:
        matplotlib.use("Agg")
except Exception:
    matplotlib.use("Agg")
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
import random
from scipy.ndimage import gaussian_filter
from scipy import signal
import numpy as np
from loader import load_LFP, load_lightpulses
from preprocessing import downsampling, filtering, get_main_channel, pre_post_condition
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_ind
from scipy.signal import welch
from scipy.ndimage import gaussian_filter


def _save_svg(fig, hint, out_dir=None, dpi=200):
    if out_dir is None:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{hint}.svg")
    fig.canvas.draw_idle()
    fig.savefig(path, format="svg", dpi=dpi, bbox_inches="tight")
    print(f"[SAVED] {path}")


def compute_peak_aligned_segments(
    up_indices, time_s, LFP_array, dt,
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
        # Feature-index -> Zeit (s)
        t_up = time_s[int(up)]

        # Zeit -> Rohsamples
        start_idx = int((t_up + offset_start) / dt)
        end_idx   = int((t_up + offset_end) / dt)

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


    # helpers 
    def _empty_states(Total_power, time_s, dt, align_pre, align_len):
        # Total_power ist Feature-Länge; time_s kann Raw-Länge haben -> nicht für Shapes verwenden
        Total_power = np.asarray(Total_power)
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
            "up_state_binary": np.zeros_like(Total_power, dtype=bool),  # Feature-Länge!
            "t_feat": np.array([], dtype=float),
            "dt_feat": np.nan,
            "Spontaneous_UP_raw": np.array([], dtype=int),
            "Spontaneous_DOWN_raw": np.array([], dtype=int),
            "Pulse_triggered_UP_raw": np.array([], dtype=int),
            "Pulse_triggered_DOWN_raw": np.array([], dtype=int),
            "Pulse_associated_UP_raw": np.array([], dtype=int),
            "Pulse_associated_DOWN_raw": np.array([], dtype=int),
        }

    # --- ALIGN: robust (align_pre/align_post can be seconds OR samples) ---

    def _looks_like_samples(a, dt):
        # If a is "large" but corresponds to a reasonable number of seconds when multiplied by dt,
        # it's almost certainly already in samples.
        try:
            a = float(a)
        except Exception:
            return False
        return (a >= 5) and (a * dt < 10.0)  # e.g. 320 samples * 0.00156 = 0.5s

    if _looks_like_samples(align_pre, dt) and _looks_like_samples(align_post, dt):
        align_pre_samp  = int(round(align_pre))
        align_post_samp = int(round(align_post))
        align_pre_sec   = align_pre_samp * dt
        align_post_sec  = align_post_samp * dt
    else:
        align_pre_sec   = float(align_pre)
        align_post_sec  = float(align_post)
        align_pre_samp  = int(round(align_pre_sec  / dt))
        align_post_samp = int(round(align_post_sec / dt))

    align_len_expected = align_pre_samp + align_post_samp

    if align_len != align_len_expected:
        print(f"[ALIGN-FIX] align_len {align_len} -> {align_len_expected}")
        align_len = align_len_expected

    print("[DBG ALIGN]",
        "align_len", align_len,
        "pre_samp", align_pre_samp, "post_samp", align_post_samp,
        "pre_sec", align_pre_sec, "post_sec", align_post_sec)


    # 1) Bandpower aus Spektrogramm
    freqs = Spect_dat[2]
    S = np.asarray(Spect_dat[0], float)  # dB
    t_feat = np.asarray(Spect_dat[1], float) 

    dt_feat = float(np.median(np.diff(t_feat))) if (t_feat.size >= 2 and np.all(np.isfinite(t_feat))) else float(dt)
    t0_feat = float(t_feat[0]) if t_feat.size else 0.0

    #     # align_pre/align_post are in seconds in this pipeline -> convert to samples for indexing
    # align_pre_samp  = int(round(align_pre  / dt))
    # align_post_samp = int(round(align_post / dt))

    # # keep align_len as given, but if it's inconsistent, enforce it
    # if align_len != (align_pre_samp + align_post_samp):
    #     align_len = align_pre_samp + align_post_samp

        # align_pre/align_post are seconds -> convert to samples



    print("Spect_dat lens/types:",
        type(Spect_dat), len(Spect_dat),
        type(Spect_dat[0]), type(Spect_dat[1]), type(Spect_dat[2]))

    print("S shape:", np.shape(Spect_dat[0]))
    print("t entry:", Spect_dat[1])
    print("t shape:", np.shape(Spect_dat[1]))
    print("f shape:", np.shape(Spect_dat[2]))

    # falls t ein Array ist:
    if np.ndim(Spect_dat[1]) == 1 and len(Spect_dat[1]) > 3:
        print("t[0:5]=", Spect_dat[1][:5], "t[-1]=", Spect_dat[1][-1],
            "dt~", np.median(np.diff(Spect_dat[1])))



    # 
    f_lo, f_hi = 0.5, 4.0
    band_mask = (freqs >= f_lo) & (freqs <= f_hi)
    if band_mask.sum() > 0:
        print("[BAND-EFF]", freqs[band_mask].min(), freqs[band_mask].max(), "bins", band_mask.sum())
    else:
        print("[BAND-EFF] EMPTY!")


    # dB -> linear, stabil clippen
    S_clip = np.clip(S[band_mask, :], -120, 30)
    linear = 10 ** (S_clip / 10.0)

    # robust gegen Hot-Bins (median statt mean)
    Total_power = np.median(linear, axis=0)
    # optionale Dynamik-Kompression (macht Thresholding stabiler)
    Total_power = np.log10(Total_power + 1e-30)



    print("Total_power stats:", np.min(Total_power), np.max(Total_power), np.isnan(Total_power).sum())

    # 2) Smooth 
    # 2) Smooth in *Sekunden* (weil t_feat ~ dt ist)
    # Ziel: 100 ms Glättung (passt oft gut für UP/DOWN)
    smooth_s = 0.05
    sigma_bins = max(1, int(round(smooth_s / dt_feat)))  # dt_feat später gesetzt? -> siehe unten
    # Falls dt_feat hier noch nicht existiert, nimm median diff von t_feat:
    # dt_feat_tmp = float(np.median(np.diff(t_feat))) if (t_feat.size > 1) else float(dt)
    # sigma_bins = max(1, int(round(smooth_s / dt_feat_tmp)))

    Total_power_smooth = gaussian_filter1d(Total_power, sigma=sigma_bins)

    # # robust z-score auf Feature-Achse
    # med = np.median(Total_power_smooth)
    # mad = np.median(np.abs(Total_power_smooth - med)) + 1e-30
    # robust_std = 1.4826 * mad

    # z = (Total_power_smooth - med) / robust_std
    x = Total_power_smooth

    # baseline = "untere" Zustände (ruhig / DOWN-lastig)
    q = np.quantile(x, 0.15)
    base = x[x <= q]
    if base.size < 10:
        base = x  # fallback

    mu = np.median(base)
    mad = np.median(np.abs(base - mu)) + 1e-30
    robust_std = 1.4826 * mad

    z = (x - mu) / robust_std


    # k_hi = 3.0   # Start-Schwelle (strenger)
    # k_lo = 2.5   # Halte-Schwelle (lockerer)

    # hi = z > k_hi
    # lo = z > k_lo

    # # hysteresis reconstruction: UP startet bei hi, läuft weiter solange lo true ist
    # up_state_binary = np.zeros_like(hi, dtype=bool)
    # active = False
    # for i in range(len(hi)):
    #     if not active and hi[i]:
    #         active = True
    #     if active and not lo[i]:
    #         active = False
    #     up_state_binary[i] = active



    # max_up_fraction = 0.60  # wenn >60% UP → zu permissiv
    # if up_state_binary.mean() > max_up_fraction:
    #     # Schwelle schrittweise anziehen, bis UP-Anteil plausibel wird
    #     for k_hi_try, k_lo_try in [(3.5, 3.0), (4.0, 3.5), (4.5, 4.0), (5.0, 4.5)]:
    #         hi = z > k_hi_try
    #         lo = z > k_lo_try

    #         up_tmp = np.zeros_like(hi, dtype=bool)
    #         active = False
    #         for i in range(len(hi)):
    #             if not active and hi[i]:
    #                 active = True
    #             if active and not lo[i]:
    #                 active = False
    #             up_tmp[i] = active

    #         if up_tmp.mean() <= max_up_fraction:
    #             up_state_binary = up_tmp
    #             k_hi, k_lo = k_hi_try, k_lo_try
    #             break
    def _hysteresis(z, k_hi, k_lo):
        hi = z > k_hi
        lo = z > k_lo
        out = np.zeros_like(hi, dtype=bool)
        active = False
        for i in range(len(hi)):
            if (not active) and hi[i]:
                active = True
            if active and (not lo[i]):
                active = False
            out[i] = active
        return out

    # Zielbereich für UP-Anteil (Feature-Zeitachse)
    target_min = 0.05
    target_max = 0.50

    # Startwerte
    k_hi = 3.5
    k_lo = 3.0

    # Wir passen k an (hoch -> weniger UP, runter -> mehr UP)
    for _ in range(20):
        up_tmp = _hysteresis(z, k_hi, k_lo)
        frac = float(up_tmp.mean())

        # passt -> fertig
        if target_min <= frac <= target_max:
            up_state_binary = up_tmp
            break

        # zu viel UP -> strenger
        if frac > target_max:
            k_hi += 0.5
            k_lo += 0.5

        # zu wenig UP -> lockerer
        else:  # frac < target_min
            k_hi -= 0.5
            k_lo -= 0.5
            k_hi = max(1.5, k_hi)
            k_lo = max(1.0, k_lo)
    else:
        up_state_binary = up_tmp

    print(f"[AUTO-THR] k_hi={k_hi:.2f} k_lo={k_lo:.2f} fracUP={up_state_binary.mean():.3f}")
    print("[Z] min/med/max:", float(np.nanmin(z)), float(np.nanmedian(z)), float(np.nanmax(z)))
    print("[Z] p95/p99:", float(np.nanpercentile(z, 95)), float(np.nanpercentile(z, 99)))



    # 4) Gap-closing
    # dt für up_state_binary muss zur Feature-Zeitachse passen:
    # time_s = np.asarray(time_s)
    # if time_s.size >= 2 and np.all(np.isfinite(time_s)):
    #     dt_feat = float(np.median(np.diff(t_feat)))
    # else:
    #     dt_feat = float(dt)  # fallback

    min_gap_s = 0.05
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

    # 5) Transitions + Dauerfilter 
    up_transitions = np.where(np.diff(up_state_binary.astype(int)) == 1)[0]
    down_transitions = np.where(np.diff(up_state_binary.astype(int)) == -1)[0]

    if down_transitions.size and up_transitions.size and down_transitions[0] < up_transitions[0]:
        down_transitions = down_transitions[1:]
    if up_transitions.size > down_transitions.size:
        up_transitions = up_transitions[:-1]

    min_up_len_s = 0.7
    if not up_transitions.size or not down_transitions.size:
        UP_start_i = np.array([], dtype=int)
        DOWN_start_i = np.array([], dtype=int)
    else:
        filtered_UP = []
        filtered_DOWN = []
        for u, d in zip(up_transitions, down_transitions):
            # duration = time_s[d] - time_s[u]
            duration = t_feat[d] - t_feat[u]

            if duration >= min_up_len_s:
                filtered_UP.append(u)
                filtered_DOWN.append(d)

        UP_start_i = np.array(filtered_UP, dtype=int)
        DOWN_start_i = np.array(filtered_DOWN, dtype=int)

    # 6) NO-UP Gate
    min_up_fraction = 0.03
    min_up_onsets = 2

    print("[BAND] f_lo,f_hi =", f_lo, f_hi)
    #print("[THR] thr =", threshold, "smooth min/max =", Total_power_smooth.min(), Total_power_smooth.max())
    print("[UP%] frac UP =", up_state_binary.mean(), "| dt_feat =", dt_feat, "min_gap_bins =", min_gap)
    print("[UP] onsets =", len(UP_start_i))
    print("[time_s] = ", time_s)
    print("t_feat = ", t_feat)

    if (up_state_binary.mean() < min_up_fraction) or (len(UP_start_i) < min_up_onsets):
        print("[STATES] no reliable UP states -> return empty dict")
        return _empty_states(Total_power, time_s, dt, align_pre, align_len)

    # 7) Puls-Logik (paarbasiert, pro UP genau ein Label)
    # Gewuenschte Prioritaet:
    #   triggered  : Pulse nahe UP-Onset (±trig_win_s), auch wenn sie frueh im UP liegen
    #   associated : nur "spaete" Pulse im UP (ab t_up + assoc_min_delay_s)
    #   spontaneous: keine Pulse in den obigen Fenstern
    p1 = np.asarray(pulse_times_1 if pulse_times_1 is not None else [], float)
    p2 = np.asarray(pulse_times_2 if pulse_times_2 is not None else [], float)
    Pulse_times_array = np.sort(np.concatenate([p1, p2])) if (p1.size or p2.size) else np.array([], float)
    Pulse_times_array = Pulse_times_array[np.isfinite(Pulse_times_array)]

    trig_win_s = 0.35
    assoc_min_delay_s = 0.20
    n_up = int(min(len(UP_start_i), len(DOWN_start_i)))
    mask_assoc = np.zeros(n_up, dtype=bool)
    mask_trig = np.zeros(n_up, dtype=bool)

    if Pulse_times_array.size and n_up:
        for i in range(n_up):
            u = int(UP_start_i[i])
            d = int(DOWN_start_i[i])
            t_up = float(t_feat[u])
            t_dn = float(t_feat[d])

            has_near = np.any(
                (Pulse_times_array >= (t_up - trig_win_s)) &
                (Pulse_times_array <= (t_up + trig_win_s))
            )
            has_assoc_late = np.any(
                (Pulse_times_array >= (t_up + assoc_min_delay_s)) &
                (Pulse_times_array <= t_dn)
            )

            if has_near:
                mask_trig[i] = True
            elif has_assoc_late:
                mask_assoc[i] = True

    # paar-konsistente Klassen
    Pulse_associated_UP = UP_start_i[:n_up][mask_assoc]
    Pulse_associated_DOWN = DOWN_start_i[:n_up][mask_assoc]

    Pulse_triggered_UP = UP_start_i[:n_up][mask_trig]
    Pulse_triggered_DOWN = DOWN_start_i[:n_up][mask_trig]

    mask_sp = ~(mask_assoc | mask_trig)
    Spontaneous_UP = UP_start_i[:n_up][mask_sp]
    Spontaneous_DOWN = DOWN_start_i[:n_up][mask_sp]

    print(
        f"[CLASSIFY] total={n_up} spont={len(Spontaneous_UP)} "
        f"trig={len(Pulse_triggered_UP)} assoc={len(Pulse_associated_UP)} "
        f"(trig_win={trig_win_s:.2f}s, assoc_delay={assoc_min_delay_s:.2f}s)"
    )

    # 8) Dauer/Stats
    if len(Spontaneous_UP) > 1 and len(Pulse_triggered_UP) > 1:
        # Duration_Spontaneous = time_s[Spontaneous_DOWN] - time_s[Spontaneous_UP]
        Duration_Spontaneous = t_feat[Spontaneous_DOWN] - t_feat[Spontaneous_UP]
        Duration_Pulse_Triggered = t_feat[Pulse_triggered_DOWN] - t_feat[Pulse_triggered_UP]
        Duration_Pulse_Associated = t_feat[Pulse_associated_DOWN] - t_feat[Pulse_associated_UP] if len(Pulse_associated_UP) else np.array([])

        # Duration_Pulse_Associated = time_s[Pulse_associated_DOWN] - time_s[Pulse_associated_UP] if len(Pulse_associated_UP) else np.array([])
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
            max0 = len(t_feat) - int(4.0/dt_feat)
            Ctrl_UP[i] = random.randint(0, max(0, max0))
            #Ctrl_UP[i] = random.randint(0, len(t_feat) - int(4 / dt_feat))
            Ctrl_DOWN[i] = Ctrl_UP[i] + int(Pulse_triggered_DOWN[i] - Pulse_triggered_UP[i])

    #  9) Peaks / aligned arrays
    Spon_UP_array = np.zeros((len(Spontaneous_UP), align_len))
    Spon_UP_peak_aligned_array = np.full((len(Spontaneous_UP), align_len), np.nan)
    # UP_Time = np.arange(align_len) * dt - align_pre * dt
    # UP_Time = np.arange(align_len) * dt - align_pre
    UP_Time = (np.arange(align_len) - align_pre_samp) * dt


    Spon_Peaks = []

    for i_Spon in range(len(Spontaneous_UP)):
        # Feature-Index → Zeit (s)
        # t_feat = time_s[Spontaneous_UP[i_Spon]]

        # # Zeit → Rohsample-Index
        # start_idx = int((t_feat - 0.75) / dt)
        # end_idx   = int((t_feat + 2.0) / dt)

        t_up = t_feat[Spontaneous_UP[i_Spon]]

        # Zeit relativ zum Start -> Sample-Index
        t0 = t_feat[0]
        start_idx = int(((t_up - t0) - 0.75) / dt)
        end_idx   = int(((t_up - t0) + 2.0) / dt)


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
            distance = max(1, int(0.1 / dt)),              # 100 ms
            height   = 2.0 * np.nanstd(V_filt)               # etwas strenger
            # oder statt height:
            # prominence = 2.0 * np.nanstd(V_filt)
        )
        peaks += lo

        if len(peaks) > 0:
            peak_global = start_idx + peaks[0]
            if i_Spon < 5:
                print("[PEAKCHK]",
                    "t_up", float(t_up),
                    "start_idx", int(start_idx),
                    "peak_rel", int(peaks[0]),
                    "peak_global", int(peak_global),
                    "n_time", int(LFP_array.shape[1]),
                    "in_bounds", 0 <= peak_global < LFP_array.shape[1])

            Spon_Peaks.append(float(peak_global))

            # a0 = peaks[0] - align_pre
            # a1 = peaks[0] + align_post
            a0 = int(peaks[0] - align_pre_samp)
            a1 = int(peaks[0] + align_post_samp)

            if a0 >= 0 and a1 <= len(current_data):
                Spon_UP_peak_aligned_array[i_Spon] = current_data[a0:a1]
        else:
            Spon_Peaks.append(np.nan)

    Spon_Peaks = np.array(Spon_Peaks, dtype=float)
    print("[DBG ALIGN]", "align_len", align_len,
      "pre_samp", align_pre_samp, "post_samp", align_post_samp,
      "Spon_arr shape", Spon_UP_peak_aligned_array.shape)

    # Trigger peaks via helper
    Trig_Peaks, Trig_UP_peak_aligned_array = compute_peak_aligned_segments(
        Pulse_triggered_UP, time_s, LFP_array, dt,
        b_lp, a_lp, b_hp, a_hp,
        align_pre_samp, align_post_samp, align_len,
        offset_start=0.2, offset_end=2.0,
        search_start_s=0.2, search_end_s=2.0,
        min_peak_spacing_s=0.1
    )
    print("freqs min/max/df:", freqs.min(), freqs.max(), np.median(np.diff(freqs)))
    print("band bins:", band_mask.sum(), "f_lo/f_hi:", f_lo, f_hi)
    print("effective band:", freqs[band_mask].min(), freqs[band_mask].max())


    UP_start_raw   = (
        np.asarray(np.round((t_feat[UP_start_i] - t0_feat) / dt), int)
        if UP_start_i.size else np.array([], int)
    )

    DOWN_start_raw = (
        np.asarray(np.round((t_feat[DOWN_start_i] - t0_feat) / dt), int)
        if DOWN_start_i.size else np.array([], int)
    )

    Spont_UP_raw = (
        np.asarray(np.round((t_feat[Spontaneous_UP] - t0_feat) / dt), int)
        if Spontaneous_UP.size else np.array([], int)
    )

    Spont_DN_raw = (
        np.asarray(np.round((t_feat[Spontaneous_DOWN] - t0_feat) / dt), int)
        if Spontaneous_DOWN.size else np.array([], int)
    )

    Trig_UP_raw  = (
        np.asarray(np.round((t_feat[Pulse_triggered_UP] - t0_feat) / dt), int)
        if Pulse_triggered_UP.size else np.array([], int)
    )

    Trig_DN_raw  = (
        np.asarray(np.round((t_feat[Pulse_triggered_DOWN] - t0_feat) / dt), int)
        if Pulse_triggered_DOWN.size else np.array([], int)
    )

    Assoc_UP_raw = (
        np.asarray(np.round((t_feat[Pulse_associated_UP] - t0_feat) / dt), int)
        if Pulse_associated_UP.size else np.array([], int)
    )

    Assoc_DN_raw = (
        np.asarray(np.round((t_feat[Pulse_associated_DOWN] - t0_feat) / dt), int)
        if Pulse_associated_DOWN.size else np.array([], int)
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
        "t_feat": t_feat,
        "dt_feat": dt_feat,
        "Spontaneous_UP_raw": Spont_UP_raw,
        "Spontaneous_DOWN_raw": Spont_DN_raw,
        "Pulse_triggered_UP_raw": Trig_UP_raw,
        "Pulse_triggered_DOWN_raw": Trig_DN_raw,
        "Pulse_associated_UP_raw": Assoc_UP_raw,
        "Pulse_associated_DOWN_raw": Assoc_DN_raw,
    }


def Generate_CSD_mean_from_onsets(
    onsets,
    signal_array,
    dt,
    pre_s=0.3,
    post_s=0.3,
    clip_to_down=None,   # optional: DOWN-Indices, um das Fenster innerhalb des UP zu halten
    baseline_pre_s=0.2,  # baseline vor Onset (Sekunden), pro Event/Kanal abziehen
    min_keep_frac=0.8,   # verwerfe stark verkuerzte Events (z.B. durch clip_to_down)
    return_stack=False,
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
    target_T = int(pre + post)
    min_keep_T = int(max(1, round(float(min_keep_frac) * target_T)))

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

        if csd is None or getattr(csd, "ndim", 0) != 2 or (not np.isfinite(csd).any()):
            continue

        tlen = int(csd.shape[1])
        if tlen < min_keep_T:
            continue

        # Baseline-Korrektur (pro Kanal): Mittel aus Pre-Onset-Teil abziehen.
        if baseline_pre_s is not None and baseline_pre_s > 0 and tlen > 0:
            n_pre = int(round(float(baseline_pre_s) / float(dt)))
            n_pre = max(1, min(n_pre, tlen))
            base = np.nanmean(csd[:, :n_pre], axis=1, keepdims=True)
            csd = csd - base

        # Auf Ziellaenge bringen: vorne ist stets korrekt ausgerichtet, tail ggf. NaN.
        csd_pad = np.full((csd.shape[0], target_T), np.nan, dtype=float)
        copy_T = min(target_T, tlen)
        csd_pad[:, :copy_T] = csd[:, :copy_T]
        csd_segments.append(csd_pad)

    if not csd_segments:
        print("[CSD] kein gültiges Event -> gebe None zurück")
        return None

    csd_stack = np.stack(csd_segments, axis=0)  # (n_events, n_ch_csd, target_T)
    csd_mean = np.nanmean(csd_stack, axis=0)                            # (n_ch_csd, min_T)

    print(f"[CSD] {len(csd_segments)} Events, mean CSD shape={csd_mean.shape}")
    if return_stack:
        return csd_mean, csd_stack
    return csd_mean


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



def _fdr_bh(p_vals):
    """Benjamini-Hochberg FDR-Korrektur (returns q-values)."""
    p = np.asarray(p_vals, float).ravel()
    q = np.full_like(p, np.nan, dtype=float)
    m = p.size
    if m == 0:
        return q
    finite = np.isfinite(p)
    if not np.any(finite):
        return q
    pv = p[finite]
    n = pv.size
    order = np.argsort(pv)
    p_sorted = pv[order]
    ranks = np.arange(1, n + 1, dtype=float)
    q_sorted = p_sorted * n / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    qv = np.empty_like(pv)
    qv[order] = q_sorted
    q[finite] = qv
    return q.reshape(np.asarray(p_vals).shape)


def compare_spectra(
    pulse_windows,
    spont_windows,
    dt,
    ignore_start_s=0.0,
    baseline_mode=None,  # None | "pre_onset_db"
):
    def _spectra_from_windows(windows):
        specs = []
        freqs_ref = None
        for trial in windows:
            x = np.asarray(trial, float)
            if x.size < 8 or not np.isfinite(x).any():
                continue

            if baseline_mode == "pre_onset_db":
                mid = int(x.size // 2)
                pre = x[:mid]
                post = x[mid:]
                if ignore_start_s > 0:
                    post = post[int(ignore_start_s / dt):]
                if pre.size < 8 or post.size < 8:
                    continue
                nps = int(min(256, pre.size, post.size))
                f, p_post = welch(post, fs=1 / dt, nperseg=nps)
                _, p_pre = welch(pre, fs=1 / dt, nperseg=nps)
                spec = 10.0 * np.log10((p_post + 1e-20) / (p_pre + 1e-20))
            else:
                if ignore_start_s > 0:
                    x = x[int(ignore_start_s / dt):]
                if x.size < 8:
                    continue
                nps = int(min(256, x.size))
                f, spec = welch(x, fs=1 / dt, nperseg=nps)

            if freqs_ref is None:
                freqs_ref = f
            else:
                m = min(freqs_ref.size, f.size, spec.size)
                freqs_ref = freqs_ref[:m]
                spec = spec[:m]
                specs = [s[:m] for s in specs]
            specs.append(spec)

        if freqs_ref is None or len(specs) == 0:
            raise ValueError("no valid spectra windows")
        return np.asarray(specs, float), freqs_ref

    pulse_spec, freqs = _spectra_from_windows(pulse_windows)
    spont_spec, _ = _spectra_from_windows(spont_windows)

    m = min(pulse_spec.shape[1], spont_spec.shape[1], freqs.size)
    pulse_spec = pulse_spec[:, :m]
    spont_spec = spont_spec[:, :m]
    freqs = freqs[:m]

    pulse_mean = np.mean(pulse_spec, axis=0)
    spont_mean = np.mean(spont_spec, axis=0)
    pulse_median = np.median(pulse_spec, axis=0)
    spont_median = np.median(spont_spec, axis=0)

    # n-gematchte Mittelwerte (Subsampling aus der größeren Gruppe)
    n_tr = pulse_spec.shape[0]
    n_sp = spont_spec.shape[0]
    n_match = int(min(n_tr, n_sp))
    rng = np.random.default_rng(0)
    if n_match >= 1:
        idx_tr = rng.choice(n_tr, size=n_match, replace=False)
        idx_sp = rng.choice(n_sp, size=n_match, replace=False)
        pulse_mean_matched = np.mean(pulse_spec[idx_tr, :], axis=0)
        spont_mean_matched = np.mean(spont_spec[idx_sp, :], axis=0)
    else:
        pulse_mean_matched = np.full(m, np.nan, float)
        spont_mean_matched = np.full(m, np.nan, float)

    _, p_vals = ttest_ind(spont_spec, pulse_spec, axis=0, equal_var=False, nan_policy='omit')
    p_vals_fdr = _fdr_bh(p_vals)
    meta = {
        "n_spont": int(n_sp),
        "n_trig": int(n_tr),
        "n_match": int(n_match),
        "spont_median": spont_median,
        "trig_median": pulse_median,
        "spont_mean_matched": spont_mean_matched,
        "trig_mean_matched": pulse_mean_matched,
    }
    return freqs, spont_mean, pulse_mean, p_vals, p_vals_fdr, meta

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
