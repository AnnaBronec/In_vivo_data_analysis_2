#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ========= Imports =========
import os, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless, safe for batch exports
import matplotlib.pyplot as plt
# do NOT call plt.ion() in batch mode

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

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
    compare_spectra
)


# + Plotter-Funktionen
from plotter import (
    plot_all_channels,
    plot_spont_up_mean,
    plot_upstate_duration_comparison,
    plot_upstate_amplitude_blocks_colored,
)

# ========= Params =========
DOWNSAMPLE_FACTOR = 50
HIGH_CUTOFF = 10
LOW_CUTOFF  = 2

# --- Defaults (werden vom Wrapper überschrieben) ---
_DEFAULT_SESSION = "/home/ananym/Code/In_vivo_data_analysis/Data/FOR ANNA IN VIVO/DRD cross/2017-8-9_13-52-30onePulse200msX20per15s"
BASE_PATH   = globals().get("BASE_PATH", _DEFAULT_SESSION)
if "LFP_FILENAME" in globals():
    LFP_FILENAME = globals()["LFP_FILENAME"]
else:
    _base_tag = os.path.basename(os.path.normpath(BASE_PATH))
    LFP_FILENAME = f"{_base_tag}.csv"

SAVE_DIR = BASE_PATH
BASE_TAG = os.path.splitext(os.path.basename(LFP_FILENAME))[0]
os.makedirs(SAVE_DIR, exist_ok=True)

# ========= Load LFP =========
LFP_df, ch_names, lfp_meta = load_LFP_new(BASE_PATH, LFP_FILENAME)
assert "time" in LFP_df.columns, "CSV braucht eine Spalte 'time'."
time_full = LFP_df["time"].to_numpy(float)
print("[INFO] CSV rows:", len(LFP_df),
      "time range:", float(LFP_df["time"].iloc[0]), "->", float(LFP_df["time"].iloc[-1]))

# ========= Pulses =========
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

# ========= Channels -> pri_* =========
chan_cols = [c for c in LFP_df.columns if c not in ("time", "stim", "din_1", "din_2")]
assert len(chan_cols) > 0, "Keine Kanalspalten gefunden."
LFP_df_ds = pd.DataFrame({"timesamples": time_full})
for i, col in enumerate(chan_cols):
    LFP_df_ds[f"pri_{i}"] = pd.to_numeric(LFP_df[col], errors="coerce")
NUM_CHANNELS = len(chan_cols)

# ========= Downsample =========
time_s, dt, LFP_array, pulse_times_1, pulse_times_2 = _ds_fun(
    DOWNSAMPLE_FACTOR, LFP_df_ds, NUM_CHANNELS,
    pulse_times_1=pulse_times_1_full,
    pulse_times_2=pulse_times_2_full,
    snap_pulses=True
)
assert LFP_array.shape[0] == NUM_CHANNELS
assert LFP_array.shape[1] == len(time_s)
print(f"[DS] time {time_s[0]:.3f}->{time_s[-1]:.3f}s, N={len(time_s)}, dt={dt:.6f}s, "
      f"LFP_array={LFP_array.shape}, p1={len(pulse_times_1)}, p2={len(pulse_times_2)}")

# ========= Stimulus-Fenster bestimmen & ALLES croppen =========
def _stim_window(p1, p2):
    ts = []
    if p1 is not None and len(p1): ts.append([np.min(p1), np.max(p1)])
    if p2 is not None and len(p2): ts.append([np.min(p2), np.max(p2)])
    if not ts:
        return None  # kein Crop
    t0 = min(x[0] for x in ts)
    t1 = max(x[1] for x in ts)
    # ein kleines Polster ist oft hilfreich:
    pad = 0.0  # z.B. 0.5 wenn gewünscht
    return t0 - pad, t1 + pad

win = _stim_window(pulse_times_1, pulse_times_2)
if win is not None:
    t0, t1 = win
    # Indizes im Zeitvektor
    i0 = int(np.searchsorted(time_s, t0, side="left"))
    i1 = int(np.searchsorted(time_s, t1, side="right"))
    i0 = max(0, min(i0, len(time_s)))
    i1 = max(i0+1, min(i1, len(time_s)))

    # Zeit/Signale croppen (ALLE synchron!)
    time_s       = time_s[i0:i1]
    LFP_array    = LFP_array[:, i0:i1]
    main_channel = LFP_array[0, :]  # falls du später einen anderen Hauptkanal willst, setz das nach get_main_channel

    # Pulse-Zeiten aufs Fenster begrenzen (Sekunden bleiben Sekunden)
    if pulse_times_1 is not None:
        pulse_times_1 = pulse_times_1[(pulse_times_1 >= time_s[0]) & (pulse_times_1 <= time_s[-1])]
    if pulse_times_2 is not None:
        pulse_times_2 = pulse_times_2[(pulse_times_2 >= time_s[0]) & (pulse_times_2 <= time_s[-1])]

    print(f"[CROP] window {t0:.3f}–{t1:.3f} s -> time_s len={len(time_s)}, LFP_array={LFP_array.shape}, p1={len(pulse_times_1)}, p2={len(pulse_times_2)}")
else:
    print("[CROP] no pulses -> no cropping")




# ========= Preprocess + Spectrogram =========

# === Preprocess + Spectrogram (robust gegen falsche Kanal-Indizes) ===
b_lp, a_lp, b_hp, a_hp = filtering(HIGH_CUTOFF, LOW_CUTOFF, dt)

# === Hauptkanal sicher nach dem Cropping bestimmen ===
try:
    mc = get_main_channel(DOWNSAMPLE_FACTOR, LFP_df_ds, NUM_CHANNELS)
except Exception as e:
    print(f"[WARN] get_main_channel failed ({e}) -> fallback to channel 0")
    mc = 0

# Wichtig: Hauptkanal IMMER aus dem (gecroppten) LFP_array nehmen
if np.isscalar(mc):
    ch_idx = int(mc)
    if not (0 <= ch_idx < NUM_CHANNELS):
        ch_idx = max(0, min(NUM_CHANNELS - 1, ch_idx))
else:
    # Falls get_main_channel ein Array zurückgegeben hat: ignorieren
    # und den variantenstärksten Kanal im JETZTIGEN (gecroppten) Array wählen
    ch_idx = int(np.nanargmax(np.nanvar(LFP_array, axis=1))) if NUM_CHANNELS > 1 else 0

main_channel = LFP_array[ch_idx, :]
print(f"[INFO] main_channel from cropped LFP_array, ch_idx={ch_idx}, len={len(main_channel)}, time_len={len(time_s)}")




print(f"[INFO] NUM_CHANNELS={NUM_CHANNELS}, main_channel_len={len(main_channel)}")

pre, post, win_len, align_pre, align_post, align_len = pre_post_condition(dt)
Spect_dat = Run_spectrogram(main_channel, time_s)

# ========= State detection =========
try:
    Up = classify_states(
        Spect_dat, time_s, pulse_times_1, pulse_times_2, dt,
        main_channel, LFP_array, b_lp, a_lp, b_hp, a_hp,
        align_pre, align_post, align_len
    )
except IndexError as e:
    print(f"[WARN] classify_states skipped due to IndexError: {e}")
    # Minimal-Dummy, damit die restlichen Plots laufen:
    Up = {
        "Spontaneous_UP":        np.array([], dtype=int),
        "Spontaneous_DOWN":      np.array([], dtype=int),
        "Pulse_triggered_UP":    np.array([], dtype=int),
        "Pulse_triggered_DOWN":  np.array([], dtype=int),
        "Pulse_associated_UP":   np.array([], dtype=int),
        "Pulse_associated_DOWN": np.array([], dtype=int),
        "Spon_Peaks":            np.array([], dtype=float),
        "Total_power":           None,
        "up_state_binary":       None,
        # Optional-Felder, falls später abgefragt:
        "UP_start_i":            np.array([], dtype=int),
        "DOWN_start_i":          np.array([], dtype=int),
    }

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

# ========= Extras für Plots =========
pulse_windows = extract_upstate_windows(Pulse_triggered_UP, LFP_array, dt, window_s=1.0)
spont_windows = extract_upstate_windows(Spontaneous_UP,    LFP_array, dt, window_s=1.0)
freqs = spont_mean = pulse_mean = p_vals = None
try:
    freqs, spont_mean, pulse_mean, p_vals = compare_spectra(
        pulse_windows, spont_windows, dt, ignore_start_s=0.3
    )
except Exception as e:
    print("[WARN] spectra compare skipped:", e)

CSD_spont = CSD_trig = None
if NUM_CHANNELS >= 7:
    try:
        Trig_Peaks = Up.get("Trig_Peaks", np.array([], float))
        CSD_spont  = Generate_CSD_mean(Spon_Peaks, LFP_array, dt)
        CSD_trig   = Generate_CSD_mean(Trig_Peaks,  LFP_array, dt)
    except Exception as e:
        print("[WARN] CSD skipped:", e)
else:
    print(f"[INFO] CSD skipped: only {NUM_CHANNELS} channels (<7).")

# ========= Ax-fähige Mini-Plotter =========
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

    # 4) Pulsezeiten (falls vorhanden) – in aktuelle y-Limits zeichnen
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
    ax.set_ylabel("LFP (a.u.)")
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
    if ax is None: fig, ax = plt.subplots(figsize=(8,3))
    else:         fig = ax.figure
    half = int(0.5/dt); traces=[]
    for pk in Spon_Peaks:
        if np.isnan(pk): continue
        pk=int(pk); s=pk-half; e=pk+half
        if s>=0 and e<=len(main_channel): traces.append(main_channel[s:e])
    if not traces:
        ax.text(0.5,0.5,"no spontaneous peaks", ha="center", va="center", transform=ax.transAxes); return fig
    tr=np.vstack(traces); m=np.nanmean(tr,0); se=np.nanstd(tr,0)/np.sqrt(tr.shape[0]); t=(np.arange(-half,half)*dt)
    ax.plot(t,m,lw=2); ax.fill_between(t,m-se,m+se,alpha=0.3); ax.axvline(0,ls="--",lw=1)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("LFP (a.u.)"); ax.set_title("Spontaneous UP – mean ± SEM")
    return fig

def upstate_duration_compare_ax(PU, PD, SU, SD, dt, ax=None):
    if ax is None: fig, ax = plt.subplots(figsize=(8,3))
    else:         fig = ax.figure
    trig=(PD-PU)*dt if len(PU) else np.array([]); spon=(SD-SU)*dt if len(SU) else np.array([])
    labels,vals=[],[]
    if len(spon): labels.append("Spont"); vals.append(spon)
    if len(trig): labels.append("Trig");  vals.append(trig)
    if not vals:
        ax.text(0.5,0.5,"no UP durations", ha="center", va="center", transform=ax.transAxes); return fig
    ax.boxplot(vals, labels=labels, whis=[5,95], showfliers=False)
    ax.set_ylabel("Duration (s)"); ax.set_title("UP durations")
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
        ax.set_ylabel("Power (a.u.)")
        ax.set_title(title)
    except Exception as e:
        ax.text(0.5, 0.5, f"Power-Plot fehlgeschlagen:\n{e}", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    return fig


def Power_spectrum_compare_ax(freqs, spont_mean, pulse_mean, p_vals=None, alpha=0.05, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,3))
    else:
        fig = ax.figure

    # Robustheit
    if freqs is None or spont_mean is None or pulse_mean is None or len(freqs) == 0:
        ax.text(0.5, 0.5, "no spectra", ha="center", va="center", transform=ax.transAxes)
        return fig

    ax.plot(freqs, spont_mean, label="Spontan", lw=2)
    ax.plot(freqs, pulse_mean, label="Getriggert", lw=2)

    # Signifikanz-Bänder (optional)
    import numpy as np
    if p_vals is not None and np.size(p_vals) == np.size(freqs):
        sig = (p_vals < alpha)
        if np.any(sig):
            idx = np.where(sig)[0]
            start = idx[0]
            for i in range(1, len(idx) + 1):
                if i == len(idx) or idx[i] != idx[i-1] + 1:
                    ax.axvspan(freqs[start], freqs[idx[i-1]], alpha=0.12)
                    if i < len(idx):
                        start = idx[i]

    ax.set_xlabel("Hz")
    ax.set_ylabel("Power (a.u.)")
    ax.set_title("Power (Spontan vs. Getriggert)")
    ax.legend()
    return fig


def CSD_compare_side_by_side_ax(
    CSD_spont, CSD_trig, dt,
    *, dz_um=100.0, align_pre=0.5, align_post=0.5,
    cmap="viridis", sat_pct=95, interp="bilinear",  # <- NEU: Look-Regler
    contours=False, n_contours=10,                 # <- optional: Konturen
    ax=None, title="CSD (Spon vs. Trig)"
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    else:
        fig = ax.figure

    # Robustheit
    if not (isinstance(CSD_spont, np.ndarray) and CSD_spont.ndim == 2 and
            isinstance(CSD_trig,  np.ndarray) and CSD_trig.ndim  == 2):
        ax.text(0.5, 0.5, "no CSD", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    # Gemeinsame Dynamik (robustes Clipping) + Zero Center
    stack = np.concatenate([CSD_spont.ravel(), CSD_trig.ravel()])
    stack = stack[np.isfinite(stack)]
    if stack.size == 0:
        ax.text(0.5, 0.5, "invalid CSD values", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    vmax = float(np.nanpercentile(np.abs(stack), sat_pct))
    if vmax <= 0 or not np.isfinite(vmax):
        vmax = 1.0
    vmin = -vmax
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    # Tiefe in mm, 0 mm oben
    n_ch_sp, _ = CSD_spont.shape
    n_ch_tr, _ = CSD_trig.shape
    depth_mm_max_sp = (n_ch_sp - 1) * (dz_um / 1000.0)
    depth_mm_max_tr = (n_ch_tr - 1) * (dz_um / 1000.0)

    # Gemeinsame Zeitachse
    t_min, t_max = -float(align_pre), float(align_post)

    # Inset-Achsen + Colorbar
    ax_left  = ax.inset_axes([0.00, 0.0, 0.45, 1.0])
    ax_right = ax.inset_axes([0.50, 0.0, 0.45, 1.0])
    cax      = ax.inset_axes([0.955, 0.1, 0.02, 0.8])
    ax.set_axis_off()

    # Spont
    imL = ax_left.imshow(
        CSD_spont, aspect="auto", origin="upper",
        extent=[t_min, t_max, 0.0, depth_mm_max_sp],
        cmap=cmap, norm=norm, interpolation=interp
    )
    ax_left.set_title("Spontaneous", fontsize=10)
    ax_left.set_xlabel("Zeit (s)")
    ax_left.set_ylabel("Tiefe (mm)")
    ax_left.set_xlim(t_min, t_max)

    # Trig
    imR = ax_right.imshow(
        CSD_trig, aspect="auto", origin="upper",
        extent=[t_min, t_max, 0.0, depth_mm_max_tr],
        cmap=cmap, norm=norm, interpolation=interp
    )
    ax_right.set_title("Triggered", fontsize=10)
    ax_right.set_xlabel("Zeit (s)")
    ax_right.set_yticks([])
    ax_right.set_xlim(t_min, t_max)

    # Optional: Konturen (für “Paper”-Haptik)
    if contours:
        try:
            levels = np.linspace(vmin, vmax, n_contours)
            ax_left.contour(CSD_spont, levels=levels, colors="k",
                            linewidths=0.3, origin="upper",
                            extent=[t_min, t_max, 0.0, depth_mm_max_sp])
            ax_right.contour(CSD_trig, levels=levels, colors="k",
                             linewidths=0.3, origin="upper",
                             extent=[t_min, t_max, 0.0, depth_mm_max_tr])
        except Exception:
            pass

    cb = fig.colorbar(imR, cax=cax)
    cb.set_label("CSD (a.u.)", rotation=90)

    # Gesamttitel
    ax_left.set_title(title, fontsize=11, pad=22)
    return fig

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

    # Verzeichnisse ableiten:
    #   summary_path = …/For David/<Parent>/<Experiment>/upstate_summary.csv
    exp_dir       = os.path.dirname(summary_path)
    parent_dir    = os.path.dirname(exp_dir)        # …/For David/<Parent>
    for_david_dir = os.path.dirname(parent_dir)     # …/For David

    # --- (1) Rollup pro Parent-Ordner ---
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

    # --- (2) Rollup auf For-David-Ebene (alle Parents zusammen) ---
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


#def export_vertical_stacked_pages_vector(base_tag, save_dir, plot_specs,
#                                         cols=2, rows_per_page=3,
#                                         tile_w=10.2, tile_h=3.8,
#                                         also_save_each_svg=False):
#    """
#    Zeichnet JEDE Plot-Spezifikation direkt in ein übergebenes Ax (vektorbasiert),
#    baut Seiten mit rows_per_page Zeilen x cols Spalten und schreibt eine Multi-Page-PDF.
#    plot_specs: Liste von callables der Form:  lambda ax: <zeichne in ax>
#    """
#    os.makedirs(save_dir, exist_ok=True)
#    out_pdf = os.path.join(save_dir, f"{base_tag}_ALL_PLOTS_STACKED.pdf")#
#
#    def draw_into_ax(ax, spec):
#        try:
#            spec(ax)  # <- spec erwartet genau EIN Argument: ax
#        except Exception as e:
#            ax.text(0.5, 0.5, f"Plot error:\n{e}", ha="center", va="center", transform=ax.transAxes)
#            ax.set_axis_off()#

    # optional: pro Kachel auch ein Einzel-SVG speichern (vektorbasiert)
#    if also_save_each_svg:
#        for i, spec in enumerate(plot_specs, 1):
#            fig, ax = plt.subplots(figsize=(8, 3))
#            draw_into_ax(ax, spec)
##            out_svg = os.path.join(save_dir, f"{base_tag}_plot_{i:02d}.svg")
#            fig.savefig(out_svg, format="svg", bbox_inches="tight")
#            plt.close(fig)
#            print(f"[SAVED] {out_svg}")##
#
#    per_page = cols * rows_per_page
#    with PdfPages(out_pdf) as pdf:
#        for start in range(0, len(plot_specs), per_page):
#            chunk = plot_specs[start:start+per_page]
#            rows = rows_per_page if len(chunk) > cols else max(1, (len(chunk) + cols - 1) // cols)#
#
#            fig_w, fig_h = 3.2 * cols, 3.2 * rows
#            fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
#            if rows == 1 and cols == 1:
#                axes = np.array([[axes]])
#            elif rows == 1:
#                axes = np.array([axes])
#            elif cols == 1:
#                axes = np.array([[ax] for ax in axes])#

#            k = 0
#            for r in range(rows):
#                for c in range(cols):
#                    ax = axes[r, c]
#                    if k < len(chunk):
#                        draw_into_ax(ax, chunk[k])
#                    else:
#                        ax.axis("off")
#                    k += 1

 #           fig.suptitle(f"{base_tag} — Seite {start//per_page + 1}", y=0.995)
 #           fig.tight_layout(rect=[0, 0, 1, 0.98])
 #           pdf.savefig(fig, bbox_inches="tight")
 #           plt.close(fig)

  #  print(f"[PDF] geschrieben: {out_pdf}")






# ========= Helfer (rasterize 0-Arg-Plots & PDF-Grid) =========
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
    return images

# ====== Custom 1-Page Layout (1 groß, 2 klein, 1 groß) ======
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
                main_channel, time_s,
                Spontaneous_UP, Spontaneous_DOWN,
                Pulse_triggered_UP, Pulse_triggered_DOWN,
                Pulse_associated_UP, Pulse_associated_DOWN,
                pulse_times_1=pulse_times_1,
                pulse_times_2=pulse_times_2,
                ax=ax_top,
                title="Main channel with UP classification"
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

        # MIDDLE RIGHT – Power spectrum compare (fallback, falls Daten fehlen)
        ax_midR = fig.add_subplot(gs[1, 1])
        try:
            if (freqs is not None) and (spont_mean is not None) and (pulse_mean is not None) and len(freqs):
                Power_spectrum_compare_ax(freqs, spont_mean, pulse_mean, p_vals=p_vals, ax=ax_midR)
            else:
                ax_midR.text(0.5, 0.5, "no spectra", ha="center", va="center", transform=ax_midR.transAxes)
                ax_midR.set_axis_off()
        except NameError:
            # falls die Funktion in deiner Datei fehlt
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
                dz_um=dz_um, align_pre=align_pre, align_post=align_post,
                cmap=cmap, sat_pct=sat_pct, interp=interp, contours=contours,
                ax=ax_bottom,
                title="CSD (Spont vs. Trig; gemeinsame Zeitachse, 0 mm oben)"
            )
        except Exception as e:
            ax_bottom.text(0.5, 0.5, f"Plot error (CSD):\n{e}", ha="center", va="center", transform=ax_bottom.transAxes)
            ax_bottom.set_axis_off()

        fig.suptitle(base_tag, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"[PDF] geschrieben: {out_pdf}")


# ==== AUFRUF (statt export_vertical_stacked_pages_vector(...)) ====
# export_onepage_custom(
#     BASE_TAG, SAVE_DIR,
#     main_channel=main_channel, time_s=time_s,
#     Spontaneous_UP=Spontaneous_UP, Spontaneous_DOWN=Spontaneous_DOWN,
#     Pulse_triggered_UP=Pulse_triggered_UP, Pulse_triggered_DOWN=Pulse_triggered_DOWN,
#     Pulse_associated_UP=Pulse_associated_UP, Pulse_associated_DOWN=Pulse_associated_DOWN,
#     pulse_times_1=pulse_times_1, pulse_times_2=pulse_times_2,
#     dt=dt, freqs=freqs, spont_mean=spont_mean, pulse_mean=pulse_mean, p_vals=p_vals,
#     CSD_spont=CSD_spont, CSD_trig=CSD_trig,
#     align_pre=align_pre, align_post=align_post,
#     dz_um=100.0, cmap="turbo", sat_pct=90, interp="bilinear", contours=True
# )


from matplotlib import gridspec

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
    """
    Zeichnet den Hauptkanal (LFP) und markiert Bereiche:
      grün = Spontaneous UP
      blau = Triggered UP
      orange = Associated UP
    Zusätzlich: Pulse als vlines, optionale Peak-Marker.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np

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
    ax.set_ylabel("LFP (a.u.)")

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

def Power_spectrum_compare_ax(freqs, spont_mean, pulse_mean, p_vals=None,
                              alpha=0.05, ax=None, title="Power (Spon vs. Trig)"):
    import numpy as np
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8,3))
    else:
        fig = ax.figure

    # Existenz + Arrays glätten
    if freqs is None or spont_mean is None or pulse_mean is None:
        ax.text(0.5,0.5,"no spectra", ha="center", va="center", transform=ax.transAxes)
        return fig
    freqs      = np.asarray(freqs).ravel()
    spont_mean = np.asarray(spont_mean).ravel()
    pulse_mean = np.asarray(pulse_mean).ravel()

    # Längen abgleichen, NaNs filtern
    m = min(freqs.size, spont_mean.size, pulse_mean.size)
    if m < 3:
        ax.text(0.5,0.5,"invalid/empty spectra", ha="center", va="center", transform=ax.transAxes)
        return fig
    freqs, spont_mean, pulse_mean = freqs[:m], spont_mean[:m], pulse_mean[:m]
    good = np.isfinite(freqs) & np.isfinite(spont_mean) & np.isfinite(pulse_mean)
    if good.sum() < 3:
        ax.text(0.5,0.5,"spectra have too many NaNs", ha="center", va="center", transform=ax.transAxes)
        return fig
    freqs, spont_mean, pulse_mean = freqs[good], spont_mean[good], pulse_mean[good]

    # Plot
    ax.plot(freqs, spont_mean, label="Spontan",  lw=2)
    ax.plot(freqs, pulse_mean, label="Getriggert", lw=2)

    # Optional: Signifikanzschattierung
    if p_vals is not None:
        p_vals = np.asarray(p_vals).ravel()[:len(freqs)]
        sig = np.isfinite(p_vals) & (p_vals < alpha)
        if sig.any():
            idx = np.where(sig)[0]
            starts = [idx[0]]
            for i in range(1, len(idx)):
                if idx[i] != idx[i-1] + 1:
                    starts.append(idx[i])
            ends = []
            for s in starts[1:] + [None]:
                ends.append(idx[np.where(idx < s)[0][-1]] if s is not None else idx[-1])
            for s, e in zip(starts, ends):
                ax.axvspan(freqs[s], freqs[e], alpha=0.12, lw=0)

    ax.set_xlabel("Frequenz (Hz)")
    ax.set_ylabel("Power (a.u.)")
    ax.set_title(title)
    ax.legend()
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
    ax.set_xlabel("Hz"); ax.set_ylabel("Power (a.u.)")
    ax.set_title("Power (Spontan vs. Getriggert)"); ax.legend()
    return fig




# ========= plot_specs (ax-basierte Lambdas; NICHT überschreiben) =========
# plot_specs = [
#     # 1) Hauptkanal mit Klassifikationen (der gewünschte Plot)
#     # 1) Hauptkanal mit Klassifikationen (der gewünschte Plot mit Pulsen)
#     lambda ax: plot_up_classification_ax(
#         main_channel, time_s,
#         Spontaneous_UP, Spontaneous_DOWN,
#         Pulse_triggered_UP, Pulse_triggered_DOWN,
#         Pulse_associated_UP, Pulse_associated_DOWN,
#         pulse_times_1=pulse_times_1,           # <— HINZU
#         pulse_times_2=pulse_times_2,           # <— HINZU
#         #Spon_Peaks=Spon_Peaks,                 # optional
#         #Trig_Peaks=Up.get("Trig_Peaks", np.array([], int)),  # optional
#         ax=ax
#     ),

#      lambda ax: _blank_ax(ax),
#  # Zeile 2 — zwei zusammen
#     lambda ax: upstate_duration_compare_ax(
#         Pulse_triggered_UP, Pulse_triggered_DOWN,
#         Spontaneous_UP, Spontaneous_DOWN, dt, ax=ax
#     ),
#     # Power spectrum nur anhängen, wenn vorhanden
#     (lambda ax: Power_spectrum_compare_ax(freqs, spont_mean, pulse_mean, p_vals=p_vals, ax=ax))
#       if (freqs is not None and spont_mean is not None and pulse_mean is not None and len(freqs))
#       else (lambda ax: _blank_ax(ax, "no spectra")),

#     # Zeile 3 — groß (CSD)
#     (lambda ax: CSD_compare_side_by_side_ax(
#         CSD_spont, CSD_trig, dt,
#         dz_um=100.0, align_pre=align_pre, align_post=align_post,
#         cmap="turbo", sat_pct=90, interp="bilinear", contours=True,
#         ax=ax, title="CSD (Spont vs. Trig; gemeinsame Zeitachse, 0 mm oben)"
#     )) if (
#         CSD_spont is not None and getattr(CSD_spont, "ndim", 0) == 2 and
#         CSD_trig  is not None and getattr(CSD_trig,  "ndim",  0) == 2
#     ) else (lambda ax: _blank_ax(ax, "no CSD")),
#     lambda ax: _blank_ax(ax),
# ]

# # 6) Power-Spektrum (nur falls vorhanden)
# if (freqs is not None) and (spont_mean is not None) and (pulse_mean is not None) and len(freqs):
#     plot_specs.append(lambda ax: Power_spectrum_compare_ax(
#         freqs, spont_mean, pulse_mean, p_vals=p_vals, ax=ax
#     ))

# # 7) CSD: zwei Panels, gleiche Zeitachse, 0 mm oben
# if (CSD_spont is not None and getattr(CSD_spont, "ndim", 0) == 2 and
#     CSD_trig  is not None and getattr(CSD_trig,  "ndim",  0) == 2):
#     plot_specs.append(lambda ax: CSD_compare_side_by_side_ax(
#         CSD_spont, CSD_trig, dt,
#         dz_um=100.0,                # ggf. an dein Kanalabstand anpassen
#         align_pre=align_pre,        # kommt aus pre_post_condition(dt)
#         align_post=align_post,
#         cmap="turbo",
#         sat_pct=90,
#         interp="bilinear",
#         contours=True,  
#         ax=ax,
#         title="CSD (Spont vs. Trig)"
#     ))






# Export (
#export_vertical_stacked_pages_vector(
#    BASE_TAG, SAVE_DIR, plot_specs,
#    cols=2, rows_per_page=2,
#    tile_w=5.6, tile_h=4.0,
#    also_save_each_svg=True
#)


# ========= Layout definieren (Zeilen) =========
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
        Pulse_triggered_UP, Pulse_triggered_DOWN,
        Spontaneous_UP, Spontaneous_DOWN, dt, ax=ax
    ),
     lambda ax: Power_spectrum_compare_ax(
        freqs, spont_mean, pulse_mean, p_vals=p_vals, ax=ax
    )],

    # REIHE 3: CSD (volle Breite)
    [lambda ax: CSD_compare_side_by_side_ax(
        CSD_spont, CSD_trig, dt,
        dz_um=100.0,                     # bei Bedarf anpassen
        align_pre=align_pre, align_post=align_post,
        cmap="turbo", sat_pct=90, interp="bilinear",
        contours=True,
        ax=ax,
        title="CSD (Spon vs. Trig; gemeinsame Zeitachse, 0 mm oben)"
    )],
]


def _write_summary_csv():
    import csv, io
    # --- 1) Zielpfad
    summary_path = os.path.join(BASE_PATH, "upstate_summary.csv")

    # --- 2) Delimiter erkennen (falls Datei existiert), sonst Standard = ';'
    delimiter = ';'
    if os.path.isfile(summary_path):
        with open(summary_path, "r", newline="", encoding="utf-8") as f:
            head = f.read(4096)
        try:
            dialect = csv.Sniffer().sniff(head, delimiters=[",",";","\t","|"])
            delimiter = dialect.delimiter
        except Exception:
            pass  # bleibt bei ';'

    # --- 3) Feldnamen (Schema)
    FIELDNAMES = [
        "Parent","Experiment","Dauer [s]","Samplingrate [Hz]","Kanäle",
        "Pulse count 1","Pulse count 2",
        "Upstates total","triggered","spon","associated",
        "Downstates total","UP/DOWN ratio",
        "Mean UP Dauer [s]","Mean UP Dauer Triggered [s]","Mean UP Dauer Spontaneous [s]",
        "Datum Analyse",
    ]

    # --- 4) Helfer: numpy/NaN -> plain
    def _py(v):
        try:
            import numpy as _np
            if isinstance(v, (_np.floating, _np.float32, _np.float64)):
                f = float(v);  return "" if (f != f) else f  # NaN -> ""
            if isinstance(v, (_np.integer,)): return int(v)
        except Exception:
            pass
        if v is None: return ""
        if isinstance(v, float): return "" if (v != v) else round(v, 6)
        return v

    # --- 5) aktuelle Zeile bauen
    experiment_name = os.path.basename(BASE_PATH)
    parent_folder   = os.path.basename(os.path.dirname(BASE_PATH))

    def _pairs(Up_states, time_vec):
        UP_i   = np.array(Up_states.get("UP_start_i",   []), dtype=int)
        DOWN_i = np.array(Up_states.get("DOWN_start_i", []), dtype=int)
        if DOWN_i.size == 0:
            sUP = np.array(Up_states.get("Spontaneous_UP",       []), dtype=int)
            sDN = np.array(Up_states.get("Spontaneous_DOWN",     []), dtype=int)
            tUP = np.array(Up_states.get("Pulse_triggered_UP",   []), dtype=int)
            tDN = np.array(Up_states.get("Pulse_triggered_DOWN", []), dtype=int)
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

    # ---- Debug: zeig die Zeile im Log
    print("[SUMMARY] target:", summary_path)
    print("[SUMMARY] delimiter:", repr(delimiter))
    print("[SUMMARY] row:", {k: _py(v) for k,v in row.items()})

    # --- 6) vorhandene Zeilen laden & aufs Schema mappen
    rows = []
    if os.path.isfile(summary_path):
        with open(summary_path, "r", newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f, delimiter=delimiter)
            for r in rdr:
                rows.append({k: r.get(k, "") for k in FIELDNAMES})

    # --- 7) updaten oder anhängen (Match: Parent+Experiment)
    updated = False
    for r in rows:
        if r.get("Experiment","") == experiment_name and r.get("Parent","") == parent_folder:
            for k in FIELDNAMES:
                r[k] = _py(row.get(k, r.get(k, "")))
            updated = True
            break
    if not updated:
        rows.append({k: _py(row.get(k, "")) for k in FIELDNAMES})

    # --- 8) zurückschreiben mit erkanntem Delimiter
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


def export_with_layout(base_tag, save_dir, layout_rows, rows_per_page=3, also_save_each_svg=True):
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
            elif len(row) == 2:
                for spec in row:
                    fig, ax = plt.subplots(figsize=(5, 3.4))
                    draw_into_ax(ax, spec)
                    fig.savefig(os.path.join(save_dir, f"{base_tag}_plot_{k:02d}.svg"),
                                format="svg", bbox_inches="tight")
                    plt.close(fig); k += 1

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

    print(f"[PDF] geschrieben: {out_pdf}")




# ========= Export aufrufen =========
export_with_layout(
    BASE_TAG, SAVE_DIR, layout_rows,
    rows_per_page=3,          # 3 Zeilen -> alles auf eine Seite
    also_save_each_svg=True
)

