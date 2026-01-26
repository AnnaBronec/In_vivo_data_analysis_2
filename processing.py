
import os, math
import numpy as np
_np = np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from pathlib import Path
import re

#Konstanten
DOWNSAMPLE_FACTOR = 50
HIGH_CUTOFF = 10
LOW_CUTOFF  = 2
ANALYSE_IN_AU = True
HTML_IN_uV    = True
DEFAULT_FS_XDAT = 32000.0   #ist das richtig?

_DEFAULT_SESSION = "/home/ananym/Code/In_vivo_data_analysis/Data/FOR ANNA IN VIVO/"
BASE_PATH   = globals().get("BASE_PATH", _DEFAULT_SESSION)

UNIT_LABEL = "µV/mm²"          
PSD_UNIT_LABEL = "µV²/Hz"


if "LFP_FILENAME" in globals():
    LFP_FILENAME = globals()["LFP_FILENAME"]
else:
    _base_tag = os.path.basename(os.path.normpath(BASE_PATH))
    LFP_FILENAME = f"{_base_tag}.csv"

SAVE_DIR = BASE_PATH
BASE_TAG = os.path.splitext(os.path.basename(LFP_FILENAME))[0]
os.makedirs(SAVE_DIR, exist_ok=True)

LOGFILE = os.path.join(SAVE_DIR, "runlog.txt")





def _counts_to_uV(x, bits, vpp, gain):
    # x = integer/float "counts" (LSB), vpp = Volt p-p
    lsb_volt = float(vpp) / (2**bits)          # Volt pro LSB
    return (np.asarray(x, float) * lsb_volt / float(gain)) * 1e6  # -> µV

def _volts_to_uV(x):
    return np.asarray(x, float) * 1e6

def convert_df_to_uV(df, mode):
    df = df.copy()
    chan_cols_orig = [c for c in df.columns if c not in ("time","stim","din_1","din_2")]
    if mode == "uV":
        # bereits µV – nichts tun
        return df
    elif mode == "volts":
        for c in chan_cols_orig:
            df[c] = _volts_to_uV(pd.to_numeric(df[c], errors="coerce"))
        return df
    elif mode == "counts":
        for c in chan_cols_orig:
            g = PER_CH_GAIN.get(c, PREAMP_GAIN)
            df[c] = _counts_to_uV(pd.to_numeric(df[c], errors="coerce"), ADC_BITS, ADC_VPP, g)
        return df
    else:
        raise ValueError(f"Unbekannter CALIB_MODE: {mode}")

def _decimate_xy(x, Y, max_points=40000):
    """Reduziert Punktezahl, damit SVGs klein bleiben."""
    import numpy as np
    if max_points is None or len(x) <= max_points:
        return x, Y
    step = int(np.ceil(len(x) / max_points))
    return x[::step], Y[:, ::step]



def _ensure_main_channel(LFP_array, preferred_idx=10):
    """
    Liefert (main_channel, used_idx).
    Bevorzugt preferred_idx, sonst Kanal 0.
    Unabhängig von good_idx, damit früh nutzbar (z.B. fürs Spektrogramm).
    """
    num_ch = int(LFP_array.shape[0])
    if isinstance(preferred_idx, int) and 0 <= preferred_idx < num_ch:
        return LFP_array[preferred_idx, :], preferred_idx
    return LFP_array[0, :], 10


def _ensure_seconds(ts, time_ref, fs_xdat=DEFAULT_FS_XDAT):
    """
    Bringt ts (Pulszeiten) in die gleiche Einheit wie time_ref (Sekunden).
    Erkennt 'zu große' Werte heuristisch und teilt dann durch fs_xdat.
    """

    if ts is None: 
        return None
    ts = np.asarray(ts, float)
    if ts.size == 0 or time_ref is None or len(time_ref) == 0:
        return ts
    # Heuristik: Wenn Pulse deutlich außerhalb der time_s-Skala liegen -> in Samples
    tr_min, tr_max = float(time_ref[0]), float(time_ref[-1])
    if np.nanmax(ts) > 100.0 * max(1.0, tr_max):   # sehr konservativ
        return ts / float(fs_xdat)
    return ts

def _safe_crop_to_pulses(time_s, LFP_array, p1, p2, p1_off, p2_off, pad=0.5):

    t = np.asarray(time_s, float)
    if t.size == 0:
        print("[CROP] skip: empty time_s")
        return time_s, LFP_array, p1, p2, p1_off, p2_off

    tmin, tmax = float(t[0]), float(t[-1])

    def _clamp(ts):
        if ts is None:
            return None
        ts = np.asarray(ts, float)
        if ts.size == 0:
            return ts
        return ts[(ts >= tmin) & (ts <= tmax)]

    p1c     = _clamp(p1)
    p2c     = _clamp(p2)
    p1offc  = _clamp(p1_off)
    p2offc  = _clamp(p2_off)

    if ((p1c is None or p1c.size == 0) and (p2c is None or p2c.size == 0)):
        print("[CROP] no ON pulses in range -> no cropping")
        return time_s, LFP_array, p1, p2, p1_off, p2_off

    spans = []
    if p1c is not None and p1c.size:
        spans.append((float(np.min(p1c)), float(np.max(p1c))))
    if p2c is not None and p2c.size:
        spans.append((float(np.min(p2c)), float(np.max(p2c))))
    if not spans:
        print("[CROP] no valid spans -> no cropping")
        return time_s, LFP_array, p1, p2, p1_off, p2_off

    t0 = max(min(s[0] for s in spans) - pad, tmin)
    t1 = min(max(s[1] for s in spans) + pad, tmax)
    if not (t1 > t0):
        print(f"[CROP] invalid window {t0}..{t1} -> no cropping")
        return time_s, LFP_array, p1, p2, p1_off, p2_off

    i0 = int(np.searchsorted(t, t0, side="left"))
    i1 = int(np.searchsorted(t, t1, side="right"))
    i0 = max(0, min(i0, t.size))
    i1 = max(i0 + 1, min(i1, t.size))

    time_new = time_s[i0:i1]
    LFP_new  = LFP_array[:, i0:i1]

    def _keep_in(ts):
        if ts is None:
            return None
        ts = np.asarray(ts, float)
        if ts.size == 0:
            return ts
        return ts[(ts >= time_new[0]) & (ts <= time_new[-1])]

    p1_new    = _keep_in(p1c)
    p2_new    = _keep_in(p2c)
    p1off_new = _keep_in(p1offc)
    p2off_new = _keep_in(p2offc)

    print(f"[CROP] window {t0:.3f}–{t1:.3f} s -> time_s len={len(time_new)}, "
          f"LFP_array={LFP_new.shape}, p1={0 if p1_new is None else len(p1_new)}, "
          f"p2={0 if p2_new is None else len(p2_new)}, "
          f"p1_off={0 if p1off_new is None else len(p1off_new)}, "
          f"p2_off={0 if p2off_new is None else len(p2off_new)}")

    return time_new, LFP_new, p1_new, p2_new, p1off_new, p2off_new

def _empty_updict():
    import numpy as np
    ZI = np.array([], dtype=int); ZF = np.array([], dtype=float)
    return {
        "Spontaneous_UP": ZI, "Spontaneous_DOWN": ZI,
        "Pulse_triggered_UP": ZI, "Pulse_triggered_DOWN": ZI,
        "Pulse_associated_UP": ZI, "Pulse_associated_DOWN": ZI,
        "Spon_Peaks": ZF, "Trig_Peaks": ZF,
        "UP_start_i": ZI, "DOWN_start_i": ZI,
        "Total_power": None, "up_state_binary": None,
    }

def _clip_pairs(U, D, n):
    U = np.asarray(U, int); D = np.asarray(D, int)
    m = min(U.size, D.size)
    if m == 0: return U[:0], D[:0]
    U, D = U[:m], D[:m]
    mask = (U >= 0) & (D > U) & (D <= n)
    return U[mask], D[mask]


# --- Events/Pulse boundary-sicher machen ---
def _clip_events_to_bounds(pulse_times, time_s, pre_s, post_s):
    import numpy as np
    if pulse_times is None: 
        return np.array([], dtype=float)
    t = np.asarray(pulse_times, float)
    if t.size == 0 or len(time_s) == 0:
        return np.array([], dtype=float)
    lo = float(time_s[0]) + float(pre_s)
    hi = float(time_s[-1]) - float(post_s)
    if hi <= lo:
        return np.array([], dtype=float)
    return t[(t >= lo) & (t <= hi)]




def _upstate_amplitudes(signal, up_idx, down_idx):
    """
    Misst pro UP-Event die Amplitude (max - min) im Rohsignal.
    up_idx/down_idx: Sample-Indizes in 'signal' (wie aus classify_states).
    Rückgabe: np.ndarray [n_events] (float), NaN-frei gefiltert.
    """
    import numpy as np
    sig = np.asarray(signal, float)
    U = np.asarray(up_idx, dtype=int)
    D = np.asarray(down_idx, dtype=int)
    m = min(U.size, D.size)
    if m == 0:
        return np.array([], dtype=float)

    U, D = U[:m], D[:m]
    # chronologisch sortieren (optional)
    order = np.argsort(U)
    U, D = U[order], D[order]

    amps = []
    n = sig.size
    for u, d in zip(U, D):
        if not (0 <= u < n and 0 < d <= n and d > u):
            continue
        seg = sig[u:d]
        seg = seg[np.isfinite(seg)]
        if seg.size == 0:
            continue
        amps.append(float(np.nanmax(seg) - np.nanmin(seg)))
    return np.array(amps, dtype=float)



def _sem(x):
    import numpy as np
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return np.nanstd(x) / np.sqrt(max(1, x.size)) if x.size else np.nan

def _even_subsample(idx, k):
    idx = np.asarray(idx, int)
    if idx.size <= k: return idx
    pos = np.linspace(0, idx.size-1, k).round().astype(int)
    return idx[pos]



def _check_peak_indices(label, peaks, n):
    import numpy as np
    p = np.asarray(peaks, int)
    bad = np.sum((p < 0) | (p >= n))
    print(f"[DIAG] {label}: count={p.size}, out_of_bounds={bad} (n_time={n})")


def crop_up_intervals(UP, DOWN, dt, start_s=0.3, end_s=1.0):
    """
    Schneidet UP-Intervalle zu:
    neuer Bereich = [UP + start_s  ...  UP + end_s], aber niemals über DOWN hinaus.
    Gibt zwei Arrays zurück: cropped_UP, cropped_DOWN
    """
    UP = np.asarray(UP, int)
    DOWN = np.asarray(DOWN, int)
    m = min(len(UP), len(DOWN))

    if m == 0:
        return np.array([], int), np.array([], int)

    UP = UP[:m]
    DOWN = DOWN[:m]

    cropped_UP = []
    cropped_DOWN = []

    start_offset = int(round(start_s / dt))
    end_offset   = int(round(end_s / dt))

    for u, d in zip(UP, DOWN):
        new_u = u + start_offset
        new_d = min(u + end_offset, d)

        if new_d > new_u:   # sonst skip
            cropped_UP.append(new_u)
            cropped_DOWN.append(new_d)

    return np.array(cropped_UP, int), np.array(cropped_DOWN, int)


def compute_refractory_period(UP_indices, DOWN_indices, time_s):
    """
    Berechnet die Refraktärzeit:
    Zeit vom Ende eines UP-Zustands (DOWN) bis zum Beginn des
    nächsten UP-Zustands (UP), in Sekunden.

    UP_indices, DOWN_indices: Arrays mit Indizes in time_s
    time_s: Vektor der Zeitstempel (Sekunden)
    """
    UP   = np.asarray(UP_indices,   int)
    DOWN = np.asarray(DOWN_indices, int)

    m = min(len(UP), len(DOWN))
    if m < 2:
        return np.array([], float)

    UP   = UP[:m]
    DOWN = DOWN[:m]

    # Sicherheit: nach Zeit sortieren
    order = np.argsort(time_s[UP])
    UP    = UP[order]
    DOWN  = DOWN[order]

    refrac = []
    for i in range(m - 1):
        t_off = time_s[DOWN[i]]      # Ende des aktuellen UP
        t_on  = time_s[UP[i+1]]      # Beginn des nächsten UP
        dt_ref = t_on - t_off
        if dt_ref >= 0:
            refrac.append(dt_ref)

    return np.array(refrac, float)


def compute_refractory_any_to_type(
    Spontaneous_UP, Spontaneous_DOWN,
    Pulse_triggered_UP, Pulse_triggered_DOWN,
    Pulse_associated_UP, Pulse_associated_DOWN,
    time_s
):
    """
    Refraktärzeit relativ zum ENDE des vorherigen UP-Zustands (egal welcher Typ)
    bis zum BEGINN des nächsten UP-Zustands eines bestimmten Typs.

    Gibt zurück:
        refrac_any_to_spont : np.array, Sekunden
        refrac_any_to_trig  : np.array, Sekunden

    Definition:
        Für jeden UP_k (k >= 1):
            prev = UP_{k-1} (Typ egal)
            curr = UP_k
            dt = onset(curr) - offset(prev)
            -> falls curr 'spont'  : in refrac_any_to_spont
               falls curr 'trig'   : in refrac_any_to_trig
               falls curr 'assoc'  : wird ignoriert
    """
    # alles in Arrays bringen
    Spontaneous_UP   = np.asarray(Spontaneous_UP,   int)
    Spontaneous_DOWN = np.asarray(Spontaneous_DOWN, int)
    Pulse_triggered_UP   = np.asarray(Pulse_triggered_UP,   int)
    Pulse_triggered_DOWN = np.asarray(Pulse_triggered_DOWN, int)
    Pulse_associated_UP   = np.asarray(Pulse_associated_UP,   int)
    Pulse_associated_DOWN = np.asarray(Pulse_associated_DOWN, int)

    up_list   = []
    down_list = []
    type_list = []

    for label, U, D in [
        ("spont", Spontaneous_UP,      Spontaneous_DOWN),
        ("trig",  Pulse_triggered_UP,  Pulse_triggered_DOWN),
        ("assoc", Pulse_associated_UP, Pulse_associated_DOWN),
    ]:
        m = min(len(U), len(D))
        if m == 0:
            continue
        U = U[:m]
        D = D[:m]
        up_list.append(U)
        down_list.append(D)
        type_list.extend([label] * m)

    if not up_list:
        return np.array([], float), np.array([], float)

    up_all   = np.concatenate(up_list)
    down_all = np.concatenate(down_list)
    types    = np.array(type_list, dtype=object)

    # chronologisch nach UP-Onset sortieren
    order   = np.argsort(time_s[up_all])
    up_all   = up_all[order]
    down_all = down_all[order]
    types    = types[order]

    refrac_any_to_spont = []
    refrac_any_to_trig  = []

    for i in range(1, len(up_all)):
        t_off_prev = time_s[down_all[i-1]]   # Ende des vorherigen UP
        t_on_curr  = time_s[up_all[i]]       # Beginn des aktuellen UP
        dt_ref = t_on_curr - t_off_prev
        if dt_ref < 0:
            # sollte eigentlich nicht passieren, aber zur Sicherheit
            continue

        if types[i] == "spont":
            refrac_any_to_spont.append(dt_ref)
        elif types[i] == "trig":
            refrac_any_to_trig.append(dt_ref)
        # 'assoc' ignorieren wir als "Zielt yp"

    return np.array(refrac_any_to_spont, float), np.array(refrac_any_to_trig, float)


def pulse_to_up_latencies(pulse_times, up_indices, time_s, max_win_s=1.0):
    """
    Berechnet die Latenz zwischen Puls und Beginn des UP-Zustands.
    Für jeden UP-Index wird der *letzte Puls davor* gesucht (innerhalb von max_win_s),
    und die Differenz (UP_time - pulse_time) in Sekunden zurückgegeben.
    """
    if pulse_times is None or len(pulse_times) == 0 or len(up_indices) == 0:
        return np.array([], float)

    pulse_times = np.asarray(pulse_times, float)
    up_indices  = np.asarray(up_indices, int)

    # Zeitpunkte der UP-Onsets
    up_t = time_s[up_indices]
    latencies = []

    for t_up in up_t:
        # alle Pulse, die vor diesem UP liegen
        mask = pulse_times <= t_up
        if not mask.any():
            continue
        t_p = pulse_times[mask][-1]    # letzter Puls vor UP

        lat = t_up - t_p
        # Optionales Fenster: nur Pulse, die "in der Nähe" liegen
        if 0.0 <= lat <= max_win_s:
            latencies.append(lat)

    return np.array(latencies, float)



def upstate_amplitude_compare_ax(
    spont_amp, trig_amp,
    ax=None,
    title="UP Amplitude (max-min): Spontan vs. Getriggert"
):

    spont_amp = np.asarray(spont_amp, float)
    trig_amp  = np.asarray(trig_amp,  float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
    else:
        fig = ax.figure

    data, labels = [], []
    if spont_amp.size:
        data.append(spont_amp); labels.append("Spontan")
    if trig_amp.size:
        data.append(trig_amp); labels.append("Getriggert")

    if not data:
        ax.text(0.5, 0.5, "no UP amplitudes", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    ax.boxplot(data, labels=labels, whis=[5, 95], showfliers=False)
    ax.set_ylabel(f"Amplitude ({UNIT_LABEL})")
    ax.set_title(title)
    ax.grid(alpha=0.15, linestyle=":")
    return fig


def pulse_to_up_latency_hist_ax(latencies, ax=None, bins=30):

    latencies = np.asarray(latencies, float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3.0))
    else:
        fig = ax.figure

    if latencies.size == 0:
        ax.text(0.5, 0.5, "no Pulse→UP latencies", 
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    # in Millisekunden umrechnen
    lat_ms  = latencies * 1000.0
    mean_ms = float(np.nanmean(lat_ms))

    # Histogramm
    ax.hist(lat_ms, bins=bins, alpha=0.8, label="Einzel-Latenzen")

    # vertikale Linie beim Mittelwert
    ax.axvline(mean_ms, linestyle="--", linewidth=2,
               label=f"Mean = {mean_ms:.1f} ms")

    ax.set_xlabel("Pulse→UP Latenz (ms)")
    ax.set_ylabel("Anzahl")
    ax.set_title(f"Pulse→UP Latenzen (Mean = {mean_ms:.1f} ms)")

    # Textbox oben rechts mit Mean + n
    ax.text(
        0.98, 0.95,
        f"Mean = {mean_ms:.1f} ms\nn = {lat_ms.size}",
        transform=ax.transAxes,
        ha="right", va="top"
    )

    ax.legend()
    return fig


def upstate_duration_compare_ax(
    Trig_UP_crop, Trig_DOWN_crop,
    Spon_UP_crop, Spon_DOWN_crop,
    dt, ax=None
):


    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
    else:
        fig = ax.figure

    Trig_UP   = np.asarray(Trig_UP_crop,   int)
    Trig_DOWN = np.asarray(Trig_DOWN_crop, int)
    Spon_UP   = np.asarray(Spon_UP_crop,   int)
    Spon_DOWN = np.asarray(Spon_DOWN_crop, int)

    m_trig = min(len(Trig_UP), len(Trig_DOWN))
    m_spon = min(len(Spon_UP), len(Spon_DOWN))

    trig_dur = (Trig_DOWN[:m_trig] - Trig_UP[:m_trig]) * dt if m_trig > 0 else np.array([], float)
    spon_dur = (Spon_DOWN[:m_spon] - Spon_UP[:m_spon]) * dt if m_spon > 0 else np.array([], float)

    data, labels = [], []
    if spon_dur.size:
        data.append(spon_dur); labels.append("Spontan")
    if trig_dur.size:
        data.append(trig_dur); labels.append("Getriggert")

    if not data:
        ax.text(0.5, 0.5, "no UP durations", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    ax.boxplot(data, labels=labels, whis=[5, 95], showfliers=False)
    ax.set_ylabel("Dauer (s)")
    ax.set_title("UP-Dauern (cropped 0.3–1.0 s)")
    ax.grid(alpha=0.15, linestyle=":")
    return fig

def refractory_compare_ax(refrac_spont, refrac_trig, ax=None, title="Refraktärzeit bis zum nächsten UP"):
    import numpy as np
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
    else:
        fig = ax.figure

    refrac_spont = np.asarray(refrac_spont, float)
    refrac_trig  = np.asarray(refrac_trig,  float)

    data, labels = [], []
    if refrac_spont.size:
        data.append(refrac_spont)
        labels.append("Spontan")
    if refrac_trig.size:
        data.append(refrac_trig)
        labels.append("Getriggert")

    if not data:
        ax.text(0.5, 0.5, "keine Refraktärzeiten", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    ax.boxplot(
        data,
        labels=labels,
        whis=[5, 95],
        showfliers=False
    )
    ax.set_ylabel("Refraktärzeit bis nächster UP (s)")
    ax.set_title(title)
    ax.grid(alpha=0.15, linestyle=":")

    # kleine n-Angabe
    txt = []
    if refrac_spont.size:
        txt.append(f"Spontan: n={refrac_spont.size}")
    if refrac_trig.size:
        txt.append(f"Getriggert: n={refrac_trig.size}")
    ax.text(
        0.98, 0.95,
        "\n".join(txt),
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", alpha=0.7)
    )

    return fig


def CSD_single_panel_ax(
    CSD,
    dt,
    *,
    z_mm=None,
    align_pre=0.5,
    align_post=0.5,
    ax=None,
    title="CSD",
    cmap="Spectral_r",      # klassisches Blau-Rot
    sat_pct=95,
    smooth_sigma=(2.0, 3.0),   # (depth, time) Glättung
    flip_y=True
):
    """
    Paper-freundliche Darstellung eines einzelnen CSD:
    - glatte Darstellung
    - lineare, zero-centered Skala
    - klassisches Blau-Rot-Layout
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.figure

    if CSD is None or not isinstance(CSD, np.ndarray) or CSD.ndim != 2 or CSD.size == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "no CSD", ha="center", va="center", transform=ax.transAxes)
        return fig

    # 1) float + optional smoothing
    A = np.asarray(CSD, float)
    if smooth_sigma is not None:
        A = gaussian_filter(A, sigma=smooth_sigma)

    # 2) ggf. Tiefe invertieren (superficial oben)
    if flip_y:
        A = A[::-1, :]
        if z_mm is not None:
            z_plot = np.asarray(z_mm)[::-1]
        else:
            z_plot = None
    else:
        z_plot = np.asarray(z_mm) if z_mm is not None else None

    # 3) robuste Skala
    vals = np.abs(A[np.isfinite(A)])
    if vals.size == 0:
        vmax = 1.0
    else:
        vmax = float(np.nanpercentile(vals, sat_pct))
        if vmax <= 0 or not np.isfinite(vmax):
            vmax = 1.0
    vmin = -vmax
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    # 4) Zeitachse
    n_t = A.shape[1]
    t = np.linspace(-float(align_pre), float(align_post), n_t)

    # 5) extent für imshow
    if z_plot is not None:
        z0, z1 = float(z_plot[0]), float(z_plot[-1])
    else:
        z0, z1 = 0.0, float(A.shape[0] - 1)

    im = ax.imshow(
        A,
        aspect="auto",
        origin="upper",
        extent=[t[0], t[-1], z0, z1],
        cmap=cmap,
        norm=norm,
        interpolation="bilinear",   # weich, „paper“-Look
    )

    ax.set_xlabel("Zeit (s)")
    ax.set_ylabel("Tiefe (mm)" if z_plot is not None else "Tiefe (arb.)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="CSD (a.u.)")

    # optional: dünne vertikale Linie bei 0 s
    ax.axvline(0, color="k", lw=0.5, alpha=0.7)

    return fig


def _as_valid_idx(arr, n):
    if arr is None:
        return None
    a = np.asarray(arr)
    if np.issubdtype(a.dtype, np.floating):
        return None
    a = a.astype(int, copy=False)
    a = a[(a >= 0) & (a < n)]
    return a

def _build_rollups(summary_path, out_name="upstate_summary_ALL.csv"):

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

    exp_dir       = os.path.dirname(summary_path)
    parent_dir    = os.path.dirname(exp_dir)        
    for_david_dir = os.path.dirname(parent_dir)     

    #Rollup pro Parent-Ordner 
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

    # Rollup (alle Parents zusammen) 
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

