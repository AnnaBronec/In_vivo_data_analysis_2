
import os
import re
import numpy as np
_np = np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import glob

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
    return LFP_array[0, :], 0


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




def _upstate_amplitudes(signal, up_idx, down_idx, p_hi=95, p_lo=5):
    """
    Misst pro UP-Event die Amplitude als p95–p5 des Segments (robust gegen Spikes).
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
        amps.append(float(np.percentile(seg, p_hi) - np.percentile(seg, p_lo)))
    return np.array(amps, dtype=float)



def _sem(x):
    import numpy as np
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return np.nanstd(x) / np.sqrt(max(1, x.size)) if x.size else np.nan

def compute_mua_rate(signal_raw, fs_raw=32000.0, hp_hz=300.0, threshold_sigma=3.5,
                     refractory_ms=1.0, return_times=False):
    """Threshold-Crossing MUA aus Rohdaten (32 kHz).
    HP >hp_hz Hz → negative Kreuzungen bei -threshold_sigma × MAD → Firing-Rate [Hz].
    refractory_ms: Mindestabstand zwischen zwei Spikes (Standard 1 ms = 32 Samples bei 32 kHz).
    Mit return_times=True: gibt (rate, spike_times_s) zurück, Zeiten relativ zum Signalstart."""
    from scipy.signal import butter, sosfiltfilt
    _nan = (np.nan, np.array([], dtype=float)) if return_times else np.nan
    sig = np.asarray(signal_raw, dtype=float)
    sig = sig[np.isfinite(sig)]
    if sig.size < int(fs_raw * 0.5):
        return _nan
    sos = butter(3, hp_hz / (fs_raw / 2.0), btype="high", output="sos")
    filt = sosfiltfilt(sos, sig)
    noise = np.median(np.abs(filt)) / 0.6745
    if noise < 1e-12:
        return (0.0, np.array([], dtype=float)) if return_times else 0.0
    thr = -threshold_sigma * noise
    crossing_idx = np.where((filt[:-1] > thr) & (filt[1:] <= thr))[0] + 1
    if refractory_ms > 0 and crossing_idx.size > 1:
        ref_samples = int(round(refractory_ms * fs_raw / 1000.0))
        keep = np.ones(crossing_idx.size, dtype=bool)
        last = crossing_idx[0]
        for i in range(1, crossing_idx.size):
            if crossing_idx[i] - last < ref_samples:
                keep[i] = False
            else:
                last = crossing_idx[i]
        crossing_idx = crossing_idx[keep]
    rate = float(crossing_idx.size / (sig.size / fs_raw))
    if return_times:
        return rate, crossing_idx.astype(float) / fs_raw
    return rate


def count_spikes_per_upstate(spike_times_s, up_idx, down_idx, time_s):
    """
    Zählt MUA-Spikes (Aktionspotenziale) innerhalb jedes Up-Zustands.

    Parameter
    ---------
    spike_times_s : 1D-Array
        Absolute Spike-Zeiten in Sekunden (Output von compute_mua_rate mit return_times=True).
    up_idx, down_idx : int-Arrays
        Indizes des Beginns (UP) und Endes (DOWN) jedes Up-Zustands in time_s.
    time_s : 1D-Array
        Zeitvektor in Sekunden.

    Gibt zurück
    -----------
    counts : int-Array (Länge = min(len(up_idx), len(down_idx)))
        Anzahl der Spikes pro Up-Zustand.
    rates_hz : float-Array
        Lokale Firing-Rate in Hz = counts / Dauer [s].
    durations_s : float-Array
        Dauer jedes Up-Zustands in Sekunden.
    """
    spike_times_s = np.sort(np.asarray(spike_times_s, float))
    up_idx   = np.asarray(up_idx,   int)
    down_idx = np.asarray(down_idx, int)
    time_s   = np.asarray(time_s,   float)

    m = min(len(up_idx), len(down_idx))
    counts       = np.zeros(m, dtype=int)
    rates_hz     = np.full(m, np.nan)
    durations_s  = np.full(m, np.nan)

    if m == 0:
        return counts, rates_hz, durations_s

    n_time = len(time_s)
    up_idx   = np.clip(up_idx[:m],   0, n_time - 1)
    down_idx = np.clip(down_idx[:m], 0, n_time - 1)

    if spike_times_s.size > 0:
        for i in range(m):
            t_start = float(time_s[up_idx[i]])
            t_end   = float(time_s[down_idx[i]])
            dur = t_end - t_start
            if dur <= 0:
                continue
            durations_s[i] = dur
            i_lo = int(np.searchsorted(spike_times_s, t_start, side="left"))
            i_hi = int(np.searchsorted(spike_times_s, t_end,   side="left"))
            counts[i]   = i_hi - i_lo
            rates_hz[i] = float(counts[i]) / dur
    else:
        for i in range(m):
            t_start = float(time_s[up_idx[i]])
            t_end   = float(time_s[down_idx[i]])
            dur = t_end - t_start
            if dur > 0:
                durations_s[i] = dur

    return counts, rates_hz, durations_s


def mua_spikes_per_upstate_hist_ax(
    spont_counts, spont_rates=None,
    ax=None,
    title="MUA: Aktionspotenziale pro spontanem Up-Zustand",
):
    """
    Histogramm der Spike-Anzahl pro spontanem Up-Zustand (eine Session).
    Zeigt Median, Mittelwert und n.  spont_rates optional für zweite Achse.
    """
    counts = np.asarray(spont_counts, float)
    valid  = counts[np.isfinite(counts)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
    else:
        fig = ax.figure

    if valid.size == 0:
        ax.text(0.5, 0.5, "keine MUA-Spike-Daten",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    # Ganzzahlige Bins (Spike-Anzahl = integer)
    max_c = int(valid.max())
    bins = np.arange(-0.5, max_c + 1.5, 1.0)
    ax.hist(valid, bins=bins, color="#4C78A8", alpha=0.80, edgecolor="white", linewidth=0.5)

    mean_v   = float(np.nanmean(valid))
    median_v = float(np.nanmedian(valid))
    ax.axvline(mean_v,   color="#E15759", lw=1.8, linestyle="--", label=f"Mittel={mean_v:.1f}")
    ax.axvline(median_v, color="#59A14F", lw=1.8, linestyle=":",  label=f"Median={median_v:.1f}")

    ax.set_xlabel("Spikes pro Up-Zustand")
    ax.set_ylabel("Anzahl Up-Zustände")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.18, linestyle=":")

    # Textbox: Kennzahlen
    sem_v = float(np.nanstd(valid) / np.sqrt(valid.size))
    txt = (
        f"n={valid.size}\n"
        f"µ={mean_v:.1f} ± {sem_v:.1f} (SEM)\n"
        f"Median={median_v:.1f}\n"
        f"Min={int(valid.min())}  Max={int(valid.max())}"
    )
    if spont_rates is not None:
        rates = np.asarray(spont_rates, float)
        r_val = rates[np.isfinite(rates)]
        if r_val.size:
            txt += f"\nFiring-Rate (µ)={np.nanmean(r_val):.1f} Hz"
    ax.text(
        0.97, 0.96, txt,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=0.78),
    )
    return fig


def _even_subsample(idx, k):
    idx = np.asarray(idx, int)
    if idx.size <= k: return idx
    pos = np.linspace(0, idx.size-1, k).round().astype(int)
    return idx[pos]


def _mwu_stats(spont, trig, alpha=0.05):
    """
    Mann-Whitney-U + Cliff's delta fuer Spontan vs Getriggert.
    """
    sp = np.asarray(spont, float)
    tr = np.asarray(trig, float)
    sp = sp[np.isfinite(sp)]
    tr = tr[np.isfinite(tr)]
    out = {
        "n_sp": int(sp.size),
        "n_tr": int(tr.size),
        "p": np.nan,
        "delta": np.nan,
        "significant": False,
    }
    if sp.size < 2 or tr.size < 2:
        return out
    try:
        from scipy.stats import mannwhitneyu
        _, p = mannwhitneyu(sp, tr, alternative="two-sided")
        out["p"] = float(p)
        out["significant"] = bool(np.isfinite(p) and (p < float(alpha)))
    except Exception:
        pass
    try:
        gt = np.sum(sp[:, None] > tr[None, :])
        lt = np.sum(sp[:, None] < tr[None, :])
        out["delta"] = float((gt - lt) / float(sp.size * tr.size))
    except Exception:
        pass
    return out


def _p_to_sig_label(p, alpha=0.05):
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < float(alpha):
        return "*"
    return "n.s."


def _annotate_sig_2groups(ax, x1, x2, p, alpha=0.05):
    label = _p_to_sig_label(p, alpha=alpha)
    if label == "n/a":
        return
    y0, y1 = ax.get_ylim()
    yr = (y1 - y0) if (np.isfinite(y1 - y0) and (y1 > y0)) else max(abs(y1), 1.0)
    h = 0.04 * yr
    y = y1 + 0.01 * yr
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, color="black", clip_on=False)
    ax.text((x1 + x2) * 0.5, y + h, label, ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(y0, y1 + 0.14 * yr)



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
    time_s = np.asarray(time_s, float)
    n_time = int(time_s.size)
    if n_time < 2:
        return np.array([], float)

    m = min(len(UP), len(DOWN))
    if m < 2:
        return np.array([], float)

    UP   = UP[:m]
    DOWN = DOWN[:m]
    valid = (
        np.isfinite(UP) & np.isfinite(DOWN) &
        (UP >= 0) & (UP < n_time) &
        (DOWN > UP) & (DOWN <= n_time)
    )
    UP = UP[valid]
    DOWN = DOWN[valid]
    if UP.size < 2:
        return np.array([], float)

    # Sicherheit: nach Zeit sortieren
    order = np.argsort(time_s[UP])
    UP    = UP[order]
    DOWN  = DOWN[order]
    dt_time = float(np.median(np.diff(time_s))) if n_time >= 2 else np.nan

    refrac = []
    for i in range(UP.size - 1):
        d_i = int(DOWN[i])
        d_clip = min(d_i, n_time - 1)
        t_off = float(time_s[d_clip])      # Ende des aktuellen UP (DOWN ist exklusiver Rand)
        if d_i == n_time and np.isfinite(dt_time):
            t_off += dt_time
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


def pulse_to_event_latencies(pulse_times, event_indices, time_s, max_win_s=1.0):
    """
    Berechnet die Latenz zwischen Puls und Event-Beginn.
    Fuer jeden Event-Index wird der letzte Puls davor gesucht (innerhalb von max_win_s),
    und die Differenz (Event-Zeit - Puls-Zeit) in Sekunden zurueckgegeben.
    """
    if pulse_times is None or len(pulse_times) == 0 or len(event_indices) == 0:
        return np.array([], float)

    pulse_times = np.asarray(pulse_times, float)
    event_indices = np.asarray(event_indices, int)

    # Zeitpunkte der Event-Onsets
    event_t = time_s[event_indices]
    latencies = []

    for t_event in event_t:
        # alle Pulse, die vor diesem Event liegen
        mask = pulse_times <= t_event
        if not mask.any():
            continue
        t_p = pulse_times[mask][-1]    # letzter Puls vor Event

        lat = t_event - t_p
        # Optionales Fenster: nur Pulse, die "in der Nähe" liegen
        if 0.0 <= lat <= max_win_s:
            latencies.append(lat)

    return np.array(latencies, float)


def pulse_to_up_latencies(pulse_times, up_indices, time_s, max_win_s=1.0):
    """
    Rueckwaertskompatibler Wrapper fuer Pulse->UP-Latenzen.
    """
    return pulse_to_event_latencies(
        pulse_times,
        up_indices,
        time_s,
        max_win_s=max_win_s,
    )



def upstate_amplitude_compare_ax(
    spont_amp, trig_amp,
    ax=None,
    title="UP Amplitude (max-min, mean): Spontan vs. Getriggert",
    y_limits=None,
):

    spont_amp = np.asarray(spont_amp, float)
    trig_amp  = np.asarray(trig_amp,  float)
    sp_valid = spont_amp[np.isfinite(spont_amp)]
    tr_valid = trig_amp[np.isfinite(trig_amp)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
    else:
        fig = ax.figure

    data, labels = [], []
    if sp_valid.size:
        data.append(sp_valid); labels.append("Spontan")
    if tr_valid.size:
        data.append(tr_valid); labels.append("Getriggert")

    if not data:
        ax.text(0.5, 0.5, "no UP amplitudes", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    means = [float(np.nanmean(d)) for d in data]
    x = np.arange(1, len(data) + 1, dtype=float)
    bars = ax.bar(x, means, width=0.55, color=["#4C78A8", "#F58518"][:len(data)], alpha=0.85)
    for i, d in enumerate(data):
        ax.text(
            x[i],
            means[i],
            f"  n={len(d)}\n  mean={means[i]:.2f}",
            va="bottom",
            ha="left",
            fontsize=8,
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel(f"Amplitude ({UNIT_LABEL})")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.15, linestyle=":")
    st = _mwu_stats(sp_valid, tr_valid, alpha=0.05)
    txt = (
        f"n_sp={st['n_sp']}, n_tr={st['n_tr']}\n"
        f"MWU p={(st['p'] if np.isfinite(st['p']) else np.nan):.2e}"
        if np.isfinite(st["p"]) else
        f"n_sp={st['n_sp']}, n_tr={st['n_tr']}\nMWU p=na"
    )
    txt += f"\nsignifikant (a=0.05): {'ja' if st['significant'] else 'nein'}"
    if np.isfinite(st["delta"]):
        txt += f"\nCliff's d={st['delta']:.2f}"
    if np.isfinite(st["p"]):
        txt += f"\nSig: {_p_to_sig_label(st['p'], alpha=0.05)}"
    ax.text(
        0.98, 0.95, txt,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.75)
    )
    if sp_valid.size and tr_valid.size and np.isfinite(st["p"]):
        _annotate_sig_2groups(ax, 1, 2, st["p"], alpha=0.05)
    if y_limits is not None:
        ax.set_ylim(y_limits)
    return fig


def pulse_to_up_latency_hist_ax(latencies, ax=None, bins=30, event_label="UP", title=None):

    latencies = np.asarray(latencies, float)
    event_label = str(event_label)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3.0))
    else:
        fig = ax.figure

    if latencies.size == 0:
        ax.text(0.5, 0.5, f"no Pulse->{event_label} latencies",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    # in Millisekunden umrechnen
    lat_ms  = latencies * 1000.0
    mean_ms = float(np.nanmean(lat_ms))
    plot_title = title or f"Pulse->{event_label} latencies (Mean = {mean_ms:.1f} ms)"

    # Histogramm
    ax.hist(lat_ms, bins=bins, alpha=0.8, label="Einzel-Latenzen")

    # vertikale Linie beim Mittelwert
    ax.axvline(mean_ms, linestyle="--", linewidth=2,
               label=f"Mean = {mean_ms:.1f} ms")

    ax.set_xlabel(f"Pulse->{event_label} latency (ms)")
    ax.set_ylabel("Anzahl")
    ax.set_title(plot_title)

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
    Trig_UP, Trig_DOWN,
    Spon_UP, Spon_DOWN,
    dt, ax=None
):


    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
    else:
        fig = ax.figure

    Trig_UP   = np.asarray(Trig_UP,   int)
    Trig_DOWN = np.asarray(Trig_DOWN, int)
    Spon_UP   = np.asarray(Spon_UP,   int)
    Spon_DOWN = np.asarray(Spon_DOWN, int)

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
    ax.set_title("UP-Dauern (Onset→Offset, uncropped)")
    ax.grid(alpha=0.15, linestyle=":")

    # Statistik (robust, non-parametrisch) + Effektgröße
    spon_valid = spon_dur[np.isfinite(spon_dur)]
    trig_valid = trig_dur[np.isfinite(trig_dur)]
    p_val = np.nan
    delta = np.nan
    if spon_valid.size >= 2 and trig_valid.size >= 2:
        try:
            from scipy.stats import mannwhitneyu
            _, p_val = mannwhitneyu(spon_valid, trig_valid, alternative="two-sided")
        except Exception:
            p_val = np.nan

        # Cliff's delta: >0 => spontan tendenziell länger, <0 => trig länger
        try:
            gt = np.sum(spon_valid[:, None] > trig_valid[None, :])
            lt = np.sum(spon_valid[:, None] < trig_valid[None, :])
            delta = (gt - lt) / float(spon_valid.size * trig_valid.size)
        except Exception:
            delta = np.nan

    txt = (
        f"n_sp={spon_valid.size}, n_tr={trig_valid.size}\n"
        f"MWU p={p_val:.2e}" if np.isfinite(p_val) else
        f"n_sp={spon_valid.size}, n_tr={trig_valid.size}\nMWU p=na"
    )
    txt += f"\nsignifikant (a=0.05): {'ja' if (np.isfinite(p_val) and p_val < 0.05) else 'nein'}"
    if np.isfinite(delta):
        txt += f"\nCliff's d={delta:.2f}"
    if np.isfinite(p_val):
        txt += f"\nSig: {_p_to_sig_label(p_val, alpha=0.05)}"
    ax.text(
        0.98, 0.95, txt,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.75)
    )
    if spon_valid.size and trig_valid.size and np.isfinite(p_val):
        _annotate_sig_2groups(ax, 1, 2, p_val, alpha=0.05)
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
    sp_valid = refrac_spont[np.isfinite(refrac_spont)]
    tr_valid = refrac_trig[np.isfinite(refrac_trig)]

    data, labels = [], []
    if sp_valid.size:
        data.append(sp_valid)
        labels.append("Spontan")
    if tr_valid.size:
        data.append(tr_valid)
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

    st = _mwu_stats(sp_valid, tr_valid, alpha=0.05)
    txt = [f"Spontan: n={st['n_sp']}", f"Getriggert: n={st['n_tr']}"]
    if np.isfinite(st["p"]):
        txt.append(f"MWU p={st['p']:.2e}")
    else:
        txt.append("MWU p=na")
    txt.append(f"signifikant (a=0.05): {'ja' if st['significant'] else 'nein'}")
    if np.isfinite(st["delta"]):
        txt.append(f"Cliff's d={st['delta']:.2f}")
    if np.isfinite(st["p"]):
        txt.append(f"Sig: {_p_to_sig_label(st['p'], alpha=0.05)}")
    ax.text(
        0.98, 0.95,
        "\n".join(txt),
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", alpha=0.7)
    )
    if sp_valid.size and tr_valid.size and np.isfinite(st["p"]):
        _annotate_sig_2groups(ax, 1, 2, st["p"], alpha=0.05)

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

    def _infer_group(row):
        exp = str(row.get("Experiment", "") or "").strip()
        parent = str(row.get("Parent", "") or "").strip()
        src = exp if exp else parent
        s = src.upper()
        for key in ("TLX", "DRD", "PV", "CHR2", "CHR", "WT", "GFP"):
            if key in s:
                return key
        m = re.match(r"([A-Z]{2,}\d*)", s)
        if m:
            return m.group(1)
        return "OTHER"

    def _to_num(series):
        return pd.to_numeric(series, errors="coerce")

    def _write_group_compare(df_rollup, out_dir, stem):
        if df_rollup is None or df_rollup.empty:
            return

        d = df_rollup.copy()
        d["Group"] = d.apply(_infer_group, axis=1)
        d["triggered"] = _to_num(d.get("triggered"))
        d["spon"] = _to_num(d.get("spon"))
        d["associated"] = _to_num(d.get("associated"))
        d["Mean UP Dauer [s]"] = _to_num(d.get("Mean UP Dauer [s]"))
        d["UP rate total [Hz]"] = _to_num(d.get("UP rate total [Hz]"))

        for c in ("triggered", "spon", "associated"):
            if c not in d or d[c].isna().all():
                d[c] = 0.0
        total = d["triggered"].fillna(0.0) + d["spon"].fillna(0.0) + d["associated"].fillna(0.0)
        total = total.where(total > 0, np.nan)
        d["frac_triggered"] = d["triggered"] / total
        d["frac_associated"] = d["associated"] / total

        grp = d.groupby("Group", dropna=False).agg(
            n_sessions=("Experiment", "count"),
            mean_frac_triggered=("frac_triggered", "mean"),
            mean_frac_associated=("frac_associated", "mean"),
            mean_up_duration_s=("Mean UP Dauer [s]", "mean"),
            mean_up_rate_hz=("UP rate total [Hz]", "mean"),
        ).reset_index()
        if grp.empty:
            return

        grp = grp.sort_values("n_sessions", ascending=False)
        out_csv = os.path.join(out_dir, f"{stem}__group_compare.csv")
        _write_semicolon(out_csv, grp)
        print(f"[SUMMARY][GROUP] {out_csv}")

        fig, axs = plt.subplots(2, 2, figsize=(11, 7))
        axs = axs.ravel()
        panels = [
            ("mean_frac_triggered", "frac_triggered", "Mean triggered fraction"),
            ("mean_frac_associated", "frac_associated", "Mean associated fraction"),
            ("mean_up_duration_s", "Mean UP Dauer [s]", "Mean UP duration [s]"),
            ("mean_up_rate_hz", "UP rate total [Hz]", "Mean UP rate [Hz]"),
        ]
        for ax, (col_mean, col_raw, title) in zip(axs, panels):
            groups = grp["Group"].astype(str).tolist()
            x = np.arange(len(groups), dtype=float)
            data = []
            for gname in groups:
                vals = pd.to_numeric(d.loc[d["Group"] == gname, col_raw], errors="coerce").to_numpy(float)
                vals = vals[np.isfinite(vals)]
                data.append(vals)

            valid_pairs = [(i, arr) for i, arr in enumerate(data) if arr.size > 0]
            if valid_pairs:
                pos = [i for i, _ in valid_pairs]
                arrs = [arr for _, arr in valid_pairs]
                bp = ax.boxplot(
                    arrs,
                    positions=pos,
                    widths=0.5,
                    whis=[5, 95],
                    showfliers=False,
                    patch_artist=True,
                )
                for patch in bp["boxes"]:
                    patch.set(facecolor="#4C78A8", alpha=0.22, edgecolor="#4C78A8")
                for med in bp["medians"]:
                    med.set(color="#4C78A8", linewidth=1.8)
                for whisk in bp["whiskers"]:
                    whisk.set(color="#4C78A8", linewidth=1.2)
                for cap in bp["caps"]:
                    cap.set(color="#4C78A8", linewidth=1.2)

            for i, vals in enumerate(data):
                if vals.size == 0:
                    continue
                if vals.size == 1:
                    xx = np.array([x[i]], dtype=float)
                else:
                    xx = np.linspace(x[i] - 0.16, x[i] + 0.16, vals.size)
                ax.scatter(xx, vals, s=24, color="#111111", alpha=0.8, zorder=3, linewidths=0)
                ax.text(x[i], np.nanmax(vals), f"n={vals.size}", ha="center", va="bottom", fontsize=8, color="#222222")

            ax.set_title(title)
            ax.set_xticks(np.arange(len(groups), dtype=float))
            xt = [f"{g}\n(n={int(n)})" for g, n in zip(groups, grp["n_sessions"])]
            ax.set_xticklabels(xt, rotation=0, ha="center")
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(f"{stem}: Group comparison (session means)", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out_png = os.path.join(out_dir, f"{stem}__group_compare.png")
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[SUMMARY][GROUP] {out_png}")

    def _load_metric_events(summary_files, metric):
        rows = []
        for sp in summary_files:
            sess_dir = os.path.dirname(sp)
            df_one = _read_any(sp)
            if df_one is None or df_one.empty:
                continue
            r = df_one.iloc[-1].to_dict()
            grp = _infer_group(r)

            if metric == "up_duration_s":
                files = sorted(glob.glob(os.path.join(sess_dir, "*__upstate_durations.csv")))
                if not files:
                    continue
                try:
                    dfm = pd.read_csv(files[-1])
                    st = dfm.get("group", pd.Series([], dtype=str)).astype(str).str.lower().str.strip()
                    st = st.replace({"spont": "spontaneous", "trig": "triggered", "trigger": "triggered"})
                    val = pd.to_numeric(dfm.get("duration_s", pd.Series([], dtype=float)), errors="coerce")
                    mask = np.isfinite(val) & st.isin(["spontaneous", "triggered"])
                    for s, v in zip(st[mask], val[mask]):
                        rows.append({"session_dir": sess_dir, "group": grp, "state": str(s), metric: float(v)})
                except Exception:
                    continue

            elif metric == "up_amplitude":
                files = sorted(glob.glob(os.path.join(sess_dir, "*__upstate_amplitudes.csv")))
                if not files:
                    continue
                try:
                    dfm = pd.read_csv(files[-1])
                    st = dfm.get("group", pd.Series([], dtype=str)).astype(str).str.lower().str.strip()
                    st = st.replace({"spont": "spontaneous", "trig": "triggered", "trigger": "triggered"})
                    val = pd.to_numeric(dfm.get("amplitude", pd.Series([], dtype=float)), errors="coerce")
                    mask = np.isfinite(val) & st.isin(["spontaneous", "triggered"])
                    for s, v in zip(st[mask], val[mask]):
                        rows.append({"session_dir": sess_dir, "group": grp, "state": str(s), metric: float(v)})
                except Exception:
                    continue

            elif metric == "mahal_k3":
                files = sorted(glob.glob(os.path.join(sess_dir, "*__pca_similarity_joint.csv")))
                if not files:
                    continue
                try:
                    dfm = pd.read_csv(files[-1])
                    st = dfm.get("event_type", pd.Series([], dtype=str)).astype(str).str.lower().str.strip()
                    st = st.replace({"spont": "spontaneous", "trig": "triggered", "trigger": "triggered"})
                    val = pd.to_numeric(dfm.get("mahal_k3", pd.Series([], dtype=float)), errors="coerce")
                    mask = np.isfinite(val) & st.isin(["spontaneous", "triggered"])
                    for s, v in zip(st[mask], val[mask]):
                        rows.append({"session_dir": sess_dir, "group": grp, "state": str(s), metric: float(v)})
                except Exception:
                    continue

            elif metric == "mua_spikes_spont":
                # Spikes pro spontanem Up-Zustand – nur spontaneous
                files = sorted(glob.glob(os.path.join(sess_dir, "*__mua_spikes_per_upstate.csv")))
                if not files:
                    continue
                try:
                    dfm = pd.read_csv(files[-1])
                    st  = dfm.get("group", pd.Series([], dtype=str)).astype(str).str.lower().str.strip()
                    val = pd.to_numeric(dfm.get("n_spikes", pd.Series([], dtype=float)), errors="coerce")
                    mask = np.isfinite(val) & (st == "spontaneous")
                    for v in val[mask]:
                        rows.append({"session_dir": sess_dir, "group": grp, "state": "spontaneous",
                                     metric: float(v)})
                except Exception:
                    continue

            elif metric == "mua_rate_hz_spont":
                # Firing-Rate innerhalb spontaner Up-Zustände
                files = sorted(glob.glob(os.path.join(sess_dir, "*__mua_spikes_per_upstate.csv")))
                if not files:
                    continue
                try:
                    dfm = pd.read_csv(files[-1])
                    st  = dfm.get("group", pd.Series([], dtype=str)).astype(str).str.lower().str.strip()
                    val = pd.to_numeric(dfm.get("firing_rate_hz", pd.Series([], dtype=float)), errors="coerce")
                    mask = np.isfinite(val) & (st == "spontaneous")
                    for v in val[mask]:
                        rows.append({"session_dir": sess_dir, "group": grp, "state": "spontaneous",
                                     metric: float(v)})
                except Exception:
                    continue

        return pd.DataFrame(rows)

    def _group_pair_box_scatter(ax, df, col, title, ylabel):
        if df is None or df.empty or col not in df.columns:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return

        d = df.copy()
        d[col] = pd.to_numeric(d[col], errors="coerce")
        d["state"] = d["state"].astype(str).str.lower()
        d = d[np.isfinite(d[col]) & d["state"].isin(["spontaneous", "triggered"])]
        if d.empty:
            ax.text(0.5, 0.5, "no valid values", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return

        groups = sorted(
            d["group"].dropna().unique().tolist(),
            key=lambda g: (
                -(d.loc[d["group"] == g, "session_dir"].nunique()),
                str(g)
            ),
        )
        if not groups:
            ax.text(0.5, 0.5, "no groups", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return

        base_x = np.arange(len(groups), dtype=float)
        states = [("spontaneous", -0.18, "#2ca02c"), ("triggered", 0.18, "#1f77b4")]
        rng = np.random.default_rng(0)

        for st, off, color in states:
            plot_data = []
            plot_pos = []
            for i, g in enumerate(groups):
                arr = pd.to_numeric(d.loc[(d["group"] == g) & (d["state"] == st), col], errors="coerce").to_numpy(float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    continue
                plot_data.append(arr)
                plot_pos.append(base_x[i] + off)

            if plot_data:
                bp = ax.boxplot(
                    plot_data,
                    positions=plot_pos,
                    widths=0.28,
                    whis=[5, 95],
                    showfliers=False,
                    patch_artist=True,
                )
                for patch in bp["boxes"]:
                    patch.set(facecolor=color, alpha=0.28, edgecolor=color)
                for med in bp["medians"]:
                    med.set(color=color, linewidth=1.8)
                for whisk in bp["whiskers"]:
                    whisk.set(color=color, linewidth=1.2)
                for cap in bp["caps"]:
                    cap.set(color=color, linewidth=1.2)

                for pos, arr in zip(plot_pos, plot_data):
                    xx = np.full(arr.size, pos, dtype=float)
                    if arr.size > 1:
                        xx = xx + (rng.random(arr.size) - 0.5) * 0.12
                    ax.scatter(xx, arr, s=20, alpha=0.75, color=color, linewidths=0, zorder=3)
                    ax.text(pos, np.nanmax(arr), f"n={arr.size}", ha="center", va="bottom", fontsize=7, color=color)

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(base_x)
        ax.set_xticklabels(groups, rotation=0, ha="center")
        ax.grid(axis="y", alpha=0.25, ls=":")
        ax.plot([], [], color="#2ca02c", lw=2, label="Spontaneous")
        ax.plot([], [], color="#1f77b4", lw=2, label="Triggered")
        ax.legend(loc="best", frameon=False, fontsize=8)

    def _group_spont_box_scatter(ax, df, col, title, ylabel):
        """
        Boxplot + Scatterplot einer Metrik für spontane Up-Zustände,
        eine Box pro Bedingungsgruppe (Ordnernamen → TLX, DRD, WT etc.).
        Jeder Punkt = ein Up-Zustand-Event.
        """
        if df is None or df.empty or col not in df.columns:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return

        d = df.copy()
        d[col] = pd.to_numeric(d[col], errors="coerce")
        d = d[np.isfinite(d[col])]
        if d.empty:
            ax.text(0.5, 0.5, "no valid values", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return

        # Gruppen nach Anzahl Sessions absteigend sortieren
        groups = sorted(
            d["group"].dropna().unique().tolist(),
            key=lambda g: (-(d.loc[d["group"] == g, "session_dir"].nunique()), str(g)),
        )
        if not groups:
            ax.text(0.5, 0.5, "no groups", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return

        # Eine Farbe pro Gruppe
        cmap = plt.get_cmap("tab10")
        base_x = np.arange(len(groups), dtype=float)
        rng = np.random.default_rng(0)

        plot_data, plot_pos, plot_colors = [], [], []
        for i, g in enumerate(groups):
            arr = pd.to_numeric(d.loc[d["group"] == g, col], errors="coerce").to_numpy(float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            color = cmap(i % 10)
            plot_data.append(arr)
            plot_pos.append(base_x[i])
            plot_colors.append(color)

        if not plot_data:
            ax.set_axis_off()
            return

        bp = ax.boxplot(
            plot_data, positions=plot_pos, widths=0.45,
            whis=[5, 95], showfliers=False, patch_artist=True,
        )
        for patch, color in zip(bp["boxes"], plot_colors):
            patch.set(facecolor=color, alpha=0.30, edgecolor=color)
        for med, color in zip(bp["medians"], plot_colors):
            med.set(color=color, linewidth=2.0)
        for el in bp["whiskers"] + bp["caps"]:
            el.set(linewidth=1.2, color="gray")

        for pos, arr, color in zip(plot_pos, plot_data, plot_colors):
            xx = np.full(arr.size, pos) + (rng.random(arr.size) - 0.5) * 0.22
            ax.scatter(xx, arr, s=14, alpha=0.55, color=color, linewidths=0, zorder=3)
            n_sess = d.loc[d["group"] == groups[int(round(pos))], "session_dir"].nunique()
            ax.text(pos, np.nanmax(arr),
                    f"n_ev={arr.size}\nn_sess={n_sess}",
                    ha="center", va="bottom", fontsize=7, color=color)

        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_xticks(base_x)
        ax.set_xticklabels(groups, rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.25, ls=":")

    def _write_parent_group_compare_pdf(parent_dir, summary_files):
        d_dur     = _load_metric_events(summary_files, "up_duration_s")
        d_amp     = _load_metric_events(summary_files, "up_amplitude")
        d_mah     = _load_metric_events(summary_files, "mahal_k3")
        d_spk     = _load_metric_events(summary_files, "mua_spikes_spont")
        d_spk_hz  = _load_metric_events(summary_files, "mua_rate_hz_spont")

        any_data = not (d_dur.empty and d_amp.empty and d_mah.empty
                        and d_spk.empty and d_spk_hz.empty)
        if not any_data:
            print("[SUMMARY][GROUP][PDF] skipped: no event-level rows")
            return

        if not d_dur.empty:
            out_csv = os.path.join(parent_dir, "upstate_summary_ALL_parent__group_event_up_durations.csv")
            _write_semicolon(out_csv, d_dur)
            print(f"[SUMMARY][GROUP][PDF] {out_csv}")
        if not d_amp.empty:
            out_csv = os.path.join(parent_dir, "upstate_summary_ALL_parent__group_event_up_amplitudes.csv")
            _write_semicolon(out_csv, d_amp)
            print(f"[SUMMARY][GROUP][PDF] {out_csv}")
        if not d_mah.empty:
            out_csv = os.path.join(parent_dir, "upstate_summary_ALL_parent__group_event_mahal.csv")
            _write_semicolon(out_csv, d_mah)
            print(f"[SUMMARY][GROUP][PDF] {out_csv}")
        if not d_spk.empty:
            out_csv = os.path.join(parent_dir, "upstate_summary_ALL_parent__group_mua_spikes_spont.csv")
            _write_semicolon(out_csv, d_spk)
            print(f"[SUMMARY][GROUP][PDF] {out_csv}")
        if not d_spk_hz.empty:
            out_csv = os.path.join(parent_dir, "upstate_summary_ALL_parent__group_mua_rate_hz_spont.csv")
            _write_semicolon(out_csv, d_spk_hz)
            print(f"[SUMMARY][GROUP][PDF] {out_csv}")

        out_pdf = os.path.join(parent_dir, "upstate_summary_ALL_parent__group_compare.pdf")
        with PdfPages(out_pdf) as pdf:
            # Seite 1: Dauer / Amplitude / Mahalanobis (bisherige Plots)
            fig, axs = plt.subplots(3, 1, figsize=(10, 12))
            _group_pair_box_scatter(
                axs[0], d_dur, "up_duration_s",
                "UP duration (spontaneous vs triggered)",
                "duration [s]"
            )
            _group_pair_box_scatter(
                axs[1], d_amp, "up_amplitude",
                "UP amplitude (spontaneous vs triggered)",
                "amplitude [a.u.]"
            )
            _group_pair_box_scatter(
                axs[2], d_mah, "mahal_k3",
                "Mahalanobis distance in JOINT PCA space",
                "mahal_k3"
            )
            fig.suptitle("Parent Group Compare", y=0.995)
            fig.tight_layout(rect=[0, 0, 1, 0.98])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Seite 2: MUA Spikes pro spontanem Up-Zustand (Gruppenvergleich)
            if not d_spk.empty or not d_spk_hz.empty:
                fig2, axs2 = plt.subplots(2, 1, figsize=(10, 9))
                _group_spont_box_scatter(
                    axs2[0], d_spk, "mua_spikes_spont",
                    "MUA: Aktionspotenziale pro spontanem Up-Zustand (Gruppenvergleich)",
                    "Spikes pro Up-Zustand"
                )
                _group_spont_box_scatter(
                    axs2[1], d_spk_hz, "mua_rate_hz_spont",
                    "MUA: Firing-Rate in spontanen Up-Zuständen (Gruppenvergleich)",
                    "Firing-Rate [Hz]"
                )
                fig2.suptitle("MUA Spikes in spontanen Up-Zuständen – Gruppenvergleich", y=0.995)
                fig2.tight_layout(rect=[0, 0, 1, 0.98])
                pdf.savefig(fig2, bbox_inches="tight")
                plt.close(fig2)

        print(f"[SUMMARY][GROUP][PDF] {out_pdf}")

    def _up_rate_overview_ax(ax, rate_per_min, rate_hz, title, y_limits=None):
        vals = np.array([rate_per_min, rate_hz], dtype=float)
        labels = ["total [/min]", "total [Hz]"]
        x = np.arange(2, dtype=float)
        if not np.isfinite(vals).any():
            ax.text(0.5, 0.5, "no valid UP-rate values", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return
        ax.bar(x, vals, color=["#4C78A8", "#59A14F"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_ylabel("value")
        ax.set_title(title)
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                ax.text(float(xi), float(v), f"{float(v):.3g}", ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.25, ls=":")
        if y_limits is not None:
            ax.set_ylim(y_limits)

    def _write_parent_up_rate_overview_pdf(parent_dir, summary_files):
        def _nat_session_key(path_str):
            sess = os.path.basename(os.path.dirname(path_str))
            m = re.match(r"^\s*(\d+)", str(sess))
            if m:
                return (0, int(m.group(1)), str(sess).lower())
            return (1, str(sess).lower())

        rows = []
        for sp in sorted(summary_files, key=_nat_session_key):
            sess_dir = os.path.dirname(sp)
            sess_name = os.path.basename(sess_dir)
            df_one = _read_any(sp)
            if df_one is None or df_one.empty:
                rows.append({
                    "session_dir": sess_dir,
                    "session_name": sess_name,
                    "rate_per_min": np.nan,
                    "rate_hz": np.nan,
                })
                continue
            r = df_one.iloc[-1].to_dict()
            rate_hz = pd.to_numeric(pd.Series([r.get("UP rate total [Hz]", np.nan)]), errors="coerce").iloc[0]
            rate_pm = pd.to_numeric(pd.Series([r.get("UP rate total [/min]", np.nan)]), errors="coerce").iloc[0]
            rows.append({
                "session_dir": sess_dir,
                "session_name": sess_name,
                "rate_per_min": float(rate_pm) if np.isfinite(rate_pm) else np.nan,
                "rate_hz": float(rate_hz) if np.isfinite(rate_hz) else np.nan,
            })

        if not rows:
            print("[SUMMARY][UP-RATE][PDF] skipped: no summary files")
            return

        # Einheitliche Y-Achse fuer alle Panels berechnen (sessionuebergreifend).
        all_rates = []
        for row in rows:
            vals = np.array([row["rate_per_min"], row["rate_hz"]], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                all_rates.append(vals)
        global_ylim = None
        if all_rates:
            vals = np.concatenate(all_rates)
            if vals.size:
                vmin = float(np.nanmin(vals))
                vmax = float(np.nanmax(vals))
                if np.isfinite(vmin) and np.isfinite(vmax):
                    if np.isclose(vmin, vmax):
                        pad = max(1e-6, abs(vmax) * 0.1)
                        global_ylim = (vmin - pad, vmax + pad)
                    else:
                        span = vmax - vmin
                        y_lo = min(0.0, vmin - 0.05 * span)
                        y_hi = vmax + 0.15 * span
                        global_ylim = (y_lo, y_hi)

        out_pdf = os.path.join(parent_dir, "upstate_summary_ALL_parent__up_rate_panels.pdf")
        n = len(rows)
        ncols = 2
        nrows_page = 3
        per_page = ncols * nrows_page

        with PdfPages(out_pdf) as pdf:
            for start in range(0, n, per_page):
                chunk = rows[start:start + per_page]
                fig, axs = plt.subplots(nrows_page, ncols, figsize=(11, 12))
                axs = np.asarray(axs).reshape(-1)
                for ax, row in zip(axs, chunk):
                    title = f"{row['session_name']} — UP rate (total): [/min] vs [Hz]"
                    _up_rate_overview_ax(
                        ax, row["rate_per_min"], row["rate_hz"], title, y_limits=global_ylim
                    )
                for ax in axs[len(chunk):]:
                    ax.axis("off")
                fig.suptitle("UP-rate overview per subfolder", y=0.995)
                fig.tight_layout(rect=[0, 0, 1, 0.98])
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        print(f"[SUMMARY][UP-RATE][PDF] {out_pdf}")

    def _write_parent_up_amplitude_overview_pdf(parent_dir, summary_files):
        def _nat_session_key(path_str):
            sess = os.path.basename(os.path.dirname(path_str))
            m = re.match(r"^\s*(\d+)", str(sess))
            if m:
                return (0, int(m.group(1)), str(sess).lower())
            return (1, str(sess).lower())

        rows = []
        for sp in sorted(summary_files, key=_nat_session_key):
            sess_dir = os.path.dirname(sp)
            sess_name = os.path.basename(sess_dir)
            amp_files = sorted(glob.glob(os.path.join(sess_dir, "*__upstate_amplitudes.csv")), key=os.path.getmtime)
            spont = np.array([], dtype=float)
            trig = np.array([], dtype=float)
            if amp_files:
                try:
                    dfm = pd.read_csv(amp_files[-1])
                    st = dfm.get("group", pd.Series([], dtype=str)).astype(str).str.lower().str.strip()
                    st = st.replace({"spont": "spontaneous", "trig": "triggered", "trigger": "triggered"})
                    amp = pd.to_numeric(dfm.get("amplitude", pd.Series([], dtype=float)), errors="coerce").to_numpy(float)
                    spont = amp[(st.to_numpy() == "spontaneous") & np.isfinite(amp)]
                    trig = amp[(st.to_numpy() == "triggered") & np.isfinite(amp)]
                except Exception:
                    spont = np.array([], dtype=float)
                    trig = np.array([], dtype=float)
            rows.append({
                "session_name": sess_name,
                "spont_amp": np.asarray(spont, dtype=float),
                "trig_amp": np.asarray(trig, dtype=float),
            })

        if not rows:
            print("[SUMMARY][UP-AMP][PDF] skipped: no summary files")
            return

        # Einheitliche Y-Achse fuer alle Panels berechnen (sessionuebergreifend).
        all_amp = []
        for row in rows:
            if row["spont_amp"].size:
                all_amp.append(row["spont_amp"])
            if row["trig_amp"].size:
                all_amp.append(row["trig_amp"])
        global_ylim = None
        if all_amp:
            vals = np.concatenate(all_amp)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                vmin = float(np.nanmin(vals))
                vmax = float(np.nanpercentile(vals, 99))
                if np.isfinite(vmin) and np.isfinite(vmax):
                    if np.isclose(vmin, vmax):
                        pad = max(1e-6, abs(vmax) * 0.1)
                        global_ylim = (vmin - pad, vmax + pad)
                    else:
                        span = vmax - vmin
                        y_lo = min(0.0, vmin - 0.05 * span)
                        y_hi = vmax + 0.15 * span
                        global_ylim = (y_lo, y_hi)

        out_pdf = os.path.join(parent_dir, "upstate_summary_ALL_parent__up_amplitude_panels.pdf")
        n = len(rows)
        ncols = 2
        nrows_page = 3
        per_page = ncols * nrows_page

        with PdfPages(out_pdf) as pdf:
            for start in range(0, n, per_page):
                chunk = rows[start:start + per_page]
                fig, axs = plt.subplots(nrows_page, ncols, figsize=(11, 12))
                axs = np.asarray(axs).reshape(-1)
                for ax, row in zip(axs, chunk):
                    title = f"{row['session_name']}\nUP Amplitude (max-min, mean): Spontan vs. Getriggert"
                    upstate_amplitude_compare_ax(
                        row["spont_amp"], row["trig_amp"], ax=ax, title=title, y_limits=global_ylim
                    )
                for ax in axs[len(chunk):]:
                    ax.axis("off")
                fig.suptitle("UP-amplitude overview per subfolder", y=0.995)
                fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=3.5)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        print(f"[SUMMARY][UP-AMP][PDF] {out_pdf}")

    def _write_parent_mua_trend_pdf(parent_dir, summary_files):
        def _nat_key(p):
            s = os.path.basename(os.path.dirname(p))
            m = re.match(r"^\s*(\d+)", s)
            return (0, int(m.group(1)), s.lower()) if m else (1, s.lower())

        rows = []
        for sp in sorted(summary_files, key=_nat_key):
            sess_name = os.path.basename(os.path.dirname(sp))
            try:
                df = pd.read_csv(sp, sep=None, engine="python")
                rate = pd.to_numeric(df.get("MUA rate [Hz]", pd.Series([np.nan])), errors="coerce").iloc[-1]
            except Exception:
                rate = np.nan
            rows.append({"session_name": sess_name, "mua_hz": float(rate) if pd.notna(rate) else np.nan})

        rows = [r for r in rows if np.isfinite(r["mua_hz"])]
        if not rows:
            return

        labels = [r["session_name"] for r in rows]
        x      = np.arange(len(labels))
        vals   = np.array([r["mua_hz"] for r in rows])

        fig_w = max(8, len(labels) * 0.9)
        fig, ax = plt.subplots(figsize=(fig_w, 5))
        ax.plot(x, vals, color="#2ca02c", marker="o", linewidth=1.5, markersize=7, label="MUA rate")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("MUA Firing Rate (Hz)")
        ax.set_title("MUA Firing Rate pro Session", fontsize=11)
        ymax = float(np.nanpercentile(vals, 99))
        yspan = max(ymax, 1e-6)
        ax.set_ylim(0, ymax + 0.20 * yspan)
        ax.grid(alpha=0.2, linestyle=":")
        ax.legend(fontsize=9)
        fig.tight_layout()
        out_pdf = os.path.join(parent_dir, "upstate_summary_ALL_parent__mua_trend.pdf")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"[SUMMARY][MUA-TREND][PDF] {out_pdf}")

    def _write_parent_up_amplitude_trend_pdf(parent_dir, summary_files):
        def _nat_session_key(path_str):
            sess = os.path.basename(os.path.dirname(path_str))
            m = re.match(r"^\s*(\d+)", str(sess))
            if m:
                return (0, int(m.group(1)), str(sess).lower())
            return (1, str(sess).lower())

        rows = []
        for sp in sorted(summary_files, key=_nat_session_key):
            sess_dir = os.path.dirname(sp)
            sess_name = os.path.basename(sess_dir)
            amp_files = sorted(glob.glob(os.path.join(sess_dir, "*__upstate_amplitudes.csv")), key=os.path.getmtime)
            spont = np.array([], dtype=float)
            trig = np.array([], dtype=float)
            if amp_files:
                try:
                    dfm = pd.read_csv(amp_files[-1])
                    st = dfm.get("group", pd.Series([], dtype=str)).astype(str).str.lower().str.strip()
                    st = st.replace({"spont": "spontaneous", "trig": "triggered", "trigger": "triggered"})
                    amp = pd.to_numeric(dfm.get("amplitude", pd.Series([], dtype=float)), errors="coerce").to_numpy(float)
                    spont = amp[(st.to_numpy() == "spontaneous") & np.isfinite(amp)]
                    trig  = amp[(st.to_numpy() == "triggered")   & np.isfinite(amp)]
                except Exception:
                    pass
            rows.append({
                "session_name": sess_name,
                "spont_mean": float(np.nanmean(spont)) if spont.size else np.nan,
                "trig_mean":  float(np.nanmean(trig))  if trig.size  else np.nan,
                "spont_std":  float(np.nanstd(spont))  if spont.size else np.nan,
                "trig_std":   float(np.nanstd(trig))   if trig.size  else np.nan,
                "spont_raw":  spont.tolist(),
                "trig_raw":   trig.tolist(),
            })

        if not rows:
            return

        labels     = [r["session_name"] for r in rows]
        x          = np.arange(len(labels))
        spont_vals = np.array([r["spont_mean"] for r in rows])
        trig_vals  = np.array([r["trig_mean"]  for r in rows])

        all_raw = np.array([v for r in rows
                            for v in r.get("spont_raw", []) + r.get("trig_raw", [])
                            if np.isfinite(v)])
        if all_raw.size:
            vmax = float(np.nanpercentile(all_raw, 99))
            vmin = float(np.nanmin(all_raw))
            span = max(vmax - vmin, 1e-6)
            ylim = (min(0.0, vmin - 0.05 * span), vmax + 0.20 * span)
        else:
            ylim = None

        fig_w = max(8, len(labels) * 0.9)
        fig, ax = plt.subplots(figsize=(fig_w, 5))

        spont_stds = np.array([r.get("spont_std", np.nan) for r in rows])
        trig_stds  = np.array([r.get("trig_std",  np.nan) for r in rows])
        sp_ok = np.isfinite(spont_vals)
        tr_ok = np.isfinite(trig_vals)

        # Einzelwerte als kleine, halbtransparente Punkte
        for i, r in enumerate(rows):
            raw_sp = r.get("spont_raw", [])
            raw_tr = r.get("trig_raw",  [])
            if raw_sp:
                ax.scatter([i] * len(raw_sp), raw_sp,
                           color="#4C78A8", alpha=0.25, s=12, zorder=2, linewidths=0)
            if raw_tr:
                ax.scatter([i] * len(raw_tr), raw_tr,
                           color="#F58518", alpha=0.25, s=12, zorder=2, linewidths=0)

        # Mittelwert als großer Punkt mit Standardabweichung
        if sp_ok.any():
            ax.errorbar(x[sp_ok], spont_vals[sp_ok], yerr=spont_stds[sp_ok],
                        color="#4C78A8", marker="o", linewidth=1.5, markersize=8,
                        capsize=3, label="Spontan", zorder=3)
        if tr_ok.any():
            ax.errorbar(x[tr_ok], trig_vals[tr_ok], yerr=trig_stds[tr_ok],
                        color="#F58518", marker="o", linewidth=1.5, markersize=8,
                        capsize=3, label="Getriggert", zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel(f"Amplitude ({UNIT_LABEL})")
        ax.set_title("UP Amplitude pro Session", fontsize=11)
        if ylim:
            ax.set_ylim(ylim)
        ax.grid(alpha=0.2, linestyle=":")
        ax.legend(fontsize=9)

        fig.tight_layout()
        out_pdf = os.path.join(parent_dir, "upstate_summary_ALL_parent__up_amplitude_trend.pdf")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"[SUMMARY][UP-AMP-TREND][PDF] {out_pdf}")

    def _write_global_all_trend_pdf(root_dir):
        """Kombiniert alle Parent-Trend-Plots in root_dir in eine einzige PDF."""
        def _name_key(name):
            m = re.match(r"^\s*(\d+)", str(name))
            return (0, int(m.group(1)), str(name).lower()) if m else (1, str(name).lower())

        parent_dirs = sorted(
            [e.path for e in os.scandir(root_dir) if e.is_dir()],
            key=lambda p: _name_key(os.path.basename(p))
        )

        entries = []
        for pd_path in parent_dirs:
            session_dirs = sorted(
                [e for e in os.scandir(pd_path) if e.is_dir()],
                key=lambda e: _name_key(e.name)
            )
            rows = []
            for entry in session_dirs:
                amp_files = sorted(glob.glob(os.path.join(entry.path, "*__upstate_amplitudes.csv")), key=os.path.getmtime)
                spont = np.array([], dtype=float)
                trig  = np.array([], dtype=float)
                if amp_files:
                    try:
                        dfm = pd.read_csv(amp_files[-1])
                        st = (dfm.get("group", pd.Series([], dtype=str))
                                 .astype(str).str.lower().str.strip()
                                 .replace({"spont": "spontaneous", "trig": "triggered",
                                           "trigger": "triggered"}))
                        amp = pd.to_numeric(
                            dfm.get("amplitude", pd.Series([], dtype=float)), errors="coerce"
                        ).to_numpy(float)
                        spont = amp[(st.to_numpy() == "spontaneous") & np.isfinite(amp)]
                        trig  = amp[(st.to_numpy() == "triggered")   & np.isfinite(amp)]
                    except Exception:
                        pass
                spont_mean = float(np.nanmean(spont)) if spont.size else np.nan
                trig_mean  = float(np.nanmean(trig))  if trig.size  else np.nan
                if np.isfinite(spont_mean) or np.isfinite(trig_mean):
                    rows.append({
                        "session_name": entry.name,
                        "spont_mean": spont_mean,
                        "trig_mean":  trig_mean,
                        "spont_std":  float(np.nanstd(spont)) if spont.size else np.nan,
                        "trig_std":   float(np.nanstd(trig))  if trig.size  else np.nan,
                        "spont_raw":  spont.tolist(),
                        "trig_raw":   trig.tolist(),
                    })
            if rows:
                entries.append((os.path.basename(pd_path), rows))

        if not entries:
            return

        out_pdf = os.path.join(root_dir, "ALL_trend_amplitude.pdf")
        with PdfPages(out_pdf) as pdf:
            for folder_name, rows in entries:
                labels     = [r["session_name"] for r in rows]
                x          = np.arange(len(labels))
                spont_vals = np.array([r["spont_mean"] for r in rows])
                trig_vals  = np.array([r["trig_mean"]  for r in rows])
                spont_stds = np.array([r.get("spont_std", np.nan) for r in rows])
                trig_stds  = np.array([r.get("trig_std",  np.nan) for r in rows])
                sp_ok = np.isfinite(spont_vals)
                tr_ok = np.isfinite(trig_vals)

                all_raw = np.array([v for r in rows
                                    for v in r.get("spont_raw", []) + r.get("trig_raw", [])
                                    if np.isfinite(v)])
                if all_raw.size:
                    vmax = float(np.nanpercentile(all_raw, 99))
                    vmin = float(np.nanmin(all_raw))
                    span = max(vmax - vmin, 1e-6)
                    ylim = (min(0.0, vmin - 0.05 * span), vmax + 0.20 * span)
                else:
                    ylim = None

                fig_w = max(9, len(labels) * 0.9)
                fig, ax = plt.subplots(figsize=(fig_w, 5))

                # Einzelwerte als kleine, halbtransparente Punkte
                for i, r in enumerate(rows):
                    raw_sp = r.get("spont_raw", [])
                    raw_tr = r.get("trig_raw",  [])
                    if raw_sp:
                        ax.scatter([i] * len(raw_sp), raw_sp,
                                   color="#4C78A8", alpha=0.25, s=12, zorder=2, linewidths=0)
                    if raw_tr:
                        ax.scatter([i] * len(raw_tr), raw_tr,
                                   color="#F58518", alpha=0.25, s=12, zorder=2, linewidths=0)

                # Mittelwert als großer Punkt mit Standardabweichung
                if sp_ok.any():
                    ax.errorbar(x[sp_ok], spont_vals[sp_ok], yerr=spont_stds[sp_ok],
                                color="#4C78A8", marker="o", linewidth=1.5, markersize=8,
                                capsize=3, label="Spontan", zorder=3)
                if tr_ok.any():
                    ax.errorbar(x[tr_ok], trig_vals[tr_ok], yerr=trig_stds[tr_ok],
                                color="#F58518", marker="o", linewidth=1.5, markersize=8,
                                capsize=3, label="Getriggert", zorder=3)

                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
                ax.set_ylabel(f"Amplitude ({UNIT_LABEL})")
                ax.set_title(folder_name, fontsize=12, fontweight="bold", pad=8)
                if ylim:
                    ax.set_ylim(ylim)
                ax.grid(alpha=0.2, linestyle=":")
                ax.legend(fontsize=9)
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        print(f"[SUMMARY][ALL-TREND][PDF] {out_pdf}  ({len(entries)} Parent-Ordner)")

    def _write_global_mua_trend_pdf(root_dir):
        """Eine PDF mit einem MUA-Trend-Plot pro Parent-Ordner, gespeichert in root_dir."""
        def _name_key(name):
            m = re.match(r"^\s*(\d+)", str(name))
            return (0, int(m.group(1)), str(name).lower()) if m else (1, str(name).lower())

        parent_dirs = sorted(
            [e.path for e in os.scandir(root_dir) if e.is_dir()],
            key=lambda p: _name_key(os.path.basename(p))
        )

        entries = []
        for pd_path in parent_dirs:
            summary_files = sorted(glob.glob(os.path.join(pd_path, "*", "upstate_summary.csv")))
            rows = []
            for sp in summary_files:
                sess_name = os.path.basename(os.path.dirname(sp))
                try:
                    df = pd.read_csv(sp, sep=None, engine="python")
                    rate = pd.to_numeric(
                        df.get("MUA rate [Hz]", pd.Series([np.nan])), errors="coerce"
                    ).iloc[-1]
                except Exception:
                    rate = np.nan
                if pd.notna(rate) and np.isfinite(float(rate)):
                    rows.append({"session_name": sess_name, "mua_hz": float(rate)})
            if rows:
                entries.append((os.path.basename(pd_path), rows))

        if not entries:
            return

        out_pdf = os.path.join(root_dir, "ALL_trend_mua.pdf")
        with PdfPages(out_pdf) as pdf:
            for folder_name, rows in entries:
                labels = [r["session_name"] for r in rows]
                x      = np.arange(len(labels))
                vals   = np.array([r["mua_hz"] for r in rows])

                fig_w = max(9, len(labels) * 0.9)
                fig, ax = plt.subplots(figsize=(fig_w, 5))
                ax.plot(x, vals, color="#2ca02c", marker="o",
                        linewidth=1.5, markersize=7, label="MUA rate")
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
                ax.set_ylabel("MUA Firing Rate (Hz)")
                ax.set_title(folder_name, fontsize=12, fontweight="bold", pad=8)
                ymax = float(np.nanpercentile(vals, 99))
                ax.set_ylim(0, ymax + 0.20 * max(ymax, 1e-6))
                ax.grid(alpha=0.2, linestyle=":")
                ax.legend(fontsize=9)
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        print(f"[SUMMARY][ALL-MUA-TREND][PDF] {out_pdf}  ({len(entries)} Parent-Ordner)")

    def _load_mua_spikes_rows_for_sessions(session_dirs):
        """
        Hilfsfunktion: Liest *__mua_spikes_per_upstate.csv für jede Session,
        filtert auf spontaneous, gibt sortierte Liste von Row-Dicts zurück.
        """
        def _nat_key(path_str):
            s = os.path.basename(os.path.dirname(str(path_str)))
            m = re.match(r"^\s*(\d+)", s)
            return (0, int(m.group(1)), s.lower()) if m else (1, s.lower())

        rows = []
        for sess_dir in sorted(session_dirs, key=_nat_key):
            sess_name = os.path.basename(sess_dir)
            spk_files = sorted(
                glob.glob(os.path.join(sess_dir, "*__mua_spikes_per_upstate.csv")),
                key=os.path.getmtime,
            )
            counts = np.array([], dtype=float)
            rates  = np.array([], dtype=float)
            if spk_files:
                try:
                    dfm = pd.read_csv(spk_files[-1])
                    st  = dfm.get("group", pd.Series([], dtype=str)).astype(str).str.lower().str.strip()
                    mask_sp = (st == "spontaneous")
                    counts = pd.to_numeric(
                        dfm.get("n_spikes", pd.Series([], dtype=float)), errors="coerce"
                    ).to_numpy(float)[mask_sp]
                    rates  = pd.to_numeric(
                        dfm.get("firing_rate_hz", pd.Series([], dtype=float)), errors="coerce"
                    ).to_numpy(float)[mask_sp]
                    counts = counts[np.isfinite(counts)]
                    rates  = rates[np.isfinite(rates)]
                except Exception:
                    pass
            rows.append({
                "session_name": sess_name,
                "mean":     float(np.nanmean(counts)) if counts.size else np.nan,
                "std":      float(np.nanstd(counts))  if counts.size else np.nan,
                "raw":      counts.tolist(),
                "rate_mean": float(np.nanmean(rates)) if rates.size else np.nan,
                "rate_std":  float(np.nanstd(rates))  if rates.size else np.nan,
                "rate_raw":  rates.tolist(),
            })
        return rows

    def _draw_mua_spikes_trend_ax(ax, rows, title, ylabel="Spikes pro spontanem Up-Zustand"):
        """
        Zeichnet den Trend-Plot (Einzelpunkte + Mittelwert±SD) in ax.
        Analog zu _write_parent_up_amplitude_trend_pdf.
        """
        if not rows:
            ax.text(0.5, 0.5, "keine Daten", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()
            return

        labels = [r["session_name"] for r in rows]
        x      = np.arange(len(labels))
        means  = np.array([r["mean"] for r in rows])
        stds   = np.array([r["std"]  for r in rows])

        all_raw = np.array([v for r in rows for v in r.get("raw", []) if np.isfinite(v)])
        if all_raw.size:
            vmax = float(np.nanpercentile(all_raw, 99))
            vmin = max(0.0, float(np.nanmin(all_raw)))
            span = max(vmax - vmin, 1e-6)
            ylim = (max(0.0, vmin - 0.05 * span), vmax + 0.25 * span)
        else:
            ylim = None

        # Einzelwerte (halbtransparent)
        for i, r in enumerate(rows):
            raw = r.get("raw", [])
            if raw:
                ax.scatter([i] * len(raw), raw,
                           color="#4C78A8", alpha=0.22, s=14, zorder=2, linewidths=0)

        # Mittelwert ± SD
        ok = np.isfinite(means)
        if ok.any():
            ax.errorbar(x[ok], means[ok], yerr=stds[ok],
                        color="#4C78A8", marker="o", linewidth=1.8, markersize=8,
                        capsize=3, label="Spontan (µ ± SD)", zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
        if ylim:
            ax.set_ylim(ylim)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2, linestyle=":")
        ax.legend(fontsize=9, frameon=False)

    def _write_parent_mua_spikes_trend_pdf(parent_dir, summary_files):
        """
        Trend-PDF pro Parent-Ordner: Ø MUA-Spikes pro spontanem Up-Zustand, Session für Session.
        Gespeichert als upstate_summary_ALL_parent__mua_spikes_trend.pdf
        """
        sess_dirs = [os.path.dirname(sp) for sp in summary_files]
        rows = _load_mua_spikes_rows_for_sessions(sess_dirs)
        rows = [r for r in rows if np.isfinite(r["mean"]) or r["raw"]]
        if not rows:
            print("[SUMMARY][MUA-SPIKES-TREND] keine Daten → PDF übersprungen")
            return

        fig_w = max(8, len(rows) * 0.9)
        fig, axes = plt.subplots(2, 1, figsize=(fig_w, 9))

        _draw_mua_spikes_trend_ax(
            axes[0], rows,
            title="MUA Spikes pro spontanem Up-Zustand",
            ylabel="Spikes / Up-Zustand",
        )
        # zweite Achse: Firing-Rate
        rows_hz = [{**r, "mean": r["rate_mean"], "std": r["rate_std"], "raw": r["rate_raw"]}
                   for r in rows]
        _draw_mua_spikes_trend_ax(
            axes[1], rows_hz,
            title="MUA Firing-Rate in spontanen Up-Zuständen",
            ylabel="Firing-Rate [Hz]",
        )

        fig.suptitle(os.path.basename(parent_dir), fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out_pdf = os.path.join(parent_dir, "upstate_summary_ALL_parent__mua_spikes_trend.pdf")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"[SUMMARY][MUA-SPIKES-TREND][PDF] {out_pdf}")

    def _write_global_mua_spikes_trend_pdf(root_dir):
        """
        Globale Trend-PDF (analog ALL_trend_amplitude.pdf):
        Eine Seite pro Parent-Ordner, zeigt Ø MUA-Spikes/spontaner Up-Zustand pro Session.
        Gespeichert als ALL_trend_mua_spikes.pdf in root_dir.
        """
        def _name_key(name):
            m = re.match(r"^\s*(\d+)", str(name))
            return (0, int(m.group(1)), str(name).lower()) if m else (1, str(name).lower())

        parent_dirs = sorted(
            [e.path for e in os.scandir(root_dir) if e.is_dir()],
            key=lambda p: _name_key(os.path.basename(p)),
        )

        entries = []
        for pd_path in parent_dirs:
            sess_dirs = sorted(
                [e.path for e in os.scandir(pd_path) if e.is_dir()],
                key=lambda p: _name_key(os.path.basename(p)),
            )
            rows = _load_mua_spikes_rows_for_sessions(sess_dirs)
            rows = [r for r in rows if np.isfinite(r["mean"]) or r["raw"]]
            if rows:
                entries.append((os.path.basename(pd_path), rows))

        if not entries:
            print("[SUMMARY][ALL-MUA-SPIKES-TREND] keine Daten → PDF übersprungen")
            return

        out_pdf = os.path.join(root_dir, "ALL_trend_mua_spikes.pdf")
        with PdfPages(out_pdf) as pdf:
            for folder_name, rows in entries:
                fig_w = max(9, len(rows) * 0.9)
                fig, axes = plt.subplots(2, 1, figsize=(fig_w, 9))
                _draw_mua_spikes_trend_ax(
                    axes[0], rows,
                    title="MUA Spikes pro spontanem Up-Zustand",
                    ylabel="Spikes / Up-Zustand",
                )
                rows_hz = [{**r, "mean": r["rate_mean"], "std": r["rate_std"], "raw": r["rate_raw"]}
                           for r in rows]
                _draw_mua_spikes_trend_ax(
                    axes[1], rows_hz,
                    title="MUA Firing-Rate in spontanen Up-Zuständen",
                    ylabel="Firing-Rate [Hz]",
                )
                fig.suptitle(folder_name, fontsize=13, fontweight="bold")
                fig.tight_layout(rect=[0, 0, 1, 0.97])
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        print(f"[SUMMARY][ALL-MUA-SPIKES-TREND][PDF] {out_pdf}  ({len(entries)} Parent-Ordner)")

    def _needs_update(out_path, sources):
        if not os.path.exists(out_path):
            return True
        t_out = os.path.getmtime(out_path)
        return any(os.path.getmtime(s) > t_out for s in sources if os.path.exists(s))

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
        _write_group_compare(r, parent_dir, "upstate_summary_ALL_parent")
        _pdf_p = lambda n: os.path.join(parent_dir, n)
        if _needs_update(_pdf_p("upstate_summary_ALL_parent__group_compare.pdf"), files_parent):
            _write_parent_group_compare_pdf(parent_dir, files_parent)
        else:
            print("[SUMMARY][SKIP] group_compare PDF aktuell")
        if _needs_update(_pdf_p("upstate_summary_ALL_parent__up_rate_panels.pdf"), files_parent):
            _write_parent_up_rate_overview_pdf(parent_dir, files_parent)
        else:
            print("[SUMMARY][SKIP] up_rate PDF aktuell")
        if _needs_update(_pdf_p("upstate_summary_ALL_parent__up_amplitude_panels.pdf"), files_parent):
            _write_parent_up_amplitude_overview_pdf(parent_dir, files_parent)
        else:
            print("[SUMMARY][SKIP] up_amplitude PDF aktuell")
        if _needs_update(_pdf_p("upstate_summary_ALL_parent__up_amplitude_trend.pdf"), files_parent):
            _write_parent_up_amplitude_trend_pdf(parent_dir, files_parent)
        else:
            print("[SUMMARY][SKIP] amplitude_trend PDF aktuell")
        if _needs_update(_pdf_p("upstate_summary_ALL_parent__mua_trend.pdf"), files_parent):
            _write_parent_mua_trend_pdf(parent_dir, files_parent)
        else:
            print("[SUMMARY][SKIP] mua_trend PDF aktuell")
        if _needs_update(_pdf_p("upstate_summary_ALL_parent__mua_spikes_trend.pdf"), files_parent):
            _write_parent_mua_spikes_trend_pdf(parent_dir, files_parent)
        else:
            print("[SUMMARY][SKIP] mua_spikes_trend PDF aktuell")
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
        _write_group_compare(r_all, for_david_dir, "upstate_summary_ALL_global")
        _pdf_fd = lambda n: os.path.join(for_david_dir, n)
        if _needs_update(_pdf_fd("ALL_trend_amplitude.pdf"), files_all):
            _write_global_all_trend_pdf(for_david_dir)
        else:
            print("[SUMMARY][SKIP] ALL_trend_amplitude PDF aktuell")
        if _needs_update(_pdf_fd("ALL_trend_mua.pdf"), files_all):
            _write_global_mua_trend_pdf(for_david_dir)
        else:
            print("[SUMMARY][SKIP] ALL_trend_mua PDF aktuell")
        if _needs_update(_pdf_fd("ALL_trend_mua_spikes.pdf"), files_all):
            _write_global_mua_spikes_trend_pdf(for_david_dir)
        else:
            print("[SUMMARY][SKIP] ALL_trend_mua_spikes PDF aktuell")
        try:
            import importlib.util as _ilu
            from pathlib import Path as _Path
            _dur_path = _Path(__file__).with_name("merge_trend_duration_pdfs.py")
            if _dur_path.exists():
                _spec = _ilu.spec_from_file_location("merge_trend_duration_pdfs", str(_dur_path))
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                _mod.ROOT_DIR = for_david_dir
                _mod.OUT_PDF  = os.path.join(for_david_dir, "ALL_trend_duration.pdf")
                _mod.main()
        except Exception as _e:
            print(f"[TREND-DURATION] übersprungen: {_e}")
        try:
            import importlib.util as _ilu
            from pathlib import Path as _Path
            _pca_path = _Path(__file__).with_name("pca_upstates.py")
            if _pca_path.exists():
                _spec = _ilu.spec_from_file_location("pca_upstates", str(_pca_path))
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                _mod.ROOT_DIR = for_david_dir
                _mod.OUT_PDF  = os.path.join(for_david_dir, "ALL_pca_upstates.pdf")
                _mod.main()
        except Exception as _e:
            print(f"[PCA-UPSTATES] übersprungen: {_e}")
    else:
        print("[SUMMARY][ROLLUP For David] keine Quellen gefunden")
