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



def downsampling(downsample_factor: int, df, num_channels):
    # Downsampling: jede n-te Zeile behalten
    df_down = df.iloc[::downsample_factor].reset_index(drop=True)
    # Zeitvektor neu berechnen (hier sinnvoller: direkt aus 'timesamples' oder timestamps)
    time_s = df_down['timesamples'].to_numpy() 
    dt = np.mean(np.diff(time_s))
    # Kanalnamen extrahieren
    channel_cols = [col for col in df.columns if col.startswith('pri_')][:num_channels]
    # Daten extrahieren, transponieren: shape (channels, time)
    LFP_array = df_down[channel_cols].to_numpy().T


    # ===== Pulse aus downsampled Daten extrahieren =====
    din1_ds = df_down["din_1"].to_numpy()
    din2_ds = df_down["din_2"].to_numpy()

    rising_edges_1 = np.where((din1_ds[:-1] == 0) & (din1_ds[1:] == 1))[0]
    rising_edges_2 = np.where((din2_ds[:-1] == 0) & (din2_ds[1:] == 2))[0]

    pulse_times_1 = time_s[rising_edges_1]
    pulse_times_2 = time_s[rising_edges_2]


    return (time_s, dt, LFP_array, pulse_times_1, pulse_times_2)


import numpy as np

def downsampling_old(
    downsample_factor: int,
    df,
    num_channels: int,
    pulse_times_1=None,
    pulse_times_2=None,
    snap_pulses: bool = True
):
    """
    Downsampling per Zeilenselektion (df.iloc[::factor]) – wie in deiner alten Version.
    Erwartet DataFrame mit:
      - 'timesamples' (Sekunden, monoton)
      - 'pri_0'..'pri_{N-1}'
      - optional: 'din_1', 'din_2' (altes Format)

    Für neue Struktur ohne din_1/din_2:
      - Übergib pulse_times_1/2 (in Sekunden, relativ zum Recording-Start).
      - Diese werden (optional) auf das downsampled-Raster gesnappt.
    """

    # --- Downsample DataFrame ---
    df_down = df.iloc[::downsample_factor].reset_index(drop=True)

    # --- Zeitvektor & dt ---
    time_s = df_down['timesamples'].to_numpy()
    if time_s.size < 2:
        raise ValueError("Zu wenige Samples nach Downsampling.")
    dt = float(np.mean(np.diff(time_s)))

    # --- Kanäle in (channels, time) ---
    channel_cols = [c for c in df.columns if c.startswith('pri_')][:num_channels]
    LFP_array = df_down[channel_cols].to_numpy().T  # shape (C, T)

    # --- Puls-Extraktion (alt) oder Mapping (neu) ---
    def _snap_to_grid(times, grid):
        """Snap Zeiten (s) auf nächstliegenden Wert im grid (time_s)."""
        times = np.asarray(times, dtype=float)
        if times.size == 0:
            return times
        # auf Bereich beschränken
        mask = (times >= grid[0]) & (times <= grid[-1])
        times = times[mask]
        if times.size == 0:
            return times
        idx = np.searchsorted(grid, times)
        idx = np.clip(idx, 1, len(grid)-1)
        left = grid[idx-1]
        right = grid[idx]
        choose_left = (times - left) <= (right - times)
        return np.where(choose_left, left, right)

    if ('din_1' in df_down.columns) and ('din_2' in df_down.columns):
        # --- Altes Verhalten: Pulse aus digitalen Kanälen holen ---
        din1_ds = df_down["din_1"].to_numpy()
        din2_ds = df_down["din_2"].to_numpy()

        rising_edges_1 = np.where((din1_ds[:-1] == 0) & (din1_ds[1:] == 1))[0]
        rising_edges_2 = np.where((din2_ds[:-1] == 0) & (din2_ds[1:] == 2))[0]

        pulse_times_1_out = time_s[rising_edges_1]
        pulse_times_2_out = time_s[rising_edges_2]

    else:
        # --- Neue Struktur: Pulse werden extern geliefert ---
        pt1 = np.asarray(pulse_times_1 if pulse_times_1 is not None else [], dtype=float)
        pt2 = np.asarray(pulse_times_2 if pulse_times_2 is not None else [], dtype=float)

        if snap_pulses:
            pulse_times_1_out = _snap_to_grid(pt1, time_s)
            pulse_times_2_out = _snap_to_grid(pt2, time_s)
        else:
            # nur auf Datenbereich beschneiden
            def _clip(times, grid):
                m = (times >= grid[0]) & (times <= grid[-1])
                return times[m]
            pulse_times_1_out = _clip(pt1, time_s)
            pulse_times_2_out = _clip(pt2, time_s)

    return (time_s, dt, LFP_array, pulse_times_1_out, pulse_times_2_out)





def get_main_channel(downsample_factor: int, df, num_channels):
    # Downsampling: jede n-te Zeile behalten
    df_down = df.iloc[::downsample_factor].reset_index(drop=True)
    time_s = df_down['timesamples'].to_numpy() 
    main_channel = df_down["pri_10"].to_numpy()
    main_channel.shape == time_s.shape
    main_channel = main_channel-np.mean(main_channel)

    return(main_channel)

def filtering(high_cutoff, low_cutoff, dt):
    srate = 1/dt
    nyq = 0.5 * srate
    high = high_cutoff / nyq
    low = low_cutoff / nyq
    b_lp, a_lp = signal.butter(5, high, btype='low', analog=False)
    b_hp, a_hp = signal.butter(5, low, btype='high', analog=False)
    return b_lp, a_lp, b_hp, a_hp

def pre_post_condition(dt):
    pre = int(0.75 / dt)
    post = int(2 / dt)
    win_len = pre + post

    align_pre = int(0.5 / dt)
    align_post = int(1.5 / dt)
    align_len = align_pre + align_post

    return(pre, post, win_len, align_pre, align_post, align_len)






