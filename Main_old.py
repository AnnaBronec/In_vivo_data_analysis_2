

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
from preprocessing import downsampling, filtering, get_main_channel, pre_post_condition, downsampling_old
from plotter import plot_all_channels, plot_spont_up_mean, Total_power_plot, tests, plot_upstate_amplitudes, plot_upstate_amplitude_mean, plot_upstate_amplitude_blocks_colored
from state_detection_old import classify_states, Generate_CSD_mean, Spectrum, extract_upstate_windows, compute_spectra, compare_spectra, plot_contrast_heatmap, average_amplitude_in_upstates, compute_spectra, plot_CSD_comparison, plot_LFP_average_around_peaks
from loader_old import load_LFP_new, load_lightpulses_simple


DOWNSAMPLE_FACTOR = 50
NUM_CHANNELS = 17
HIGH_CUTOFF = 10
LOW_CUTOFF  = 2

BASE_PATH = "/home/ananym/Code/In_vivo_data_analysis/Data"
LFP_FILENAME = "2017-8-9_13-52-30onePulse200msX20per15s_combined_drdcross.csv"
LIGHT_PULSES = "2017-8-9_13-52-30onePulse200msX20per15s_Stimuli.csv"   # dein Dateiname
# LFP (neue Struktur → altes Format)
LFP_df, channel_names, lfp_meta = load_LFP_new(BASE_PATH, LFP_FILENAME)
NUM_CHANNELS = len([c for c in LFP_df.columns if c.startswith("pri_")])

# Sampling-Rate
time_raw_s = LFP_df["timesamples"].to_numpy()
dt_raw = np.mean(np.diff(time_raw_s))
sampling_rate = 1.0 / dt_raw
print(f"Sampling-Rate: {sampling_rate:.2f} Hz")

# Lichtpulse aus separater Datei (ein Kanal)
pulse_times_1 = load_lightpulses_simple(BASE_PATH, LIGHT_PULSES, lfp_meta=lfp_meta)
pulse_times_2 = np.array([], dtype=float)  # kein zweiter Kanal vorhanden



time_s, dt, LFP_array, pulse_times_1, pulse_times_2 = downsampling_old(
    DOWNSAMPLE_FACTOR, LFP_df, NUM_CHANNELS,
    pulse_times_1=pulse_times_1,   # aus LightPulses.csv
    pulse_times_2=np.array([]),    # ein Kanal → leer
    snap_pulses=True               # auf Raster legen (praktisch für Index-Logik)
)


b_lp, a_lp, b_hp, a_hp = filtering(HIGH_CUTOFF, LOW_CUTOFF, dt)
main_channel = get_main_channel(DOWNSAMPLE_FACTOR,LFP_df, NUM_CHANNELS)
pre, post, win_len, align_pre, align_post, align_len = pre_post_condition(dt)

#Define Spectogramm
Spect_dat = Run_spectrogram(main_channel,time_s)

#state detection 
Up_states = classify_states(Spect_dat, time_s, pulse_times_1, pulse_times_2, dt, main_channel, LFP_array, b_lp, a_lp, b_hp, a_hp, align_pre, align_post, align_len)

# Access results
Spontaneous_UP = Up_states["Spontaneous_UP"]
Spontaneous_DOWN = Up_states["Spontaneous_DOWN"]
Pulse_triggered_UP = Up_states["Pulse_triggered_UP"]
Pulse_triggered_DOWN = Up_states["Pulse_triggered_DOWN"]
Pulse_associated_UP = Up_states["Pulse_associated_UP"]
Pulse_associated_DOWN = Up_states["Pulse_associated_DOWN"]
Spon_UP_array = Up_states["Spon_UP_array"]
Spon_UP_peak_alligned_array = Up_states["Spon_UP_peak_alligned_array"]
Spon_Peaks = Up_states["Spon_Peaks"]
UP_Time = Up_states["UP_Time"]
Total_power = Up_states["Total_power"]
UP_start_i = Up_states["UP_start_i"]
up_state_binary  = Up_states["up_state_binary "]



pulse_windows = extract_upstate_windows(Pulse_triggered_UP, LFP_array, dt, window_s=1.0)
spont_windows = extract_upstate_windows(Spontaneous_UP, LFP_array, dt, window_s=1.0)
freqs, spont_mean, pulse_mean, p_vals = compare_spectra(pulse_windows, spont_windows, dt)
plot_contrast_heatmap(pulse_windows, spont_windows, dt)


#Plots
plot_all_channels(NUM_CHANNELS, time_s, LFP_array)
plot_spont_up_mean(
    main_channel,        # Signal
    time_s,              # Zeitvektor
    dt,                  # Sampling-Intervall in Sekunden → HIER FEHLTE ES!
    Spon_Peaks,          # Peak-Positionen
    up_state_binary,     # Array mit 0/1 für UP-Zustände
    pulse_times_1,       # Puls-Typ 1
    pulse_times_2,       # Puls-Typ 2
    Pulse_triggered_UP,  # Triggered UP Start-Indices
    Pulse_triggered_DOWN,# Triggered UP End-Indices
    Spontaneous_UP,       # Spontane UP Start-Indices
    Spontaneous_DOWN
)
#tests(Spect_dat, time_s, Total_power, UP_start_i)

all_ups = np.concatenate((Pulse_triggered_UP, Spontaneous_UP))
all_times = time_s[all_ups]
sort_idx = np.argsort(all_times)
all_ups_sorted = all_ups[sort_idx]
all_downs_sorted = np.concatenate((Pulse_triggered_DOWN, Spontaneous_DOWN))[sort_idx]

N = len(all_ups_sorted)
if N == 0:
    print("⛔ Keine UP-Zustände gefunden – überspringe Amplitudenanalyse.")
else:
    start_idx = 1
    end_idx = N  # oder min(N, 7) etc.

    average_amplitude_in_upstates(
        main_channel=main_channel,
        time_s=time_s,
        UP_start_i=all_ups_sorted,
        DOWN_start_i=all_downs_sorted,
        start_idx=start_idx,
        end_idx=end_idx
    )



#plot_upstate_amplitudes(
#    main_channel=main_channel,
#    UP_start_i=all_ups_sorted,
#    DOWN_start_i=all_downs_sorted,
#    start_idx=3,
#    end_idx=10
#)

#plot_upstate_amplitude_mean(
 #   main_channel=main_channel,
#    UP_start_i=all_ups_sorted,
#    DOWN_start_i=all_downs_sorted,
#    start_idx=10,
#    end_idx=62
#)

#results = plot_upstate_amplitude_blocks_colored(
#    main_channel=main_channel,
#    UP_start_i=all_ups_sorted,
#    DOWN_start_i=all_downs_sorted,
#    index_blocks=[(0, 1)],
#    filename=LFP_FILENAME
#)



compare_spectra(pulse_windows, spont_windows, dt, ignore_start_s=0.3)
Triggered_Peaks = Up_states["Trig_Peaks"]
CSD_spont = Generate_CSD_mean(Up_states["Spon_Peaks"], LFP_array, dt)
CSD_trig = Generate_CSD_mean(Triggered_Peaks, LFP_array, dt)
plot_CSD_comparison(CSD_spont, CSD_trig, dt)



