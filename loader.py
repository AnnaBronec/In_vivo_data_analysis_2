
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

def load_LFP(base_path: str, filename: str):
    col_list = ["timestamps", "timesamples"] + [f"pri_{i}" for i in range(17)] + ["din_1"] + ["din_2"]
    filepath = os.path.join(base_path, filename)
    return pd.read_csv(filepath, usecols=col_list)

def load_lightpulses (base_path: str, filename: str):
    filepath = os.path.join(base_path, filename)
    ds = pd.read_csv(filepath) 
    ds_shifted = ds + time_s[0]
    return ds["Light pulses"].to_numpy()


def load_LFP_from_CSC_csv(base_path, filename):

    filepath = os.path.join(base_path, filename)
    df = pd.read_csv(filepath)

    # Alle Value-Spalten extrahieren
    value_cols = [col for col in df.columns if col.endswith("_values")]
    ts_cols = [col for col in df.columns if col.endswith("_timestamps")]

    # Stelle sicher, dass alle Kanäle die gleiche Länge haben
    num_samples = len(df)
    num_channels = len(value_cols)

    LFP_array = np.zeros((num_channels, num_samples))
    for i, col in enumerate(value_cols):
        LFP_array[i, :] = df[col].values

    # Nutze die Zeitstempel der ersten Channel-Spalte
    time_raw = df[ts_cols[0]].values
    time_s = (time_raw - time_raw[0]) / 1e6  # Annahme: Zeit in Mikrosekunden
    channel_names = value_cols

    return time_s, LFP_array, channel_names


