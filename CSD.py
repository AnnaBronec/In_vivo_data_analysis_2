'''
Current source density equation

CSD(x,t) = sigma*(2*(x,t)-(x+deltax,t)-(x-deltax,t))/((deltax)^2)

Where:

sigma = extracellular conductivity (default = 0.3S/m)
deltax = distance between neighbouring electrodes

so 

(x,t) gives the Voltage (LFP) at a given electrode (x), at a given timestep (t), and x+deltax and x-deltax
are the voltages at the electrode before and electrode after x

So construct an array that is x total by t total (i.e. 14,10000 for 16 electrodes and 10s of data sampled at 1Khz 
because you need plus and minus one electrode so the first and last electrode are discarded)
 
'''
import numpy as np
from scipy import signal


import numpy as np
from scipy import signal
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter

def CSD_calc(V_Array, dt,
             dz_um=100.0,          # Elektrodenabstand in µm
             hp_hz=0.5, lp_hz=30,  # zeitliche Bandbegrenzung pro Kanal
             sg_win=None, sg_poly=3,  # Savitzky–Golay entlang Kanälen (ungerades Fenster)
             smooth_ch=0.8, smooth_t=0.4,  # optionale 2D-Glättung (Kanäle, Zeit)
             sigma_S_per_m=0.3,    # Leitfähigkeit σ (S/m) für physikalische Skalierung
             detrend='mean'        # 'mean' oder None
            ):
    """
    V_Array: 2D ndarray [channels x time]
    dt:      Sekunden pro Sample
    Rückgabe: CSD [channels x time] (gleiche Kanalzahl wie Input)
    """
    V = np.asarray(V_Array, dtype=float)  # kopie, nicht in-place
    n_ch, n_t = V.shape
    Fs = 1.0 / dt
    nyq = 0.5 * Fs

    # 0) Detrend pro Kanal (vor Filtern)
    if detrend == 'mean':
        V = V - np.mean(V, axis=1, keepdims=True)

    # 1) zeitliche Filter (sicher mit SOS)
    if lp_hz is not None and lp_hz < nyq:
        sos_lp = signal.butter(5, lp_hz/nyq, btype='low', output='sos')
        V = signal.sosfiltfilt(sos_lp, V, axis=1)
    if hp_hz is not None and hp_hz > 0:
        sos_hp = signal.butter(5, hp_hz/nyq, btype='high', output='sos')
        V = signal.sosfiltfilt(sos_hp, V, axis=1)

    # 2) Glättung entlang der Kanalachse (Savitzky–Golay)
    if sg_win is None:
        # sinnvolles Standardfenster relativ zur Kanalzahl (ungerade!)
        sg_win = max(5, (n_ch // 4) * 2 + 1)  # ~25% der Kanäle, mindestens 5
        sg_win = min(sg_win, n_ch - (1 - n_ch % 2))  # nicht > n_ch, ungerade
        if sg_win % 2 == 0: sg_win = max(5, sg_win - 1)
    if sg_win >= 5 and sg_win <= n_ch and sg_win % 2 == 1:
        V_smooth = savgol_filter(V, window_length=sg_win, polyorder=sg_poly, axis=0, mode='interp')
    else:
        V_smooth = V

    # 3) zweite räuml. Ableitung ∂²V/∂z² (z entlang Kanalachse)
    dz = dz_um * 1e-6  # µm -> m
    second_diff = np.diff(V_smooth, n=2, axis=0) / (dz**2)

    # auf gleiche Kanalzahl bringen (Pad mit Randwerten)
    csd = -sigma_S_per_m * np.pad(second_diff, ((1,1),(0,0)), mode='edge')

    # 4) optionale 2D-Glättung (Kanäle x Zeit), kleine Sigmas gegen "Fleckigkeit"
    if (smooth_ch and smooth_ch > 0) or (smooth_t and smooth_t > 0):
        csd = gaussian_filter(csd, sigma=(smooth_ch, smooth_t), mode='nearest')

    return csd
