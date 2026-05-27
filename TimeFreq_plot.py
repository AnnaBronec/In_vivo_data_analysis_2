import numpy as np
import matplotlib.pyplot as plt
'''
Perform the spectrogram function to get the power/frequency matrix for this burst
Take 4s window and return normalized matrix array giving power freq signal for 2s 

Input: 4s field potential array
Output: 2s time-freq plot
'''


# Fixed parameters 
# Parameters for spectrogram
dt = 5e-3
srate=1/dt
num_frex = 20
range_cycles = [1  ,8]
min_freq = 0.5
max_freq = 10
frex = np.linspace(min_freq,max_freq,num = num_frex)
t_wav  = np.arange(-2,(2-(1/srate)),(1/srate))
nCycs = np.logspace(np.log10(range_cycles[0]),np.log10(range_cycles[-1]),num = num_frex)
half_wave = (len(t_wav)-1)/2
Win_T = np.arange(-2+dt,2-dt,dt)

def Compute_spectrogram(y,t_win):

	nKern = len(t_wav)
	nData = len(y)
	nConv = nKern + nData - 1
	hw    = int(half_wave)

	# FFT of data once
	dataX = np.fft.fft(y, nConv)

	# All wavelets at once: shape (num_frex, nKern)
	s        = nCycs / (2 * np.pi * frex)                              # (num_frex,)
	wavelets = (np.exp(2j * np.pi * frex[:, None] * t_wav[None, :])
	            * np.exp(-t_wav[None, :]**2 / (2 * s[:, None]**2)))   # (num_frex, nKern)

	# Batch FFT → multiply → batch IFFT
	waveletX = np.fft.fft(wavelets, n=nConv, axis=1)                  # (num_frex, nConv)
	As       = np.fft.ifft(waveletX * dataX[None, :], n=nConv, axis=1)# (num_frex, nConv)

	# Trim edges identically to original per-frequency trim
	tf = np.abs(As[:, hw + 1 : nConv - hw]) ** 2                      # (num_frex, nData-1)

	Spect_Out = [tf, frex, t_win[0:-1]]
	print('Test out: ',np.shape(Spect_Out[0]),np.shape(Spect_Out[2]))
	return Spect_Out


#Spect = spectrogram(Win_EEG_FF[0:-1:25],Win_T[0:-1:25])
def Run_spectrogram(V_Signal,t_signal):
	Spect = Compute_spectrogram(V_Signal,t_signal)
	# There are several outputs for spectrogram, we only want the first Spect[0]
	TF = Spect[0]
	# Normalise power for each frequency band (skip row 0, vectorised)
	means   = TF[1:].mean(axis=1, keepdims=True)
	TF[1:] = 100 * (TF[1:] - means) / means
	
	#plt.contourf(t_signal[0:-1],frex,TF)
	#plt.show()
	# Assign this power/freq matrix (TF) to the 3D matrix Burst_specta 
	Spect_dat = [TF, t_signal[0:-1], frex]
	return Spect_dat

