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
min_freq = 0.01
max_freq = 10
frex = np.linspace(min_freq,max_freq,num = num_frex)
t_wav  = np.arange(-2,(2-(1/srate)),(1/srate))
nCycs = np.logspace(np.log10(range_cycles[0]),np.log10(range_cycles[-1]),num = num_frex)
half_wave = (len(t_wav)-1)/2
Win_T = np.arange(-2+dt,2-dt,dt)

def Compute_spectrogram(y,t_win):
		
	# FFT parameters
	nKern = len(t_wav)
	nData = len(y)
	nConv = nKern+nData-1
	# Convert data to frequency domain
	dataX   = np.fft.fft( y ,nConv )
	tf = np.zeros((num_frex,len(y)-1)) #np.zeros((num_frex,len(t_win)-1))
	#tf = np.zeros((num_frex,nConv-1))

	# Loop through each frequency.
	for fi in range(0,num_frex):
		s = nCycs[fi]/(2*np.pi*frex[fi]);
		# Wavelet function	
		wavelet = np.exp(2*complex(0,1)*np.pi*frex[fi]*t_wav) * np.exp(-t_wav**2/(2*s**2))
		# Wavelet function in the frequency domain
		waveletX = np.fft.fft(wavelet,nConv);
		waveR = [wr.real for wr in wavelet]
		waveI = [wi.imag for wi in wavelet]
		#plt.plot(t_wav,waveR)
		#plt.show()

		# Multiply the fourier transform of the wavelet by the fourier transform of the data
		# then compute the inverse fourier transform to convert back to the time domain.
		As = np.fft.ifft(np.multiply(waveletX,dataX),nConv)
		As = As[int(half_wave)+1:-int(half_wave)]
		#print(np.shape(tf[fi,:]))
		#print(np.shape(As))
		tf[fi,:] = abs(As)**2

		# for i in range(1,num_frex):
	   	#	tf[i,:] = 10*np.log10(np.divide(tf[i,:],np.mean(tf[i,:])))
		#	tf[i,:] = 100 * np.divide(np.subtract(tf[i,:],np.mean(tf[i,:])), np.mean(tf[i,:]))
		Spect_Out = [tf, frex, t_win[0:-1]]
	print('Test out: ',np.shape(Spect_Out[0]),np.shape(Spect_Out[2]))
	return Spect_Out


#Spect = spectrogram(Win_EEG_FF[0:-1:25],Win_T[0:-1:25])
def Run_spectrogram(V_Signal,t_signal):
	DownSample = int(0.001/np.diff(t_signal[0:2]))
	print(np.diff(t_signal[0:2]))
	Spect = Compute_spectrogram(V_Signal,t_signal)#Win_T[0:-1:25])
	# There are several outputs for spectrogram, we only want the first Spect[0]
	TF = Spect[0]
	# Normalise power for each frequency band
	for m in range(1,num_frex):
	#	tf[i,:] = 10*np.log10(np.divide(tf[i,:],np.mean(tf[i,:])))
		TF[m,:] = 100 * np.divide(np.subtract(TF[m,:],np.mean(TF[m,:])), np.mean(TF[m,:]))
	
	#plt.contourf(t_signal[0:-1],frex,TF)
	#plt.show()
	# Assign this power/freq matrix (TF) to the 3D matrix Burst_specta 
	Spect_dat = [TF, t_signal[0:-1], frex]
	return Spect_dat

