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



def CSD_calc(V_Array,dt):
    Depth = np.arange(0,1400,100)
    sigma = 0.3 #S/m
    Time = np.arange(-0.5,0.5-dt,dt)
    CSD = np.zeros((len(Depth),len(Time)))
    srate = 1 / dt
    nyq = 0.5 * srate

    # Create lowpass filter
    high_cutoff = 30
    high = high_cutoff / nyq
    b_lp, a_lp = signal.butter(5, high, btype='low', analog=False)

    # Highpass filter
    low_cutoff = 0.5
    low = low_cutoff / nyq
    b_hp, a_hp = signal.butter(5, low, btype='high', analog=False)

    for j in range(len(Depth)):
        V_Array[j,:] = signal.filtfilt(b_lp, a_lp, V_Array[j,:])
        V_Array[j,:] = signal.filtfilt(b_hp, a_hp, V_Array[j,:])


    for i in range(1,len(Depth)-1):
        for ii in range(1, len(Time) -1):
            #CSD[i-1,ii] = sigma*(2*(V_Array[i,ii])-(V_Array[i+1,ii])-(V_Array[i-1,ii]))/((0.000001)**2)
            CSD[i-1,ii] = (-2*(V_Array[i,ii])+(V_Array[i+1,ii])+(V_Array[i-1,ii]))/((0.000001)**2)

    return CSD