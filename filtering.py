import numpy as np
import matplotlib.pyplot as plt
import wave
from scipy.io import wavfile
from scipy.signal import find_peaks
from scipy.signal import peak_widths
from pydub import AudioSegment
import os
from utils import *
from scipy.optimize import curve_fit

path = os.path.dirname(os.path.realpath(__file__))
file =  path + '/audioTests/test.wav'
cutfile = path + '/audioTests/cutTest2.wav'

def cleanFFT(FFT):
    mean = np.mean(np.abs(FFT))
    std = np.std(np.abs(FFT))
    threshold = mean-3*std
    filteredFFT = np.where(np.abs(FFT) > threshold, FFT, 0)
    freqs = np.fft.rfftfreq(len(FFT), d=1/SPLITRATE)

    #cutting off high notes
    for i in range(len(freqs)):
        if freqs[i] < -2000 or freqs[i] > 2000:
            freqs.pop(i)
            filteredFFT.pop(i)
    
    return filteredFFT


#function for indexing elements in a numpy array
def ind(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx[0]

def tune(FFT,samplerate):
    '''
    "autotune" - basically matches every peak freq to the nearest 'real' note from utils.py
    and shifts the peak to that note
    
    
    '''

    #get the real part of the FFT
    reFFT = np.real(FFT)
    freqs = np.fft.rfftfreq(len(FFT),d=1/samplerate)

    #find peaks and peak widths
    peaks,properties = find_peaks(reFFT)
    peakwidths,_, _, _  = peak_widths(reFFT,peaks) #FWHM in # of samples

    #get # of samples that are included in this peak
    peakstds = [int(x)//2 for x in peakwidths]

    peakinfo = dict(zip(peaks,peakstds))

    for index,std in peakinfo.items():
        try:
            intensitySlice = reFFT[index-std:index+std+1]
            freqSlice = freqs[index-std:index+std+1]
            peakIntensity = reFFT[index]
            peakFreq = freqs[index]

            #find the absolute frequency which is closest to the peak freq
            newFreq = min(notes, key=lambda x: abs(x - peakFreq))
            #now do this again but to find the closest value in the freq list itself
            newFreqinFreqs = min(freqs,key = lambda x: abs(x-newFreq))
            newIndex = ind(freqs,newFreqinFreqs)
            #shift the intensity peak to the new peak freq
            reFFT[index-std:index+std+1] = [0]*len(intensitySlice)
            reFFT[newIndex-std:newIndex+std+1] = intensitySlice
        except:
            continue

    #replace the real part of the fft with the tuned part
    FFT.real = reFFT

    return FFT


# tunedFFT = tune(fft,sampleRate)

# # Compute the mean and standard deviation of the FFT coefficients
# mean_fft = np.mean(np.abs(tunedFFT))
# std_fft = np.std(np.abs(tunedFFT))

# # Threshold for noise removal (adjust as needed)
# threshold = mean_fft + 2 * std_fft  # Example threshold: 3 standard deviations above the mean

# filteredFFT = np.where(np.abs(tunedFFT) > threshold, tunedFFT, 0)*0.0001 #factor to stop it from sounding insane


# cleanAudio = np.fft.irfft(filteredFFT)

# wavfile.write(path + '/audioTests/tunedaudio1.wav', sampleRate, cleanAudio)

def plotFFT(filename):


    sampleRate, data = wavfile.read(filename)
    # Compute the rfft (real fft)
    FFT = np.fft.rfft(data,n = len(data))

    # Compute the mean and standard deviation of the FFT coefficients
    mean_fft = np.mean(np.abs(FFT))
    std_fft = np.std(np.abs(FFT))

    # Threshold for noise removal (adjust as needed)
    threshold = mean_fft + 2 * std_fft  # Example threshold: 3 standard deviations above the mean

    # Calculate the frequency bins

    binnedFreq = np.fft.rfftfreq(len(FFT),d = 1/sampleRate)

    posbinnedFreq = binnedFreq[:len(binnedFreq)//2]
    posFFT = FFT[:len(binnedFreq)//2]


    cutbinnedFreq = [freq for freq in posbinnedFreq if np.abs(freq) < 2000]
    cutFFT = posFFT[:len(cutbinnedFreq)]
    print(cutbinnedFreq)

    # Remove noise by zeroing out FFT coefficients below the threshold
    filteredFFT = np.where(np.abs(cutFFT) > threshold, cutFFT, 0)*0.0001 #factor to stop it from sounding insane

    # print(binnedFreq)
    # peaks,properties = find_peaks(np.abs(cutFilteredFFT),width = 10)

    cleanAudio = np.fft.irfft(posFFT)

    wavfile.write(path + '/audioTests/outputAudio13.wav', sampleRate, cleanAudio)


    #Plot the magnitude spectrum
    plt.figure(figsize=(10, 4))
    plt.plot(cutbinnedFreq,np.abs(filteredFFT))
    #plt.plot(cutbinnedFreq[peaks],cutFilteredFFT[peaks],'x')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Frequency Spectrum')
    plt.grid(True)
    plt.show()


#plotFFT(cutfile)

#GetFreqSpectrum(cutfile)





def fitGaussian(freqs,intensities,std,peakIntensity,peakFreq):
    def gauss(x,C,mean,sigma):
        return C*np.exp(-(x-mean)**2/(2*sigma**2))

    guess = [peakIntensity,peakFreq,2]

    fit = curve_fit()