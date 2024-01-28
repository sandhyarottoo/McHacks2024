import numpy as np
import matplotlib.pyplot as plt
import wave
from scipy.io import wavfile
from scipy.signal import find_peaks
from pydub import AudioSegment
import os

path = os.path.dirname(os.path.realpath(__file__))
file =  path + '/audioTests/test.wav'
cutfile = path + '/audioTests/cutTest2.wav'


def cleanFFT(FFT):
    return FFT


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



