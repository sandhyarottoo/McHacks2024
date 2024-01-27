import numpy as np
import matplotlib.pyplot as plt
import wave
from scipy.io import wavfile
from pydub import AudioSegment
import os

path = os.path.dirname(os.path.realpath(__file__))
file =  path + '/audioTests/test.wav'
cutfile = path + '/audioTests/cutTest.wav'

def CutTo10Secs(file):
    audio = AudioSegment.from_file(file)
    cutAudio = audio[0.00:3000]
    cutAudio.export(cutfile, format="wav")

CutTo10Secs(file)

def splitFile(sampleRate,data):
    print(sampleRate)
    timeSampleRate = 1/sampleRate
    print(timeSampleRate)
    npoints = len(data)
    print(npoints)
    #get time in ms
    timeMS= npoints*timeSampleRate*1000
    newRate = timeMS/10
    print(newRate)
    return int(newRate)


def GetFreqSpectrum(filename):
    # Read the WAV file into 
    sampleRate, fulldata = wavfile.read(filename)
    splitRate = splitFile(sampleRate,fulldata)
    freqlist = []
    for i in range(0,len(fulldata),splitRate):
        data = fulldata[i:i+splitRate]
        FFT = np.fft.rfft(data)
    
        # Calculate the frequency bins
        binnedFreq = np.fft.rfftfreq(len(FFT), d=1/splitRate)

        freqlist.append(binnedFreq[:len(binnedFreq)//2])
    return None



def cleanFFT(FFT,binnedFreq):
    return None

def plotFFT(filename):


    sampleRate, data = wavfile.read(filename)
    # Compute the rfft (real fft)
    FFT = np.fft.rfft(data)

    # Compute the mean and standard deviation of the FFT coefficients
    mean_fft = np.mean(np.abs(FFT))
    std_fft = np.std(np.abs(FFT))

    # Threshold for noise removal (adjust as needed)
    threshold = mean_fft + 2 * std_fft  # Example threshold: 3 standard deviations above the mean

    # Remove noise by zeroing out FFT coefficients below the threshold
    FFT_filtered = np.where(np.abs(FFT) > threshold, FFT, 0)
    
    # Calculate the frequency bins
    binnedFreq = np.fft.rfftfreq(len(FFT), d=1/sampleRate)

    cleanAudio = np.fft.irfft(FFT_filtered)

    wavfile.write(path + '/audioTests/output_audio.wav', sampleRate, cleanAudio)


    
    # Plot the magnitude spectrum
    # plt.figure(figsize=(10, 4))
    # plt.plot(binnedFreq[:len(binnedFreq)//2], np.abs(FFT_filtered)[:len(binnedFreq)//2])
    # plt.xlim(0,5000)
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Magnitude')
    # plt.title('Frequency Spectrum')
    # plt.grid(True)
    # plt.show()


plotFFT(cutfile)

#GetFreqSpectrum(cutfile)



