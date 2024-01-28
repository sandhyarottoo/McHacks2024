import numpy as np
import matplotlib.pyplot as plt
import wave
from scipy.io import wavfile
from scipy.signal import find_peaks
from pydub import AudioSegment
import os
from utils import *

path = os.path.dirname(os.path.realpath(__file__))
file =  path + '/audioTests/test.wav'
cutfile = path + '/audioTests/cutTest2.wav'

def CutTo10Secs(file):
    audio = AudioSegment.from_file(file)
    cutAudio = audio[0.00:10000]
    cutAudio.export(cutfile, format="wav")

#CutTo10Secs(file)

def splitFile(sampleRate,data):
    print(sampleRate)
    timeSampleRate = 1/sampleRate
    print(timeSampleRate)
    npoints = len(data)
    print(npoints)
    #get time in ms
    timeMS= npoints*timeSampleRate*1000
    newRate = timeMS/100
    print(newRate)
    return int(newRate)

def getFFT(data):
    FFT = np.fft.rfft(data)
    mean = np.mean(np.abs(FFT))
    std = np.std(np.abs(FFT))
    threshold = mean-2*std
    filteredFFT = np.where(np.abs(FFT) > threshold, FFT, 0)
    return FFT

def GetFreqSpectrum(filename):
    # Read the WAV file into 
    sampleRate, fulldata = wavfile.read(filename)
    ogFFT = np.fft.rfft(fulldata)
    splitRate = splitFile(sampleRate,fulldata)
    FFTarray = []
    for i in range(0,len(fulldata),splitRate):
        data = fulldata[i:i+splitRate]
        FFT = getFFT(data)

        FFTarray.append(FFT)
    FFTarray = np.array(FFTarray)
    return FFTarray,ogFFT
    # print(fft)
    # freq = np.fft.rfftfreq(len(fft),d = 1/sampleRate)
    # plt.plot(freq[:len(freq)//2],np.abs(fft)[:len(freq)//2])
    # plt.show()

array,ogFFT = GetFreqSpectrum(cutfile)

def reconstruct(array,sampleRate):
    reconstructed = []
    for el in array:
        reconstructed.extend(np.fft.irfft(el))

    print(reconstructed[1])
    # if FFT != ogFFT:
    #     print('OOPS')
    #     print(FFT[1],ogFFT[1])
    #     print(len(FFT),len(ogFFT))
    # binnedFreq = np.fft.rfftfreq(len(FFT),d = 1/sampleRate)


    #peaks,properties = find_peaks(np.abs(cutFilteredFFT),width = 10)

    ogmean = np.mean(np.abs(ogFFT))
    ogstd = np.std(np.abs(ogFFT))
    ogthreshold = ogmean-2*ogstd
    ogfilteredFFT = np.where(np.abs(ogFFT) > ogthreshold, ogFFT, 0)*0.0001

    cleanAudio = np.array(reconstructed)*0.0001
    ogAudio = np.fft.irfft(ogfilteredFFT)

    wavfile.write(path + '/audioTests/testoutput11.wav', sampleRate, cleanAudio)
    wavfile.write(path + '/audioTests/testoutput7.wav', sampleRate, ogAudio)



reconstruct(array,44100)
    
    # #Plot the magnitude spectrum
    # plt.figure(figsize=(10, 4))
    # plt.plot(cutbinnedFreq[:len(cutbinnedFreq)//2], np.abs(cutFilteredFFT)[:len(cutbinnedFreq)//2])
    # #plt.plot(cutbinnedFreq[peaks],cutFilteredFFT[peaks],'x')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Magnitude')
    # plt.title('Frequency Spectrum')
    # plt.grid(True)
    # plt.show()

    