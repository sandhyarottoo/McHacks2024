import numpy as np
import matplotlib.pyplot as plt
import wave
from scipy.io import wavfile
from scipy.signal import find_peaks
from pydub import AudioSegment
import os
from utils import *
from filtering import *

path = os.path.dirname(os.path.realpath(__file__))
file =  path + '/audioTests/test.wav'
cutfile = path + '/audioTests/cutTest2.wav'

def CutToXSecs(file,x):
    audio = AudioSegment.from_file(file)
    cutAudio = audio[0.00:x*1000]
    cutAudio.export(cutfile, format="wav")

#CutTo10Secs(file)

def splitFile(sampleRate,data):
    timeSampleRate = 1/sampleRate
    npoints = len(data)
    #get time in ms
    timeMS= npoints*timeSampleRate*1000
    newRate = timeMS//100
    return int(newRate)

def getFFT(data):
    FFT = np.fft.rfft(data)
    FFT = cleanFFT(FFT)
    return FFT

def GetFreqSpectrum(filename):
    # Read the WAV file into 
    sampleRate, fulldata = wavfile.read(filename)
    ogFFT = np.fft.rfft(fulldata)
    splitRate = splitFile(sampleRate,fulldata)
    realFFT,imFFT = [],[]
    for i in range(0,len(fulldata),splitRate):
        data = fulldata[i:i+splitRate]
        FFT = getFFT(data)
        realFFT.append(FFT)

    # realFFT = np.array(realFFT,dtype ='float')
    # print(realFFT[0])
    # imFFT = np.array(imFFT,dtype ='float')
    return np.array(realFFT)





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

    wavfile.write(path + '/audioTests/testoutput12.wav', sampleRate, cleanAudio)
    wavfile.write(path + '/audioTests/testoutput7.wav', sampleRate, ogAudio)



#reconstruct(array,SAMPLERATE)


    