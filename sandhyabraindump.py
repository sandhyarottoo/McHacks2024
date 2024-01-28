import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import *


#FWHM = 2.35 sigma

testData = np.array([0,0,0,0,2,5,4,3,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,2,4,5,7,8,6,4,3,0,0,0,0,0])

peaks,properties = find_peaks(testData)
peakwidths,_, _, _  = peak_widths(testData,peaks) #FWHM in # of samples


peakstds = [int(x)//2 for x in peakwidths]
print(peaks,peakstds)

peakinfo = dict(zip(peaks,peakstds))

for index,std in peakinfo.items():



plt.plot(testData)
plt.plot(peaks,testData[peaks],'x')
plt.show()



