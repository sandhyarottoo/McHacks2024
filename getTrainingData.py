from pydub import AudioSegment
import os
from createArray import *
import h5py
import pickle


path = os.path.dirname(os.path.realpath(__file__)) + '/trainingdata/'

files = ['J.S. Bach_ Partita for Violin Solo No. 1 in B Minor, BWV 1002 - 6. Double.wav', 'J.S. Bach_ Sonata for Violin Solo No. 1 in G Minor, BWV 1001 - 3. Siciliana.wav', 'J.S. Bach_ Sonata for Violin Solo No. 2 in A Minor, BWV 1003 - 3. Andante.wav',
          'J.S. Bach_ Partita for Violin Solo No. 1 in B Minor, BWV 1002 - 7. Tempo di Bourreé.wav', 'J.S. Bach_ Partita for Violin Solo No. 1 in B Minor, BWV 1002 - 5. Sarabande.wav', 'J.S. Bach_ Partita for Violin Solo No. 1 in B Minor, BWV 1002 - 2. Double.wav', 
          'Hilary Hahn - J.S. Bach_ Partita for Violin Solo No. 1 in B Minor, BWV 1002 - 4. Doubl....wav', 'J.S. Bach_ Sonata for Violin Solo No. 2 in A Minor, BWV 1003 - 1. Grave.wav', 'j-s-bach-sonata-for-violin-solo-no-1-in-g-minor-bwv-1001-1-adagio-audio-(mp3convert.org).wav',
            'J.S. Bach_ Partita for Violin Solo No. 1 in B Minor, BWV 1002 - 8. Double.wav', 'J.S. Bach_ Sonata for Violin Solo No. 2 in A Minor, BWV 1003 - 4. Allegro.wav', 'J.S. Bach_ Partita for Violin Solo No. 1 in B Minor, BWV 1002 - 3. Courante.wav', 'j-s-bach-sonata-for-violin-solo-no-1-in-g-minor-bwv-1001-2-fuga-allegro-(mp3convert.org).wav', 
            'J.S. Bach_ Partita for Violin Solo No. 1 in B Minor, BWV 1002 - 1. Allemande.wav']
files = [path+el for el in files]

dReal,dIm = np.array(len(files)),np.array(len(files))
for i in range(len(files)):
    file = files[i]
    spec = GetFreqSpectrum(file)
    print(spec)
    real = np.real(spec)
    im = np.imag(spec)
    dReal[i] = real
    dIm[i] = im
    
with h5py.File('trainingdata.h5','w') as f:
    f.create_dataset('real',dReal)
    f.create_dataset('im',dIm)

f.close()





