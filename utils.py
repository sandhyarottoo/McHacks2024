import tensorflow as tf
import os

#from C0 to C7, A = 440 Hz https://pages.mtu.edu/~suits/notefreqs.html
notes = [16.35,17.32,18.35,19.45,20.60,21.83,23.12,24.50,25.96,27.50,
         29.14,30.87,32.70,34.65,36.71,38.89,41.20,43.65,46.25,49.00,
         51.91,55.00,58.27,61.74,65.41,69.30,73.42,77.78,82.41,87.31,
         92.50,98.00,103.83,110.00,116.54,123.47,130.81,138.59,146.83,
         155.56,164.81,174.61,185.00,196.00,207.65,220.00,233.08,246.94,
         261.63,277.18,293.66,311.13,329.63,349.23,369.99,392.00,415.30,
         440.0,466.16,493.88,523.25,554.37,587.33,622.25,659.25,698.46,
         739.99,830.61,880.00,932.77,1046.50,1108.73,1174.66,1244.51,
         1318.51,1396.91,1479.98,1567.98,1661.22,1760.00,1864.66,1975.53,
         2093.00]


SAMPLERATE = 44100
FREQUENCIES = 2000
TIMES = 100
SPLITRATE = 30

def setupCheckpoint(model, optimizer, direc):
    current_directory = os.path.dirname(os.path.realpath(__file__))
    checkpoint_directory = os.path.join(current_directory, f'checkpoints/{direc}')
    # creating checkpoint along with manager to save and load
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpointManager = tf.train.CheckpointManager(checkpoint, checkpoint_directory, max_to_keep=4)

    return checkpoint, checkpointManager

def restoreCheckpoint(checkpoint, checkpointManager):
    checkpoint.restore(checkpointManager.latest_checkpoint).expect_partial()
