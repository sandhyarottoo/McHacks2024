import numpy as np
import tensorflow as tf
import keras
from keras import layers
from utils import FREQUENCIES as FREQUENCIES
from utils import TIMES as TIMES
GENERATEDTIMES = 100
FREQUENCYKERNELSIZE = 5
TIMEKERNELSIZE = 2
GENERATORFILTERS = 1

def make_generator():
    inputs = keras.Input(shape=(FREQUENCIES,TIMES))
    frequency_convolution_list = list() 
    for _ in range(TIMES):
        frequency_convolution_list.append(tf.conv1D(filters = GENERATORFILTERS, kernel_size = FREQUENCYKERNELSIZE,strides=1,padding = 'same',activation='RELU')(inputs))
    concatenate_frequencies = tf.keras.layers.Concatenate()(frequency_convolution_list)














































def make_discriminator():
    inputs = inputs