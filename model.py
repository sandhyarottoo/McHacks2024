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
    inputs = keras.Input(shape=(TIMES, FREQUENCIES))
    outputs = inputs

    frequency_convolution_list = list() 

    for i in range(TIMES):
        
        get_row = tf.keras.layers.Lambda(lambda x: x[:, i:i+1, :])(input_layer)
        # Reshape row to match Conv1D requirements (shape will be (4000, 1))
        reshape_row = tf.keras.layers.Reshape((input_shape[1], 1))(get_row)

        tf.conv1D(filters = GENERATORFILTERS, kernel_size = FREQUENCYKERNELSIZE,
                                                    strides=1,padding = 'same',activation='RELU')(inputs)

        frequency_convolution_list.append()
        
        
        
        # Create a Conv1D layer for each row
        # Adjust filters, kernel_size, and activation according to your needs
        conv_layer = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='valid', activation='relu')(reshape_row)

    concatenate_frequencies = tf.keras.layers.Concatenate()(frequency_convolution_list)














































def make_discriminator():
    inputs = inputs