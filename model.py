import numpy as np
import tensorflow as tf
import keras
from keras import layers
from utils import FREQUENCIES as FREQUENCIES
from utils import TIMES as TIMES


def make_generator(generator_filters = 1, generated_times = 100, frequency_kernel_size = 5, time_kernel_size = 2):

    inputs = keras.Input(shape=(TIMES, FREQUENCIES))
    outputs = inputs

    for i in range(generated_times):
        counter = 0

        # run the 1D convs along the TIMES most recent rows 
        frequency_convolution_list = list() 

        for j in range(counter, TIMES):
            # Get a row then reshape it to match Conv1D requirements (shape will be (4000, 1))
            get_row = tf.keras.layers.Lambda(lambda x: x[:, j:j+1, :])(outputs)
            reshaped_row = tf.keras.layers.Reshape((FREQUENCIES, 1))(get_row)
            frequency_convolution_list.append(tf.conv1D(filters = generator_filters, kernel_size = frequency_kernel_size,
                                                        strides=1,padding = 'same', activation='RELU')(reshaped_row))
            
        concatenate_frequencies = tf.keras.layers.Concatenate()(frequency_convolution_list)

        # run the 1D convs along the columns of the TIMES most recent rows
        time_convolution_list = list()

        for j in range(counter, TIMES):
            get_row = tf.keras.layers.Lambda(lambda x: x[:, j:j+1, :])(outputs)
            reshaped_row = tf.keras.layers.Reshape((FREQUENCIES, 1))(get_row)
            frequency_convolution_list.append(tf.conv1D(filters = generator_filters, kernel_size = frequency_kernel_size,
                                                        strides=1,padding = 'same', activation='RELU')(reshaped_row))



        counter += 1














































def make_discriminator(initial_filters = 64,number_convolution_layers=4,neurons_per_dense_layer=1024):
    inputs = tf.keras.Input(shape=(TIMES, FREQUENCIES))
    curr_shape = (TIMES, FREQUENCIES)
    filters = initial_filters
    conv = tf.keras.layers.Conv2D(filters=filters,kernel_size=3,padding='same',activation='relu')(inputs)
    for _ in range(number_convolution_layers-1):
        initial_filters*=2
        conv = tf.keras.layers.Conv2D(filters=filters,kernel_size=3,padding='same',activation='relu')(conv)

if __name__ == '__main__':
    make_discriminator(initial_filters = 64,number_convolution_layers=4,neurons_per_dense_layer=1024)