import numpy as np
import tensorflow as tf
import keras
from keras import layers
from utils import FREQUENCIES as FREQUENCIES
from utils import TIMES as TIMES


def make_generator(generator_filters = 1, generated_times = 2, frequency_kernel_size = 5, 
                   time_kernel_size = 2, new_layers = 5):

    inputs = keras.Input(shape=(TIMES, FREQUENCIES))
    outputs = inputs

    for counter in range(0, generated_times, new_layers):

        # run the 1D convs along the TIMES most recent rows 
        frequency_convolution_list = list() 

        for j in range(counter, TIMES + counter):
            # Get a row then reshape it to match Conv1D requirements (shape will be (4000, 1))
            get_row = layers.Lambda(lambda x: x[:, j, :])(outputs)
            reshaped_row = layers.Reshape((FREQUENCIES, 1))(get_row)
            assert reshaped_row.shape[1:] == (FREQUENCIES, 1)

            frequency_convolution_list.append(layers.Conv1D(filters = generator_filters, kernel_size = frequency_kernel_size,
                                                        strides=1,padding = 'same', activation='relu')(reshaped_row))
            
        concatenated_frequencies = layers.Concatenate(axis=1)(frequency_convolution_list)
        concatenated_frequencies = layers.Flatten()(concatenated_frequencies)

        # run the 1D convs along the columns of the TIMES most recent rows
        time_convolution_list = list()

        for j in range(FREQUENCIES):
            get_row = tf.expand_dims(layers.Lambda(lambda x: x[:, counter:TIMES+counter, j])(outputs), axis=-1)
            assert get_row.shape[1:] == (TIMES, 1)

            time_convolution_list.append(layers.Conv1D(filters = generator_filters, kernel_size = time_kernel_size,
                                                        strides=1,padding = 'same', activation='relu')(get_row))

        concatenated_times = layers.Concatenate(axis=1)(time_convolution_list)
        concatenated_times = layers.Flatten()(concatenated_times)

        # get the first vector that will be sent to the dense portion
        vector = layers.Concatenate(axis=1)([concatenated_frequencies, concatenated_times])
        assert vector.shape[1:] == (TIMES*FREQUENCIES*generator_filters*2)

        vector = layers.Dense(units=TIMES*FREQUENCIES*generator_filters, activation='relu')(vector)
        vector = layers.Dense(units=TIMES*FREQUENCIES*generator_filters//2, activation='relu')(vector)
        new_rows = layers.Dense(units=FREQUENCIES*new_layers, activation='relu')(vector)
        new_rows = layers.Reshape((new_layers, FREQUENCIES))(new_rows)

        outputs = layers.Concatenate(axis=1)([outputs, new_rows])
        assert outputs.shape[1:] == (TIMES + (counter+1)*new_layers, FREQUENCIES)
    
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model




        
















































def make_discriminator(initial_filters = 64,number_convolution_layers=2,neurons_per_dense_layer=1024):
    inputs = tf.keras.Input(shape=(TIMES, FREQUENCIES, 1))
    filters = initial_filters
    batch = inputs
    assert(batch.shape[1:] == (TIMES, FREQUENCIES,1))
    for _ in range(number_convolution_layers):
        convolution = layers.Conv2D(filters=filters*2,kernel_size=3,padding='same',activation='relu')(batch)
        avgpool = layers.AveragePooling2D(padding='same')(convolution)
        batch = layers.BatchNormalization()(avgpool)
    assert(batch.shape[1:] == (int(TIMES/(2**number_convolution_layers)),int( FREQUENCIES/(2**number_convolution_layers)),filters*2))
    flatten = layers.Flatten()(batch)
    dense = layers.Dense(units = neurons_per_dense_layer, activation = 'relu')(flatten)
    assert(dense.shape[1:] == (neurons_per_dense_layer))
    return dense

if __name__ == '__main__':
    dense = make_discriminator(initial_filters = 16,number_convolution_layers=2,neurons_per_dense_layer=256)
    # make_generator()