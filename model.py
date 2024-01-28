import numpy as np
import tensorflow as tf
import keras
from keras import layers
from utils import FREQUENCIES as FREQUENCIES
from utils import TIMES as TIMES

def make_generator(freq_filters = 1, time_filters = 1, generated_times = 2, freq_kernel_size = 5, 
                   time_kernel_size = 2, new_layers = 5):

    inputs = keras.Input(shape=(TIMES, FREQUENCIES))
    outputs = inputs

    # making the lists of conv layers to be reused

    freq_kernels = list()
    for _ in range(TIMES):
        freq_kernels.append(layers.Conv1D(filters = freq_filters, kernel_size = freq_kernel_size,
                                                        strides=1,padding = 'same', activation='relu'))
    
    time_kernels = list()
    for _ in range(int(FREQUENCIES/10)):
        time_kernels.append(layers.Conv1D(filters = time_filters, kernel_size = time_kernel_size,
                                                        strides=1, padding = 'same', activation='relu'))

    for counter in range(0, generated_times, new_layers):

        # run the 1D convs along the TIMES most recent rows 
        freq_conv_output_list = list() 

        for j in range(counter, TIMES + counter):
            # Get a row then reshape it to match Conv1D requirements (shape will be (4000, 1))
            get_row = layers.Lambda(lambda x: x[:, j, :])(outputs)
            reshaped_row = layers.Reshape((FREQUENCIES, 1))(get_row)
            assert reshaped_row.shape[1:] == (FREQUENCIES, 1)

            freq_conv_output_list.append(freq_kernels[j-counter](reshaped_row))
            
        concatenated_frequencies = layers.Concatenate(axis=1)(freq_conv_output_list)
        concatenated_frequencies = layers.Flatten()(concatenated_frequencies)

        # run the 1D convs along the columns of the TIMES most recent rows
        time_conv_output_list = list()

        for j in range(FREQUENCIES):
            get_column = tf.expand_dims(layers.Lambda(lambda x: x[:, counter:TIMES+counter, j])(outputs), axis=-1)
            assert get_column.shape[1:] == (TIMES, 1)

            time_conv_output_list.append(time_kernels[j//10](get_column))

        concatenated_times = layers.Concatenate(axis=1)(time_conv_output_list)
        concatenated_times = layers.Flatten()(concatenated_times)

        # get the first vector that will be sent to the dense portion
        vector = layers.Concatenate(axis=1)([concatenated_frequencies, concatenated_times])
        assert vector.shape[1:] == (TIMES*FREQUENCIES*(freq_filters + time_filters))

        vector = layers.Dense(units=TIMES*FREQUENCIES*(freq_filters + time_filters)//2, activation='relu')(vector)
        vector = layers.Dense(units=TIMES*FREQUENCIES*freq_filters//4, activation='relu')(vector)
        new_rows = layers.Dense(units=FREQUENCIES*new_layers, activation='relu')(vector)
        new_rows = layers.Reshape((new_layers, FREQUENCIES))(new_rows)

        outputs = layers.Concatenate(axis=1)([outputs, new_rows])
        assert outputs.shape[1:] == (TIMES + (counter+1)*new_layers, FREQUENCIES)
    
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


def make_full_generator():

    re_gen = make_generator()
    im_gen = make_generator()

    inputs = keras.Input(shape=(2, TIMES, FREQUENCIES))
    tensor1 = layers.Lambda(lambda x: x[:, 0, :, :])(inputs)
    tensor2 = layers.Lambda(lambda x: x[:, 1, :, :])(inputs)

    real_part = tf.expand_dims(re_gen(tensor1), axis=0)
    im_part = tf.expand_dims(im_gen(tensor2), axis=0)

    combined = layers.Concatenate()([real_part, im_part])

    model = keras.Model(inputs=inputs, outputs=combined)

    return model



















































def make_discriminator(initial_filters = 64,number_convolution_layers=2,neurons_per_dense_layer=1024, number_dense_layers = 3):
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
    for _ in range(number_dense_layers):
        dense = layers.Dense(units = neurons_per_dense_layer, activation = 'relu')(dense)
    assert(dense.shape[1:] == (neurons_per_dense_layer))
    output = layers.Dense(units = 1, activation = None)(dense)
    model = keras.Model(inputs=inputs, outputs=output)
    return model

if __name__ == '__main__':
    critic = make_discriminator(initial_filters = 32,number_convolution_layers=2,neurons_per_dense_layer=256,number_dense_layers=3)
    critic.summary()
    # make_generator()
