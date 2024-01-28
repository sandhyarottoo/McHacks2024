import numpy as np
import tensorflow as tf
import keras
from keras import layers
from utils import FREQUENCIES as FREQUENCIES
from utils import TIMES as TIMES
OPTIMIZER = keras.optimizers.RMSprop(learning_rate=0.00005)
def w_loss(true_labels, predicted_labels):
    return keras.backed(true_labels* predicted_labels)

TIMES = 10


def make_generator(freq_filters = 1, time_filters = 1, generated_times = 2, freq_kernel_size = 5, 
                   time_kernel_size = 2, new_layers = 2):
    TIMES = 10
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

    row_reshaper = layers.Reshape((FREQUENCIES, 1))
    final_reshaper = layers.Reshape((new_layers, FREQUENCIES))
    flattener = layers.Flatten()
    ax1_concat = layers.Concatenate(axis=1)

    # run the 1D convs along the TIMES most recent rows 
    freq_conv_output_list = list() 

    for j in range(0, TIMES):
        # Get a row then reshape it to match Conv1D requirements (shape will be (4000, 1))
        get_row = outputs[:, j, :]
        reshaped_row = row_reshaper(get_row)
        assert reshaped_row.shape[1:] == (FREQUENCIES, 1)

        freq_conv_output_list.append(freq_kernels[j](reshaped_row))
        
    concatenated_frequencies = ax1_concat(freq_conv_output_list)
    concatenated_frequencies = flattener(concatenated_frequencies)

    # run the 1D convs along the columns of the TIMES most recent rows
    time_conv_output_list = list()

    for j in range(FREQUENCIES):
        get_column = tf.expand_dims(outputs[:, 0:TIMES, j], axis=-1)
        assert get_column.shape[1:] == (TIMES, 1)

        time_conv_output_list.append(time_kernels[j//10](get_column))

    concatenated_times = ax1_concat(time_conv_output_list)
    concatenated_times = flattener(concatenated_times)

    # get the first vector that will be sent to the dense portion
    vector = ax1_concat([concatenated_frequencies, concatenated_times])
    
    assert vector.shape[1:] == (TIMES*FREQUENCIES*(freq_filters + time_filters))

    vector = layers.Dense(units=800, activation='relu')(vector)
    vector = layers.Dense(units=600, activation='relu')(vector)
    new_rows = layers.Dense(units=FREQUENCIES*new_layers, activation='relu')(vector)
    new_rows = final_reshaper(new_rows)
    
    model = keras.Model(inputs=inputs, outputs=new_rows)

    return model


def make_full_generator():

    re_gen = make_generator()
    im_gen = make_generator()

    inputs = keras.Input(shape=(2, TIMES, FREQUENCIES))
    tensor1 = layers.Lambda(lambda x: x[:, 0, :, :])(inputs)
    tensor2 = layers.Lambda(lambda x: x[:, 1, :, :])(inputs)

    real_part = tf.expand_dims(re_gen(tensor1), axis=0)
    im_part = tf.expand_dims(im_gen(tensor2), axis=0)

    combined = layers.Concatenate(axis=1)([real_part, im_part])

    model = keras.Model(inputs=inputs, outputs=combined)

    return model

def make_GAN(generator, discriminator):
    discriminator.trainable = False
    model = keras.models.Sequential([generator, discriminator])
    model.compile(loss=w_loss, optimizer=OPTIMIZER)
    return model



def make_discriminator(initial_filters = 32,number_convolution_layers=2,neurons_per_dense_layer=128, number_dense_layers = 3):
    inputs = tf.keras.Input(shape=(2, 2*TIMES, FREQUENCIES))
    filters = initial_filters
    batch = inputs
    for _ in range(number_convolution_layers):
        convolution = layers.Conv2D(filters=filters,kernel_size=3,padding='same',activation='relu',data_format='channels_first')(batch)
        avgpool = layers.AveragePooling2D(padding='same',data_format='channels_first')(convolution)
        batch = layers.BatchNormalization()(avgpool)
    flatten = layers.Flatten()(batch)
    dense = layers.Dense(units = neurons_per_dense_layer, activation = 'relu')(flatten)
    for _ in range(number_dense_layers):
        dense = layers.Dense(units = neurons_per_dense_layer, activation = 'relu')(dense)
    output = layers.Dense(units = 1, activation = None)(dense)
    model = keras.Model(inputs=inputs, outputs=output)
    model.compile(loss=w_loss,optimizer=OPTIMIZER)
    return model

if __name__ == '__main__':
    critic = make_discriminator(initial_filters = 32,number_convolution_layers=2,neurons_per_dense_layer=128,number_dense_layers=3)
    generator = make_generator()
    gan = make_GAN(generator=generator,discriminator=critic)
