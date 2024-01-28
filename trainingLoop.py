import numpy as np
import tensorflow as tf
import keras
import model 
import utils
import os

continue_training = False

generator = model.make_full_generator()
gen_checkpoint, gen_checkpoint_manager = utils.setupCheckpoint(generator, keras.optimizers.Adam(), 'test_gen')

discriminator = model.make_discriminator()
disc_checkpoint, disc_checkpoint_manager = utils.setupCheckpoint(discriminator, keras.optimizers.Adam(), 'test_disc')

if continue_training:
    utils.restoreCheckpoint(gen_checkpoint, gen_checkpoint_manager)
    utils.restoreCheckpoint(disc_checkpoint, disc_checkpoint_manager)

# assume data has shape (N, 2, timesteps, frequencies)

data = 0

def train_GAN(generator, discriminator,gan,epochs=10,batch_number =32,number_steps = 320):
    return