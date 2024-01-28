import numpy as np
import tensorflow as tf
import keras
import model 
import utils
import h5py
import os



data = 0
with h5py.File('trainingdata.h5', 'r') as hf:
    print(hf.get('im'))
    print(hf.keys())

def train_GAN(generator, discriminator,gan,epochs=10,batch_size =32,number_steps = 320,times_to_generate_two_rows = 50):
    for _ in range(epochs):
        for _ in range(number_steps):
            true_songs,generated_songs = np.array(batch_size,2,utils.TIMES*2,utils.FREQUENCIES),np.array(batch_size,2,utils.TIMES*2,utils.FREQUENCIES)
            with tf.GradientTape() as tape: #keep track of gradient over all gradient cycles
                for index in range(batch_size):
                    song = np.ones(2,utils.TIMES,utils.FREQUENCIES)
                    start_of_song = np.random.randint(0,len(song)-201) # start at a random point in the song
                    beginning_song = np.expand_dims(song[start_of_song:start_of_song+100],axis = 0) #generator expects batch size,inthis case 1
                    context = np.copy(beginning_song) #context from which the generator builds the new rows
                    generated_song = np.empty(shape=(1,2,utils.TIMES*2,utils.FREQUENCIES))
                    generated_song[:, :, :utils.TIMES, :] = beginning_song
                    for i in range(0, times_to_generate_two_rows, 2):
                        generated_song[:, :,utils.TIMES + i:utils.TIMES + i+2, :] = generator(tf.Tensor(context))
                        context = generated_song[:, :, i:i+utils.TIMES, :]
                    generated_songs[index] = generated_song
                    true_songs[index] = np.expand_dims(song[start_of_song:start_of_song+200],axis=0) #
            discriminator.trainable = True
            critic_positive_loss = discriminator.train_on_batch(true_songs, -np.ones((batch_size,1)))
            critic_negative_loss = discriminator.train_on_batch(generated_songs,np.ones((batch_size,1)))
            discriminator.trainable = False
            gradients = tape.gradients(critic_negative_loss, gan_model.trainable_variables)
            model.OPTIMIZER.apply_gradients(zip(gradients,gan_model.trainable_variables))
    return


if __name__ == '__main__':
    # # continue_training = False

    # generator = model.make_full_generator()
    # gen_checkpoint, gen_checkpoint_manager = utils.setupCheckpoint(generator, keras.optimizers.Adam(), 'test_gen')

    # discriminator = model.make_discriminator()
    # disc_checkpoint, disc_checkpoint_manager = utils.setupCheckpoint(discriminator, keras.optimizers.Adam(), 'test_disc')

    # if continue_training:
    #     utils.restoreCheckpoint(gen_checkpoint, gen_checkpoint_manager)
    #     utils.restoreCheckpoint(disc_checkpoint, disc_checkpoint_manager)

    # assume data has shape (N, 2, timesteps, frequencies)
    critic = model.make_discriminator()
    generator = model.make_full_generator()
    gan_model = model.make_GAN(generator=generator,discriminator=critic)
    train_GAN(generator=generator,discriminator=critic,gan = gan_model)
