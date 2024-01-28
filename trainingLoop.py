import numpy as np
import tensorflow as tf
import keras
import model 
import utils
import os

re_generator = model.make_generator()
re_checkpoint, re_checkpoint_manager = utils.setupCheckpoint()
im_generator = model.make_generator()
discriminator = model.make_discriminator()

