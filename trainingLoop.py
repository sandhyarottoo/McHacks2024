import numpy as np
import tensorflow as tf
import keras
import model 

re_generator = model.make_generator()
im_generator = model.make_generator()

discriminator = model.make_discriminator()