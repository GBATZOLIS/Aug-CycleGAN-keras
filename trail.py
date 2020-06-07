# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 20:23:11 2020

@author: Georgios
"""

import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.layers import Input
#K = tf.keras.backend

#y_curl = tf.squeeze(tf.random.categorical(tf.math.log(0.5*np.ones((1,2))), 1)) 
#y_curl = K.get_value(y_curl)
#print(y_curl)




x = np.random.randint(2)
print(x)
"""

x = Input((10, 16))
out = Dense(100)(x)
out1 = Dense(64)(out)
out2 = Dense(64)(out)
model = Model(inputs = x, outputs=[out1, out2])


print(model.summary())

inp = tf.random.normal((10,16))
output = model(inp)


print(output[1].shape)
"""