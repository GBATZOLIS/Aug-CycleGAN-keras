from PIL import Image
from math import floor, log2
import numpy as np
import time
from functools import partial
from random import random
import os

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
import tensorflow as tf
import tensorflow.keras.backend as K

from conv_mod import *

im_size = 256
latent_size = 512
BATCH_SIZE = 16
directory = "Earth"

cha = 24

n_layers = int(log2(im_size) - 1)

mixed_prob = 0.9

def noise(n):
    return np.random.normal(0.0, 1.0, size = [n, latent_size]).astype('float32')

def noiseList(n):
    return [noise(n)] * n_layers

def mixedList(n):
    tt = int(random() * n_layers)
    p1 = [noise(n)] * tt
    p2 = [noise(n)] * (n_layers - tt)
    return p1 + [] + p2

def nImage(n):
    return np.random.uniform(0.0, 1.0, size = [n, im_size, im_size, 1]).astype('float32')


#Loss functions
def gradient_penalty(samples, output, weight):
    gradients = K.gradients(output, samples)[0]
    gradients_sqr = K.square(gradients)
    gradient_penalty = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))

    # (weight / 2) * ||grad||^2
    # Penalize the gradient norm
    return K.mean(gradient_penalty) * weight

def hinge_d(y_true, y_pred):
    return K.mean(K.relu(1.0 + (y_true * y_pred)))

def w_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


#Lambdas
def crop_to_fit(x):

    height = x[1].shape[1]
    width = x[1].shape[2]

    return x[0][:, :height, :width, :]

def upsample(x):
    return K.resize_images(x,2,2,"channels_last",interpolation='bilinear')

def upsample_to_size(x):
    y = im_size // x.shape[2]
    x = K.resize_images(x, y, y, "channels_last",interpolation='bilinear')
    return x


#Blocks


def d_block(inp, fil, p = True):

    res = Conv2D(fil, 1, kernel_initializer = 'he_uniform')(inp)

    out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')(inp)
    out = LeakyReLU(0.2)(out)
    out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')(out)
    out = LeakyReLU(0.2)(out)

    out = add([res, out])

    if p:
        out = AveragePooling2D()(out)

    return out

def to_rgb(inp, style):
    size = inp.shape[2]
    x = Conv2DMod(3, 1, kernel_initializer = VarianceScaling(200/size), demod = False)([inp, style])
    return x

def from_rgb(inp, conc = None):
    fil = int(im_size * 4 / inp.shape[2])
    z = AveragePooling2D()(inp)
    x = Conv2D(fil, 1, kernel_initializer = 'he_uniform')(z)
    if conc is not None:
        x = concatenate([x, conc])
    return x, z


def generator():
        # === Style Mapping ===

        S = Sequential()

        S.add(Dense(512, input_shape = [latent_size]))
        S.add(LeakyReLU(0.2))
        S.add(Dense(512))
        S.add(LeakyReLU(0.2))
        S.add(Dense(512))
        S.add(LeakyReLU(0.2))
        S.add(Dense(512))
        S.add(LeakyReLU(0.2))


        # === Generator ===

        #Inputs
        inp_style = []

        for i in range(n_layers):
            inp_style.append(Input([512]))

        inp_noise = Input([im_size, im_size, 1])

        #Latent
        x = Lambda(lambda x: x[:, :1] * 0 + 1)(inp_style[0])

        outs = []

        #Actual Model
        x = Dense(4*4*4*cha, activation = 'relu', kernel_initializer = 'random_normal')(x)
        x = Reshape([4, 4, 4*cha])(x)

        x, r = g_block(x, inp_style[0], inp_noise, 32 * cha, u = False)  #4
        outs.append(r)

        x, r = g_block(x, inp_style[1], inp_noise, 16 * cha)  #8
        outs.append(r)

        x, r = g_block(x, inp_style[2], inp_noise, 8 * cha)  #16
        outs.append(r)

        x, r = g_block(x, inp_style[3], inp_noise, 6 * cha)  #32
        outs.append(r)

        x, r = g_block(x, inp_style[4], inp_noise, 4 * cha)   #64
        outs.append(r)

        x, r = g_block(x, inp_style[5], inp_noise, 2 * cha)   #128
        outs.append(r)

        x, r = g_block(x, inp_style[6], inp_noise, 1 * cha)   #256
        outs.append(r)

        x = add(outs)

        x = Lambda(lambda y: y/2 + 0.5)(x) #Use values centered around 0, but normalize to [0, 1], providing better initialization

        G = Model(inputs = inp_style + [inp_noise], outputs = x)

        return G



    
#model = generator()
#print(model.summary())

def g_block(inp, istyle, filters):
    init = RandomNormal(stddev=0.02)
    
    rgb_style = Dense(filters, kernel_initializer = init)(istyle)
    
    style1 = Dense(inp.shape[-1], kernel_initializer = init)(istyle)
    out = Conv2DMod(filters = filters, kernel_size = 3, padding = 'same', kernel_initializer = init)([inp, style1])
    out = LeakyReLU(0.2)(out)

    style2 = Dense(filters, kernel_initializer = init)(istyle)
    out = Conv2DMod(filters = filters, kernel_size = 3, padding = 'same', kernel_initializer = init)([out, style2])
    out = LeakyReLU(0.2)(out)
    
    rgb_out = Conv2DMod(3, 1, kernel_initializer = init, demod = False)([out, rgb_style])
    
    return out, rgb_out


x=Input((100,100,3))
latent=Input((10,))


outs = []
out=x

for i in range(4):
    out, rgb_out = g_block(inp=out, istyle=latent, filters=64)
    outs.append(rgb_out)

out_image = add(outs)

model=Model(inputs=[x,latent], outputs=out_image)
print(model.summary())