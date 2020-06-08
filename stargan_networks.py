# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 00:01:08 2020

@author: Georgios
"""

#This file contains the networks used for StarGAN v2
from tensorflow.keras.layers import Input, Concatenate, Reshape
from tensorflow.keras.models import Model

from stargan_modules import *


def G(inp_shape, style_size):
    x = Input(inp_shape)
    s = Input((style_size,))
    output = generator(x, s)
    model = Model(inputs = [x, s], outputs = output, name='Generator')
    return model

def E(inp_shape, style_size, domains):
    x = Input(inp_shape)
    outputs = encoder(x, D=style_size, K=domains)
    model = Model(inputs=x, outputs=outputs, name='Encoder')
    return model

def D(inp_shape, domains):
    x = Input(inp_shape)
    outputs = encoder(x, D=1, K=domains, discriminator_use=True)
    model = Model(inputs=x, outputs=outputs, name='Discriminator')
    return model

def F(latent_size, style_size, domains):
    z = Input((latent_size,))
    s_outputs = mapping_network(z, D=style_size, K=domains) #K output branches - one style code for each domain
    model = Model(inputs=z, outputs = s_outputs, name = 'Mapping Network')
    return model