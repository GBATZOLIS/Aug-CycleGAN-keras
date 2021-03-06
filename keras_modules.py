# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 18:04:27 2020

@author: Georgios
"""

from tensorflow.keras.layers import add, ZeroPadding2D, Reshape, Conv2D, Add, LeakyReLU, Activation, Input,DepthwiseConv2D, Dense, Lambda, BatchNormalization, Conv2DTranspose
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.python.keras.engine.network import Network
from tensorflow.keras.optimizers import Adam


from keras_custom_layers import AdaInstanceNormalization, InstanceNormalization

import tensorflow as tf

from conv_mod import *
import scipy.stats as st
import numpy as np

    
    
"""General useful modules"""
#-----------------------------------------------------------------------------------
# Extending the ZeroPadding2D layer to do reflection padding instead.
class ReflectionPadding2D(ZeroPadding2D):
    def call(self, x, mask=None):
        pattern = [[0, 0],
                   [self.top_pad, self.bottom_pad],
                   [self.left_pad, self.right_pad],
                   [0, 0]]
        return tf.pad(x, pattern, mode='REFLECT')
#------------------------------------------------------------------------------------- 
        
"""Modules for stochastic image generators: G_AB and G_BA"""
#----------------------------------------------------------------------------------
     
#I need to create a conditional instance normalisation layer.
#To get inspiration on how to do it properly look at the batch normalisation source code. 

            
def CondInstanceNorm(image, noise, x_dim, z_dim):
    init = RandomNormal(stddev=0.02)
    
    
    shift_conv = Conv2D(filters = x_dim, kernel_size=1, padding='same', kernel_initializer = init)(noise)
    shift_conv = LeakyReLU(alpha=0.2)(shift_conv)

    scale_conv = Conv2D(filters = x_dim, kernel_size=1, padding='same', kernel_initializer = init)(noise)
    scale_conv = LeakyReLU(alpha=0.2)(scale_conv)

    image = AdaInstanceNormalization()([image, shift_conv, scale_conv])
    
    return image
    

def CINResnetBlock(image, noise, filters):  
    def conv_block(image, noise, filters):
        init = RandomNormal(stddev=0.02)
        
        style = Dense(image.shape[-1], kernel_initializer = init)(noise)
        image = Conv2DMod(filters = filters, kernel_size = 3, padding = 'same', kernel_initializer = init)([image, style])
        image = LeakyReLU(alpha=0.2)(image)
        
        return image
    
    out_image = conv_block(image, noise, filters)
    out_image = Add()([out_image, image])
    out_image = LeakyReLU(alpha=0.2)(out_image)
    
    return out_image

def CINResnetGenerator(image, noise, ngf, nlatent):
    #ngf: number of generator filters
    #nlatent: the dimensionality of the latent space
    
    """
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
    
    latent = Reshape((nlatent,))(noise)
    outs = []

    out=image
    
    for i in range(4):
        out, rgb_out = g_block(inp=out, istyle=latent, filters=64)
        outs.append(rgb_out)
    
    out_image = add(outs)
    """
    
    init = RandomNormal(stddev=0.02)
    
    noise = Reshape((nlatent,))(noise)
    image = Lambda(lambda x: 2*x - 1, output_shape=lambda x:x)(image)
    
    image = Conv2D(filters = ngf, kernel_size=7, padding='same', kernel_initializer = init)(image)
    image = LeakyReLU(alpha=0.2)(image)
    
    image = Conv2D(filters = ngf, kernel_size=3, padding='same', kernel_initializer = init)(image)
    image = LeakyReLU(alpha=0.2)(image)
    
    image = Conv2D(filters = 2*ngf, kernel_size=3, padding='same', kernel_initializer = init)(image)
    image = LeakyReLU(alpha=0.2)(image)
    
    for i in range(3):
        image = CINResnetBlock(image, noise, 2*ngf)

    style4 = Dense(image.shape[-1], kernel_initializer = init)(noise)
    image = Conv2DMod(filters = 2*ngf, kernel_size = 3, padding = 'same', kernel_initializer = init)([image, style4])
    image = LeakyReLU(alpha=0.2)(image)
    
    style5 = Dense(image.shape[-1], kernel_initializer = init)(noise)
    image = Conv2DMod(filters = ngf, kernel_size = 3, padding = 'same', kernel_initializer = init)([image, style5])
    image = LeakyReLU(alpha=0.2)(image)
       
    image = Conv2D(filters = 3, kernel_size=7, padding='same', kernel_initializer = init)(image)
    image = Activation('tanh')(image)
    image = Lambda(lambda x: 0.5*x + 0.5, output_shape=lambda x:x)(image)
    
    return image
    
#--------------------------------------------------------------------------------------------------------------------   


"""Modules for encoders: E_A and E_B"""

def LatentEncoder(concat_A_B, nef, z_dim):
    init = RandomNormal(stddev=0.02)
    
    concat_A_B = Conv2D(filters=nef, kernel_size=3, strides=2, padding='same', kernel_initializer = init)(concat_A_B)
    concat_A_B = LeakyReLU(alpha=0.2)(concat_A_B)
    
    concat_A_B = Conv2D(filters=2*nef, kernel_size=3, strides=2, padding='same', kernel_initializer = init)(concat_A_B)
    concat_A_B = BatchNormalization(axis=-1)(concat_A_B)
    concat_A_B = LeakyReLU(alpha=0.2)(concat_A_B)
    
    concat_A_B = Conv2D(filters=4*nef, kernel_size=3, strides=2, padding='same', kernel_initializer = init)(concat_A_B)
    concat_A_B = BatchNormalization(axis=-1)(concat_A_B)
    concat_A_B = LeakyReLU(alpha=0.2)(concat_A_B)
    
    concat_A_B = Conv2D(filters=8*nef, kernel_size=3, strides=2, padding='same', kernel_initializer = init)(concat_A_B)
    concat_A_B = BatchNormalization(axis=-1)(concat_A_B)
    concat_A_B = LeakyReLU(alpha=0.2)(concat_A_B)
    
    concat_A_B = Conv2D(filters=8*nef, kernel_size=3, strides=2, padding='same', kernel_initializer = init)(concat_A_B)
    concat_A_B = BatchNormalization(axis=-1)(concat_A_B)
    concat_A_B = LeakyReLU(alpha=0.2)(concat_A_B)
    
    concat_A_B = Conv2D(filters=8*nef, kernel_size=3, strides=2, padding='same', kernel_initializer = init)(concat_A_B)
    concat_A_B = BatchNormalization(axis=-1)(concat_A_B)
    concat_A_B = LeakyReLU(alpha=0.2)(concat_A_B)
    
    concat_A_B = Conv2D(filters=8*nef, kernel_size=3, strides=2, padding='same', kernel_initializer = init)(concat_A_B)
    concat_A_B = BatchNormalization(axis=-1)(concat_A_B)
    concat_A_B = LeakyReLU(alpha=0.2)(concat_A_B)
    
    """
    concat_A_B = Conv2D(filters=8*nef, kernel_size=3, strides=1, padding='same', kernel_initializer = init)(concat_A_B)
    concat_A_B = BatchNormalization(axis=-1)(concat_A_B)
    concat_A_B = LeakyReLU(alpha=0.2)(concat_A_B)
    """

    encoding = Conv2D(filters=z_dim, kernel_size=1, strides=1, padding='valid', kernel_initializer = init)(concat_A_B)
    
    return encoding

#----------------------------------------------------------------------------------
"""Discriminator Modules for domains A and B"""

def img_domain_critic(img, ndf=64):
    init = RandomNormal(stddev=0.02)
    
    kw=4
    img = Conv2D(filters=ndf, kernel_size=kw, strides=2, padding='same', kernel_initializer = init)(img)
    img = LeakyReLU(alpha=0.2)(img)
    
    img = Conv2D(filters=2*ndf, kernel_size=kw, strides=2, padding='same', kernel_initializer = init)(img)
    img = BatchNormalization(axis=-1)(img)
    img = LeakyReLU(alpha=0.2)(img)
    
    img = Conv2D(filters=4*ndf, kernel_size=kw, strides=1, padding='same', kernel_initializer = init)(img)
    img = BatchNormalization(axis=-1)(img)
    img = LeakyReLU(alpha=0.2)(img)
    
    img = Conv2D(filters=4*ndf, kernel_size=kw, strides=1, padding='same', kernel_initializer = init)(img)
    img = BatchNormalization(axis=-1)(img)
    img = LeakyReLU(alpha=0.2)(img)
    
    img = Conv2D(filters=1, kernel_size=kw, strides=1, padding='same', kernel_initializer = init)(img)
    
    return img

def noise_domain_critic(noise, ndf=64):
    init = RandomNormal(stddev=0.02)
    
    noise = Dense(units = ndf, kernel_initializer=init)(noise)
    noise = BatchNormalization(axis = -1)(noise)
    noise = LeakyReLU(alpha=0.2)(noise)
    
    noise = Dense(units = ndf, kernel_initializer=init)(noise)
    noise = BatchNormalization(axis = -1)(noise)
    noise = LeakyReLU(alpha=0.2)(noise)
    
    noise = Dense(units = ndf, kernel_initializer=init)(noise)
    noise = BatchNormalization(axis = -1)(noise)
    noise = LeakyReLU(alpha=0.2)(noise)
    
    noise = Dense(units = 1)(noise)
    return noise

def noise_mapping_func(noise, nlatent):
    init = RandomNormal(stddev=0.02)
    
    noise = Reshape((nlatent,))(noise)
    
    noise = Dense(units = nlatent, kernel_initializer=init)(noise)
    noise = LeakyReLU(alpha=0.2)(noise)
    noise = Dense(units = nlatent, kernel_initializer=init)(noise)
    noise = LeakyReLU(alpha=0.2)(noise)
    noise = Dense(units = nlatent, kernel_initializer=init)(noise)
    noise = LeakyReLU(alpha=0.2)(noise)
    noise = Dense(units = nlatent, kernel_initializer=init)(noise)
    noise = LeakyReLU(alpha=0.2)(noise)
    
    noise = Reshape((1, 1, nlatent))(noise)
    
    return noise
    

def blur(img_shape):
    def gauss_kernel(kernlen=21, nsig=3, channels=3):
        interval = (2*nsig+1.)/(kernlen)
        x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw/kernel_raw.sum()
        out_filter = np.array(kernel, dtype = np.float32)
        out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
        out_filter = np.repeat(out_filter, channels, axis = 2)
        return out_filter
    
    kernel_size=21
    blur_kernel_weights = gauss_kernel()
    
    image = Input(img_shape)
    g_layer = DepthwiseConv2D(kernel_size, use_bias=False, padding='same')
    image_processed = g_layer(image)
    
    g_layer.set_weights([blur_kernel_weights])
    g_layer.trainable = False
    model = Model(inputs = image, outputs = image_processed, name='blur')
    model.compile(loss='mse',  optimizer=Adam(lr=0.0002, beta_1=0.5))
    return model
    
    
    


    