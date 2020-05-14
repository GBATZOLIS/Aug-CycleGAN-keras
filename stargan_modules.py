# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:29:13 2020

@author: Georgios
"""

#StarGAN modules

from tensorflow.keras.layers import *
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.models import Model
from custom_layers import *
from conv_mod import *
import math

def res_block(image, dim_in, dim_out, normalize=False, downsample=False):
    learned_sc = dim_in != dim_out
    
    def shortcut(x):
        if learned_sc:
            x = Conv2D(filters=dim_out, kernel_size = 1, padding='same', kernel_initializer=he_uniform())(x)
        
        if downsample:
            x = AveragePooling2D(pool_size = 2, padding='same')(x)
        
        return x
    
    def residual(x):
        if normalize:
            x = InstanceNormalization(axis=-1)(x)
            
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=dim_in, kernel_size = 3, padding='same', kernel_initializer=he_uniform())(x)
        
        if downsample:
            x = AveragePooling2D(pool_size = 2, padding='same')(x)
        
        if normalize:
            x = InstanceNormalization(axis=-1)(x)
        
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=dim_out, kernel_size = 3, padding='same', kernel_initializer=he_uniform())(x)
        
        return x
    
    x1 = shortcut(image)
    x2 = residual(image)
    
    res_img = Add()([x1,x2])
    
    res_img = Lambda(lambda x: x/math.sqrt(2), output_shape=lambda x:x)(res_img)
    
    return res_img

def mod_res_block(image, style, dim_in, dim_out, w_hpf=0, upsample=False):
    learned_sc = dim_in != dim_out
    
    def mod_conv_block(image, s, filters):
        style = Dense(image.shape[-1], kernel_initializer = he_uniform())(s)
        image = Conv2DMod(filters = filters, kernel_size = 3, padding = 'same', kernel_initializer = he_uniform())([image, style])
        image = LeakyReLU(alpha=0.2)(image)
        return image
    
    def shortcut(x):
        if upsample:
            x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)
        
        if learned_sc:
            x = Conv2D(filters=dim_out, kernel_size = 1, padding='same', kernel_initializer=he_uniform())(x)
        
        return x
    
    def residual(x, s):
        x = mod_conv_block(x, s, dim_out)
        
        if upsample:
            x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)
        
        x = mod_conv_block(x, s, dim_out)
        
        return x
    
    out = residual(image, style)
    if w_hpf==0:
        short = shortcut(image)
        
        out = Add()([out, short])
        out = Lambda(lambda x: x/math.sqrt(2), output_shape=lambda x:x)(out)
    
    return out

def generator(image, style):
    bf=16
    out = Conv2D(filters=bf, kernel_size = 1, padding='same', kernel_initializer=he_uniform())(image)
    #-----------------------------------------------------------------------------------
    out = res_block(out, dim_in = bf, dim_out = 2*bf, normalize=False, downsample=True)
    out = res_block(out, dim_in = 2*bf, dim_out = 4*bf, normalize=False, downsample=True)
    out = res_block(out, dim_in = 4*bf, dim_out = 8*bf, normalize=False, downsample=True)
    out = res_block(out, dim_in = 8*bf, dim_out = 8*bf, normalize=False, downsample=True)
    #---------------------------------------------------------------------------------
    out = res_block(out, dim_in = 8*bf, dim_out = 8*bf, normalize=False, downsample=False)
    out = res_block(out, dim_in = 8*bf, dim_out = 8*bf, normalize=False, downsample=False)
    out = mod_res_block(out, style, dim_in=8*bf, dim_out=8*bf, upsample=False)
    out = mod_res_block(out, style, dim_in=8*bf, dim_out=8*bf, upsample=False)
    #----------------------------------------------------------------------------------
    out = mod_res_block(out, style, dim_in=8*bf, dim_out=8*bf, upsample=True)
    out = mod_res_block(out, style, dim_in=8*bf, dim_out=8*bf, upsample=True)
    out = mod_res_block(out, style, dim_in=8*bf, dim_out=8*bf, upsample=True)
    out = mod_res_block(out, style, dim_in=8*bf, dim_out=8*bf, upsample=True)
    #--------------------------------------------------------------------------------
    out = Conv2D(filters=3, kernel_size=1, padding='same', kernel_initializer=he_uniform())(out)
    
    return out

def encoder(image, dim_out, domains=1):
    """this network can be used for both the encoders and the discriminators"""
    #For the encoder: use dim_out = latent_size
    #For the discriminator: use dim_out = 1
    
    bf=16
    out = Conv2D(filters=bf, kernel_size = 1, padding='same', kernel_initializer=he_uniform())(image)
    #------------------------------------------------------------------------------------------------
    out = res_block(out, dim_in=bf, dim_out=2*bf, normalize=True, downsample=True)
    out = res_block(out, dim_in=2*bf, dim_out=4*bf, normalize=True, downsample=True)
    out = res_block(out, dim_in=4*bf, dim_out=8*bf, normalize=True, downsample=True)
    out = res_block(out, dim_in=8*bf, dim_out=8*bf, normalize=True, downsample=True)
    out = res_block(out, dim_in=8*bf, dim_out=8*bf, normalize=True, downsample=True)
    out = res_block(out, dim_in=8*bf, dim_out=8*bf, normalize=True, downsample=True)
    #--------------------------------------------------------------------------------------------
    out = LeakyReLU(0.2)(out)
    out = Conv2D(filters=8*bf, kernel_size=4, padding='valid', kernel_initializer=he_uniform())(out)
    out = LeakyReLU(0.2)(out)
    out = Flatten()(out)
    out = Dense(dim_out*domains)(out)
    return out
    
    
    
    
#image = Input((256,256,3))
#y = encoder(image, dim_out=64)
#model = Model(inputs=image, outputs=y)
#print(model.summary())