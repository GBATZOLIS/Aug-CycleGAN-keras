# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:29:13 2020

@author: Georgios
"""

#StarGAN modules

from tensorflow.keras.layers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.models import Model
from custom_layers import *
from conv_mod import *
import math

def res_block(image, dim_in, dim_out, normalize=False, downsample=False):
    learned_sc = dim_in != dim_out
    
    def shortcut(x):
        if learned_sc:
            init = tf.keras.initializers.HeUniform()
            x = Conv2D(filters=dim_out, kernel_size = 1, padding='same', use_bias=False, kernel_initializer=init)(x)
        
        if downsample:
            x = AveragePooling2D(pool_size = 2, padding='same')(x)
        
        return x
    
    def residual(x):
        init = tf.keras.initializers.HeUniform()
        
        if normalize:
            x = InstanceNormalization(axis=-1)(x)
            
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=dim_in, kernel_size = 3, padding='same', use_bias=False, kernel_initializer=init)(x)
        
        if downsample:
            x = AveragePooling2D(pool_size = 2, padding='same')(x)
        
        if normalize:
            x = InstanceNormalization(axis=-1)(x)
        
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=dim_out, kernel_size = 3, padding='same', use_bias=False, kernel_initializer=init)(x)
        
        return x
    
    x1 = shortcut(image)
    x2 = residual(image)
    
    res_img = Add()([x1,x2])
    
    res_img = Lambda(lambda x: x/math.sqrt(2), output_shape=lambda x:x)(res_img) #unit variance
    
    return res_img


def AdaIN(image, style):
    
    num_features = image.shape[-1]
    
    gamma = Dense(num_features)(style)
    gamma = Reshape(shape=(gamma.shape[0], 1, 1, gamma.shape[1]))(gamma)
    
    beta = Dense(num_features)(style)
    beta = Reshape(shape=(beta.shape[0], 1, 1, beta.shape[1]))(beta)
    
    image = AdaInstanceNormalization()([image, beta, gamma]) #scale is initialised at 1
    
    return image

    
def AdaResBlock(image, style, dim_in, dim_out, w_hpf=0, upsample=False):
    learned_sc = dim_in != dim_out
    
    def shortcut(x):
        if upsample:
            x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)
        
        if learned_sc:
            init = tf.keras.initializers.HeUniform()
            x = Conv2D(filters=dim_out, kernel_size = 1, padding='same', use_bias=False, kernel_initializer=init)(x)
        
        return x
    
    def residual(x, s):
        init = tf.keras.initializers.HeUniform()
        
        x = AdaIN(x, s)
        x = LeakyReLU(alpha=0.2)(x)
        
        if upsample:
            x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)
        
        x = Conv2D(filters=dim_out, kernel_size=3, padding='same', use_bias=False, kernel_initializer=init)(x)
        x = AdaIN(x, s)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=dim_out, kernel_size=3, padding='same', use_bias=False, kernel_initializer=init)(x)
        
        return x
    
    out = residual(image, style)
    if w_hpf==0:
        short = shortcut(image)
        
        out = Add()([out, short])
        out = Lambda(lambda x: x/math.sqrt(2), output_shape=lambda x:x)(out)
    
    return out

def generator(image, style):
    def add_block(image1, image2):
        image = Add()([image1, image2])
        image = Lambda(lambda x: x/math.sqrt(2), output_shape=lambda x:x)(image)
        return image
    
    init = RandomNormal(stddev=0.02) 
    bf=16
    out1 = Conv2D(filters=bf, kernel_size = 1, padding='same', kernel_initializer=init)(image)
    #-----------------------------------------------------------------------------------
    out2 = res_block(out1, dim_in = bf, dim_out = 2*bf, normalize=False, downsample=True)
    out3 = res_block(out2, dim_in = 2*bf, dim_out = 4*bf, normalize=False, downsample=True)
    out4 = res_block(out3, dim_in = 4*bf, dim_out = 8*bf, normalize=False, downsample=True)
    out = res_block(out4, dim_in = 8*bf, dim_out = 8*bf, normalize=False, downsample=True)
    #---------------------------------------------------------------------------------
    out = res_block(out, dim_in = 8*bf, dim_out = 8*bf, normalize=False, downsample=False)
    out = res_block(out, dim_in = 8*bf, dim_out = 8*bf, normalize=False, downsample=False)
    out = mod_res_block(out, style, dim_in=8*bf, dim_out=8*bf, upsample=False)
    out = mod_res_block(out, style, dim_in=8*bf, dim_out=8*bf, upsample=False)
    #----------------------------------------------------------------------------------
    out5 = mod_res_block(out, style, dim_in=8*bf, dim_out=8*bf, w_hpf=1, upsample=True)
    out5 = add_block(out4, out5)
    out6 = mod_res_block(out5, style, dim_in=8*bf, dim_out=4*bf, w_hpf=1, upsample=True)
    out6 = add_block(out3, out6)
    out7 = mod_res_block(out6, style, dim_in=4*bf, dim_out=2*bf, w_hpf=1, upsample=True)
    out7 = add_block(out2, out7)
    out8 = mod_res_block(out7, style, dim_in=2*bf, dim_out=bf, w_hpf=1, upsample=True)
    #--------------------------------------------------------------------------------
    #out8 = mod_res_block(out8, style, dim_in=bf, dim_out=bf, w_hpf=0, upsample=False)
    
    out9 = add_block(out8, out1)
    
    out9 = mod_res_block(out9, style, dim_in=bf, dim_out=bf, w_hpf=0, upsample=False)
    out9 = mod_res_block(out9, style, dim_in=bf, dim_out=bf, w_hpf=0, upsample=False)
    out9 = mod_res_block(out9, style, dim_in=bf, dim_out=bf, w_hpf=0, upsample=False)
    
    
    
    out = Conv2D(filters=3, kernel_size=1, padding='same', kernel_initializer=init)(out9)
    
    return out

def encoder(image, D, K=2):
    """this network can be used for both the encoders and the discriminators"""
    #For the encoder: use D = style_size
    #For the discriminator: use D = 1
    
    init = RandomNormal(stddev=0.02)
    bf=16
    out = Conv2D(filters=bf, kernel_size = 1, padding='same', kernel_initializer=init)(image)
    #------------------------------------------------------------------------------------------------
    out = res_block(out, dim_in=bf, dim_out=2*bf, normalize=True, downsample=True)
    out = res_block(out, dim_in=2*bf, dim_out=4*bf, normalize=True, downsample=True)
    out = res_block(out, dim_in=4*bf, dim_out=8*bf, normalize=True, downsample=True)
    out = res_block(out, dim_in=8*bf, dim_out=8*bf, normalize=True, downsample=True)
    out = res_block(out, dim_in=8*bf, dim_out=8*bf, normalize=True, downsample=True)
    out = res_block(out, dim_in=8*bf, dim_out=8*bf, normalize=True, downsample=True)
    #--------------------------------------------------------------------------------------------
    out = LeakyReLU(0.2)(out)
    out = Conv2D(filters=8*bf, kernel_size=4, padding='valid', kernel_initializer=init)(out)
    out = LeakyReLU(0.2)(out)
    out = Flatten()(out)
    
    outs = []
    for domain in range(K):
        domain_out = Dense(D)(out)
        outs.append(domain_out)
        
    return outs
    
    
    
    
#image = Input((256,256,3))
#y = encoder(image, dim_out=64)
#model = Model(inputs=image, outputs=y)
#print(model.summary())