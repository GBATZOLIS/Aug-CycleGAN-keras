# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:29:13 2020

@author: Georgios
"""

#StarGAN modules
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.models import Model
from custom_layers import *
from conv_mod import *
import math

def ResBlock(image, dim_in, dim_out, normalize=False, downsample=False):
    learned_sc = dim_in != dim_out
    
    def shortcut(x):
        if learned_sc:
            init = tf.keras.initializers.he_uniform()
            x = Conv2D(filters=dim_out, kernel_size = 1, padding='same', use_bias=False, kernel_initializer=init)(x)
        
        if downsample:
            x = AveragePooling2D(pool_size = 2, padding='same')(x)
        
        return x
    
    def residual(x):
        init = tf.keras.initializers.he_uniform()
        
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
    init = tf.keras.initializers.he_uniform()
    
    num_features = image.shape[-1]
    
    gamma = Dense(num_features, kernel_initializer = init)(style)
    gamma = Reshape(target_shape=(1, 1, -1))(gamma)
    
    beta = Dense(num_features, kernel_initializer = init)(style)
    beta = Reshape(target_shape=(1, 1, -1))(beta)
    
    image = AdaInstanceNormalization()([image, beta, gamma]) #scale is initialised at 1
    
    return image

    
def AdaResBlock(image, style, dim_in, dim_out, _shortcut=True, upsample=False):
    learned_sc = dim_in != dim_out
    
    def style_block(image, noise, filters):
        init = tf.keras.initializers.he_uniform()
        style = Dense(image.shape[-1], kernel_initializer = init)(noise)
        image = Conv2DMod(filters = filters, kernel_size = 3, padding = 'same', kernel_initializer = init)([image, style])
        image = LeakyReLU(alpha=0.2)(image)
        return image
    
    def shortcut(x):
        if upsample:
            x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)
        
        if learned_sc:
            init = tf.keras.initializers.he_uniform()
            x = Conv2D(filters=dim_out, kernel_size = 1, padding='same', use_bias=False, kernel_initializer=init)(x)
        
        return x
    
    def residual(x, s):
        init = tf.keras.initializers.he_uniform()
        
        #x = AdaIN(x, s)
        #x = LeakyReLU(alpha=0.2)(x)
        
        x = style_block(x, s, dim_out) #new
        
        if upsample:
            x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)
        
        #x = Conv2D(filters=dim_out, kernel_size=3, padding='same', use_bias=False, kernel_initializer=init)(x)
        #x = AdaIN(x, s)
        #x = LeakyReLU(alpha=0.2)(x)
        #x = Conv2D(filters=dim_out, kernel_size=3, padding='same', use_bias=False, kernel_initializer=init)(x)
        
        x = style_block(x, s, dim_out) #new

        return x
    
    out = residual(image, style)
    if _shortcut:
        short = shortcut(image)
        
        out = Add()([out, short])
        out = Lambda(lambda x: x/math.sqrt(2), output_shape=lambda x:x)(out)
    
    return out

def generator(image, style):    
    init = tf.keras.initializers.he_uniform()
    
    bf=32
    
    #------------------------------------------------------------------------------------
    out1 = Conv2D(filters=bf, kernel_size = 1, padding='same', kernel_initializer=init)(image)
    out1 = ResBlock(out1, dim_in = bf, dim_out = bf, normalize=False, downsample=False)
    out1_forward, out1_skip = tf.split(out1, num_or_size_splits=2, axis=-1) #channel splitting
    
    #out1_forward channels = bf/2
    #-------------------------------------------------------------------------------------
    
    """downsampling"""
    
    #----------------------------------------------------------------------------------
    out2 = ResBlock(out1_forward, dim_in = bf//2, dim_out = 2*bf, normalize=False, downsample=True)
    out2 = ResBlock(out2, dim_in = 2*bf, dim_out = 2*bf, normalize=False, downsample=False)
    out2_forward, out2_skip = tf.split(out2, num_or_size_splits=2, axis=-1)
    #out2_forward channels = bf
    #-------------------------------------------------------------------------------------
    out3 = ResBlock(out2_forward, dim_in = bf, dim_out = 4*bf, normalize=False, downsample=True)
    out3 = ResBlock(out3, dim_in = 4*bf, dim_out = 4*bf, normalize=False, downsample=False)
    out3_forward, out3_skip = tf.split(out3, num_or_size_splits=2, axis=-1)
    #out3_forward channels = 2*bf
    #-------------------------------------------------------------------------------
    out4 = ResBlock(out3_forward, dim_in = 2*bf, dim_out = 8*bf, normalize=False, downsample=True)
    
    """bottleneck"""
    out4 = ResBlock(out4, dim_in = 8*bf, dim_out = 8*bf, normalize=False, downsample=False)
    out4 = ResBlock(out4, dim_in = 8*bf, dim_out = 8*bf, normalize=False, downsample=False)
    out4 = AdaResBlock(out4, style, dim_in = 8*bf, dim_out = 8*bf, _shortcut=True, upsample=False)
    out4 = AdaResBlock(out4, style, dim_in = 8*bf, dim_out = 8*bf, _shortcut=True, upsample=False)

    """upsampling"""
    out5_forward = AdaResBlock(out4, style, dim_in = 8*bf, dim_out = 2*bf, _shortcut=False, upsample=True)
    out5_concat = Concatenate(axis=-1)([out5_forward, out3_skip])
    #out5_concat channels = 4*bf
    out5 = AdaResBlock(out5_concat, style, dim_in = 4*bf, dim_out = 4*bf, _shortcut=False, upsample=False)
    
    #----------------------------------------------------------------
    
    out6_forward = AdaResBlock(out5, style, dim_in = 4*bf, dim_out = bf, _shortcut=False, upsample=True)
    out6_concat = Concatenate(axis=-1)([out6_forward, out2_skip])
    #out6_concat channels = 2*bf
    out6 = AdaResBlock(out6_concat, style, dim_in = 2*bf, dim_out = 2*bf, _shortcut=False, upsample=False)
    
    #----------------------------------------------------------------
    out7_forward = AdaResBlock(out6, style, dim_in = 2*bf, dim_out = bf//2, _shortcut=False, upsample=True)
    out7_concat = Concatenate(axis=-1)([out7_forward, out1_skip])
    #out7_concat channels = bf
    out7 = AdaResBlock(out7_concat, style, dim_in = bf, dim_out = bf, _shortcut=False, upsample=False)
    
    out7 = AdaResBlock(out7, style, dim_in = bf, dim_out = bf, _shortcut=False, upsample=False) #new
    out7 = AdaResBlock(out7, style, dim_in = bf, dim_out = bf, _shortcut=False, upsample=False) #new
    #out7 = AdaResBlock(out7, style, dim_in = bf, dim_out = bf, _shortcut=True, upsample=False) #new
    #out7 = AdaResBlock(out7, style, dim_in = bf, dim_out = bf, _shortcut=True, upsample=False) #new
    
    output = Conv2D(filters = 3, kernel_size = 1, padding = 'same', kernel_initializer = init)(out7)

    return output

def mapping_network(z, D, K):
    #D: number of dimensions of the style code
    #K: number of domains - number of output branches
    
    def dense_block(inp, out_nodes, act=True):
        init = tf.keras.initializers.he_uniform()
        out = Dense(out_nodes, kernel_initializer = init)(inp)
        if act:
            out = LeakyReLU(alpha=0.2)(out)
        return out
    
    #shared part
    shared = dense_block(z, 512)
    shared = dense_block(shared, 512)
    shared = dense_block(shared, 512)
    shared = dense_block(shared, 512)
    
    #K specific output branches
    outputs=[]
    for i in range(K):
        branch_output = dense_block(shared, 512)
        branch_output = dense_block(branch_output, 512)
        branch_output = dense_block(branch_output, 512)
        branch_output = dense_block(branch_output, D)
        outputs.append(branch_output)
    
    return outputs
        
    
def encoder(image, D, K=2, discriminator_use=False):
    """this network can be used for both the encoders and the discriminators"""
    #D: number of dimensions of the style code
    # : - For the encoder: use D = style_size
    # : - For the discriminator: use D = 1
    
    #K: number of domains - number of output branches
    
    
    init = tf.keras.initializers.he_uniform()
    bf = 32 #bf stands for base filter
    out = Conv2D(filters = bf, kernel_size = 1, padding = 'same', kernel_initializer = init)(image)
    #------------------------------------------------------------------------------------------------
    out = ResBlock(out, dim_in=bf, dim_out=2*bf, normalize=True, downsample=True)
    out = ResBlock(out, dim_in=2*bf, dim_out=4*bf, normalize=True, downsample=True)
    out = ResBlock(out, dim_in=4*bf, dim_out=8*bf, normalize=True, downsample=True)
    out = ResBlock(out, dim_in=8*bf, dim_out=8*bf, normalize=True, downsample=True)
    out = ResBlock(out, dim_in=8*bf, dim_out=8*bf, normalize=True, downsample=True)
    out = ResBlock(out, dim_in=8*bf, dim_out=8*bf, normalize=True, downsample=True)
    #--------------------------------------------------------------------------------------------
    out = LeakyReLU(0.2)(out)
    out = Conv2D(filters=8*bf, kernel_size=4, padding='valid', kernel_initializer=init)(out)
    out = LeakyReLU(0.2)(out)
    out = Flatten()(out)
    
    outs = []
    for domain in range(K):
        if discriminator_use:
            domain_out = Dense(D, activation='sigmoid', kernel_initializer=init)(out)
        else:
            domain_out = Dense(D, kernel_initializer=init)(out)
        
        outs.append(domain_out)
        
    return outs


"""
image = Input((256,256,3))
outs = encoder(image, D=64, K=2)
model = Model(inputs = image, outputs = outs, name = 'Encoder')
print(model.summary())
"""


"""
z=Input((16,))
out_K = mapping_network(z = z, D = 64, K=2)  
model = Model(inputs=z, outputs=out_K, name='mapping network')
print(model.summary())
"""
    

"""
image = Input((256,256,3))
style = Input((64,))
output = generator(image, style)
model = Model(inputs=[image, style], outputs=output, name='gen')
print(model.summary())
"""
