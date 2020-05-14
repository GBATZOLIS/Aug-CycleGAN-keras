# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 17:56:58 2020

@author: Georgios
"""

from tensorflow.keras.layers import Input, Concatenate, Reshape
from tensorflow.keras.models import Model
from modules import CINResnetGenerator, Alternative_Encoder, LatentEncoder, styleGAN_disc, img_domain_critic, noise_domain_critic, noise_mapping_func
from tensorflow.python.keras.engine.network import Network
from stargan_modules import *
    
#keras models needed for keras model Augmented CycleGAN

def G_AB(img_shape, latent_shape):
    image = Input(img_shape)
    noise = Input((latent_shape[-1],))
    #output = CINResnetGenerator(image, noise, filters=16, nlatent=latent_shape[-1])
    #output = celeb_generator(image, noise, filters=16, nlatent=latent_shape[-1])
    output = generator(image, noise)
    model = Model(inputs=[image, noise], outputs=output, name='GAB')
    return model

def G_BA(img_shape, latent_shape):
    image = Input(img_shape)
    noise = Input((latent_shape[-1],))
    #output = CINResnetGenerator(image, noise, filters=16, nlatent=latent_shape[-1])
    #output = celeb_generator(image, noise, filters=16, nlatent=latent_shape[-1])
    output = generator(image, noise)
    model = Model(inputs=[image, noise], outputs=output, name='GBA')
    return model

def E_A(img_shape, latent_shape):
    imgA = Input(img_shape)
    imgB = Input(img_shape)
    
    #-----------------------------------------------------------------------
    #concat_A_B = Concatenate(axis=-1)([imgA, imgB])
    #encoding = LatentEncoder(concat_A_B, nef=32, z_dim = latent_shape[-1])
    #-----------------------------------------------------------------------
    
    #concat_A_B = Concatenate(axis=-1)([imgA, imgB])
    encoding = Alternative_Encoder(imgA, imgB, latent_shape[-1])
    
    #concat_A_B = Concatenate(axis=-1)([imgA, imgB])
    #encoding = encoder(concat_A_B, latent_shape[-1])
    
    model = Model(inputs=[imgA, imgB], outputs=encoding, name='EA')
    
    return model

def E_B(img_shape, latent_shape):
    imgA = Input(img_shape)
    imgB = Input(img_shape)
    
    #---------------------------------------------------------------------
    #concat_A_B = Concatenate(axis=-1)([imgA, imgB])
    #encoding = LatentEncoder(concat_A_B, nef=32, z_dim = latent_shape[-1])
    #--------------------------------------------------------------------
    #concat_B_A = Concatenate(axis=-1)([imgB, imgA])
    encoding = Alternative_Encoder(imgB, imgA, latent_shape[-1])
    
    #concat_B_A = Concatenate(axis=-1)([imgB, imgA])
    #encoding = encoder(concat_B_A, latent_shape[-1])
    
    model = Model(inputs=[imgA, imgB], outputs=encoding, name='EB')
    
    return model

def D_A(img_shape):
    img=Input(img_shape)
    result = img_domain_critic(img)
    #result = encoder(img, dim_out=1)
    model = Model(inputs=img, outputs=result, name='DA')
    return model

def D_B(img_shape):
    img=Input(img_shape)
    result = img_domain_critic(img)
    #result = encoder(img, dim_out=1)
    model = Model(inputs=img, outputs=result, name='DB')
    return model

def D_Za(latent_shape):
    Za = Input((latent_shape[-1],))
    #reshaped_Za = Reshape((latent_shape[-1],))(Za)
    result = noise_domain_critic(Za)
    model = Model(inputs=Za, outputs=result, name='DZa')
    return model

def D_Zb(latent_shape):
    Zb = Input((latent_shape[-1],))
    #reshaped_Zb = Reshape((latent_shape[-1],))(Zb)
    result = noise_domain_critic(Zb)
    model = Model(inputs=Zb, outputs=result, name='DZb')
    return model

def N_map(latent_shape, domain):
    n = Input(latent_shape)
    w = noise_mapping_func(n, latent_shape[-1])
    model = Model(inputs=n, outputs=w, name='NoiseMap_'+domain)
    return model

model=G_AB((256,256,3), (1,1,32))
print(model.summary())

#model = D_A((256,256,3))
#print(model.summary())