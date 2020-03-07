# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 17:56:58 2020

@author: Georgios
"""

from keras.layers import Input, Concatenate, Reshape, Dense
from keras.models import Model
from keras_modules import CINResnetGenerator, LatentEncoder, img_domain_critic, noise_domain_critic, latent_mapping_network
from keras.engine.network import Network


#keras models needed for keras model Augmented CycleGAN

def G_AB(img_shape, latent_shape):
    image = Input(img_shape)
    noise = Input(latent_shape)
    output = CINResnetGenerator(image, noise, ngf=32, nlatent=latent_shape)
    model = Model(inputs=[image, noise], outputs=output, name='GAB')
    return model

def G_BA(img_shape, latent_shape):
    image = Input(img_shape)
    noise = Input(latent_shape)
    output = CINResnetGenerator(image, noise, ngf=32, nlatent=latent_shape)
    model = Model(inputs=[image, noise], outputs=output, name='GBA')
    return model

def E_A(img_shape, latent_shape):
    imgA = Input(img_shape)
    imgB = Input(img_shape)
    
    concat_A_B = Concatenate(axis=-1)([imgA, imgB])
    encoding = LatentEncoder(concat_A_B, nef=32, z_dim = latent_shape)
    
    model = Model(inputs=[imgA, imgB], outputs=encoding, name='EA')
    return model

def E_B(img_shape, latent_shape):
    imgA = Input(img_shape)
    imgB = Input(img_shape)
    
    concat_A_B = Concatenate(axis=-1)([imgA, imgB])
    encoding = LatentEncoder(concat_A_B, nef=32, z_dim = latent_shape)
    
    model = Model(inputs=[imgA, imgB], outputs=encoding, name='EB')
    return model

def D_A(img_shape):
    img=Input(img_shape)
    result = img_domain_critic(img)
    
    model = Model(inputs=img, outputs=result, name='DA')
    static_model = Network(inputs=img, outputs=result, name='DA_static')
    
    return model, static_model

def D_B(img_shape):
    img=Input(img_shape)
    result = img_domain_critic(img)
    
    model = Model(inputs=img, outputs=result, name='DB')
    static_model = Network(inputs=img, outputs=result, name='DB_static')
    
    return model, static_model

def D_Za(latent_shape, name):
    
    Za = Input(latent_shape)
    reshaped_Za = Reshape((latent_shape[-1],))(Za)
    result = noise_domain_critic(reshaped_Za)
    
    model = Model(inputs=Za, outputs=result, name=name)
    static_model = Network(inputs=Za, outputs=result, name=name+'_static')
    
    return model, static_model

def D_Zb(latent_shape, name):
    
    Zb = Input(latent_shape)
    reshaped_Zb = Reshape((latent_shape[-1],))(Zb)
    result = noise_domain_critic(reshaped_Zb)
    
    model = Model(inputs=Zb, outputs=result, name=name)
    static_model = Network(inputs=Zb, outputs=result, name=name+'_static')
    
    return model, static_model

def latent_map(latent_shape, name):
    #This network maps initial latent distribution (e.g. Normal) to final latent distribution
    #it is a simple concatentation of dense layers
    
    z = Input(latent_shape)
    reshaped_z = Reshape((latent_shape[-1],))(z)
    out = latent_mapping_network(reshaped_z)
    reshaped_out = Reshape(latent_shape)(out)
    
    model = Model(inputs=z, outputs=reshaped_out, name=name)
    
    return model
    

    