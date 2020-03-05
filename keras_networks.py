# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 17:56:58 2020

@author: Georgios
"""

from keras.layers import Input, Concatenate, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras_modules import CINResnetGenerator, LatentEncoder, img_domain_critic, noise_domain_critic
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

def D_Za(latent_shape):
    
    Za = Input(latent_shape)
    reshaped_Za = Reshape((latent_shape[-1],))(Za)
    result = noise_domain_critic(reshaped_Za)
    
    model = Model(inputs=Za, outputs=result, name='DZa')
    static_model = Network(inputs=Za, outputs=result, name='DZa_static')
    
    return model, static_model

def D_Zb(latent_shape):
    
    Zb = Input(latent_shape)
    reshaped_Zb = Reshape((latent_shape[-1],))(Zb)
    result = noise_domain_critic(reshaped_Zb)
    
    model = Model(inputs=Zb, outputs=result, name='DZb')
    static_model = Network(inputs=Zb, outputs=result, name='DZb_static')
    
    return model, static_model


def combined_cyclic_A(G_AB, G_BA, E_A, E_B, D_B, D_Za, img_shape, latent_shape):
        D_B.trainable=False
        D_Za.trainable=False
        
        G_AB.trainable = True
        G_BA.trainable = False
        
        E_A.trainable = True
        E_B.trainable = False
        
        a=Input(img_shape)
        z_b=Input(latent_shape)
        
        #---------------------------------
        b_hat = G_AB([a, z_b])
        valid_b_hat = D_B(b_hat)
        
        z_a_hat = E_A([a, b_hat])
        valid_z_a_hat = D_Za(z_a_hat)
        #---------------------------------
        
        a_cyc = G_BA([b_hat, z_a_hat])
        z_b_cyc = E_B([a, b_hat])
        
        model = Model(inputs=[a, z_b], outputs=[valid_b_hat, valid_z_a_hat, a_cyc, z_b_cyc], name='combined_cyclic_A')
        model.compile(loss=['mse', 'mse', 'mae', 'mae'], loss_weights=[1,1,1,1], optimizer=Adam(0.0002, 0.5))
        return model


def combined_cyclic_B(G_AB, G_BA, E_A, E_B, D_A, D_Zb, img_shape, latent_shape):
    D_A.trainable=False
    D_Zb.trainable=False
    
    G_AB.trainable = False
    G_BA.trainable = True
    
    E_A.trainable = False
    E_B.trainable = True
        
    
    b=Input(img_shape)
    z_a = Input(latent_shape)
    
    #---------------------------------
    a_hat = G_BA([b, z_a])
    valid_a_hat = D_A(a_hat)
    
    z_b_hat = E_B([a_hat, b])
    valid_z_b_hat = D_Zb(z_b_hat)
    #---------------------------------
        
    b_cyc = G_AB([a_hat, z_b_hat])
    z_a_cyc = E_A([a_hat, b])
    
    model = Model(inputs = [b, z_a], outputs = [valid_a_hat, valid_z_b_hat, b_cyc, z_a_cyc], name='combined_cyclic_B')
    model.compile(loss=['mse', 'mse', 'mae', 'mae'], loss_weights=[1,1,1,1], optimizer=Adam(0.0002, 0.5))
    return model


"""
model = D_A((100,100,3))
print(model.summary())
"""

"""  
model = E_A((100,100,3), (1,1,16))
print(model.summary())
    

model = G_AB((100,100,3), (1,1,16))
print(model.summary())
"""
    