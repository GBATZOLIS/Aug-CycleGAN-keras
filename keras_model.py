# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:04:32 2020

@author: Georgios
"""

#augmented cycleGAN from scratch in KERAS


#Implementation of the MCinCGAN paper

#laod required modules
import datetime
import numpy as np
import os
import pickle
import shutil
from glob import glob
from zipfile import ZipFile

#load functions and classes from other .py files within the repository
from data_loader import DataLoader
#from evaluator import evaluator 


#keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model
from keras_modules import blur
from keras_networks import G_AB, G_BA, E_A, E_B, D_A, D_B, D_Za, D_Zb, N_map
from keras_evaluator import evaluator
from tensorflow.keras.losses import mse
import tensorflow as tf
from lpips import lpips


#visualisation packages
import matplotlib.pyplot as plt


  
def discriminator_loss(real, generated):
    # Multiplied by 0.5 so that it will train at half-speed
    return (tf.reduce_mean(mse(tf.ones_like(real), real)) + tf.reduce_mean(mse(tf.zeros_like(generated), generated))) * 0.5

# Measures how real the discriminator believes the fake image is
def gen_loss(validity):
    return tf.reduce_mean(mse(tf.ones_like(validity), validity))
        
def L1_loss(image1, image2):
    return tf.reduce_mean(tf.abs(image1 - image2))
        
class AugCycleGAN(object):
    def __init__(self, img_shape, latent_shape, resume=False):
        self.img_shape = img_shape
        self.latent_shape = latent_shape
        
        #--------------log settings---------------------
        self.train_info={}
        
        #-----------------LOSSES------------------------
        self.train_info['losses'] = {}
        self.train_info['losses']['sup']={}
        self.train_info['losses']['sup']['dist_a']=[]
        self.train_info['losses']['sup']['perc_a']=[]
        self.train_info['losses']['sup']['dist_b']=[]
        self.train_info['losses']['sup']['perc_b']=[]
        
        self.train_info['losses']['unsup']={}
        #ADVERSARIAL LOSSES FOR DISCRIMINATORS
        self.train_info['losses']['unsup']['D_A']=[]
        self.train_info['losses']['unsup']['D_B']=[]
        self.train_info['losses']['unsup']['D_Za']=[]
        self.train_info['losses']['unsup']['D_Zb']=[]
        #ADVERSARIAL LOSSES FOR GENERATORS
        self.train_info['losses']['unsup']['G_A']=[]
        self.train_info['losses']['unsup']['G_B']=[]
        self.train_info['losses']['unsup']['E_A']=[]
        self.train_info['losses']['unsup']['E_B']=[]
        #RECONSTRUCTIONS LOSSES
        self.train_info['losses']['unsup']['rec_a_dist']=[]
        self.train_info['losses']['unsup']['rec_a_perc']=[]
        self.train_info['losses']['unsup']['rec_b_dist']=[]
        self.train_info['losses']['unsup']['rec_b_perc']=[]
        self.train_info['losses']['unsup']['rec_Za']=[]
        self.train_info['losses']['unsup']['rec_Zb']=[]
        #BLUR LOSSES (preserve low frequencies between mapped and initial image)
        self.train_info['losses']['unsup']['blur_ab']=[] #from domain A to domain B
        self.train_info['losses']['unsup']['blur_ba']=[] #from domain B to domain A
        
        #regularisation losses
        self.train_info['losses']['reg']={}
        self.train_info['losses']['reg']['ppl_G_AB']=[]
        self.train_info['losses']['reg']['ppl_G_BA']=[]
        self.train_info['losses']['reg']['ms_G_AB']=[]
        self.train_info['losses']['reg']['ms_G_BA']=[]
        
        
        #-------------PERFORMANCE EVALUATION----------------------
        self.train_info['performance'] = {}
        self.train_info['performance']['eval_points'] = []
        #Every N batches (usually 100) we calculate LPIPS and SSIM on test data
        #We use M images and K output images for each image for the evaluation
        #At the end of each epoch we calculate LPIPS and SSIM 
        #on the entire test dataset using 20 output images for each image
        #We want to get as less noisy estimation of the performance of the model as possible
        #The first list holds the sample interval measurements, the second holds the ep measurements
        self.train_info['performance']['ssim_mean']=[[],[]]
        self.train_info['performance']['ssim_std']=[[],[]]
        self.train_info['performance']['lpips_mean']=[[],[]]
        self.train_info['performance']['lpips_std']=[[],[]]


        #configure data loader
        self.data_loader = DataLoader(img_res=(self.img_shape[0], self.img_shape[1]))
        
        #MISCALLENIA
        #perceptual paths length parameters
        self.pl_mean_G_AB = 0.
        self.pl_mean_G_BA = 0.
        
        #instantiate the LPIPS loss object
        self.lpips = lpips(self.img_shape)
        self.lpips.create_model()
        
        self.G_AB = G_AB(img_shape, latent_shape)
        self.G_BA = G_BA(img_shape, latent_shape)
        self.E_A = E_A(img_shape, latent_shape)
        self.E_B = E_B(img_shape, latent_shape)
        self.D_A = D_A(img_shape)
        self.D_B = D_B(img_shape)
        self.D_Za = D_Za(latent_shape) 
        self.D_Zb = D_Zb(latent_shape)
        self.blurring = blur(img_shape)
            
        if resume==True:
            self.G_AB.load_weights(glob('models/G_AB/*.h5')[-1])
            self.G_BA.load_weights(glob('models/G_BA/*.h5')[-1])
            self.E_A.load_weights(glob('models/E_A/*.h5')[-1])
            self.E_B.load_weights(glob('models/E_B/*.h5')[-1])
            self.D_A.load_weights(glob('models/D_A/*.h5')[-1])
            self.D_B.load_weights(glob('models/D_B/*.h5')[-1])
            self.D_Za.load_weights(glob('models/D_Za/*.h5')[-1])
            self.D_Zb.load_weights(glob('models/D_Zb/*.h5')[-1])
        
        self.G_AB_opt = self.G_BA_opt = Adam(lr=0.0002, beta_1=0.5)
        self.D_A_opt = self.D_B_opt = Adam(lr=0.0002, beta_1=0.5)
        self.E_A_opt = self.E_B_opt = Adam(lr=0.0002, beta_1=0.5)
        self.D_Za_opt = self.D_Zb_opt = Adam(lr=0.0002, beta_1=0.5)
    
    def supervised_step(self, a, b):     
        with tf.GradientTape(persistent=True) as tape:
            z_a_hat = self.E_A([a,b], training=True)
            a_hat = self.G_BA([b,z_a_hat], training=True)
            #sup_loss_a = 0.5*L1_loss(a,a_hat)-0.5*tf.reduce_mean(tf.image.ssim(a,a_hat, max_val=1))
            
            #sup_perc_a = self.lpips.distance(a,a_hat)
            #self.train_info['losses']['sup']['perc_a'].append(sup_perc_a)
            
            sup_dist_a = L1_loss(a,a_hat)
            self.train_info['losses']['sup']['dist_a'].append(sup_dist_a)
            
            sup_loss_a = sup_dist_a
            
            z_b_hat = self.E_B([a,b], training=True)
            b_hat = self.G_AB([a, z_b_hat], training=True)
            #sup_loss_b = 0.5*L1_loss(b,b_hat)-0.5*tf.reduce_mean(tf.image.ssim(a,a_hat, max_val=1))
            #sup_perc_b = self.lpips.distance(b,b_hat)
            #self.train_info['losses']['sup']['perc_b'].append(sup_perc_b)
            
            sup_dist_b = L1_loss(b,b_hat)
            self.train_info['losses']['sup']['dist_b'].append(sup_dist_b)
            
            sup_loss_b = sup_dist_b
   
        
        #supervised loss a
        G_BA_grads = tape.gradient(sup_loss_a, self.G_BA.trainable_variables)
        self.G_BA_opt.apply_gradients(zip(G_BA_grads, self.G_BA.trainable_variables))
        
        E_A_grads = tape.gradient(sup_loss_a, self.E_A.trainable_variables)
        self.E_A_opt.apply_gradients(zip(E_A_grads, self.E_A.trainable_variables))
        
        #supervised loss b
        G_AB_grads = tape.gradient(sup_loss_b, self.G_AB.trainable_variables)
        self.G_AB_opt.apply_gradients(zip(G_AB_grads, self.G_AB.trainable_variables))
        
        E_B_grads = tape.gradient(sup_loss_b, self.E_B.trainable_variables)
        self.E_B_opt.apply_gradients(zip(E_B_grads, self.E_B.trainable_variables))
        
        return sup_loss_a, sup_loss_b
                
    
    def step_cycle_A(self, a, b, z_a, z_b):
        
        with tf.GradientTape(persistent=True) as tape:
            #1st map
            b_hat = self.G_AB([a, z_b], training=True)
            fake_b = self.D_B(b_hat, training=True)

            b_hat_blur=self.blurring(b_hat, training=False)
            a_blur = self.blurring(a, training=False)
            
            z_a_hat = self.E_A([a, b_hat], training=True)
            fake_z_a = self.D_Za(z_a_hat, training=True)
    
    	    #2nd map
            a_cyc = self.G_BA([b_hat, z_a_hat], training=True)
            z_b_cyc = self.E_B([a, b_hat], training=True)
        
            #---------------COMPUTE LOSSES-----------------------
            D_B_loss = discriminator_loss(self.D_B(b, training=True), fake_b)
            self.train_info['losses']['unsup']['D_B'].append(D_B_loss)
            
            D_Za_loss = discriminator_loss(self.D_Za(z_a, training=True), fake_z_a)
            self.train_info['losses']['unsup']['D_Za'].append(D_Za_loss)
            
            adv_gen_B = gen_loss(fake_b)
            self.train_info['losses']['unsup']['G_B'].append(adv_gen_B)
            
            adv_gen_Za = gen_loss(fake_z_a)
            self.train_info['losses']['unsup']['E_A'].append(adv_gen_Za)
            
            rec_a_dist = L1_loss(a_cyc,a)
            self.train_info['losses']['unsup']['rec_a_dist'].append(rec_a_dist)
            
            rec_Zb = L1_loss(z_b_cyc,z_b)
            self.train_info['losses']['unsup']['rec_Zb'].append(rec_Zb)
            
            blur_ab = L1_loss(a_blur, b_hat_blur)
            self.train_info['losses']['unsup']['blur_ab'].append(blur_ab)
            
            cycle_A_Zb_loss = adv_gen_B + adv_gen_Za + rec_a_dist + rec_Zb + blur_ab

        D_B_grads = tape.gradient(D_B_loss, self.D_B.trainable_variables)
        self.D_B_opt.apply_gradients(zip(D_B_grads, self.D_B.trainable_variables))
                
        D_Za_grads = tape.gradient(D_Za_loss, self.D_Za.trainable_variables)
        self.D_Za_opt.apply_gradients(zip(D_Za_grads, self.D_Za.trainable_variables))
                
        G_AB_grads = tape.gradient(cycle_A_Zb_loss, self.G_AB.trainable_variables)
        self.G_AB_opt.apply_gradients(zip(G_AB_grads, self.G_AB.trainable_variables))

        E_A_grads = tape.gradient(cycle_A_Zb_loss, self.E_A.trainable_variables)
        self.E_A_opt.apply_gradients(zip(E_A_grads, self.E_A.trainable_variables))

		#Update G_BA and E_B only based on cycle starting from B
        G_BA_grads = tape.gradient(cycle_A_Zb_loss, self.G_BA.trainable_variables)
        self.G_BA_opt.apply_gradients(zip(G_BA_grads, self.G_BA.trainable_variables))

        E_B_grads = tape.gradient(cycle_A_Zb_loss, self.E_B.trainable_variables)
        self.E_B_opt.apply_gradients(zip(E_B_grads, self.E_B.trainable_variables))
        
        return D_B_loss, D_Za_loss, cycle_A_Zb_loss
    
    def step_cycle_B(self, a, b, z_a, z_b):
        
        
        with tf.GradientTape(persistent=True) as tape:
            #1st map
            a_hat = self.G_BA([b, z_a], training=True)
            fake_a = self.D_A(a_hat, training=True)

            a_hat_blur=self.blurring(a_hat, training=False)
            b_blur = self.blurring(b, training=False)

            z_b_hat = self.E_B([a_hat, b], training=True)
            fake_z_b = self.D_Zb(z_b_hat, training=True)
			
			#2nd map
            b_cyc = self.G_AB([a_hat, z_b_hat], training=True)
            z_a_cyc = self.E_A([a_hat, b], training=True)
            
            #----------COMPUTE LOSSES-----------
            D_A_loss = discriminator_loss(self.D_A(a, training=True), fake_a)
            self.train_info['losses']['unsup']['D_A'].append(D_A_loss)
            
            D_Zb_loss = discriminator_loss(self.D_Zb(z_b, training=True), fake_z_b)
            self.train_info['losses']['unsup']['D_Zb'].append(D_Zb_loss)
            
            adv_gen_A = gen_loss(fake_a)
            self.train_info['losses']['unsup']['G_A'].append(adv_gen_A)
            
            adv_gen_Zb = gen_loss(fake_z_b)
            self.train_info['losses']['unsup']['E_B'].append(adv_gen_Zb)
            
            rec_b_dist = L1_loss(b_cyc,b)
            self.train_info['losses']['unsup']['rec_b_dist'].append(rec_b_dist)
            
            rec_Za = L1_loss(z_a_cyc,z_a)
            self.train_info['losses']['unsup']['rec_Za'].append(rec_Za)
            
            blur_ba = L1_loss(b_blur,a_hat_blur)
            self.train_info['losses']['unsup']['blur_ba'].append(blur_ba)
            
            cycle_B_Za_loss = adv_gen_A + adv_gen_Zb + rec_b_dist + rec_Za + blur_ba

        D_A_grads = tape.gradient(D_A_loss, self.D_A.trainable_variables)
        self.D_A_opt.apply_gradients(zip(D_A_grads, self.D_A.trainable_variables))
                
        D_Zb_grads = tape.gradient(D_Zb_loss, self.D_Zb.trainable_variables)
        self.D_Zb_opt.apply_gradients(zip(D_Zb_grads, self.D_Zb.trainable_variables))
                
        G_AB_grads = tape.gradient(cycle_B_Za_loss, self.G_AB.trainable_variables)
        self.G_AB_opt.apply_gradients(zip(G_AB_grads, self.G_AB.trainable_variables))

        E_A_grads = tape.gradient(cycle_B_Za_loss, self.E_A.trainable_variables)
        self.E_A_opt.apply_gradients(zip(E_A_grads, self.E_A.trainable_variables))

		#Update G_BA and E_B only based on cycle starting from B
        G_BA_grads = tape.gradient(cycle_B_Za_loss, self.G_BA.trainable_variables)
        self.G_BA_opt.apply_gradients(zip(G_BA_grads, self.G_BA.trainable_variables))

        E_B_grads = tape.gradient(cycle_B_Za_loss, self.E_B.trainable_variables)
        self.E_B_opt.apply_gradients(zip(E_B_grads, self.E_B.trainable_variables))
        
        return D_A_loss, D_Zb_loss, cycle_B_Za_loss
    
    def mode_seeking_regularisation(self, a, b):
        with tf.GradientTape(persistent=True) as tape:
            z_b = tf.random.normal((a.shape[0], 1, 1, self.latent_shape[-1]), dtype=tf.float32)
            b_hat = self.G_AB([a,z_b], training=True)
                
            z_b_dash = tf.random.normal((a.shape[0], 1, 1, self.latent_shape[-1]), dtype=tf.float32)
            b_hat_dash = self.G_AB([a,z_b_dash], training=True)
            
            mode_seeking_rt_AB = L1_loss(b_hat, b_hat_dash)/(L1_loss(z_b, z_b_dash)+1e-8)
            mode_seeking_loss_AB = -1*mode_seeking_rt_AB
            self.train_info['losses']['reg']['ms_G_AB'].append(mode_seeking_loss_AB)
            
            #-----------------------------------------------
            z_a = tf.random.normal((b.shape[0], 1, 1, self.latent_shape[-1]), dtype=tf.float32)
            a_hat = self.G_BA([b,z_a], training=True)
                
            z_a_dash = tf.random.normal((b.shape[0], 1, 1, self.latent_shape[-1]), dtype=tf.float32)
            a_hat_dash = self.G_BA([b,z_a_dash], training=True)
            
            mode_seeking_rt_BA = L1_loss(a_hat, a_hat_dash)/(L1_loss(z_a, z_a_dash)+1e-8)
            mode_seeking_loss_BA = -1*mode_seeking_rt_BA
            self.train_info['losses']['reg']['ms_G_BA'].append(mode_seeking_loss_BA)
            
        #update the generator models G_AB and G_BA
        G_AB_grads = tape.gradient(mode_seeking_loss_AB, self.G_AB.trainable_variables)
        self.G_AB_opt.apply_gradients(zip(G_AB_grads, self.G_AB.trainable_variables))
        
        G_BA_grads = tape.gradient(mode_seeking_loss_BA, self.G_BA.trainable_variables)
        self.G_BA_opt.apply_gradients(zip(G_BA_grads, self.G_BA.trainable_variables))
            
        
    def ppl_regularisation(self, a, b):
        #every M steps we regularise the perceptual path length
        #we have 2 generators (G_AB, G_BA)
        
        with tf.GradientTape(persistent=True) as tape:
            z_b = tf.random.normal((a.shape[0], 1, 1, self.latent_shape[-1]), dtype=tf.float32)
            b_hat = self.G_AB([a,z_b], training=True)
            
            z_b_dash = z_b + 0.05*tf.random.normal((a.shape[0], 1, 1, self.latent_shape[-1]), dtype=tf.float32)
            b_hat_dash = self.G_AB([a,z_b_dash], training=True)
            
            delta_G_AB = tf.math.reduce_mean(tf.abs(b_hat-b_hat_dash), axis=[1,2,3])
            pl_lengths_G_AB = delta_G_AB
            
            ppl_loss_G_AB = tf.math.reduce_mean(tf.abs(pl_lengths_G_AB - self.pl_mean_G_AB))
            self.train_info['losses']['reg']['ppl_G_AB'].append(ppl_loss_G_AB)
            
            #---------------------------------------------------------------------------------------
            z_a = tf.random.normal((b.shape[0], 1, 1, self.latent_shape[-1]), dtype=tf.float32)
            a_hat = self.G_BA([b,z_a], training=True)
            
            z_a_dash = z_a + 0.05*tf.random.normal((b.shape[0], 1, 1, self.latent_shape[-1]), dtype=tf.float32)
            a_hat_dash = self.G_BA([b,z_a_dash], training=True)
            
            delta_G_BA = tf.math.reduce_mean(tf.abs(a_hat-a_hat_dash), axis=[1,2,3])
            pl_lengths_G_BA = delta_G_BA
            
            ppl_loss_G_BA = tf.math.reduce_mean(tf.abs(pl_lengths_G_BA - self.pl_mean_G_BA))
            self.train_info['losses']['reg']['ppl_G_BA'].append(ppl_loss_G_BA)
            
        
        #update the generator models
        ppl_G_AB_grads = tape.gradient(ppl_loss_G_AB, self.G_AB.trainable_variables)
        self.G_AB_opt.apply_gradients(zip(ppl_G_AB_grads, self.G_AB.trainable_variables))
        
        ppl_G_BA_grads = tape.gradient(ppl_loss_G_BA, self.G_BA.trainable_variables)
        self.G_BA_opt.apply_gradients(zip(ppl_G_BA_grads, self.G_BA.trainable_variables))
            
        
        if self.pl_mean_G_AB==0.:
            self.pl_mean_G_AB = tf.math.reduce_mean(pl_lengths_G_AB)
        else:
            self.pl_mean_G_AB = 0.99*self.pl_mean_G_AB + 0.01*tf.math.reduce_mean(pl_lengths_G_AB)
            
        if self.pl_mean_G_BA==0.:
            self.pl_mean_G_BA = tf.math.reduce_mean(pl_lengths_G_BA)
        else:
            self.pl_mean_G_BA = 0.99*self.pl_mean_G_BA + 0.01*tf.math.reduce_mean(pl_lengths_G_BA)
        

    def save_models(self,epoch):
        
        #save the models to intoduce resume capacity to training
        self.G_AB.save("models/G_AB/G_AB_{}.h5".format(epoch))
        self.G_BA.save("models/G_BA/G_BA_{}.h5".format(epoch))
        self.E_A.save("models/E_A/E_A_{}.h5".format(epoch))
        self.E_B.save("models/E_B/E_B_{}.h5".format(epoch))
        self.D_A.save("models/D_A/D_A_{}.h5".format(epoch))
        self.D_B.save("models/D_B/D_B_{}.h5".format(epoch))
        self.D_Za.save("models/D_Za/D_Za_{}.h5".format(epoch))
        self.D_Zb.save("models/D_Zb/D_Zb_{}.h5".format(epoch))
    
    def delete_models(self,directories):
        for directory in directories:
            os.remove(directory)
        
    def train(self, epochs, batch_size=10, sample_interval=50):
        start_time = datetime.datetime.now()
        def chop_microseconds(delta):
            #utility to help avoid printing the microseconds
            return delta - datetime.timedelta(microseconds=delta.microseconds)
        
        try:
            #create a dynamic evaluator object
            dynamic_evaluator = evaluator(self.img_shape, self.latent_shape)
            for epoch in range(epochs):
                for batch, (img_A, img_B, sup_img_A, sup_img_B) in enumerate(self.data_loader.load_batch(batch_size)):
                    img_A = tf.convert_to_tensor(img_A, dtype=tf.float32)
                    img_B = tf.convert_to_tensor(img_B, dtype=tf.float32)
                    sup_img_A = tf.convert_to_tensor(sup_img_A, dtype=tf.float32)
                    sup_img_B = tf.convert_to_tensor(sup_img_B, dtype=tf.float32)
        
                    
                        
                    for i in range(2):
                        z_a = tf.random.normal((batch_size, 1, 1, self.latent_shape[-1]), dtype=tf.float32)
                        z_b = tf.random.normal((batch_size, 1, 1, self.latent_shape[-1]), dtype=tf.float32)
                        
                        D_B_loss, D_Za_loss, cycle_A_Zb_loss = self.step_cycle_A(img_A, img_B, z_a, z_b)
                        D_A_loss, D_Zb_loss, cycle_B_Za_loss = self.step_cycle_B(img_A, img_B, z_a, z_b)
                    
                    sup_a, sup_b = self.supervised_step(sup_img_A, sup_img_B)
                    
                    #generate the noise vectors from the N(0,sigma^2) distribution
                    if batch % 10 == 0:
                        self.ppl_regularisation(img_A, img_B)
                        self.mode_seeking_regularisation(img_A, img_B)
                        
                        elapsed_time = chop_microseconds(datetime.datetime.now() - start_time)
                        print('[%d/%d][%d/%d]-[%s:%.3f %s:%.3f %s:%.3f %s:%.3f]-[%s:%.3f %s:%.3f]-[%s:%.3f %s:%.3f]-[%s:%.6f %s:%.6f %s:%.4f %s:%.4f]-[time:%s]'
                              % (epoch, epochs, batch, self.data_loader.n_batches,
                                 'D_A', D_A_loss, 'D_B', D_B_loss, 'D_Za', D_Za_loss, 'D_Zb', D_Zb_loss,
                                 'cyc_A_Zb', cycle_A_Zb_loss, 'cyc_B_Za', cycle_B_Za_loss,
                                 'sup_a', sup_a, 'sup_b', sup_b, 
                                 'ppl_AB', self.train_info['losses']['reg']['ppl_G_AB'][-1],
                                 'ppl_BA', self.train_info['losses']['reg']['ppl_G_AB'][-1], 
                                 'ms_AB',self.train_info['losses']['reg']['ms_G_AB'][-1],
                                 'ms_BA',self.train_info['losses']['reg']['ms_G_BA'][-1],
                                 elapsed_time))
    
                    if batch % 50 == 0 and not(batch==0 and epoch==0):
                        training_point = np.around(epoch+batch/self.data_loader.n_batches, 4)
                        self.train_info['performance']['eval_points'].append(training_point)
                        dynamic_evaluator.model = self.G_AB
                        #Perception and distortion evaluation
                        info = dynamic_evaluator.test(batch_size=100, num_out_imgs=10, training_point=training_point, test_type='mixed')
                        
                        
                        self.train_info['performance']['ssim_mean'][0].append(info['ssim_mean'])
                        self.train_info['performance']['ssim_std'][0].append(info['ssim_std'])
                        self.train_info['performance']['lpips_mean'][0].append(info['lpips_mean'])
                        self.train_info['performance']['lpips_std'][0].append(info['lpips_std'])
                        
                        plt.figure(figsize=(21,15))
                        plt.title('SSIM (100x10)')
                        plt.plot(self.train_info['performance']['eval_points'], self.train_info['performance']['ssim_mean'][0], label='mean')
                        plt.plot(self.train_info['performance']['eval_points'], self.train_info['performance']['ssim_std'][0], label='std')
                        plt.legend()
                        plt.savefig('progress/distortion/SSIM.png', bbox_inches='tight')
                        
                        
                        plt.figure(figsize=(21,15))
                        plt.title('LPIPS (100x10)')
                        plt.plot(self.train_info['performance']['eval_points'], self.train_info['performance']['lpips_mean'][0], label='mean')
                        plt.plot(self.train_info['performance']['eval_points'], self.train_info['performance']['lpips_std'][0], label='std')
                        plt.legend()
                        plt.savefig('progress/perception/LPIPS.png', bbox_inches='tight')
                        plt.close('all')
                        
                        
                        #save the generators
                        self.G_AB.save("models/G_AB_all/G_AB_{}_{}.h5".format(epoch, batch))
                        #save the tensorboard values
                        with open('progress/training_information/'+ 'train_info' + '.pkl', 'wb') as f:
                            pickle.dump(self.train_info, f, pickle.HIGHEST_PROTOCOL)
                
                
                
                dynamic_evaluator.model = self.G_AB #set the current G_AB model for evaluation
                #Perception and distortion evaluation on the entire test dataset
                info = dynamic_evaluator.test(batch_size=400, num_out_imgs=20, training_point=training_point, test_type='mixed')
                
                self.train_info['performance']['ssim_mean'][1].append(info['ssim_mean'])
                self.train_info['performance']['ssim_std'][1].append(info['ssim_std'])
                
                self.train_info['performance']['lpips_mean'][1].append(info['lpips_mean'])
                self.train_info['performance']['lpips_std'][1].append(info['lpips_std'])
                
                plt.figure(figsize=(21,15))
                plt.title('SSIM (test dataset)')
                plt.plot(self.train_info['performance']['ssim_mean'][1], label='mean')
                plt.plot(self.train_info['performance']['ssim_std'][1], label='std')
                plt.legend()
                plt.savefig('progress/distortion/SSIM_epoch.png', bbox_inches='tight')
                
                
                plt.figure(figsize=(21,15))
                plt.title('LPIPS (test dataset)')
                plt.plot(self.train_info['performance']['lpips_mean'][1], label='mean')
                plt.plot(self.train_info['performance']['lpips_std'][1], label='std')
                plt.legend()
                plt.savefig('progress/perception/LPIPS_epoch.png', bbox_inches='tight')
                plt.close('all')
                
                #save the models to intoduce resume capacity to training
                self.save_models(epoch)
            
        except KeyboardInterrupt:
            print('Training has been terminated manually')
            save = input("Should I save the training data?")
            
            if save=='yes' or save=='y':
                #Create a new directory under the run file to save the training information
                now = datetime.datetime.now()
                date_time = now.strftime("%m_%d_%Y__%H_%M_%S")
                os.mkdir('runs/%s' % date_time)
                #zip and save the models under '../runs/date_time/'
                shutil.make_archive('runs/%s/models' % date_time, 'zip', 'models/')
                shutil.make_archive('runs/%s/progress' % date_time, 'zip', 'progress/')
                
                #zip and save the code used
                python_paths = glob('*.py')
                with ZipFile('runs/%s/code.zip' % date_time, 'w') as code_writer:
                    for path in python_paths:
                        code_writer.write(path)
            
            clean = input("Should I clean the repository?")
            
            if clean=='y' or clean=='yes':
                #delete training data from repo main files
                G_AB_paths = glob('models/G_AB/*.h5')
                G_AB_all_paths = glob('models/G_AB_all/*.h5')
                G_BA_paths = glob('models/G_BA/*.h5')
                E_A_paths = glob('models/E_A/*.h5')
                E_B_paths = glob('models/E_B/*.h5')
                D_A_paths = glob('models/D_A/*.h5')
                D_B_paths = glob('models/D_B/*.h5')
                D_Za_paths = glob('models/D_Za/*.h5')
                D_Zb_paths = glob('models/D_Zb/*.h5')
                
                self.delete_models(G_AB_paths)
                self.delete_models(G_AB_all_paths)
                self.delete_models(G_BA_paths)
                self.delete_models(E_A_paths)
                self.delete_models(E_B_paths)
                self.delete_models(D_A_paths)
                self.delete_models(D_B_paths)
                self.delete_models(D_Za_paths)
                self.delete_models(D_Zb_paths)
            
                
            
            
model = AugCycleGAN((100,100,3), (1,1,2), resume=False)
model.train(epochs=100, batch_size = 1)



    

        

        
