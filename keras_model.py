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

#load functions and classes from other .py files within the repository
from data_loader import DataLoader
#from evaluator import evaluator 


#keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from keras_modules import blur
from keras_networks import G_AB, G_BA, E_A, E_B, D_A, D_B, D_Za, D_Zb, N_map
from keras_evaluator import evaluator
from tensorflow.keras.losses import mse
import tensorflow as tf

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
    def __init__(self, img_shape, latent_shape):
        self.img_shape = img_shape
        self.latent_shape = latent_shape
        
        #log settings
        self.eval_training_points = []
        self.avg_ssim, self.avg_min_ssim, self.avg_max_ssim = [], [], []
        self.ep_avg_ssim, self.ep_avg_min_ssim, self.ep_avg_max_ssim = [], [], []
        
        #discriminator ground-truths
        self.D_A_out_shape = (img_shape[0]//4, img_shape[1]//4, 1)
        self.D_B_out_shape = (img_shape[0]//4, img_shape[1]//4, 1)
        
        #configure data loader
        self.data_loader = DataLoader(img_res=(self.img_shape[0], self.img_shape[1]))
        
        self.G_AB = G_AB(img_shape, latent_shape)
        self.G_BA = G_BA(img_shape, latent_shape)
        self.E_A = E_A(img_shape, latent_shape)
        self.E_B = E_B(img_shape, latent_shape)
        self.D_A, self.D_A_static = D_A(img_shape)
        self.D_B, self.D_B_static = D_B(img_shape)
        self.D_Za, self.D_Za_static = D_Za(latent_shape) 
        self.D_Zb, self.D_Zb_static = D_Zb(latent_shape)
        self.blurring = blur(img_shape)
        
        self.G_AB_opt = self.G_BA_opt = Adam(lr=0.0002, beta_1=0.5)
        self.D_A_opt = self.D_B_opt = Adam(lr=0.0002, beta_1=0.5)
        self.E_A_opt = self.E_B_opt = Adam(lr=0.0002, beta_1=0.5)
        self.D_Za_opt = self.D_Zb_opt = Adam(lr=0.0002, beta_1=0.5)
    
    def supervised_step(self, a, b):     
        with tf.GradientTape(persistent=True) as tape:
            z_a_hat = self.E_A([a,b], training=True)
            a_hat = self.G_BA([b,z_a_hat], training=True)
            sup_loss_a = L1_loss(a,a_hat)
            
            z_b_hat = self.E_B([a,b], training=True)
            b_hat = self.G_AB([a, z_b_hat], training=True)
            sup_loss_b = L1_loss(b,b_hat)
                
        
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
            b_hat = self.G_AB([a, z_b], training=True)
            fake_b = self.D_B(b_hat, training=True)
            
            z_a_hat = self.E_A([a, b_hat], training=True)
            fake_z_a = self.D_Za(z_a_hat, training=True)
    
    		#2nd map
            a_cyc = self.G_BA([b_hat, z_a_hat], training=True)
            z_b_cyc = self.E_B([a, b_hat], training=True)
        
            D_B_loss = discriminator_loss(self.D_B(b, training=True), fake_b)
            D_Za_loss = discriminator_loss(self.D_Za(z_a, training=True), fake_z_a)
            
            cycle_A_Zb_loss = gen_loss(fake_b)+gen_loss(fake_z_a)+L1_loss(a_cyc,a)+L1_loss(z_b_cyc,z_b)

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
            a_hat = self.G_BA([b, z_a], training=True)
            fake_a = self.D_A(a_hat, training=True)
            
            z_b_hat = self.E_B([a_hat, b], training=True)
            fake_z_b = self.D_Zb(z_b_hat, training=True)
			
			#2nd map
            b_cyc = self.G_AB([a_hat, z_b_hat], training=True)
            z_a_cyc = self.E_A([a_hat, b], training=True)
        
            D_A_loss = discriminator_loss(self.D_A(a, training=True), fake_a)
            D_Zb_loss = discriminator_loss(self.D_Zb(z_b, training=True), fake_z_b)
            
            cycle_B_Za_loss = gen_loss(fake_a)+gen_loss(fake_z_b)+L1_loss(b_cyc,b)+L1_loss(z_a_cyc,z_a)

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
            
    def train(self, epochs, batch_size=10, sample_interval=50):
        start_time = datetime.datetime.now()
        def chop_microseconds(delta):
            #utility to help avoid printing the microseconds
            return delta - datetime.timedelta(microseconds=delta.microseconds)
        
        #create a dynamic evaluator object
        dynamic_evaluator = evaluator(self.img_shape, self.latent_shape)
        for epoch in range(epochs):
            for batch, (img_A, img_B, sup_img_A, sup_img_B) in enumerate(self.data_loader.load_batch(batch_size)):
                training_point = np.around(epoch+batch/self.data_loader.n_batches, 3)
                #generate the noise vectors from the N(0,sigma^2) distribution
                
                for i in range(4):
                    z_a = np.random.randn(batch_size, 1, 1, self.latent_shape[-1])
                    z_b = np.random.randn(batch_size, 1, 1, self.latent_shape[-1])
                    
                    D_B_loss, D_Za_loss, cycle_A_Zb_loss = self.step_cycle_A(img_A, img_B, z_a, z_b)
                    D_A_loss, D_Zb_loss, cycle_B_Za_loss = self.step_cycle_B(img_A, img_B, z_a, z_b)
                
                sup_a, sup_b = self.supervised_step(sup_img_A, sup_img_B)
                
                elapsed_time = chop_microseconds(datetime.datetime.now() - start_time)
                print('[%d/%d][%d/%d]-[%s:%.3f %s:%.3f %s:%.3f %s:%.3f]-[%s:%.3f %s:%.3f]-[%s:%.3f %s:%.3f]-[time:%s]'
                      % (epoch, epochs, batch, self.data_loader.n_batches,
                         'D_A', D_A_loss, 'D_B', D_B_loss, 'D_Za', D_Za_loss, 'D_Zb', D_Zb_loss,
                         'cyc_A_Zb', cycle_A_Zb_loss, 'cyc_B_Za', cycle_B_Za_loss,
                         'sup_a', sup_a, 'sup_b', sup_b, elapsed_time))

                if batch % 100 == 0 and not(batch==0 and epoch==0):
                    self.eval_training_points.append(training_point)
                    
                    dynamic_evaluator.model = self.G_AB
                    #Perceptual Evaluation
                    dynamic_evaluator.test(batch_size=5, num_out_imgs=5, training_point=training_point, test_type='perception')
                    #Distortion Evaluation
                    avg_ssim, avg_min_ssim, avg_max_ssim = dynamic_evaluator.test(batch_size=30, num_out_imgs=10, training_point=training_point, test_type='distortion')
                    self.avg_ssim.append(avg_ssim)
                    self.avg_min_ssim.append(avg_min_ssim)
                    self.avg_max_ssim.append(avg_max_ssim)
                    
                    plt.figure(figsize=(21,15))
                    plt.plot(self.eval_training_points, self.avg_ssim, label='Avg Mean SSIM')
                    plt.plot(self.eval_training_points, self.avg_min_ssim, label='Avg Min SSIM')
                    plt.plot(self.eval_training_points, self.avg_max_ssim, label='Avg Max SSIM')
                    plt.legend()
                    plt.savefig('progress/distortion/distortion_performance.png', bbox_inches='tight')
                    
                    #save the generators
                    self.G_AB.save("models/G_AB_{}_{}.h5".format(epoch, batch))
                    self.G_BA.save("models/G_BA_{}_{}.h5".format(epoch, batch))
            
            avg_ssim, avg_min_ssim, avg_max_ssim = dynamic_evaluator.test(batch_size=100, num_out_imgs=20, training_point=None, test_type='distortion')
            self.ep_avg_ssim.append(avg_ssim)
            self.ep_avg_min_ssim.append(avg_min_ssim)
            self.ep_avg_max_ssim.append(avg_max_ssim)
            
            plt.figure(figsize=(21,15))
            plt.plot(self.ep_avg_ssim, label='Avg Mean SSIM')
            plt.plot(self.ep_avg_min_ssim, label='Avg Min SSIM')
            plt.plot(self.ep_avg_max_ssim, label='Avg Max SSIM')
            plt.legend()
            plt.savefig('progress/distortion/ep_distortion_performance.png', bbox_inches='tight')
            
            
model = AugCycleGAN((100,100,3), (1,1,4))
model.train(epochs=10, batch_size = 1)



    

        

        