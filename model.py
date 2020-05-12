import datetime
import time
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
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model
from modules import blur
from networks import G_AB, G_BA, E_A, E_B, D_A, D_B, D_Za, D_Zb, N_map
from evaluator import evaluator
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
      
        self.train_info['performance']['avg_min_lpips']=[[],[]]
        self.train_info['performance']['avg_mean_lpips']=[[],[]]
        self.train_info['performance']['avg_max_lpips']=[[],[]]
        
        self.train_info['performance']['avg_min_ssim']=[[],[]]
        self.train_info['performance']['avg_mean_ssim']=[[],[]]
        self.train_info['performance']['avg_max_ssim']=[[],[]]
        
        self.train_info['performance']['avg_min_div']=[[],[]]
        self.train_info['performance']['avg_mean_div']=[[],[]]
        self.train_info['performance']['avg_max_div']=[[],[]]
        


        #configure data loader
        self.data_loader = DataLoader(img_res=(self.img_shape[0], self.img_shape[1]))
        
        #TRAINING PARAMETERS
        #perceptual paths length parameters
        self.pl_mean_G_AB = 0.
        self.pl_mean_G_BA = 0.
        
        #Exponential moving average parameters
        self.beta=0.99
        
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
        
        #create the inference model with exponential moving average of the weights
        self.G_AB_EMA = clone_model(self.G_AB)
        self.G_AB_EMA.set_weights(self.G_AB.get_weights())
        #---------------------
        self.G_BA_EMA = clone_model(self.G_BA)
        self.G_BA_EMA.set_weights(self.G_BA.get_weights())
        #--------------------------------------------------
        
        
        #set the optimizers of all models
        self.G_AB_opt = self.G_BA_opt = Adam(lr=0.0002, beta_1=0.5)
        self.D_A_opt = self.D_B_opt = Adam(lr=0.0002, beta_1=0.5)
        self.E_A_opt = self.E_B_opt = Adam(lr=0.0002, beta_1=0.5)
        self.D_Za_opt = self.D_Zb_opt = Adam(lr=0.0002, beta_1=0.5)
        
    
    def EMA(self,):
        for i in range(len(self.G_AB.layers)):
            up_weight = self.G_AB.layers[i].get_weights()
            old_weight = self.G_AB_EMA.layers[i].get_weights()
            new_weight = []
            for j in range(len(up_weight)):
                new_weight.append(old_weight[j] * self.beta + (1-self.beta) * up_weight[j])
            self.G_AB_EMA.layers[i].set_weights(new_weight)
            
        for i in range(len(self.G_BA.layers)):
            up_weight = self.G_BA.layers[i].get_weights()
            old_weight = self.G_BA_EMA.layers[i].get_weights()
            new_weight = []
            for j in range(len(up_weight)):
                new_weight.append(old_weight[j] * self.beta + (1-self.beta) * up_weight[j])
            self.G_BA_EMA.layers[i].set_weights(new_weight)
        

    def EMA_init(self,):
        self.G_AB_EMA.set_weights(self.G_AB.get_weights())
        self.G_BA_EMA.set_weights(self.G_BA.get_weights())

        
    def supervised_step(self, a, b):     
        with tf.GradientTape(persistent=True) as tape:
            z_a_hat = self.E_A([a,b], training=True)
            fake_z_a = self.D_Za(z_a_hat, training=False) #used for the regularisation og the E_A encoder
            a_hat = self.G_BA([b,z_a_hat], training=True)

            #sup_perc_a = self.lpips.distance(a,a_hat)
            #self.train_info['losses']['sup']['perc_a'].append(sup_perc_a)
            
            sup_dist_a = L1_loss(a,a_hat)
            self.train_info['losses']['sup']['dist_a'].append(sup_dist_a)
            
            sup_loss_a = sup_dist_a
            adv_gen_Za = gen_loss(fake_z_a)
            E_A_loss = sup_loss_a + 0.5*adv_gen_Za
            
            #---------------------------------------------------------------
            z_b_hat = self.E_B([a,b], training=True)
            fake_z_b = self.D_Zb(z_b_hat, training=False)
            b_hat = self.G_AB([a, z_b_hat], training=True)
            
            
            #sup_perc_b = self.lpips.distance(b,b_hat)
            #self.train_info['losses']['sup']['perc_b'].append(sup_perc_b)
            
            sup_dist_b = L1_loss(b,b_hat)
            self.train_info['losses']['sup']['dist_b'].append(sup_dist_b)
            
            sup_loss_b = sup_dist_b
            adv_gen_Zb = gen_loss(fake_z_b)
            E_B_loss = sup_loss_b + 0.5*adv_gen_Zb
        
        #supervised loss a
        G_BA_grads = tape.gradient(sup_loss_a, self.G_BA.trainable_variables)
        self.G_BA_opt.apply_gradients(zip(G_BA_grads, self.G_BA.trainable_variables))
        
        E_A_grads = tape.gradient(E_A_loss, self.E_A.trainable_variables)
        self.E_A_opt.apply_gradients(zip(E_A_grads, self.E_A.trainable_variables))
        
        #supervised loss b
        G_AB_grads = tape.gradient(sup_loss_b, self.G_AB.trainable_variables)
        self.G_AB_opt.apply_gradients(zip(G_AB_grads, self.G_AB.trainable_variables))
        
        E_B_grads = tape.gradient(E_B_loss, self.E_B.trainable_variables)
        self.E_B_opt.apply_gradients(zip(E_B_grads, self.E_B.trainable_variables))
        
        return sup_loss_a, sup_loss_b
                
    
    def step_cycle_A(self, a, b, z_a, z_b, ppl=False):
        
        with tf.GradientTape(persistent=True) as tape:
            #1st map
            b_hat = self.G_AB([a, z_b], training=True)
            fake_b = self.D_B(b_hat, training=True)
            
            if ppl:
                z_b_dash = z_b + 0.15*tf.random.normal((a.shape[0], self.latent_shape[-1]), dtype=tf.float32)
                b_hat_dash = self.G_AB([a,z_b_dash], training=True)
                pl_lengths_G_AB = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(b_hat-b_hat_dash), axis=[1,2,3]))
                ppl_loss_G_AB = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(pl_lengths_G_AB - self.pl_mean_G_AB)))
                self.train_info['losses']['reg']['ppl_G_AB'].append(ppl_loss_G_AB)
            
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
            
            cycle_A_Zb_loss = adv_gen_B + adv_gen_Za + rec_a_dist + rec_Zb
            
            if ppl:
                print("------------------------------------")
                print(cycle_A_Zb_loss)
                print(ppl_loss_G_AB)
                G_AB_loss = cycle_A_Zb_loss + ppl_loss_G_AB
                print(G_AB_loss)
                #do the exponential moving average update step for the mean ppl
                if self.pl_mean_G_AB==0.:
                    self.pl_mean_G_AB = tf.math.reduce_mean(pl_lengths_G_AB)
                else:
                    self.pl_mean_G_AB = 0.999*self.pl_mean_G_AB + 0.001*tf.math.reduce_mean(pl_lengths_G_AB)
            else:
                G_AB_loss = cycle_A_Zb_loss

        D_B_grads = tape.gradient(D_B_loss, self.D_B.trainable_variables)
        self.D_B_opt.apply_gradients(zip(D_B_grads, self.D_B.trainable_variables))
                
        D_Za_grads = tape.gradient(D_Za_loss, self.D_Za.trainable_variables)
        self.D_Za_opt.apply_gradients(zip(D_Za_grads, self.D_Za.trainable_variables))
 
        G_AB_grads = tape.gradient(G_AB_loss, self.G_AB.trainable_variables)
        self.G_AB_opt.apply_gradients(zip(G_AB_grads, self.G_AB.trainable_variables))

        E_A_grads = tape.gradient(cycle_A_Zb_loss, self.E_A.trainable_variables)
        self.E_A_opt.apply_gradients(zip(E_A_grads, self.E_A.trainable_variables))

		#Update G_BA and E_B only based on cycle starting from B
        G_BA_grads = tape.gradient(cycle_A_Zb_loss, self.G_BA.trainable_variables)
        self.G_BA_opt.apply_gradients(zip(G_BA_grads, self.G_BA.trainable_variables))

        E_B_grads = tape.gradient(cycle_A_Zb_loss, self.E_B.trainable_variables)
        self.E_B_opt.apply_gradients(zip(E_B_grads, self.E_B.trainable_variables))
        
        return D_B_loss, D_Za_loss, cycle_A_Zb_loss
    
    def step_cycle_B(self, a, b, z_a, z_b, ppl=False):
        #z_a2 = z_a + 0.07*tf.random.normal((b.shape[0], 1, 1, self.latent_shape[-1]), dtype=tf.float32)
        
        with tf.GradientTape(persistent=True) as tape:
            #1st map
            a_hat = self.G_BA([b, z_a], training=True)
            fake_a = self.D_A(a_hat, training=True)
            
            if ppl:
                z_a_dash = z_a + 0.15*tf.random.normal((b.shape[0], self.latent_shape[-1]), dtype=tf.float32)
                a_hat_dash = self.G_BA([b,z_a_dash], training=True)
                pl_lengths_G_BA = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(a_hat-a_hat_dash), axis=[1,2,3]))
                ppl_loss_G_BA = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(pl_lengths_G_BA - self.pl_mean_G_BA)))
                self.train_info['losses']['reg']['ppl_G_BA'].append(ppl_loss_G_BA)
            
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
            
            cycle_B_Za_loss = adv_gen_A + adv_gen_Zb + rec_b_dist + rec_Za
            
            if ppl:
                G_BA_loss = cycle_B_Za_loss + ppl_loss_G_BA
                
                #update step
                if self.pl_mean_G_BA==0.:
                    self.pl_mean_G_BA = tf.math.reduce_mean(pl_lengths_G_BA)
                else:
                    self.pl_mean_G_BA = 0.999*self.pl_mean_G_BA + 0.001*tf.math.reduce_mean(pl_lengths_G_BA)
            else:
                G_BA_loss = cycle_B_Za_loss
            

        D_A_grads = tape.gradient(D_A_loss, self.D_A.trainable_variables)
        self.D_A_opt.apply_gradients(zip(D_A_grads, self.D_A.trainable_variables))
                
        D_Zb_grads = tape.gradient(D_Zb_loss, self.D_Zb.trainable_variables)
        self.D_Zb_opt.apply_gradients(zip(D_Zb_grads, self.D_Zb.trainable_variables))
                
        G_AB_grads = tape.gradient(cycle_B_Za_loss, self.G_AB.trainable_variables)
        self.G_AB_opt.apply_gradients(zip(G_AB_grads, self.G_AB.trainable_variables))

        E_A_grads = tape.gradient(cycle_B_Za_loss, self.E_A.trainable_variables)
        self.E_A_opt.apply_gradients(zip(E_A_grads, self.E_A.trainable_variables))

		#Update G_BA and E_B only based on cycle starting from B
        G_BA_grads = tape.gradient(G_BA_loss, self.G_BA.trainable_variables)
        self.G_BA_opt.apply_gradients(zip(G_BA_grads, self.G_BA.trainable_variables))

        E_B_grads = tape.gradient(cycle_B_Za_loss, self.E_B.trainable_variables)
        self.E_B_opt.apply_gradients(zip(E_B_grads, self.E_B.trainable_variables))
        
        return D_A_loss, D_Zb_loss, cycle_B_Za_loss
    
    def mode_seeking_regularisation(self, a, b):
        with tf.GradientTape(persistent=True) as tape:
            z_b = tf.random.normal((a.shape[0], self.latent_shape[-1]), dtype=tf.float32)
            b_hat = self.G_AB([a,z_b], training=True)
                
            z_b_dash = tf.random.normal((a.shape[0], self.latent_shape[-1]), dtype=tf.float32)
            b_hat_dash = self.G_AB([a,z_b_dash], training=True)
            
            mode_seeking_loss_AB = -1*self.lpips.distance(b_hat, b_hat_dash)/(tf.norm(z_b - z_b_dash)+1e-8)
            self.train_info['losses']['reg']['ms_G_AB'].append(-1*mode_seeking_loss_AB)
            
            #-----------------------------------------------
            z_a = tf.random.normal((b.shape[0], self.latent_shape[-1]), dtype=tf.float32)
            a_hat = self.G_BA([b,z_a], training=True)
                
            z_a_dash = tf.random.normal((b.shape[0], self.latent_shape[-1]), dtype=tf.float32)
            a_hat_dash = self.G_BA([b,z_a_dash], training=True)
            
            mode_seeking_loss_BA = -1*self.lpips.distance(a_hat, a_hat_dash)/(tf.norm(z_a - z_a_dash)+1e-8)
            self.train_info['losses']['reg']['ms_G_BA'].append(-1*mode_seeking_loss_BA)
            
        #update the generator models G_AB and G_BA
        G_AB_grads = tape.gradient(mode_seeking_loss_AB, self.G_AB.trainable_variables)
        self.G_AB_opt.apply_gradients(zip(G_AB_grads, self.G_AB.trainable_variables))
        
        G_BA_grads = tape.gradient(mode_seeking_loss_BA, self.G_BA.trainable_variables)
        self.G_BA_opt.apply_gradients(zip(G_BA_grads, self.G_BA.trainable_variables))
            
        
    def ppl_regularisation(self, a, b):
        #every M steps we regularise the perceptual path length
        #we have 2 generators (G_AB, G_BA)
        
        with tf.GradientTape(persistent=True) as tape:
            z_b = tf.random.normal((a.shape[0], self.latent_shape[-1]), dtype=tf.float32)
            b_hat = self.G_AB([a,z_b], training=True)
            
            z_b_dash = z_b + 0.1*tf.random.normal((a.shape[0], self.latent_shape[-1]), dtype=tf.float32)
            b_hat_dash = self.G_AB([a,z_b_dash], training=True)
            
            pl_lengths_G_AB = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(b_hat-b_hat_dash), axis=[1,2,3]))
            
            ppl_loss_G_AB = tf.math.reduce_mean(tf.math.square(pl_lengths_G_AB - self.pl_mean_G_AB))
            self.train_info['losses']['reg']['ppl_G_AB'].append(ppl_loss_G_AB)
            
            
            #---------------------------------------------------------------------------------------
            z_a = tf.random.normal((b.shape[0], self.latent_shape[-1]), dtype=tf.float32)
            a_hat = self.G_BA([b,z_a], training=True)
            
            z_a_dash = z_a + 0.1*tf.random.normal((b.shape[0], self.latent_shape[-1]), dtype=tf.float32)
            a_hat_dash = self.G_BA([b,z_a_dash], training=True)
            
            pl_lengths_G_BA = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(a_hat-a_hat_dash), axis=[1,2,3]))
            
            ppl_loss_G_BA = tf.math.reduce_mean(tf.math.square(pl_lengths_G_BA - self.pl_mean_G_BA))
            self.train_info['losses']['reg']['ppl_G_BA'].append(ppl_loss_G_BA)
            
        
        #update the generator models
        ppl_G_AB_grads = tape.gradient(ppl_loss_G_AB, self.G_AB.trainable_variables)
        self.G_AB_opt.apply_gradients(zip(ppl_G_AB_grads, self.G_AB.trainable_variables))
        
        ppl_G_BA_grads = tape.gradient(ppl_loss_G_BA, self.G_BA.trainable_variables)
        self.G_BA_opt.apply_gradients(zip(ppl_G_BA_grads, self.G_BA.trainable_variables))
            
        
        if self.pl_mean_G_AB==0.:
            self.pl_mean_G_AB = tf.math.reduce_mean(pl_lengths_G_AB)
        else:
            self.pl_mean_G_AB = 0.999*self.pl_mean_G_AB + 0.001*tf.math.reduce_mean(pl_lengths_G_AB)
            
        if self.pl_mean_G_BA==0.:
            self.pl_mean_G_BA = tf.math.reduce_mean(pl_lengths_G_BA)
        else:
            self.pl_mean_G_BA = 0.999*self.pl_mean_G_BA + 0.001*tf.math.reduce_mean(pl_lengths_G_BA)
        

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
                for batch, (img_A, img_B) in enumerate(self.data_loader.load_unpaired_batch(batch_size, portion = 1)):
                    img_A = tf.convert_to_tensor(img_A, dtype=tf.float32)
                    img_B = tf.convert_to_tensor(img_B, dtype=tf.float32)
                    
                    #sup_img_A = tf.convert_to_tensor(sup_img_A, dtype=tf.float32)
                    #sup_img_B = tf.convert_to_tensor(sup_img_B, dtype=tf.float32)
                        
                    z_a = tf.random.normal((batch_size, self.latent_shape[-1]), dtype=tf.float32)
                    z_b = tf.random.normal((batch_size, self.latent_shape[-1]), dtype=tf.float32)
                    
                    if batch % 16 == 0:
                        ppl=True
                    else:
                        ppl=False
                        
                    D_B_loss, D_Za_loss, cycle_A_Zb_loss = self.step_cycle_A(img_A, img_B, z_a, z_b, ppl)
                    D_A_loss, D_Zb_loss, cycle_B_Za_loss = self.step_cycle_B(img_A, img_B, z_a, z_b, ppl)
                    
                    #sup_a, sup_b = self.supervised_step(sup_img_A, sup_img_B)
                    
                        
                    if batch % 10 == 0 and not(batch==0 and epoch==0):
                        self.EMA() #update the inference model with exponential moving average

                    #generate the noise vectors from the N(0,sigma^2) distribution
                    if batch % 50 == 0 and not(batch==0 and epoch==0):
                        elapsed_time = chop_microseconds(datetime.datetime.now() - start_time)
                        print('[%d/%d][%d/%d]-[%s:%.3f %s:%.3f %s:%.3f %s:%.3f]-[%s:%.3f %s:%.3f %s:%.3f %s:%.3f]-[%s:%.3f %s:%.3f %s:%.3f %s:%.3f]-[time:%s]'
                              % (epoch, epochs, batch, self.data_loader.n_batches,
                                 'D_A', D_A_loss, 
                                 'D_B', D_B_loss, 
                                 'D_Za', D_Za_loss, 
                                 'D_Zb', D_Zb_loss,
                                 'Adv_A', self.train_info['losses']['unsup']['G_A'][-1] , 
                                 'Adv_B', self.train_info['losses']['unsup']['G_B'][-1],
                                 'Adv_Za', self.train_info['losses']['unsup']['E_A'][-1],
                                 'Adv_Zb', self.train_info['losses']['unsup']['E_B'][-1],
                                 'Rec_A', self.train_info['losses']['unsup']['rec_a_dist'][-1],
                                 'Rec_B', self.train_info['losses']['unsup']['rec_b_dist'][-1],
                                 'Rec_Za', self.train_info['losses']['unsup']['rec_Za'][-1],
                                 'Rec_Zb', self.train_info['losses']['unsup']['rec_Zb'][-1],
                                 elapsed_time))
    
                    if batch % 100 == 0 and not(batch==0 and epoch==0):
                        training_point = np.around(epoch+batch/self.data_loader.n_batches, 4)
                        self.train_info['performance']['eval_points'].append(training_point)
                        
                        #set the models
                        dynamic_evaluator.G_AB = self.G_AB_EMA
                        dynamic_evaluator.G_BA = self.G_BA_EMA
                        
                        #Perception and distortion evaluation
                        
                        #time_start=time.time()
                        dynamic_evaluator.test(batch_size=5, num_out_imgs=5, training_point=training_point, test_type='visual')
                        #mixed_duration = time.time()-time_start
                        #print('Mixed Evaluation took %.3f seconds' % mixed_duration)
                        
                        """
                        self.train_info['performance']['avg_min_lpips'][0].append(info['avg_min_lpips'])
                        self.train_info['performance']['avg_mean_lpips'][0].append(info['avg_mean_lpips'])
                        self.train_info['performance']['avg_max_lpips'][0].append(info['avg_max_lpips'])
                        
                        self.train_info['performance']['avg_min_ssim'][0].append(info['avg_min_ssim'])
                        self.train_info['performance']['avg_mean_ssim'][0].append(info['avg_mean_ssim'])
                        self.train_info['performance']['avg_max_ssim'][0].append(info['avg_max_ssim'])
                        
                        self.train_info['performance']['avg_min_div'][0].append(info['avg_min_div'])
                        self.train_info['performance']['avg_mean_div'][0].append(info['avg_mean_div'])
                        self.train_info['performance']['avg_max_div'][0].append(info['avg_max_div'])
                        
                        plt.figure(figsize=(21,15))
                        plt.title('Diversity')
                        plt.plot(self.train_info['performance']['eval_points'], self.train_info['performance']['avg_min_div'][0], label='min')
                        plt.plot(self.train_info['performance']['eval_points'], self.train_info['performance']['avg_mean_div'][0], label='mean')
                        plt.plot(self.train_info['performance']['eval_points'], self.train_info['performance']['avg_max_div'][0], label='max')
                        plt.legend()
                        plt.savefig('progress/diversity/diversity.png', bbox_inches='tight')
                        
                        plt.figure(figsize=(21,15))
                        plt.title('SSIM')
                        plt.plot(self.train_info['performance']['eval_points'], self.train_info['performance']['avg_min_ssim'][0], label='min')
                        plt.plot(self.train_info['performance']['eval_points'], self.train_info['performance']['avg_mean_ssim'][0], label='mean')
                        plt.plot(self.train_info['performance']['eval_points'], self.train_info['performance']['avg_max_ssim'][0], label='max')
                        plt.legend()
                        plt.savefig('progress/distortion/SSIM.png', bbox_inches='tight')
                        
                        
                        plt.figure(figsize=(21,15))
                        plt.title('LPIPS')
                        plt.plot(self.train_info['performance']['eval_points'], self.train_info['performance']['avg_min_lpips'][0], label='min')
                        plt.plot(self.train_info['performance']['eval_points'], self.train_info['performance']['avg_mean_lpips'][0], label='mean')
                        plt.plot(self.train_info['performance']['eval_points'], self.train_info['performance']['avg_max_lpips'][0], label='max')
                        plt.legend()
                        plt.savefig('progress/perception/LPIPS.png', bbox_inches='tight')
                        
                        plt.close('all')
                        """
                        
                        #save the generators
                        self.G_AB_EMA.save("models/G_AB_all/G_AB_{}_{}.h5".format(epoch, batch))
                        self.G_BA_EMA.save("models/G_BA_all/G_BA_{}_{}.h5".format(epoch, batch))
                    
                    if batch % 100 ==0 and epoch % 10==0:
                        self.EMA_init() #restart the G_AB_EMA from the current state of the G_AB
                
                
                """
                dynamic_evaluator.model = self.G_AB_EMA #set the current G_AB_EMA model for evaluation
                #Perception and distortion evaluation on the entire test dataset
                info = dynamic_evaluator.test(batch_size=250, num_out_imgs=25, training_point=training_point, test_type='mixed')
                
                self.train_info['performance']['avg_min_lpips'][1].append(info['avg_min_lpips'])
                self.train_info['performance']['avg_mean_lpips'][1].append(info['avg_mean_lpips'])
                self.train_info['performance']['avg_max_lpips'][1].append(info['avg_max_lpips'])
                
                self.train_info['performance']['avg_min_ssim'][1].append(info['avg_min_ssim'])
                self.train_info['performance']['avg_mean_ssim'][1].append(info['avg_mean_ssim'])
                self.train_info['performance']['avg_max_ssim'][1].append(info['avg_max_ssim'])
                
                self.train_info['performance']['avg_min_div'][1].append(info['avg_min_div'])
                self.train_info['performance']['avg_mean_div'][1].append(info['avg_mean_div'])
                self.train_info['performance']['avg_max_div'][1].append(info['avg_max_div'])
                
                plt.figure(figsize=(21,15))
                plt.title('Diversity')
                plt.plot(self.train_info['performance']['avg_min_div'][1], label='min')
                plt.plot(self.train_info['performance']['avg_mean_div'][1], label='mean')
                plt.plot(self.train_info['performance']['avg_max_div'][1], label='max')
                plt.legend()
                plt.savefig('progress/diversity/diversity_epoch.png', bbox_inches='tight')
                
                plt.figure(figsize=(21,15))
                plt.title('SSIM')
                plt.plot(self.train_info['performance']['avg_min_ssim'][1], label='min')
                plt.plot(self.train_info['performance']['avg_mean_ssim'][1], label='mean')
                plt.plot(self.train_info['performance']['avg_max_ssim'][1], label='max')
                plt.legend()
                plt.savefig('progress/distortion/SSIM_epoch.png', bbox_inches='tight')
                
                
                plt.figure(figsize=(21,15))
                plt.title('LPIPS')
                plt.plot(self.train_info['performance']['avg_min_lpips'][1], label='min')
                plt.plot(self.train_info['performance']['avg_mean_lpips'][1], label='mean')
                plt.plot(self.train_info['performance']['avg_max_lpips'][1], label='max')
                plt.legend()
                plt.savefig('progress/perception/LPIPS_epoch.png', bbox_inches='tight')
                
                plt.close('all')
                """
                
                #save the models to intoduce resume capacity to training
                self.save_models(epoch)
                
                #save the tensorboard values
                with open('progress/training_information/'+ 'train_info' + '.pkl', 'wb') as f:
                    pickle.dump(self.train_info, f, pickle.HIGHEST_PROTOCOL)
            
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
            
                
            
            
model = AugCycleGAN((256,256,3), (1,1,32), resume=False)
model.train(epochs=100, batch_size = 10)