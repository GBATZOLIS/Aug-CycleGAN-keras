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
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input

from keras_modules import blur
from keras_networks import G_AB, G_BA, E_A, E_B, D_A, D_B, D_Za, D_Zb
from keras_evaluator import evaluator

#visualisation packages
import matplotlib.pyplot as plt
  
class AugCycleGAN(object):
    def __init__(self, img_shape, latent_shape):
        self.img_shape = img_shape
        self.latent_shape = latent_shape
        
        #log settings
        self.eval_training_points = []
        self.avg_ssim, self.avg_min_ssim, self.avg_max_ssim = [], [], []
        
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
        
        #Compile the Discriminators
        self.D_A.compile(loss='mse',  optimizer=Adam(lr=0.0002, beta_1=0.5))
        self.D_B.compile(loss='mse',  optimizer=Adam(lr=0.0002, beta_1=0.5))
        self.D_Za.compile(loss='mse',  optimizer=Adam(lr=0.0002, beta_1=0.5))
        self.D_Zb.compile(loss='mse',  optimizer=Adam(lr=0.0002, beta_1=0.5))
        
        """
        #Instantiate the forward and backward combined cyclic models
        
        #combined model starting from domain A
        self.c_cyclic_A = self.combined_cyclic_A() 
        self.c_cyclic_A.compile(loss=['mse', 'mse', 'mae', 'mae'], loss_weights=[1,1,1,1], optimizer=Adam(0.0002, 0.5))
        
        #combined model starting from domain B
        
        self.c_cyclic_B = self.combined_cyclic_B() 
        self.c_cyclic_B.compile(loss=['mse', 'mse', 'mae', 'mae'], loss_weights=[1,1,1,1], optimizer=Adam(0.0002, 0.5))
        """
        
        self.cyclic = self.combined_cyclic()
        self.cyclic.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mse', 'mse', 'mae', 'mae', 'mae'], loss_weights=[1,1,1,1,1,1,1,1,1,1], optimizer=Adam(0.0002, 0.5))
        
        self.sup_cyclic = self.supervised_cycle()
        self.sup_cyclic.compile(loss=['mae', 'mae'], loss_weights=[1,1], optimizer=Adam(0.0002, 0.5))
        
    def combined_cyclic(self,):
        inputs=[]
        outputs=[]
        
        self.D_B_static.trainable=False
        self.D_Za_static.trainable=False
        
        
        a=Input(self.img_shape)
        z_b=Input(self.latent_shape)
        
        #---------------------------------
        b_hat = self.G_AB([a, z_b])
        b_hat_blurred = self.blurring(b_hat)
        valid_b_hat = self.D_B_static(b_hat)
        
        z_a_hat = self.E_A([a, b_hat])
        valid_z_a_hat = self.D_Za_static(z_a_hat)
        #---------------------------------
        
        a_cyc = self.G_BA([b_hat, z_a_hat])
        z_b_cyc = self.E_B([a, b_hat])
        
        inputs.extend([a, z_b])
        outputs.extend([valid_b_hat, valid_z_a_hat, a_cyc, b_hat_blurred, z_b_cyc])

        self.D_A_static.trainable=False
        self.D_Zb_static.trainable=False
        
        b = Input(self.img_shape)
        z_a = Input(self.latent_shape)
        
        #---------------------------------
        a_hat = self.G_BA([b, z_a])
        a_hat_blurred = self.blurring(a_hat)
        valid_a_hat = self.D_A_static(a_hat)
        
        z_b_hat = self.E_B([a_hat, b])
        valid_z_b_hat = self.D_Zb_static(z_b_hat)
        #---------------------------------
        
        b_cyc = self.G_AB([a_hat, z_b_hat])
        z_a_cyc = self.E_A([a_hat, b])
        
        inputs.extend([b, z_a])
        outputs.extend([valid_a_hat, valid_z_b_hat, b_cyc, a_hat_blurred, z_a_cyc])
        
        
        model = Model(inputs = inputs, outputs = outputs, name='combined_cyclic')
        return model
    
    def supervised_cycle(self,):
        a=Input(self.img_shape)
        b=Input(self.img_shape)
        
        z_a_hat = self.E_A([a,b])
        a_hat = self.G_BA([b,z_a_hat])
        
        z_b_hat = self.E_B([a,b])
        b_hat = self.G_AB([a, z_b_hat])
        
        model = Model(inputs=[a,b], outputs=[a_hat, b_hat], name='Supervised Cyclic model')
        return model
            
        
    def generate_fake_samples(self, img_A, img_B, z_a, z_b):
        b_hat = self.G_AB.predict([img_A, z_b])
        z_a_hat = self.E_A.predict([img_A, b_hat])
        
        a_hat = self.G_BA.predict([img_B, z_a])
        z_b_hat = self.E_B.predict([a_hat, img_B])
        
        return a_hat, b_hat, z_a_hat, z_b_hat
    
        
    def train(self, epochs, batch_size=10, sample_interval=50):
        start_time = datetime.datetime.now()
        def chop_microseconds(delta):
            #utility to help avoid printing the microseconds
            return delta - datetime.timedelta(microseconds=delta.microseconds)
        
        valid_D_A = np.ones((batch_size,) + self.D_A_out_shape)
        valid_D_B = np.ones((batch_size,) + self.D_B_out_shape)
        fake_D_A = np.zeros((batch_size,) + self.D_A_out_shape)
        fake_D_B = np.zeros((batch_size,) + self.D_B_out_shape)
        
        valid_D_Za = np.ones((batch_size, 1))
        valid_D_Zb = np.ones((batch_size, 1))
        fake_D_Za = np.zeros((batch_size, 1))
        fake_D_Zb = np.zeros((batch_size, 1))
        
        #create a dynamic evaluator object
        dynamic_evaluator = evaluator(self.img_shape, self.latent_shape)
        for epoch in range(epochs):
            for batch, (img_A, img_B, sup_img_A, sup_img_B) in enumerate(self.data_loader.load_batch(batch_size)):
                training_point = np.around(epoch+batch/self.data_loader.n_batches, 3)
                
                #blur the imgA and imgB for appropriate blur supervision
                #we want to incite the mapping function to keep the low-frequencies unchanged!
                blur_img_A = self.blurring.predict(img_A)
                blur_img_B = self.blurring.predict(img_B)
                
                for noise_batch in range(2):
                    #generate the noise vectors from the N(0,sigma^2) distribution
                    z_a = np.random.randn(batch_size, 1, 1, self.latent_shape[-1])
                    z_b = np.random.randn(batch_size, 1, 1, self.latent_shape[-1])
                    
                    #Update the Discriminators with the samples from the real marginals
                    D_A_loss_real = self.D_A.train_on_batch(img_A, valid_D_A)
                    D_B_loss_real = self.D_B.train_on_batch(img_B, valid_D_B)
                    D_Za_loss_real = self.D_Za.train_on_batch(z_a, valid_D_Za)
                    D_Zb_loss_real = self.D_Zb.train_on_batch(z_b, valid_D_Zb)
                    
                    #Make the appropriate translations for training of the Discriminators on the fake samples
                    img_A_fake, img_B_fake, z_a_fake, z_b_fake = self.generate_fake_samples(img_A, img_B, z_a, z_b)
                    
                    #Update the discriminators using the fake samples
                    D_A_loss_fake = self.D_A.train_on_batch(img_A_fake, fake_D_A)
                    D_B_loss_fake = self.D_B.train_on_batch(img_B_fake, fake_D_B)
                    D_Za_loss_fake = self.D_Za.train_on_batch(z_a_fake, fake_D_Za)
                    D_Zb_loss_fake = self.D_Zb.train_on_batch(z_b_fake, fake_D_Zb)
    
    
                    
                    cc_loss = self.cyclic.train_on_batch([img_A, z_b, img_B, z_a],
                                                         [valid_D_B, valid_D_Za, img_A, blur_img_A, z_b, valid_D_A, valid_D_Zb, img_B, blur_img_B, z_a])
                    
                    #Calculate losses
                    D_A_loss_mean = (D_A_loss_real + D_A_loss_fake)/2
                    D_B_loss_mean = (D_B_loss_real + D_B_loss_fake)/2
                    D_Za_loss_mean = (D_Za_loss_real + D_Za_loss_fake)/2
                    D_Zb_loss_mean = (D_Zb_loss_real + D_Zb_loss_fake)/2
                    Adv_Img = (cc_loss[1]+cc_loss[6])/2 
                    Adv_Noise = (cc_loss[2]+cc_loss[7])/2
                    RecImg = (cc_loss[3]+cc_loss[8])/2
                    Blur_loss = (cc_loss[4]+cc_loss[9])/2
                    RecN = (cc_loss[5]+cc_loss[10])/2
                    
                    elapsed_time = datetime.datetime.now() - start_time
                    elapsed_time = chop_microseconds(elapsed_time)
                    print('[epoch:%d/%d][img_batch:%d/%d][N_batch:%d/10]--[D_A:%.3f - D_B:%.3f - D_Za:%.3f - D_Zb:%.3f ]--[Adv_Img:%.3f - Adv_N:%.3f - RecImg:%.3f - BlurImg:%.3f - RecN:%.3f]--[elapsed time:%s]' 
                          % (epoch, epochs, batch, self.data_loader.n_batches,noise_batch+1, D_A_loss_mean, D_B_loss_mean, D_Za_loss_mean, D_Zb_loss_mean,
                             Adv_Img, Adv_Noise, RecImg, Blur_loss, RecN, elapsed_time))
                
                sup_loss = self.sup_cyclic.train_on_batch([sup_img_A, sup_img_B],[sup_img_A, sup_img_B])
                print('[epoch:%d/%d][img_batch:%d/%d][--------------] [supA:%.3f - supB:%.3f]'%(epoch, epochs, batch, self.data_loader.n_batches, sup_loss[1], sup_loss[2]))
                
                if batch % 50 == 0 and not(batch==0 and epoch==0):
                    self.eval_training_points.append(training_point)
                    
                    dynamic_evaluator.model = self.G_AB
                    #Perceptual Evaluation
                    dynamic_evaluator.test(batch_size=5, num_out_imgs=5, training_point=training_point, test_type='perception')
                    #Distortion Evaluation
                    avg_ssim, avg_min_ssim, avg_max_ssim = dynamic_evaluator.test(batch_size=25, num_out_imgs=10, training_point=training_point, test_type='distortion')
                    self.avg_ssim.append(avg_ssim)
                    self.avg_min_ssim.append(avg_min_ssim)
                    self.avg_max_ssim.append(avg_max_ssim)
                    
                    plt.figure(figsize=(21,15))
                    plt.plot(self.eval_training_points, self.avg_ssim, label='Avg Mean SSIM')
                    plt.plot(self.eval_training_points, self.avg_min_ssim, label='Avg Min SSIM')
                    plt.plot(self.eval_training_points, self.avg_max_ssim, label='Avg Max SSIM')
                    plt.legend()
                    plt.savefig('progress/distortion/distortion_performance.png', bbox_inches='tight')

model = AugCycleGAN((100,100,3), (1,1,16))
model.train(epochs=10, batch_size = 1)

    

        

        