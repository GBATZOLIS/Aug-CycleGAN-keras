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
K = tf.keras.backend


#visualisation packages
import matplotlib.pyplot as plt


from stargan_networks import G,E,D,F

  
def discriminator_loss(real, generated):
    # Multiplied by 0.5 so that it will train at half-speed
    return (tf.reduce_mean(mse(tf.ones_like(real), real)) + tf.reduce_mean(mse(tf.zeros_like(generated), generated))) * 0.5

# Measures how real the discriminator believes the fake image is
def gen_loss(validity):
    return tf.reduce_mean(mse(tf.ones_like(validity), validity))
        
def L1_loss(image1, image2):
    return tf.reduce_mean(tf.abs(image1 - image2))
        
class StarGANv2(object):
    def __init__(self, img_shape, latent_size, style_size, domains, resume=False):
        self.img_shape = img_shape
        self.latent_size = latent_size
        self.style_size = latent_size
        self.domains = domains
        
        #--------------log settings---------------------
        self.train_info={}
        
        #-----------------LOSSES------------------------
        self.train_info['losses'] = {}
        self.train_info['losses']['L_adv'] = []
        self.train_info['losses']['L_sty'] = []
        self.train_info['losses']['L_ds'] = []
        self.train_info['losses']['L_cyc'] = []
        
        
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
        
        #Weights of the losses of the objective
        self.l_sty = 1
        self.l_ds = 1
        self.l_cyc = 1
        
        #instantiate the LPIPS loss object
        self.lpips = lpips(self.img_shape)
        self.lpips.create_model()
        
        self.G = G(self.img_shape, self.style_size)
        self.E = E(self.img_shape, self.style_size, self.domains)
        self.D = D(self.img_shape, self.domains)
        self.F = F(self.latent_size, self.style_size, self.domains) #F has two output branches reflecting the two different domains
            
        if resume==True:
            self.G.load_weights(glob('models/G/*.h5')[-1])
            self.E.load_weights(glob('models/E/*.h5')[-1])
            self.D.load_weights(glob('models/D/*.h5')[-1])
            self.F.load_weights(glob('models/F/*.h5')[-1])
        
        #For evaluation we use exponential moving average of all modules except D
        self.G_EMA = clone_model(self.G)
        self.G_EMA.set_weights(self.G.get_weights())
        #---------------------
        self.E_EMA = clone_model(self.E)
        self.E_EMA.set_weights(self.E.get_weights())
        #--------------------------------------------------
        self.F_EMA = clone_model(self.F)
        self.F_EMA.set_weights(self.F.get_weights())
        #------------------------------------------------------------------------------------
        
        
        #set the optimizers of all models
        self.G_opt = self.E_opt = self.D_opt  = Adam(lr=10**(-4), beta_1=0, beta_2 = 0.99)
        self.F_opt = Adam(lr=10**(-6), beta_1=0, beta_2 = 0.99)
    
    def save_models(self,epoch):
        
        #save the models to intoduce resume capacity to training
        self.G.save("models/G/G_{}.h5".format(epoch))
        self.E.save("models/E/E_{}.h5".format(epoch))
        self.F.save("models/F/F_{}.h5".format(epoch))
        self.D.save("models/D/D_{}.h5".format(epoch))
    
    def delete_models(self, directories):
        for directory in directories:
            os.remove(directory)
            
    def EMA(self,):
        #-------------------G--------------------------------
        for i in range(len(self.G.layers)):
            up_weight = self.G.layers[i].get_weights()
            old_weight = self.G_EMA.layers[i].get_weights()
            new_weight = []
            for j in range(len(up_weight)):
                new_weight.append(old_weight[j] * self.beta + (1-self.beta) * up_weight[j])
            self.G_AB_EMA.layers[i].set_weights(new_weight)
          
        #-------------------E--------------------------------
        for i in range(len(self.E.layers)):
            up_weight = self.E.layers[i].get_weights()
            old_weight = self.E_EMA.layers[i].get_weights()
            new_weight = []
            for j in range(len(up_weight)):
                new_weight.append(old_weight[j] * self.beta + (1-self.beta) * up_weight[j])
            self.G_BA_EMA.layers[i].set_weights(new_weight)
        
        #-------------------F--------------------------------
        for i in range(len(self.F.layers)):
            up_weight = self.F.layers[i].get_weights()
            old_weight = self.F_EMA.layers[i].get_weights()
            new_weight = []
            for j in range(len(up_weight)):
                new_weight.append(old_weight[j] * self.beta + (1-self.beta) * up_weight[j])
            self.G_BA_EMA.layers[i].set_weights(new_weight)
        

    def EMA_init(self,):
        self.G_EMA.set_weights(self.G.get_weights())
        self.E_EMA.set_weights(self.E.get_weights())
        self.F_EMA.set_weights(self.F.get_weights())
    
    
    def training_cycle(self, x, y):
        def L_adv(D_true, D_fake):
            output = tf.reduce_mean(tf.math.log(D_true)) + tf.reduce_mean(tf.math.log(1-D_fake))
            return output
        
        def L_sty(s_curl, s_curl_rec):
            output = tf.reduce_mean(tf.math.norm(s_curl - s_curl_rec, ord=1, axis=1))
            return output
        
        def L_ds(x_curl, x_curl_2):
            output = tf.reduce_sum(tf.math.norm(x_curl - x_curl_2, ord=1, axis=[1,2,3]))
            return ouput
        
        def L_cyc(x, cycle_cons):
            output = tf.reduce_mean(tf.math.norm(x - cycle_cons, ord=1, axis=[1,2,3]))
        
        z = tf.random.normal(shape = (x.shape[0], 16))
        y_curl = tf.squeeze(tf.random.categorical(tf.math.log(0.5*np.ones((1,2))), 1)) 
        y_curl = K.get_value(y_curl)
        
        z_2 = tf.random.normal(shape = (x.shape[0], 16))
        
        with tf.GradientTape(persistent=True) as tape:
            #Mappings
            s_curl = self.F(z, training=True)[y_curl]
            
            x_curl = self.G([x, s_curl], training=True) #main translation
            
            D_true = self.D(x, training=True)[y] 
            D_fake = self.D(x_curl, training=True)[y_curl]
            
            #for style reconstruction
            s_curl_rec = self.E(x_curl, training=True)[y_curl] 
            
            #for style diversification
            s_curl_2 = self.F(z_2, training=True)[y_curl] 
            x_curl_2 = self.G([x, s_curl_2], training=True)
            
            #for preserving source characteristics (ID)
            s_hat = self.E(x, training=True)[y]
            cycle_cons = self.G([x_curl, s_hat], training=True)
            
            #Computation of losses
            Ladv = L_adv(D_true, D_fake)
            Lsty = L_sty(s_curl, s_curl_rec)
            Lds = L_ds(x_curl, x_curl_2)
            Lcyc = L_cyc(x, cycle_cons)
            
            objective = Ladv + self.l_sty*Lsty -1*self.l_ds*Lds + self.l_cyc*Lcyc
            D_loss = -1*objective
            GFE_loss = objective
        
        
        D_grads = tape.gradient(D_loss, self.D.trainable_variables)
        self.D_opt.apply_gradients(zip(D_grads, self.D.trainable_variables))
        
        G_grads = tape.gradient(GFE_loss, self.G.trainable_variables)
        self.G_opt.apply_gradients(zip(G_grads, self.G.trainable_variables))
        
        E_grads = tape.gradient(GFE_loss, self.E.trainable_variables)
        self.E_opt.apply_gradients(zip(E_grads, self.E.trainable_variables))
        
        F_grads = tape.gradient(GFE_loss, self.F.trainable_variables)
        self.F_opt.apply_gradients(zip(F_grads, self.F.trainable_variables))   
        
        #Update the tensorboard values
        self.train_info['losses']['L_adv'].append(Ladv)
        self.train_info['losses']['L_sty'].append(Lsty)
        self.train_info['losses']['L_ds'].append(Lds)
        self.train_info['losses']['L_cyc'].append(Lcyc)
        
    def train(self, epochs, batch_size=10):
        start_time = datetime.datetime.now()
        def chop_microseconds(delta):
            #utility to help avoid printing the microseconds
            return delta - datetime.timedelta(microseconds=delta.microseconds)
        
        try:
            #create a dynamic evaluator object
            dynamic_evaluator = evaluator(self.img_shape, self.latent_shape)
            
            for it in iterations:
                y = np.random.randint(2)
                if y==0:
                    x = self.data_loader.load_data(batch_size=batch_size, dataset='train', domain='A')
                elif y==1:
                    x = self.data_loader.load_data(batch_size=batch_size, dataset='train', domain='B')
                else:
                    Exception('Wrong encoding provideed for the domains')
                
                self.training_cycle(x,y)
                
                if it % 10 == 0:
                    L_adv = self.train_info['losses']['L_adv'][-1]
                    L_sty = self.train_info['losses']['L_sty'][-1]
                    L_ds = self.train_info['losses']['L_ds'][-1]
                    L_cyc = self.train_info['losses']['L_cyc'][-1]
                    
                    report = '[%d/%d]  [%s:%.3f  %s:%.3f  %s:%.3f  %s:%.3f]' % (L_adv, L_sty, L_ds, L_cyc)
                    print(report)
                
                """
                if it % 200 = 100:
                    #evaluation based on FID and LPIPS
                    #I may need to introduce a measure which captures LPIPS between input and ouput
                    #This metric will check whether the identity of the person remains the same
                """
                    





            
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
            
                
            
# def __init__(self, img_shape, latent_size, style_size, domains, resume=False):           
model = StarGANv2((128,128,3), latent_size=16, style_size=64, domains=2, resume=False)
model.train(epochs=100, batch_size = 1)