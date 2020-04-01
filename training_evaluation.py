# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 16:05:10 2020

@author: Georgios
"""

#This file will be used for robust evaluation of the model training
#The SSIM evaluation every sample interval of batches or at the end of each epoch is very noisy
#It uses a small batch of the test data and a small number of sample latent vectors
#compromising the trade-off between training speed and on-line evaluation of the training

#This file runs off-line and robustly estimates the success of a particular training setting 
#based on perception and distortion criteria

from keras_networks import G_AB
from glob import glob
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import collections
import os

class train_eval(object):
    def __init__(self,img_shape=(100,100,3), latent_size=2):
        self.img_shape = img_shape
        self.latent_size = latent_size
        self.latent_shape = (1,1,latent_size)
    
    
    #function to perform distortion evaluation of a model
    def distortion(self, model_name, no_samples):
        #no_samples is the number of latent values to be sampled for the evaluation of the expected SSIM value
        #It is assumed that the model is saved under the file ../models/

        model = G_AB(self.img_shape, self.latent_shape) #define model architecture
        model.load_weights("models/%s" % (model_name)) #load the saved weights
        
        
        phone_paths = glob('data/testA/*')[:10]
        dslr_paths = glob('data/testB/*')[:10]

        ssim_vals=np.zeros((len(phone_paths), no_samples))

        print(ssim_vals.shape)
        i=0
        for phone_path, dslr_path in tqdm(zip(phone_paths, dslr_paths)):
            x_true = plt.imread(phone_path).astype(np.float)
            x_true = x_true/255
            x = np.expand_dims(x_true, axis=0) #form needed to pass to the network 
            
            y_true = plt.imread(dslr_path).astype(np.float)
            y_true = y_true/255
            
            for j in range(no_samples):
                z = np.random.randn(1,1,1, self.latent_size)
                y_pred = model.predict([x,z])
                ssim_vals[i,j] = ssim(y_pred[0],y_true, multichannel=True)
            i+=1
        
        return np.mean(ssim_vals), np.std(ssim_vals)
    
    def models_distortion(self, no_samples):
        
        model_names = glob('models/*.h5')
        print(model_names)
        AB_model_names = []
        for model_name in model_names:
            if model_name.split('_')[1]=='AB':
                model_name=os.path.basename(model_name)
                AB_model_names.append(model_name)
        print(AB_model_names)
        AB_model_names = sorted(AB_model_names)
        
        model_info = {}
        for model_name in AB_model_names:
            mean_ssim, std_ssim = self.distortion(model_name, no_samples)
            model_info[model_name]=[mean_ssim, std_ssim]
        
        #list the names in decreasing order wrt to first argument
        ordered_model_info=sorted(model_info.items(), key=lambda t: t[1][0])
        
        for info in ordered_model_info:
            print('%s: mean SSIM: %.4f --- std SSIM: %.4f' % (info[0], info[1][0], info[1][1]))
    

#testing of distortion function
train_evaluation = train_eval()
train_evaluation.models_distortion(10)


"""
runs=4
mean_ssim_values = np.zeros(runs)
for i in range(runs):
    ssim_vals = train_evaluation.distortion(model_name = 'G_AB_10_0.h5', no_samples=10)
    mean_ssim_values[i] = np.mean(ssim_vals)
    print(mean_ssim_values[i])


print('mean: ', np.mean(mean_ssim_values))
print('std: ', np.std(mean_ssim_values))

plt.figure(figsize=[10,8])
n, bins, patches = plt.hist(x=mean_ssim_values, bins=8, color='#0504aa',alpha=0.7, rwidth=0.85)
""" 

            
            
            

        