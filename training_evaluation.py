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
import os
from lpips import lpips
import tensorflow as tf



#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

class train_eval(object):
    def __init__(self,img_shape=(100,100,3), latent_size=2):
        self.img_shape = img_shape
        self.latent_size = latent_size
        self.latent_shape = (1,1,latent_size)
        
        self.metric = lpips((100,100,3))
        self.metric.create_model()
    
    #function to perform distortion evaluation of a model
    def distortion(self, model_name, no_samples, metric='LPIPS'):
        #no_samples is the number of latent values to be sampled for the evaluation of the expected SSIM value
        #It is assumed that the model is saved under the file ../models/

        model = G_AB(self.img_shape, self.latent_shape) #define model architecture
        model.load_weights("models/%s" % (model_name)) #load the saved weights
        model.trainable=False
        
        phone_paths = glob('data/testA/*')
        dslr_paths = glob('data/testB/*')

        distance_vals=np.zeros((len(phone_paths), no_samples))

        i=0
        
        
        print('model: %s' % model_name)
        if metric=='LPIPS':
            for phone_path, dslr_path in tqdm(zip(phone_paths, dslr_paths)):
                x_true = plt.imread(phone_path).astype(np.float)
                x_true = x_true/255
                x = np.expand_dims(x_true, axis=0) #form needed to pass to the network 
                x = tf.convert_to_tensor(x)
                
                y_true = plt.imread(dslr_path).astype(np.float)
                y_true = y_true/255
                y_true = np.expand_dims(y_true, axis=0)
                y_true = tf.convert_to_tensor(y_true)
                with tf.device('/GPU:0'):
                    self.metric.set_reference(y_true)
                    for j in range(no_samples):
                        z = tf.random.normal(shape=(1,1,1, self.latent_size))
                        y_pred = model([x,z])
                        distance_vals[i,j] = self.metric.distance(y_pred)
                i+=1
        
        elif metric=='SSIM':
            for phone_path, dslr_path in tqdm(zip(phone_paths, dslr_paths)):
                x_true = plt.imread(phone_path).astype(np.float)
                x_true = x_true/255
                x = np.expand_dims(x_true, axis=0) #form needed to pass to the network 
                y_true = plt.imread(dslr_path).astype(np.float)
                y_true = y_true/255
                for j in range(no_samples):
                    z = tf.random.normal(shape=(1,1,1, self.latent_size))
                    y_pred = model([x,z])
                    distance_vals[i,j] = ssim(y_pred[0],y_true, multichannel=True)
                    
        return np.mean(distance_vals), np.std(distance_vals)
    
    def models_distortion(self, no_samples, metric='LPIPS'):
        def sort_models(model_names):
            models_dict = {}
            max_batch=500 #need to automate this
            for model_name in model_names:
                epoch = int(model_name.split('_')[2])
                batch = int(model_name.split('_')[3].split('.')[0])
                training_point = epoch+batch/max_batch
                models_dict[model_name]=training_point
            
            sorted_models = [i[0] for i in sorted(models_dict.items(), key=lambda item: item[1])]
            return sorted_models
            
        model_names = glob('models/*.h5')
        AB_model_names = []
        for model_name in model_names:
            if model_name.split('_')[1]=='AB':
                model_name=os.path.basename(model_name)
                AB_model_names.append(model_name)
        
        AB_model_names = sort_models(AB_model_names)
        
        model_info = {}
        training_points = []
        mean_vals = []
        std_vals = []
        
        max_batch=500
        for model_name in tqdm(AB_model_names):
            #infer and save the training point
            epoch = int(model_name.split('_')[2])
            batch = int(model_name.split('_')[3].split('.')[0])
            training_point = epoch+batch/max_batch
            training_points.append(training_point) 
            
            #get model performance and save it
            mean_ssim, std_ssim = self.distortion(model_name=model_name, 
                                                  no_samples=no_samples,
                                                  metric=metric)
            mean_vals.append(mean_ssim)
            std_vals.append(std_ssim)
            model_info[model_name]=[mean_ssim, std_ssim]
        
        mean_vals=np.array(mean_vals)
        std_vals=np.array(std_vals)
        plt.figure()
        plt.plot(training_points, mean_vals+2*std_vals, label='mean+2std')
        plt.plot(training_points, mean_vals-2*std_vals, label='mean-2std')
        plt.plot(training_points, mean_vals, label='mean')
        plt.legend()
        plt.savefig('progress/training_evaluation/lpips.png', bbox_inches='tight')
        print('Evaluation graphs has been generated and saved in ../progress/training_evaluation/')
        
        
        #list the names in decreasing order wrt to first argument
        ordered_model_info=sorted(model_info.items(), key=lambda t: t[1][0])
        
        for info in ordered_model_info:
            print('%s: mean LPIPS: %.4f --- std LPIPS: %.4f' % (info[0], info[1][0], info[1][1]))
        
        with open('progress/training_evaluation/ranking.txt', 'w') as file:
            for info in ordered_model_info:
                file.write('%s: mean LPIPS: %.4f --- std LPIPS: %.4f \n' % (info[0], info[1][0], info[1][1]))
        
        print('Ranking of models has been generated and saved in .../progress/training_evaluation/ranking.txt')
     
        return ordered_model_info

#testing of distortion function
train_evaluation = train_eval()
#m,s = train_evaluation.distortion(model_name='G_AB_28_400.h5', no_samples=10)

#print(m,s)
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

            
            
            

        