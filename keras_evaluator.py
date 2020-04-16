# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 12:49:38 2020

@author: Georgios
"""

from data_loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from lpips import lpips
from glob import glob
import tensorflow as tf


class evaluator(object):
    def __init__(self, img_shape, latent_shape):
        self.img_shape = img_shape
        self.latent_shape = latent_shape
        self.latent_size = latent_shape[-1]
        self.data_loader = DataLoader(img_res=(self.img_shape[0], self.img_shape[1]))
        
        #instantiate the LPIPS loss object (used as a perceptual loss)
        self.lpips = lpips(self.img_shape)
        self.lpips.create_model()
        
        
    def test(self, batch_size, num_out_imgs, training_point, test_type):
        
        phone_paths = glob('data/testA/*.jpg')
        dslr_paths = glob('data/testB/*.jpg')
        random_indices = np.random.choice(len(phone_paths), batch_size)
        phone_paths = [phone_paths[index] for index in random_indices]
        dslr_paths = [dslr_paths[index] for index in random_indices]
        
        
        
        if test_type=='perception':
            distance_vals=np.zeros((len(phone_paths), num_out_imgs))
            i=0
            for phone_path, dslr_path in zip(phone_paths, dslr_paths):
                x_true = plt.imread(phone_path).astype(np.float)
                x_true = x_true/255
                x = np.expand_dims(x_true, axis=0) #form needed to pass to the network 
                x = tf.convert_to_tensor(x)
                
                y_true = plt.imread(dslr_path).astype(np.float)
                y_true = y_true/255
                y_true = np.expand_dims(y_true, axis=0)
                y_true = tf.convert_to_tensor(y_true)
                
                with tf.device('/GPU:0'):
                    self.lpips.set_reference(y_true)
                    for j in range(num_out_imgs):
                        z = tf.random.normal(shape=(1,1,1, self.latent_size))
                        y_pred = self.model([x,z])
                        distance_vals[i,j] = self.lpips.distance(y_pred)
                i+=1
            
            info = {}
            info['lpips_mean'] = np.mean(distance_vals)
            info['lpips_std'] = np.std(distance_vals)
            info['lpips_min'] = np.amin(distance_vals)
            info['lpips_max'] = np.amax(distance_vals)
            
            print('lpips_min: %.3f - lpips_mean : %.3f - lpips_max: %.3f - lpips_std: %.3f' % 
                  (info['lpips_min'], info['lpips_mean'], info['lpips_max'], info['lpips_std']))
            return info
        
        
        elif test_type == 'distortion':
            distance_vals=np.zeros((len(phone_paths), num_out_imgs))
            i=0
            for phone_path, dslr_path in zip(phone_paths, dslr_paths):
                x_true = plt.imread(phone_path).astype(np.float)
                x_true = x_true/255
                x = np.expand_dims(x_true, axis=0) #form needed to pass to the network 
                x = tf.convert_to_tensor(x)
                y_true = plt.imread(dslr_path).astype(np.float)
                y_true = y_true/255
                for j in range(num_out_imgs):
                    z = tf.random.normal(shape=(1,1,1, self.latent_size))
                    y_pred = self.model([x,z])
                    y_pred = tf.make_ndarray(y_pred)
                    distance_vals[i,j] = ssim(y_pred[0],y_true, multichannel=True)
                i+=1
            
            info = {}
            info['ssim_mean'] = np.mean(distance_vals)
            info['ssim_std'] = np.std(distance_vals)
            info['ssim_min'] = np.amin(distance_vals)
            info['ssim_max'] = np.amax(distance_vals)

            print('ssim_min: %.3f - ssim_mean : %.3f - ssim_max: %.3f - ssim_std: %.3f' % 
                  (info['ssim_min'], info['ssim_mean'], info['ssim_max'], info['ssim_std']))
            return info
        
        elif test_type == 'mixed':
            lpips_vals=np.zeros((len(phone_paths), num_out_imgs))
            ssim_vals=np.zeros((len(phone_paths), num_out_imgs))
            i=0
            for phone_path, dslr_path in zip(phone_paths, dslr_paths):
                x_true = plt.imread(phone_path).astype(np.float)
                x_true = x_true/255
                x = np.expand_dims(x_true, axis=0) #form needed to pass to the network 
                x = tf.convert_to_tensor(x)
                
                y_true = plt.imread(dslr_path).astype(np.float)
                y_true = y_true/255
                y_true_nd_array = np.expand_dims(y_true, axis=0)
                y_true_tensor = tf.convert_to_tensor(y_true_nd_array, dtype=tf.float32)
                
                with tf.device('/GPU:0'):
                    self.lpips.set_reference(y_true_tensor)
                    for j in range(num_out_imgs):
                        z = tf.random.normal(shape=(1,1,1, self.latent_size))
                        y_pred_tensor = self.model([x,z])
                        lpips_vals[i,j] = self.lpips.distance(y_pred_tensor)
                        
                        y_pred_nd_array = np.array(y_pred_tensor)
                        ssim_vals[i,j] = ssim(y_pred_nd_array[0],y_true_nd_array[0], multichannel=True)      
                i+=1
            
            info={}
            
            info['lpips_mean'] = np.mean(lpips_vals)
            info['lpips_std'] = np.std(lpips_vals)
            
            print('lpips_mean : %.4f - lpips_std: %.3f' % 
                  (info['lpips_mean'], info['lpips_std']))
            
            info['ssim_mean'] = np.mean(ssim_vals)
            info['ssim_std'] = np.std(ssim_vals)
            
            print('ssim_mean : %.4f - ssim_std: %.3f' % 
                  (info['ssim_mean'], info['ssim_std']))
            
            return info
        
        elif test_type=='diversity':
            i=0
            avg_ref_distances=np.zeros(batch_size)
            for phone_path in phone_paths:
                x_true = plt.imread(phone_path).astype(np.float)
                x_true = x_true/255
                x = np.expand_dims(x_true, axis=0) #form needed to pass to the network 
                x = tf.convert_to_tensor(x)
                
                
                ref_distances = np.zeros(num_out_imgs)
                with tf.device('/GPU:0'):
                    z0=tf.zeros(shape=(1,1,1, self.latent_size))
                    y_ref = self.model([x,z0])
                    self.lpips.set_reference(y_ref)
                    for j in range(num_out_imgs):
                        z = tf.random.normal(shape=(1,1,1, self.latent_size))
                        y_pred = self.model([x,z])
                        ref_distances[j] = self.lpips.distance(y_pred)
                
                avg_ref_distance = np.mean(ref_distances)
                avg_ref_distances[i]=avg_ref_distance
                i+=1
            
            avg_lpips_distance = np.mean(avg_ref_distances)
            print('diversity: %.4f' % avg_lpips_distance)
            return avg_lpips_distance
            
        else:
            return Exception('test type not valid')
            
            
        
