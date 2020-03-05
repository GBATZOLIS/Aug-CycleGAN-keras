# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 12:49:38 2020

@author: Georgios
"""

from data_loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


class evaluator(object):
    def __init__(self, img_shape, latent_shape):
        self.img_shape = img_shape
        self.latent_shape=latent_shape
        self.data_loader = DataLoader(img_res=(self.img_shape[0], self.img_shape[1]))
        
    def test(self, batch_size, num_out_imgs, training_point, test_type):
        img_A, img_B = self.data_loader.load_paired_data(batch_size=batch_size)
        
        fake_imgs_B = np.zeros((batch_size, num_out_imgs)+self.img_shape)
        for j in range(batch_size):
            fake_B = np.zeros((num_out_imgs,)+self.img_shape)
            for i in range(num_out_imgs):
                z_b = np.random.randn(1, 1, 1, self.latent_shape[-1])
                fake_B[i] = self.model.predict([np.expand_dims(img_A[j],axis=0), z_b])
            
            fake_imgs_B[j] = fake_B
        
        if test_type == 'perception':
        
            fig, axs = plt.subplots(batch_size, num_out_imgs+2, figsize=(21,15))
            
            for j in range(batch_size):
                for i in range(num_out_imgs+2):
                    ax = axs[j,i]
                    if i == 0:
                        ax.imshow(img_A[j])
                    elif i == num_out_imgs+1:
                        ax.imshow(img_B[j])
                    else:
                        ax.imshow(fake_imgs_B[j,i-1])
            
            fig.savefig("progress/perception/perc_test_%s.png" % str(training_point), bbox_inches='tight')
            plt.close("all")
            
            print('Perceptual test results have been successfully generated and saved in ../progress/perception/')
        
        elif test_type == 'distortion':
            
            avg_ssim = 0
            avg_max_ssim = 0
            avg_min_ssim = 0
            for j in range(batch_size):
                max_ssim = -100
                min_ssim = 100
                for i in range(num_out_imgs):
                    ssim_value = ssim(img_B[j],fake_imgs_B[j,i],multichannel=True)
                    
                    if ssim_value>max_ssim:
                        max_ssim = ssim_value
                    
                    if ssim_value<min_ssim:
                        min_ssim = ssim_value
                    
                    avg_ssim += ssim_value
                
                avg_max_ssim+=max_ssim
                avg_min_ssim+=min_ssim
            
                
            avg_max_ssim=avg_max_ssim/batch_size
            avg_min_ssim=avg_min_ssim/batch_size
            avg_ssim = avg_ssim/(batch_size*num_out_imgs)
            
            print('Distortion test results have been obtained')
            print('Mean SSIM: %.3f - Min SSIM: %.3f - Max SSIM: %.3f' % (avg_ssim, avg_min_ssim, avg_max_ssim))
            return avg_ssim, avg_min_ssim, avg_max_ssim
