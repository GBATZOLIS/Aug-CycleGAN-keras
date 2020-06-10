# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 11:03:06 2020

@author: Georgios
"""

#StarGAN v2 evaluator class

from data_loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt

class evaluator(object):
    def __init__(self, img_shape, latent_size, domains):
        self.img_shape = img_shape
        self.latent_size = latent_size
        self.domains = domains
        
        #instantiate the data loader class
        self.data_loader = DataLoader(img_res=(self.img_shape[0], self.img_shape[1]))
    
    def visual_performance(self, batch_size = 5, num_out_imgs = 3, training_point=100):
        img_A = self.data_loader.load_data(batch_size=batch_size, dataset='test', domain='A')
        fake_imgs_B = np.zeros((batch_size, self.domains*num_out_imgs)+self.img_shape)
        for j in range(batch_size):
            fake_B = np.zeros((self.domains*num_out_imgs,)+self.img_shape)
            for domain in range(self.domains):
                for i in range(num_out_imgs):
                    z = np.random.randn(1, self.latent_size)
                    s = self.F.predict(z)[domain]
                    fake_B[i+domain*num_out_imgs] = self.G.predict([np.expand_dims(img_A[j],axis=0), s])
                
            fake_imgs_B[j] = fake_B
        
        
        fig, axs = plt.subplots(batch_size, self.domains*num_out_imgs+1, figsize=(20,28))
            
        for j in range(batch_size):
            for i in range(self.domains*num_out_imgs+1):
                ax = axs[j,i]
                if i == 0:
                    ax.imshow(img_A[j])
                else:
                    ax.imshow(fake_imgs_B[j,i-1])
        
        fig.savefig("progress/visual_results/A2B/A2B_%s.png" % str(training_point), bbox_inches='tight')
        plt.close("all")
        print('Visual results have been generated')
    
    def FID_score():
        return
    
    def LPIPS_score():
        return