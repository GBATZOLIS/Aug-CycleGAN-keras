# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 21:24:53 2019

@author: Georgios
"""


from glob import glob
import numpy as np
import matplotlib.pyplot as plt

class DataLoader():
    def __init__(self, img_res=(100, 100)):
        #self.dataset_name = dataset_name
        self.img_res = img_res
        #self.main_path = main_path
        
    
    def get_random_patch(self, img, patch_dimension):
        if img.shape[0]==patch_dimension[0] and img.shape[1]==patch_dimension[1]:
            return img
        
        else:
            image_shape=img.shape
            image_length = img.shape[0]
            image_width = img.shape[1]
            patch_length = patch_dimension[0]
            patch_width = patch_dimension[1]
            
            if (image_length >= patch_length) and (image_width >= patch_width):
                x_max=image_shape[0]-patch_dimension[0]
                y_max=image_shape[1]-patch_dimension[1]
                x_index=np.random.randint(x_max)
                y_index=np.random.randint(y_max)
            else:
                print("Error. Not valid patch dimensions")
            
            return img[x_index:x_index+patch_dimension[0], y_index:y_index+patch_dimension[1], :]
        
    def load_data(self, domain, patch_dimension=None, batch_size=1, is_testing=False):
        data_type = r"train%s" % domain if not is_testing else "test%s" % domain
        path = glob(r'data/%s/*' % (data_type))
        batch_images = np.random.choice(path, size=batch_size)
        
        if patch_dimension==None:
            #if the patch dimension is not specified, use the training dimensions
            patch_dimension = self.img_res
            
        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            img = self.get_random_patch(img, patch_dimension)   
            imgs.append(img)

        imgs = np.array(imgs)/255

        return imgs
    
    def load_paired_data(self, batch_size=None, is_testing=True):
        #if batch_size = None, the entire test dataset is loaded.
        #This is likely to cause memory issues. This needs to be resolved later
        
        if is_testing:
            phone_paths = glob('data/testA/*')
            dslr_paths = glob('data/testB/*')
        else:
            phone_paths = glob('data/trainA/*')
            dslr_paths = glob('data/trainB/*')
            
        if batch_size:
            random_indices = np.random.choice(len(phone_paths), batch_size)
            phone_paths = [phone_paths[index] for index in random_indices]
            dslr_paths = [dslr_paths[index] for index in random_indices]
        
        
        phone_imgs=[]
        dslr_imgs=[]
        for phone_path, dslr_path in zip(phone_paths, dslr_paths):
            phone_img = self.imread(phone_path)
            phone_imgs.append(phone_img)
            dslr_img = self.imread(dslr_path)
            dslr_imgs.append(dslr_img)
        
        phone_imgs = np.array(phone_imgs)/255
        dslr_imgs = np.array(dslr_imgs)/255
        
        return phone_imgs, dslr_imgs
    
    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        path_A = glob(r'data/%sA/*' % (data_type))
        path_B = glob(r'data/%sB/*' % (data_type))
        #path_C = glob(r'data/%sB/*' % (data_type))
        
        path_A=path_A[0:10000]
        path_B=path_B[0:10000]
        #path_C=path_C[15000:20000]

        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        #random_indices = np.random.choice(len(path_A), total_samples)
        #path_A = [path_A[index] for index in random_indices]
        #path_B = [path_B[index] for index in random_indices]
        #path_C = np.random.choice(path_C, total_samples, replace=False)
        
        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)

        for i in range(self.n_batches):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            #batch_C = path_C[i*batch_size:(i+1)*batch_size]
            
            imgs_A, imgs_B= [], []
            #imgs_C=[]
            for img_A, img_B in zip(batch_A, batch_B):
                img_A = self.imread(img_A)
                img_B = self.imread(img_B)
                #img_C = self.imread(img_C)
                
                #img_A = scipy.misc.imresize(img_A, self.img_res)
                #img_B = scipy.misc.imresize(img_B, self.img_res)
                if (img_A.shape[0]>self.img_res[0]) or (img_A.shape[1]>self.img_res[1]):
                    img_A=self.get_random_patch(img_A, patch_dimension = self.img_res)
                    img_B=self.get_random_patch(img_B, patch_dimension = self.img_res)

                #if not is_testing and np.random.random() > 0.5:
                #        img_A = np.fliplr(img_A)
                #        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)
                #imgs_C.append(img_C)

            imgs_A = np.array(imgs_A)/255
            imgs_B = np.array(imgs_B)/255
            #imgs_C = np.array(imgs_C)/255

            yield imgs_A, imgs_B

    def load_img(self, path):
        img = self.imread(path)
        #img = scipy.misc.imresize(img, self.img_res)
        img = img/255
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        return plt.imread(path).astype(np.float)

