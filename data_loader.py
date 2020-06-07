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
        
        self.trainA = glob('data/%s%s/*.jpg' % ('train', 'A'))
        self.trainB = glob('data/%s%s/*.jpg' % ('train', 'B'))
        self.testA = glob('data/%s%s/*.jpg' % ('test', 'A'))
        self.testB = glob('data/%s%s/*.jpg' % ('test', 'B'))
        
    
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
        
    def load_data(self, batch_size, dataset='test', domain='A'):
        if dataset=='train' and domain=='A':
            paths = np.random.choice(self.trainA, batch_size, replace=False)
        elif dataset=='train' and domain=='B':
            paths = np.random.choice(self.trainB, batch_size, replace=False)
        elif dataset=='test' and domain=='A':
            paths = np.random.choice(self.testA, batch_size, replace=False)
        elif dataset=='test' and domain=='B':
            paths = np.random.choice(self.testB, batch_size, replace=False)
        else:
            return Exception('Incomptatible dataset or domain name')
        
        imgs = []
        for path in paths:
            img = self.imread(path)
            imgs.append(img)
        
        imgs = np.array(imgs)/255
        
        return imgs
    
    def load_paired_data(self, batch_size=None, is_testing=True):
        #if batch_size = None, the entire test dataset is loaded.
        #This is likely to cause memory issues. This needs to be resolved later
        
        if is_testing:
            phone_paths = glob('data/testA/*.jpg')
            dslr_paths = glob('data/testB/*.jpg')
        else:
            phone_paths = glob('data/trainA/*.jpg')
            dslr_paths = glob('data/trainB/*.jpg')
            
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
    
    def load_unpaired_batch(self, batch_size, dataset='train', portion=0.25):
        path_A = glob('data/%sA/*.jpg' % dataset)
        path_B = glob('data/%sB/*.jpg' % dataset)
        
        n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = n_batches * batch_size
        
        used_samples = int(portion*total_samples)
        path_A = np.random.choice(path_A, used_samples, replace=False)
        path_B = np.random.choice(path_B, used_samples, replace=False)
        
        self.n_batches = used_samples // batch_size
        
        for i in range(self.n_batches):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B= [], []
            
            for img_A, img_B in zip(batch_A, batch_B):
                img_A = self.imread(img_A)
                img_B = self.imread(img_B) 
                imgs_A.append(img_A)
                imgs_B.append(img_B)
            
            imgs_A = np.array(imgs_A)/255
            imgs_B = np.array(imgs_B)/255
            yield imgs_A, imgs_B
        
        
    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        sup_path_A = glob(r'data/%sA/*.jpg' % (data_type))
        sup_path_B = glob(r'data/%sB/*.jpg' % (data_type))

        path_A=sup_path_A[0:20000]
        path_B=sup_path_B[0:20000]
        
        sup_path_A = sup_path_A[20000:40000]
        sup_path_B = sup_path_B[20000:40000]

        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = self.n_batches * batch_size
        
        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)

        for i in range(self.n_batches):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            sup_batch_A = sup_path_A[i*batch_size:(i+1)*batch_size]
            sup_batch_B = sup_path_B[i*batch_size:(i+1)*batch_size]
            
            imgs_A, imgs_B= [], []
            sup_imgs_A, sup_imgs_B = [], []
            for img_A, img_B, sup_A, sup_B in zip(batch_A, batch_B, sup_batch_A, sup_batch_B):
                img_A = self.imread(img_A)
                img_B = self.imread(img_B) 
                sup_img_A = self.imread(sup_A)
                sup_img_B = self.imread(sup_B)
                
                if (img_A.shape[0]>self.img_res[0]) or (img_A.shape[1]>self.img_res[1]):
                    img_A=self.get_random_patch(img_A, patch_dimension = self.img_res)
                    img_B=self.get_random_patch(img_B, patch_dimension = self.img_res)

                imgs_A.append(img_A)
                imgs_B.append(img_B)
                sup_imgs_A.append(sup_img_A)
                sup_imgs_B.append(sup_img_B)

            imgs_A = np.array(imgs_A)/255
            imgs_B = np.array(imgs_B)/255
            sup_imgs_A = np.array(sup_imgs_A)/255
            sup_imgs_B = np.array(sup_imgs_B)/255
             
            yield imgs_A, imgs_B, sup_imgs_A, sup_imgs_B

    def load_img(self, path):
        img = self.imread(path)
        #img = scipy.misc.imresize(img, self.img_res)
        img = img/255
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        return plt.imread(path).astype(np.float)

