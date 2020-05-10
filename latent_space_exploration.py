# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:03:35 2020

@author: Georgios
"""

#Exploration of the latent space

from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 13:52:02 2019

@author: Georgios
"""
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time 
from skimage.metrics import structural_similarity as ssim
from keras_networks import G_AB
from PIL import Image
from lpips import lpips
  
#Search Phase inspired by Brownian motion and repel dynamics
class SwarmOptimiser(object):
    def __init__(self, n, lo_range, up_range, func, search_duration=None, swarms_factor=None):
        self.n = n
        self.lower_range = lo_range
        self.upper_range = up_range
        self.func_evals = 0 #upper limit is 10000
        
        
        #Search Phase variables
        self.M = search_duration
        self.c = swarms_factor
        
        if self.c !=None and self.M!=None:
            self.swarms = self.c**self.n
            self.swarm_positions = np.zeros((self.M, self.swarms, self.n))
            self.swarm_velocities = np.zeros((self.M, self.swarms, self.n))
            self.swarm_accelerations = np.zeros((self.M, self.swarms, self.n))
            #PSO algorithm variables
            self.pi_arr = np.zeros((self.swarms,self.n))
            self.pi_cost_vals = np.zeros(self.swarms)
        
        else:
            self.swarms=self.n
        
       
        self.g = np.zeros(self.n)
        self.g_cost = 10**9
        
        #set the optimising funvtion
        self.func = func
        
        #create a gif for the latent starting at zero
        self.images=[]
        
    def initialise_swarm_positions(self, ):
        c=self.c
        n=self.n
        #helpher functions
        def convert_dec_2_base(dec, base, n):
            #n is the length of the output vectorised based
            result=np.zeros(n)
            i=0
            while dec>0:
                remainder = dec % base
                dec = dec // base
                result[-1-i] = remainder
                i+=1
            return result

        def convert_base_2_dec(number, base):
            #number is a n-length vector each elements contains one digit of the number in base base
            result = 0
            number = number[::-1]
            for i in range(len(number)):
                result += number[i]*base**i
            
            return result
        
        #Initialise the R^n grid with the brownian particles (swarms)
        c_range = np.arange(c)
        a = (self.upper_range[0]-self.lower_range[0])/(2*c)
        co_list = np.array([a+2*c_range[i]*a for i in range(len(c_range))])
        x=np.zeros(n)
        for i in range(c**n):
            #print(x)
            swarm_position = [self.lower_range[j] + co_list[int(x[j])] for j in range(n)]
            #print(swarm_position)
            self.swarm_positions[0, i, :] = swarm_position
            x_10 = convert_base_2_dec(x, c)
            x_10 += 1
            if x_10==c**n:
                continue
            else:
                x = convert_dec_2_base(x_10, c, n)
        """       
        if self.n == 2:
            plt.figure()
            plt.scatter(self.swarm_positions[0, :, 0], self.swarm_positions[0, :, 1])
            
            plt.xlim((-2,2))
            plt.ylim((-2,2))
            plt.show()
        """
        
    def check_range(self, x):
        n = x.shape[0]
        for i in range(n):
            if x[i]<self.lower_range[i] or x[i]>self.upper_range[i]:
                return False
        
        return True
    
    def brownian(self, x0, M, dt, delta, out=None):    
        # For each element of x0, generate a sample of n numbers from a
        # normal distribution.
        random_path = np.zeros((self.M, self.n))
        current=x0
        for i in range(self.M):
            sample = np.random.normal(loc=0, scale=delta*np.sqrt(dt), size=(self.n,))
            current+=sample
            range_check = self.check_range(current)
            while range_check==False:
                current-=sample
                sample = np.random.normal(loc=0, scale=delta*np.sqrt(dt), size=(self.n,))
                current+=sample
                range_check = self.check_range(current)
            
            random_path[i,:] = current
        
        return random_path

    def search_space(self, ):
        #implement the Brownian motion with particle repelling
        #Compute the pairwise distances of the swarms
        swarms = self.swarms
        #swarm_pairs = swarms*(swarms-1)/2 #(swarms choose 2)
        #distances = np.zeros((M, swarms, swarms, n)) #Distance (vector form) matrix
        
        # The Wiener process parameter.
        delta = 1.5
        # Total time.
        T = 10.0
        # Number of steps.
        M = self.M
        # Time step size
        dt = T/M
        
        pi_arr = np.zeros((self.swarms, self.n))
        cost_vals = np.zeros(self.swarms)
        for i in range(swarms):
            if i==0: #create a gif for the latent starting at 0
                random_walk = self.brownian(self.swarm_positions[0, i, :], M, dt, delta, out=None)
                self.swarm_positions[:, i, :] = random_walk
                        
                cost_value=10**9
                pi=np.zeros(self.n)
                for j in range(M):
                    new_cost_value,frame = self.func(self.swarm_positions[j, i, :])
                    
                    self.images.append(frame) #add the frame to the gif
                    self.func_evals+=1
                    if new_cost_value < cost_value:
                        cost_value = new_cost_value
                        pi = self.swarm_positions[j, i, :]
                
                pi_arr[i] = pi
                cost_vals[i] = cost_value
            else:
                random_walk = self.brownian(self.swarm_positions[0, i, :], M, dt, delta, out=None)
                self.swarm_positions[:, i, :] = random_walk
                        
                cost_value=10**9
                pi=np.zeros(self.n)
                for j in range(M):
                    new_cost_value,_ = self.func(self.swarm_positions[j, i, :])
                    self.func_evals+=1
                    if new_cost_value < cost_value:
                        cost_value = new_cost_value
                        pi = self.swarm_positions[j, i, :]
                
                pi_arr[i] = pi
                cost_vals[i] = cost_value
        
        self.pi_arr = pi_arr
        self.pi_cost_vals = cost_vals
        
        arg_min = np.argmin(cost_vals)
        self.g = self.pi_arr[arg_min]
        self.g_cost = cost_vals[arg_min]
        
    
    def PSO(self, ):
        
        #PARTICLE SWARM OPTIMISATION
        #initialisation
        
        pso_swarms_positions = np.zeros((2000//self.swarms+10, self.swarms, self.n))
        pso_swarms_velocities = np.zeros((2000//self.swarms+10, self.swarms, self.n))
        
        search_phase=False
        if search_phase==True:
            for i in range(self.swarms):
                pso_swarms_positions[0, i, :] = self.pi_arr[i, :]
                pso_swarms_velocities[0, i, :] = 0
                #pso_swarms_velocities[0, i, :] = np.multiply(2*(self.upper_range - self.lower_range) , np.random.rand(n)) - self.upper_range + self.lower_range
        else:
            
            self.pi_arr = np.zeros((self.swarms,self.n))
            self.pi_cost_vals = np.zeros(self.swarms)
            for i in range(self.swarms):
                random_position = np.multiply(self.upper_range-self.lower_range, np.random.rand(self.n)) + self.lower_range
                pso_swarms_positions[0, i, :] = random_position
                self.pi_arr[i, :] = random_position
                
                func_value,_ = self.func(random_position)
                if func_value < self.g_cost:
                    self.g = random_position
                    self.g_cost = func_value
                pso_swarms_velocities[0, i, :] = 0  
                #pso_swarms_velocities[0, i, :] = np.multiply(2*(self.upper_range - self.lower_range) , np.random.rand(n)) - self.upper_range + self.lower_range
                
        phi_p = 0.6
        phi_g = 0.75
        omega=0.9
        it=1
        while self.func_evals<2000:
           print("iteration: ", it)
           print(self.func_evals)
           for i in range(self.swarms):
               for d in range(self.n):
                   rp = np.random.rand()
                   rg = np.random.rand()
                   pso_swarms_velocities[it, i, d] = omega*pso_swarms_velocities[it-1, i, d]+phi_p*rp*(self.pi_arr[i,d]-pso_swarms_positions[it-1, i, d]) + phi_g*rg*(self.g[d]-pso_swarms_positions[it-1, i, d])
               
            
               pso_swarms_positions[it, i, :] = pso_swarms_positions[it-1, i, :] + pso_swarms_velocities[it, i, :]
               cost_eval_i,frame = self.func(pso_swarms_positions[it, i, :])
               self.func_evals+=1
               
               if i==0:
                   for _ in range(1):
                       self.images.append(frame) #add the frame to the gif
               
              
               if  cost_eval_i < self.pi_cost_vals[i]:
                   self.pi_arr[i] = pso_swarms_positions[it, i, :]
                   self.pi_cost_vals[i] = cost_eval_i
                  
                   if cost_eval_i < self.g_cost:
                          self.g = self.pi_arr[i]
                          self.g_cost = self.pi_cost_vals[i]
           it+=1
        
        #append the last frame 10 more times
        converged_frame = self.images[-1]
        for _ in range(30):
            self.images.append(converged_frame)
        
        
        
        if self.n == 2:
            plt.figure()
            xi = np.linspace(-3, 3, 20)
            yi = np.linspace(-3, 3, 20)
            Xi, Yi = np.meshgrid(xi, yi)
            zi = np.zeros(Xi.shape)
            for i in range(Xi.shape[0]):
                for j in range(Xi.shape[1]):
                    zi[i,j],_ = self.func(np.array([Xi[i,j], Yi[i,j]]))
            plt.contourf(xi, yi, zi, levels=14, cmap="RdBu_r")
            plt.plot(pso_swarms_positions[:, 0, 0], pso_swarms_positions[:, 0, 1])
        
        
        return self.g, self.g_cost


class latent_explorer(object):
    def __init__(self,img_shape, latent_size):
        self.img_shape = img_shape
        self.latent_size = latent_size
        self.model = G_AB(img_shape, (1,1,latent_size))
        self.model.load_weights('models/G_AB_36.h5')
        
        #instantiate the LPIPS loss object
        self.lpips = lpips(self.img_shape)
        self.lpips.create_model()
        
        self.gif_images=None
    
    
    def report(self, name, a, ref=None):
        return 0
        
        
    def model_func(self, img, latent):
        return self.model.predict([img,latent])[0]
    
    def create_gif(self, name, a, ref=0, start=0, gif_type='random_walk'):
        if gif_type=='random_walk':
            images=[]
            
            z_start=start
            z_start=np.expand_dims(z_start, axis=0)
                
            
            for i in tqdm(range(300)):
                z_new=z_start+0.3*np.random.randn(1, self.latent_size)
                z_start=z_new
                fake_b = self.model_func(a, z_new)
                if type(ref)==int:
                    frame = Image.fromarray((np.concatenate((np.squeeze(a,axis=0), fake_b), axis=1) * 255).astype(np.uint8)) 
                else:
                    frame = Image.fromarray((np.concatenate((np.squeeze(a,axis=0), fake_b, ref), axis=1) * 255).astype(np.uint8))
                
                images.append(frame)
            
            print(np.linalg.norm(np.squeeze(z_start-start), ord=2))
            images[0].save('progress/gif/random/%s.gif' % (name),
                                        save_all=True, 
                                        append_images=images[::-1], 
                                        optimize=False, duration=80, loop=1) 
        
        elif gif_type=='normal':
            images=[]
            for i in range(150):
                z=np.array(tf.random.truncated_normal(shape=(1, self.latent_size)))
                #z = np.random.randn(1,self.latent_size)
                fake_b = self.model_func(a, z)
                
                if type(ref)==int:
                    frame = Image.fromarray((np.concatenate((np.squeeze(a,axis=0), fake_b), axis=1) * 255).astype(np.uint8)) 
                else:
                    frame = Image.fromarray((np.concatenate((np.squeeze(a,axis=0), fake_b, ref), axis=1) * 255).astype(np.uint8))
                
                images.append(frame)
            
            images[0].save('progress/gif/random/%s.gif' % (name),
                                        save_all=True, 
                                        append_images=images[::-1], 
                                        optimize=False, duration=100, loop=0) 
            
        
    def mode_search(self, img_A, img_B, metric='lpips'):
        if metric=='ssim':
            def func(z):
                #z=np.swapaxes(z,0,1)
                z=np.expand_dims(z,axis=0)
                #z is the latent vector
                mode_output=self.model.predict([img_A,z])[0]
                frame = Image.fromarray((np.concatenate((np.squeeze(img_A,axis=0), mode_output, img_B), axis=1) * 255).astype(np.uint8))
                return -1*ssim(img_B, mode_output, multichannel=True), frame
        
        elif metric=='lpips':
            self.lpips.set_reference(tf.convert_to_tensor(np.expand_dims(img_B,axis=0), dtype=tf.float32))
            def func(z):
                #z=np.swapaxes(z,0,1)
                z=np.expand_dims(z,axis=0)
                z=tf.convert_to_tensor(z, dtype=tf.float32)
                
                #z is the latent vector
                img_A_tensor=tf.convert_to_tensor(img_A, dtype=tf.float32)
                
                model_output=self.model([img_A_tensor,z])
                distance = self.lpips.distance(model_output)
                
                model_output_arr=np.array(model_output)
                frame = Image.fromarray((np.concatenate((np.squeeze(img_A,axis=0), model_output_arr[0], img_B), axis=1) * 255).astype(np.uint8))
                
                return distance, frame
            
        #PARTICLE SWARM OPTIMISATION
        n=self.latent_size #domain space dimension
        
        #domain range determination
        lower_range = -3*np.ones(n)
        upper_range = 3*np.ones(n)
        M=30 #duration of the search phase (number of iterations in search phase)
        c=2 #determine how many swarms are injected to the domain. swarms_factor^
        sw_optimiser = SwarmOptimiser(n, lower_range, upper_range, func, M, c)
        sw_optimiser.initialise_swarm_positions()
        sw_optimiser.search_space()
        min_val, cost = sw_optimiser.PSO()
        self.gif_images=sw_optimiser.images
        
        return min_val, cost


def get_random_patch(img, patch_dimension):
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
    

latent_exp = latent_explorer((100,100,3), 4)




names=[8, 16, 19, 58, 160, 168, 220, 263, 296, 300, 331,364,378,379,397, 433, 484,554,597,595,600,774, 783,790,
       833, 839, 847, 849, 850, 885, 913, 936, 939, 1034, 1047, 1107, 1121, 1144, 1176, 1253,
       1300, 1372, 1384, 1428, 1443, 1494, 1528, 1607, 1628, 1660, 1671, 1701, 1705, 1755, 1776, 2047, 3056
       ]



"""
for counter, name in tqdm(enumerate(names)):
    print(name)
    x_true = plt.imread('data/testA/%s.jpg'% str(name)).astype(np.float)
    x_true = x_true/255
    #x_true = get_random_patch(x_true, (500,500))
    x = np.expand_dims(x_true, axis=0)
        
    y_true = plt.imread('data/testB/%s.jpg'% str(name)).astype(np.float)
    y_true = y_true/255
    
    opt_loc, opt_value=latent_exp.mode_search(x,y_true, metric='lpips')
    print(opt_loc)
    latent_exp.create_gif(str(name)+'_opt', x, y_true, start=opt_loc, gif_type='random_walk')

"""

for name in tqdm(names):
    print(name)
    x_true = plt.imread('data/testA/%s.jpg'% str(name)).astype(np.float)
    x_true = x_true/255
    #x_true = get_random_patch(x_true, (500,500))
    x = np.expand_dims(x_true, axis=0)
            
    y_true = plt.imread('data/testB/%s.jpg'% str(name)).astype(np.float)
    y_true = y_true/255
        
    latent_exp.create_gif(str(name)+'_opt', x, y_true, start=0, gif_type='normal')







"""
info={}
for i in tqdm(range(2000)):
    info[i]={}
    x_true = plt.imread('data/testA/%s.jpg'% str(i)).astype(np.float)
    x_true = x_true/255
    x = np.expand_dims(x_true, axis=0)
    
    y_true = plt.imread('data/testB/%s.jpg'% str(i)).astype(np.float)
    y_true = y_true/255
    
    opt_loc, opt_value=latent_exp.mode_search(x,y_true)
    info[i]['opt_loc'] = opt_loc
    info[i]['opt_value'] = opt_value
    
    latent_exp.gif_images[0].save('progress/gif/lpips/%s_%s.gif'%(str(i), 'lpips'),
                                        save_all=True, 
                                        append_images=latent_exp.gif_images, 
                                        optimize=False, duration=150, loop=1)

"""

#print('optimal location: ', min_val)
#print('optimal value: ', cost)
        