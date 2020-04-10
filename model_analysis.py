# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 02:45:22 2020

@author: Georgios
"""

#This file is used for custom evaluation of the trained models

#Task1 
"""Explore the latent space and latent space efficiency"""

"""
1.) Train a model with a 2D latent space. 
2.) Evaluate G(A,z(i)) in a 2D grid of z(i) (-3,3)x(-3,3) range
3.) Evaluate SSIM and LPIPS on the grid points and plot the contour or surface plot
"""


#Load a trained model
#make predictions

from keras_networks import G_AB
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from PIL import Image
from lpips import lpips
import tensorflow as tf

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


img_shape = (800,800,3)
latent_shape=(1,1,2)

model_name = 'G_AB_61_300.h5'

model = G_AB(img_shape=img_shape, latent_shape=latent_shape) #define model architecture
model.load_weights("models/%s" % (model_name)) #load the saved weights

#y_true = plt.imread('data/testA/38.jpg').astype(np.float)
#y_true = y_true/255

x_true = plt.imread('data/full_size_test_images/6.jpg').astype(np.float)
x_true = get_random_patch(x_true, (800,800))
x_true = x_true/255
x = np.expand_dims(x_true, axis=0)

images=[]
frames=100
for _ in tqdm(range(frames)):
    z=0.3*np.array(tf.random.normal((1,1,1,2), dtype=tf.float32))
    out=model.predict([x,z])
    frame = Image.fromarray((np.concatenate((x_true, out[0]), axis=1) * 255).astype(np.uint8))
    images.append(frame)

images[0].save('progress/image.gif',
               save_all=True, append_images=images, optimize=False, duration=50, loop=0)



#-------------------code for all dimensions of latent space-------------------
"""
def add_tf_dimensions(z):
    z=np.expand_dims(z, axis=0)
    z=np.expand_dims(z, axis=0)
    z=np.expand_dims(z, axis=0)
    return z

batch_size=10
zb = np.random.randn(latent_shape[-1])


images=[]
for i in tqdm(range(200)):
    z_close = 2*np.random.randn(latent_shape[-1])
    z_close = add_tf_dimensions(z_close)
    y_pred = model.predict([x,z_close])
    im = Image.fromarray((y_pred[0] * 255).astype(np.uint8))
    images.append(im)

    
images[0].save('progress/image.gif',
               save_all=True, append_images=images[::-1], optimize=False, duration=40, loop=0)

"""











"""

z=np.zeros((1,1,1,2))
z[0,0,0,:]=np.array([-0.79,-0.964])
y_pred = model.predict([x,z])
fig, axs = plt.subplots(1, 3)
ax=axs[0]
ax.imshow(x_true)
ax=axs[1]
ax.imshow(y_pred[0])
ax=axs[2]
ax.imshow(y_true)



"""












#-------------------code for 2dimensional latent space--------------------------

"""
fig, axs = plt.subplots(1, 3)
ax=axs[0]
ax.imshow(x_true)
ax=axs[1]
ax.imshow(y_pred[0])
ax=axs[2]
ax.imshow(y_true)
"""  



"""
images=[]
x1, y1 = -2.97, 2.754
x2, y2 = 2.73471, -2.24321
slope = (y2-y1)/(x2-x1)
step=1/50
x_coord = []
y_coord = []
for k in range(abs(int((x2-x1)/step))+1):
    x_coord.append(x1+k*step)
    y_coord.append(y1+slope*k*step)
    z = np.array([x1+k*step, y1+slope*k*step])
    z=np.expand_dims(z, axis=0)
    z=np.expand_dims(z, axis=0)
    z=np.expand_dims(z, axis=0)
    y_pred = model.predict([x,z])
    print(y_pred[0].shape)
    
    im = Image.fromarray((y_pred[0] * 255).astype(np.uint8))
    images.append(im)
    
images[0].save('progress/image.gif',
               save_all=True, append_images=images, optimize=False, duration=40, loop=0)
"""



"""
#-----------------------------------------------------------------------
#create a spiral around the mode
a=0.02
b=0.15
def r(t,omega, a=0.02,b=0.15):
    return a*np.exp(b*omega*t)

x_co=[]
y_co=[]

omega=1
T=2*np.pi/omega
N=100 #steps per revolution
x_mode = -0.0668
y_mode = -1.583


images=[]

for t in tqdm(np.linspace(0, 5.5*T, int(5.5*N))):
    x_coord = r(t, omega, a, b)*np.cos(omega*t)+x_mode-a #make sure we start from the mode
    y_coord = r(t, omega, a, b)*np.sin(omega*t)+y_mode
    x_co.append(x_coord)
    y_co.append(y_coord)
    z = np.array([x_coord, y_coord])
    z=np.expand_dims(z, axis=0)
    z=np.expand_dims(z, axis=0)
    z=np.expand_dims(z, axis=0)
    y_pred = model.predict([x,z])
    
    frame=np.concatenate((x_true,y_pred[0], y_true), axis=1)
    #print(frame.shape)
    im = Image.fromarray((frame * 255).astype(np.uint8))
    images.append(im)
    
images[0].save('progress/image.gif',
               save_all=True, append_images=images[::-1], optimize=False, duration=40, loop=0)
#--------------------------------------------------------------------------------

"""




#--------------------------------------------------------------------

"""
import tensorflow as tf

metric = lpips((100,100,3))
metric.create_model()

y_true = np.expand_dims(y_true, axis=0)
y_true = tf.convert_to_tensor(y_true)
metric.set_reference(y_true)  


delta = 0.2
z1 = np.arange(-3, 3, delta)
z2 = np.arange(-3, 3, delta)
SSIM = np.zeros((len(z1), len(z2)))

x=tf.convert_to_tensor(x)
with tf.device('/GPU:0'):
    for i in tqdm(range(len(z1))):
        for j in range(len(z2)):
            z = np.array([z1[i], z2[j]])
            z=np.expand_dims(z, axis=0)
            z=np.expand_dims(z, axis=0)
            z=np.expand_dims(z, axis=0)
            z=tf.convert_to_tensor(z)
            y_pred = model([x,z])
            SSIM[i,j] = metric.distance(y_pred)
        
            #SSIM[i,j] = ssim(y_pred[0],y_true, multichannel=True)
  
plt.figure()
plt.contour(z1,z2,SSIM, levels=1000, cmap="RdBu_r")
plt.colorbar()
#plt.plot(x_co, y_co, 'ko', ms=1)

"""
