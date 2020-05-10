# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:28:42 2020

@author: Georgios
"""

#this file is going to be used to prepare that dataset

from glob import glob
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

def resize(path, width, height):
    dim = width, height
    complete_paths =  glob(path)
    #print(complete_paths)
    for path in tqdm(complete_paths):
        #print(path)
        filename = os.path.basename(path)
        img = Image.open(path)
        img.thumbnail(dim, Image.ANTIALIAS)
        img.save(path, "JPEG")
        

paths = ['data/trainA/*.jpg', 'data/trainB/*.jpg', 'data/testA/*.jpg', 'data/testB/*.jpg']

for path in paths:
    resize(path, 256, 256)