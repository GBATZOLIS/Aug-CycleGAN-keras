# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 17:08:21 2020

@author: Georgios
"""

#Create the simplest version of the LPIPS loss
#use cosine distance instead of learnable weights based on the LPIPS dataset

import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model


K = tf.keras.backend

class lpips(object):
    def __init__(self, img_shape):
        self.img_shape = img_shape
        self.model = self.create_model()
        
    def create_model(self,):
        base_model = tf.keras.applications.VGG19(input_shape=self.img_shape, 
                                                 include_top=False, 
                                                 weights='imagenet')
        
        
        conv_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']
        
        #the following code uses all convolutional layers
        """
        for layer in base_model.layers:
            layer_type = layer.name.split('_')[1][0:4]
            if layer_type=='conv':
                conv_layers.append(layer.name)
        """   
        
        outputs = [base_model.get_layer(name).output for name in conv_layers]
        
        model = Model(base_model.input, outputs)
        model.trainable=False
        
        return model
    
    def set_reference(self, x):
        x = x*255
        x = preprocess_input(x)
        out_tensors = self.model(x)
        self.reference = out_tensors
        normalised_reference = []
        for tensor_ref in out_tensors:
            tensor_ref,_ = tf.linalg.normalize(tensor_ref, axis=-1)
            normalised_reference.append(tensor_ref)
        self.normalised_reference = normalised_reference
        
    def distance(self, tensor1, reference=None):
        if reference==None:
            tensor1 = tensor1*255
            tensor1 = preprocess_input(tensor1)
            
            tensor1_outs = self.model(tensor1)
            
            distance=0
            for tensor1, tensor2 in zip(tensor1_outs, self.normalised_reference):
                H=tensor1.shape[1]
                W=tensor1.shape[2]
                tensor1,_ = tf.linalg.normalize(tensor1, axis=-1)
                diff_tensor = tf.math.squared_difference(tensor1, tensor2)
                collapsed_sum = tf.reduce_sum(diff_tensor)
                collapsed_sum_scaled = collapsed_sum/(H*W)
                distance+=collapsed_sum_scaled
            
            distance = K.get_value(distance)
            return distance
                
        else:
            #check if batch_size of tensor1 and reference are compatible
            if tensor1.shape!=reference.shape:
                return Exception('Input tensor with incompatible shapes')
            else:
                batch_size = tensor1.shape[0]
                total_distance=0
                for i in range(batch_size):
                    tensors = tf.stack([tensor1[i], reference[i]])
                    #print(tensors.shape)
                    tensors = tensors*255
                    tensors = preprocess_input(tensors)
                    out_tensors = self.model(tensors)
                    
                    distance=0
                    for tensors in out_tensors:
                        H=tensors.shape[1]
                        W=tensors.shape[2]
                        
                        tensors,_ = tf.linalg.normalize(tensors, axis=-1)
                        diff_tensor = tf.math.squared_difference(tensors[0],tensors[1])
                        collapsed_sum = tf.reduce_sum(diff_tensor)
                        collapsed_sum_scaled = collapsed_sum/(H*W)
                        distance+=collapsed_sum_scaled
                    
                    distance = K.get_value(distance)
                    
                    total_distance+=distance
                
                total_distance=total_distance/batch_size
                return total_distance


"""

metric=lpips((100,100,3))

x1 = tf.random.normal((5,100,100,3), dtype=tf.float32)
x2 = tf.random.normal((5,100,100,3), dtype=tf.float32)

print(metric.distance(x1,x2))
"""