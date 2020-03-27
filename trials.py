# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:58:45 2020

@author: Georgios
"""

#check spirals

import numpy as np
import matplotlib.pyplot as plt

def r(t,omega, a=0.02,b=0.15):
    return a*np.exp(b*omega*t)

x=[]
y=[]
omega=1
T=2*np.pi/omega
N=100 #steps per revolution

for t in np.linspace(0, 5*T, 5*N):
    x.append(r(t, omega, a=0.02,b=0.15)*np.cos(omega*t))
    y.append(r(t,omega, a=0.02,b=0.15)*np.sin(omega*t))
    
    
plt.figure()
plt.plot(x,y)