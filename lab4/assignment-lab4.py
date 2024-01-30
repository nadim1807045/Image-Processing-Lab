# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 23:59:18 2022

@author: mthff
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.fftpack import fftn, ifftn, fftshift

f = cv2.imread("period_input.jpg",0)
f = cv2.resize(f, (f.shape[0], f.shape[0]))

mean = 0
var = 10
sigma = var ** 0.5
gaussian = np.random.normal(mean, sigma, (f.shape[0], f.shape[0])) 
plt.title('gaussian noise', fontsize=10)
plt.imshow(gaussian, 'gray');
plt.show()
  
# filter
h = np.zeros((31,31))
for i in range(31):
    for j in range(31):
        if i==j:
            h[i,j]=1
for i in range(15):
    for j in range(15):
        if i==j:
            h[i,j]=0
    

# computing the number of padding on one side
padd = int(f.shape[0]//2 - h.shape[0]//2)
n = f.shape[0]
if (n % 2) == 0:
    h_pad = np.pad(h, (padd,padd-1), 'constant', constant_values=(0))
else:
    h_pad = np.pad(h, (padd,padd), 'constant', constant_values=(0))


plt.figure(figsize=(18,10))
plt.subplot(121)
plt.imshow(h, 'gray');
plt.title("filter without padding",fontsize=40)
plt.subplot(122)
plt.imshow(h_pad, 'gray'); plt.title("filter with padding",fontsize=40)
plt.show()

# computing the Fourier transforms
F = fftn(f)
H = fftn(h_pad)

plt.figure(figsize=(18,10));plt.subplot(121)
plt.imshow(fftshift(np.log(np.abs(F)+1)), 'gray')
plt.title("Fourier Trasform of input image",fontsize=10)
plt.subplot(122)
plt.imshow(fftshift(np.log(np.abs(H)+1)), 'gray')
plt.title("Fourier Trasform of Gaussian Filter",fontsize=10)
plt.show()

# convolution
G = np.multiply(F,H)

# Inverse Transform
g = fftshift(ifftn(G).real)

plt.title('image after bluring', fontsize=10)
plt.imshow(g, 'gray');
plt.show()

gaus =  fftn(gaussian)
G = G + gaus
g_ = fftshift(ifftn(G).real)

plt.figure(figsize=(18,10))
plt.subplot(121)
plt.imshow(f, 'gray');
plt.title("original image",fontsize=40)
plt.subplot(122)
plt.imshow(g, 'gray'); 
plt.title("blurred image",fontsize=40)

# ### Inverse filter
F_hat = np.divide(G,H)

'''NSR = 0.01''' 
Denoise_constant = (1 + 0.000005/(H*H))
F_hat = (F_hat / Denoise_constant)

f_hat = ifftn(F_hat).real

plt.figure(figsize=(18,10))
plt.subplot(121)
plt.imshow(g_, 'gray');

plt.subplot(122)
plt.imshow(f_hat, 'gray')
