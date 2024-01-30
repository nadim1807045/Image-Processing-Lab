# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 21:52:52 2022

@author: Zahim
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy

#Homomorphic Filter
# =============================================================================
# path = "C:\\Users\\Zahim\\Desktop\\Image Lab Test\\homo.jpg"
# 
# img = cv2.imread(path,0)
# H,W = img.shape
# # plt.imshow(img, "gray");plt.show()
# 
# img = np.float32(img)
# img = np.log(img+1)
# img = np.fft.fft2(img)
# img = np.fft.fftshift(img)
# 
# # plt.imshow(np.log(np.abs(img)), "gray");plt.show()
# 
# kernel = np.empty((H,W))
# D = np.empty((H,W))
# 
# for u in range(H):
#     for v in range(W):
#         D[u,v] = np.sqrt((u-H//2)**2+(v-W//2)**2)
# 
# GH = 1.2
# GL = 0.5
# D0 = 50
# c = 0.1
# 
# for u in range(H):
#     for v in range(W):
#         kernel[u,v] = (GH-GL)*(1-np.exp(-(c*D[u, v]**2)/(D0)**2))+GL
# 
# # plt.imshow(kernel,"hot");plt.show()
# 
# cimg = img*kernel
# cimg = np.fft.ifftshift(cimg)
# spimg = np.fft.ifft2(cimg)
# spaimg = np.exp(np.abs(spimg))-1
# 
# plt.imshow(spaimg,"gray");plt.show()
# =============================================================================

#Motion Deblur (This way not Not working)
# =============================================================================
# path = "C:\\Users\\Zahim\\Desktop\\Image Lab Test\\person_1.png"
# 
# img1 = cv2.imread(path, 0)
# H,W = img1.shape
# 
# ks = 7
# mid = ks//2
# 
# spkernel = np.zeros((ks,ks))
# out = np.zeros(img1.shape)
# 
# for i in range(ks):
#     for j in range(ks):
#         if i==j:
#             spkernel[i,j] = 1
# 
# for i in range(mid, H-mid):
#     for j in range(mid, W-mid):
#         summ = 0
#         for x in range(-mid, mid+1):
#             for y in range(-mid, mid+1):
#                 summ += spkernel[x+mid, y+mid]*img1[i-x, j-y]
#         out[i,j] = summ/ks**2
# 
# a = int(img1.shape[0]//2 - spkernel.shape[0]//2)
# spkernel_pad = np.pad(spkernel, (a,a-1), 'constant', constant_values=(0))
# img = cv2.resize(out,(spkernel_pad.shape[0],spkernel_pad.shape[1]))
# 
# img = np.float32(img)
# img = np.fft.fft2(img)
# 
# H = np.fft.fft2(spkernel_pad)
# 
# backimg = np.divide(img, H)
# 
# backimg1 = np.real(np.fft.ifft2(backimg))
# 
# plt.imshow(backimg1,"gray");plt.show()
# 
# =============================================================================














