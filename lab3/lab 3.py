# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 13:56:42 2022

@author: Zahim
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy

#Contrast Stretching
# =============================================================================
# path = "C:\\Users\\Zahim\\Desktop\\Image Lab Test\\cs.jpg"
# 
# img = cv2.imread(path, 0)
# 
# flat_img = np.array(img).flatten()
# plt.hist(flat_img, bins=256, range=[0,256])
# plt.show()
# 
# H,W = img.shape
# 
# out = deepcopy(img)
# 
# amin = img.min()
# amax = img.max()
# 
# for i in range(H):
#     for j in range(W):
#         out[i,j] = ((img[i,j]-amin)*255)//(amax-amin)
#     
# out_flat = np.array(out).flatten()
# plt.hist(out_flat, bins=256, range=[0,256])
# plt.show()
# 
# =============================================================================

#Histogram Equalization
# =============================================================================
# path = "C:\\Users\\Zahim\\Desktop\\Image Lab Test\\histogram.jpg"
# img = cv2.imread(path, 0)
# plt.imshow(img, "gray")
# plt.show()
# 
# out = deepcopy(img)
# H,W = img.shape
# flat_img = np.array(img).flatten()
# nbin = 256
# freq = [0]*nbin
# 
# plt.hist(np.array(img).flatten(), 256, [0,256])
# plt.show()
# 
# for i in flat_img:
#     freq[i] += 1
# 
# pdf = np.array(freq)/(H*W)
# 
# cdf = [0]*nbin
# cdf[0] = pdf[0]
# 
# for i in range(1, nbin):
#     cdf[i] = cdf[i-1]+pdf[i]
# 
# for i in range(0, H):
#     for j in range(0, W):
#         intensity = img[i,j]
#         out[i,j] = int(cdf[intensity]*255)
# 
# plt.hist(np.array(out).flatten(), 256, [0,256])
# plt.show()
# 
# plt.imshow(out, "gray")
# plt.show()
# =============================================================================

#Local Histogram Equalization
# =============================================================================
# path = "C:\\Users\\Zahim\\Desktop\\Image Lab Test\\hidden.jpg"
# img = cv2.imread(path, 0)
# 
# def histogramEqn(x,y):
#     patch = img[x-mid:x+mid, y-mid:y+mid]
#     flat = patch.flatten()
#     freq = np.zeros(nbin)
#     
#     for i in flat:
#         freq[i] += 1
#     
#     pdf = freq/(ws*ws)
#     
#     cdf = np.zeros(nbin)
#     cdf[0] = pdf[0]
#     
#     for i in range(1, nbin):
#         cdf[i] = cdf[i-1]+pdf[i]
#     
#     out = np.zeros(patch.shape)
#     for i in range(-mid, mid):
#         for j in range(-mid, mid):
#             intensity = patch[mid+i, mid+j]
#             out[mid+i, mid+j] = int(cdf[intensity]*255)
#     
#     OUT[x-mid:x+mid, y-mid:y+mid] = out
# 
# H,W = img.shape
# OUT = np.zeros(img.shape)
# nbin = 256
# ws = 3
# mid = ws//2
# 
# for x in range(mid, H-mid):
#     for y in range(mid, W-mid):
#        histogramEqn(x,y)
# 
# plt.imshow(OUT, "gray")
# plt.show()
# 
# =============================================================================

#Histogram Matching
# =============================================================================
# path = "C:\\Users\\Zahim\\Desktop\\Image Lab Test\\lake.jpg"
# img = cv2.imread(path, 0)
# 
# def generateGaussian(x,miu,std):
#     upper = (x-miu)**2
#     lower = 2*std**2
#     return np.exp(-upper/lower)
# 
# def findclose(val):
#     mini = 500
#     result = 0
#     for i in range(256):
#         if cdf[i]==val:
#             return i
#         elif abs(cdf[i]-val)<mini:
#             result = i
#             mini = abs(cdf[i]-val)
#         elif abs(cdf[i]-val)>mini:
#             break
#     return result
# 
# miu1 = 70
# std1 = 15
# 
# miu2 = 150
# std2 = 15
# 
# g1 = []
# for x in range(256):
#     g1.append(generateGaussian(x,miu1,std1))
# 
# g2 = []
# for x in range(256):
#     g2.append(generateGaussian(x,miu2,std2))
# 
# gnew = np.array(g1)+np.array(g2)
# 
# plt.plot(gnew);plt.show()
# 
# pdf = gnew/gnew.sum()
# 
# cdf = np.zeros(256)
# cdf[0] = pdf[0]
# 
# for i in range(1, 256):
#     cdf[i] = cdf[i-1]+pdf[i]
# 
# # plt.plot(cdf);plt.show()
# 
# freq = [0]*256
# flat_img = np.array(img).flatten()
# H,W = img.shape
# 
# for i in flat_img:
#     freq[i] += 1
# 
# pdfimg = np.array(freq)/(H*W)
# 
# cdfimg = [0]*256
# cdfimg[0] = pdfimg[0]
# 
# for i in range(1, 256):
#     cdfimg[i] = cdfimg[i-1]+pdfimg[i]
# 
# # plt.plot(freq);plt.show()
# # plt.plot(cdfimg);plt.show()
# 
# out = np.empty(img.shape)
# 
# for i in range(0,H):
#     for j in range(0,W):
#         intensity = img[i,j]
#         out[i,j] = findclose(cdfimg[intensity])
# 
# flatout = out.flatten()
# plt.hist(flatout,256,[0,256]);plt.show()
# =============================================================================

# Erlang Distribution Equation
# (lamda**k * x**(k-1)*exp(-lambda*x))/math.factorial(k-1)
# k = 90
# lamda = 0.8









