# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 06:54:25 2022

@author: Zahim
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
# =============================================================================
# 
# path = "C:\\Users\\Zahim\\Desktop\\Image Lab Test\\rubiks_cube.png"
# 
# img = cv2.imread(path, 0)
# img = cv2.resize(img, (200,200))
# H,W = img.shape
# 
# #Bilateral Filter
# ks = 5
# sSigma = ks/5
# rSigma = 1
# gKernel = np.empty((ks,ks))
# out = np.empty((H,W), np.float32)
# mid = ks//2
# 
# div = 2*sSigma**2
# for i in range(-mid, mid+1):
#     for j in range(-mid, mid+1):
#         gKernel[i+mid, j+mid] = np.exp(-((i**2+j**2)/div))
# 
# # print(gKernel)
# def iKernel(x,y):
#     iKer = np.empty((ks,ks))
#     divi = 2*rSigma**2
#     i2 = img[x,y]
#     for i in range(-mid, mid+1):
#         for j in range(-mid, mid+1):
#             i1 = img[x+i, y+j]
#             iKer[i+mid, j+mid] = exp(-(int(i2) - int(i1))**2 / divi)
#     
#     newKernel = gKernel*iKer
#     return newKernel, newKernel.sum()
# 
# for i in range(mid, H-mid):
#     for j in range(mid, W-mid):
#         kernel, ksum = iKernel(i, j)
#         summ = 0
#         for x in range(-mid, mid+1):
#             for y in range(-mid, mid+1):
#                 summ += kernel[x+mid, y+mid]*img[i-x, j-y]
#         out[i,j] = summ/ksum
# 
# plt.imshow(out, "gray")
# plt.show()
# =============================================================================

#Gaussian Filter
# =============================================================================
# ks = 9
# sSigma = ks/5 
# div = 2*sSigma**2
# gKernel = np.empty((ks,ks))
# mid = ks//2
# 
# for i in range(-mid, mid+1):
#     for j in range(-mid, mid+1):
#         gKernel[i+mid, j+mid] = np.exp(-((i**2+j**2)/div))
# 
# out = np.empty((H,W), np.float32)
# 
# for i in range(mid, H-mid):
#     for j in range(mid, W-mid):
#         summ = 0
#         for x in range(-mid, mid+1):
#             for y in range(-mid, mid+1):
#                 summ += gKernel[x+mid, y+mid]*img[i-x, j-y]
#         out[i,j] = summ
# 
# plt.imshow(out, "gray")
# plt.show()
# =============================================================================

#Median, Min, Max Filter are non linear filter Works 
#based on sorting the neighborhood values.

# Sobel Filter
# 1 0 -1     1 2 1
# 2 0 -2     0 0 0
# 1 0 -1    -1 -2 -1

# Scharr Filter
# 3 0 -3     3 10 3
# 10 0 -10   0 0 0
# 3 0 -3    -3 -10 -3

# Prewitt Filter
# 1 0 -1     1 1 1
# 1 0 -1     0 0 0
# 1 0 -1   -1 -1 -1

# Laplace Filter
# -1 -1 -1     0 -1 0
# -1 8 -1     -1 5 -1
# -1 -1 -1     0 -1 0

#Laplace Filter
path1 = "C:\\Users\\Zahim\\Desktop\\Image Lab Test\\moon.jpg"

img = cv2.imread(path1, 0)
img = cv2.resize(img, (200,200))

plt.imshow(img, "gray")
plt.show()

kernel = np.array(([0,-1,0],[-1,4,-1],[0,-1,0]), np.float32)

H,W = img.shape
mid = len(kernel)//2
out = np.empty((H,W), np.float32)

for x in range(mid, H-mid):
    for y in range(mid, W-mid):
        summ = 0
        for i in range(-mid, mid+1):
            for j in range(-mid, mid+1):
                summ+= kernel[i+mid, j+mid]*img[x-i, y-j]
        out[x,y] = int(summ)

plt.imshow(out, "gray")
plt.show()
cv2.imwrite("./LaplaceFi.jpg", out)

new = img + out
plt.imshow(new, "gray")
plt.show()
cv2.imwrite("./LaplaceD.jpg", new)


