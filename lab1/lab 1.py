# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 18:20:08 2022

@author: Zahim
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

path = "C:\\Users\\Zahim\\Desktop\\Image Lab Test\\lena.png"

img = cv2.imread(path, 0)
img = cv2.resize(img, (200,200))

plt.imshow(img, "gray")
plt.show()

kernel = np.array(([0,-1,0], [-1, 5, -1], [0,-1,0]), np.float32)

H, W = img.shape
ks = len(kernel)
mid = ks//2
n = (ks-1)//2

out = np.zeros((H,W), np.float32)

for i in range(H-mid):
    for j in range(W-mid):
        summ = 0
        for x in range(-mid, mid+1):
            for y in range(-mid, mid+1):
                summ += kernel[x+mid, y+mid]*img[i-x, j-y]
        out[i,j] = summ

plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
plt.show()


#Block Circulant matrix convolution
img = cv2.imread(path, 0)
img = cv2.resize(img, (200,200))
H,W = img.shape
img_vec = np.asarray(img)
img_vec = img_vec.reshape([H*W,1]) #Shape 4000,1

kernel = np.array(([0,-1,0], [-1, 5, -1], [0,-1,0]), np.float32)
kernel = kernel.tolist() #Convert array to list
kernel_size = len(kernel)

CmH = H*W
CmW = CmH

singleRow = list()

for i in range(kernel_size):
    singleRow.extend(kernel[i])
    rest = [0]*(W-kernel_size)
    singleRow.extend(rest)
ext = [0]*(CmW-len(singleRow))
singleRow.extend(ext) #Generating the first row

rowArray = np.array(singleRow).reshape(1, CmW) #reshape the row to 4000,1 

block_kernel = np.empty((CmH,CmW), np.float32) #4000,4000

for i in range(CmH):
    block_kernel[i] = np.append(rowArray[0, CmH-i:], rowArray[0, :CmH-i])

conv_image = block_kernel@img_vec #Matrix multiply
conv_image = conv_image.reshape([H,W])
plt.imshow(cv2.cvtColor(conv_image, cv2.COLOR_BGR2RGB))
plt.show()

