# -*- coding: utf-8 -*-
"""ImgLab1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DV3NIKUdhk-J3OSo7xym5NNLh3PwIDBk
"""

# =============================================================================
# import cv2
# import matplotlib.pyplot as plt
# 
# =============================================================================
# =============================================================================
# path = "C:\\Users\\Zahim\\Desktop\\Image Lab Test\\lena.png"
# 
# import numpy as np
# 
# img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, (200,200))
# img_H = img.shape[0]
# img_W = img.shape[1]
# 
# kernel = np.array(([0,-1,0], [-1, 5, -1], [0,-1,0]), np.float32)
# 
# kernel_size  = len(kernel)
# 
# result = np.zeros((img_H, img_W), dtype='float32')
# n = (kernel_size - 1)//2
# for x in range(img_H):
#   for y in range(img_W):
#     sum = 0
#     for i in range(kernel_size):
#       for j in range(kernel_size):
#         sum += kernel[i,j]*img[x-i-n, y-j-n]
#     result[x, y] = sum
# 
# plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
# plt.show()
# =============================================================================

#Assignment
#2D Filtering with block circulant matrix multiplication

#Import necessary packages
import numpy as np
import cv2
import matplotlib.pyplot as plt

#Load Image, resize 
path = '/content/drive/MyDrive/imageLab1/lena.png'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (200,200))
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()
img_H = img.shape[0]
img_W = img.shape[1]
img_shape = img.shape
img_vec = np.asarray(img)
img_vec = img_vec.reshape([img_H*img_W,1])

#Kernel
kernel = np.array(([0,-1,0], [-1, 5, -1], [0,-1,0]), np.float32)
kernel = kernel.tolist() #Converts to list [0,-1,0,-1,5,-1,0,-1,0]
kernel_size = len(kernel) #3

#Circulant Matrix dimension
cMatrixHeight =  img_H*img_W
cMatrixWidth = cMatrixHeight

#Constructing 1st row of the block circulant matrix 
singleRow = list()
for i in range(kernel_size):
  singleRow.extend(kernel[i])
  rest = [0]*(img_W-kernel_size)
  singleRow.extend(rest)
extras = [0]*(cMatrixWidth-len(singleRow))
singleRow.extend(extras)

#Constructing all other rows 
rowArray = np.array(singleRow).reshape(1, cMatrixWidth)
# print(rowArray.shape)
out = np.empty((cMatrixHeight, cMatrixWidth), dtype = "float32")

# print(out.shape)
for i in range(cMatrixHeight):
  out[i] = np.append(rowArray[0, cMatrixHeight-i:], rowArray[0,:cMatrixHeight-i])
# print(out)

# Multiplying matrices, reshape and create image
result = np.dot(out, img_vec)
result = result.reshape(img_shape)
plt.imshow(cv2.cvtColor(np.float32(result), cv2.COLOR_BGR2RGB))
plt.show()

# 1 1 1 1 1
# 2 2 2 2 2
# 3 3 3 3 3
# 4 4 4 4 4
# 5 5 5 5 5

# After conversion= (5*5, 1)

# 1 1 1
# 1 1 1
# 1 1 1

# #1 row of the block matrix

# 1 1 1 0 0 1 1 1 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0

