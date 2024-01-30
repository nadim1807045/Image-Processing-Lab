# -*- coding: utf-8 -*-
"""H_Filter.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UKHBLw4D2pp3TM0lnOy0x4iG5QwSfYyr
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

path = "D:\\NC files\\MainDirectory\\imageLabAssignment\\Week 4\\homo.jpg"

I = cv2.imread(path, 0)
M, N = I.shape
img = np.float32(I)
img = img + 1
img = np.log(img)
img = np.fft.fft2(img)
img = np.fft.fftshift(img)

print(img.shape)
plt.imshow((np.log(np.abs(img))), "gray")
plt.show()

GH = 1.2
GL = 0.5
D0 = 50
c = 0.1
D = np.zeros(img.shape)
for u in range(M):
    for v in range(N):
        D[u, v] = np.sqrt((u-M//2)**2+(v-N//2)**2)

H = np.zeros(img.shape)
for u in range(M):
    for v in range(N):
        H[u,v] = (GH-GL)*(1-np.exp(-(c*D[u, v]**2)/(D0)**2))+GL
plt.imshow((np.abs(H)), "gray")
plt.show()

img_f2  = img*H
img_f = np.fft.ifftshift(img_f2)
img_f = np.fft.ifft2(img_f)
img_g = np.exp(np.abs(img_f))-1

fig = plt.figure(figsize=(15,15))
fig.add_subplot(1,2,1)
plt.imshow(I, "gray")
plt.title("Original Image")
fig.add_subplot(1,2,2)
plt.imshow(img_g, "gray")
plt.title("Filtered Image")
plt.show()