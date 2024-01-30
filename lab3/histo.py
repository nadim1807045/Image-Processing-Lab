# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:33:02 2022

@author: Zahim
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

path= "C:\\Users\\Zahim\\Pictures\\Saved Pictures\\histogram.jpg"

img= cv2.imread(path, cv2.IMREAD_GRAYSCALE)

H,W = img.shape

plt.imshow(img, "gray")
plt.show()

bins=256
flat = img.flatten()
hist = np.zeros(bins)
H,W = img.shape
for pix in flat:
    hist[pix] += 1

# =============================================================================
# print(hist)
# =============================================================================

plt.plot(hist, bins)
plt.show()

pdf= np.zeros(bins)

for i in range(bins):
    pdf[i] = pdf[i]/(H*W)

plt.plot(pdf, bins)
plt.show()


cdf= np.zeros(bins)

cdf[0] = pdf[0]

for i in range(1, bins):
    cdf[i] = cdf[i-1]+pdf[i]
    
plt.plot(cdf, bins)
plt.show()



