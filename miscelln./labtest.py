# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 12:13:18 2022

@author: Zahim
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

path = "C:\\Users\\Zahim\\Desktop\\Image Lab Test\\hole.png"

img = cv2.imread(path, 0)

plt.imshow(img, "gray");plt.show()

out = np.zeros(img.shape)
out[100,55] =1

t, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

plt.imshow(img, "gray");plt.show()
plt.imshow(out, "gray");plt.show()
imgc = 1-img
plt.imshow(imgc, "gray");plt.show()


