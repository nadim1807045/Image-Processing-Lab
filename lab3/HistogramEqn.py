# -*- coding: utf-8 -*-
"""
Created on Mon May  9 21:01:33 2022

@author: Zahim
"""


import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_histogram(image, bins):
    
    hist = np.zeros(bins)
    for pix in image:
        hist[pix] += 1
    plt.plot(hist)
    plt.show()
    
    return hist



def cumsum(hist):
    
    for i in range(1, len(hist)):
        hist[i] = hist[i]+hist[i-1]

    return np.array(hist)


def normalizeCumSum(hist):
    nj = (hist - hist.min()) * 255
    N = hist.max() - hist.min()
    hist = nj / N

    hist = hist.astype('uint8')

    return hist


def visualizeOutput(img, img_new):
    fig = plt.figure(figsize=(50, 50))
    
    fig.add_subplot(1,2,1)
    plt.imshow(img, 'gray')
    
    fig.add_subplot(1,2,2)
    plt.imshow(img_new, 'gray')
    
    plt.show(block=True)
    
    plt.imshow(img_new, 'gray')
    plt.show()
    
    get_histogram(img_new, 256)


bins = 256

path = "D:\\Academic\\4th Year\\2k17\\4-1\\CSE 4128 Image Lab\\lab2\\surf.jpg"

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

plt.imshow(img, "gray")
plt.show()

img = np.asarray(img)
flat = img.flatten()

hist = get_histogram(flat, bins)

hist1 = cumsum(hist)
print(hist1)

hist2 = normalizeCumSum(hist1)
print(hist2)

img_new = hist2[flat]

img_new = np.reshape(img_new, img.shape)

visualizeOutput(img, img_new)




