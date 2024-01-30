import numpy as np
import math
import cv2
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt

path = "C:\\Users\\Zahim\\Desktop\\Image Lab Test\\rubiks_cube.png"

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (200, 200))
img_H = img.shape[0]
img_W = img.shape[1]

kernel_size = 5
sigma = kernel_size/5
rsigma = 1

out = np.empty((img_H, img_W), dtype="float32")
mid = math.floor(kernel_size / 2)

GKernel = np.zeros((kernel_size, kernel_size))
dividant = 2*pow(sigma, 2)
for i in range(-mid, mid+1):
    for j in range(-mid, mid+1):
        val = math.exp(-(pow(i, 2)+pow(j, 2))/dividant)
        GKernel[mid+i][mid+j] = round(val, 4)


def GaussianIntensityKernel(x, y):
    intensityKernel = np.zeros((kernel_size, kernel_size))
    dividant1 = 2 * pow(rsigma, 2)

    for i in range(-mid, mid+1):
        for j in range(-mid, mid+1):
            val = math.exp(-(pow(int(img[x, y]) - int(img[x+i, y+j]), 2)) / dividant1)
            intensityKernel[mid + i][mid + j] = val

    GIKernel = GKernel*intensityKernel

    return GIKernel


def bilateralFilter():
    result = np.zeros((img_H, img_W), dtype=np.float32)
    n = math.floor(kernel_size/2)
    for x in range(n, img_H-n):
      for y in range(n, img_W-n):
        sum = 0
        kernel = GaussianIntensityKernel(x, y)
        kernel_sum = kernel.sum()
        for i in range(-n, n+1):
          for j in range(-n, n+1):
            sum += kernel[i+n, j+n]*img[x-i, y-j]
        result[x, y] = sum/kernel_sum

    result = rescale_intensity(result, in_range=(0, 255))

    fig = plt.figure(figsize=(20,20))
    
    fig.add_subplot(1,2,1)
    plt.imshow(img, "gray")
    fig.add_subplot(1,2,2)
    plt.imshow(result, "gray")
    plt.show()

bilateralFilter()