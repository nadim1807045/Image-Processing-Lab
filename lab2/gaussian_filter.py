import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

path = "D:\\NC files\\MainDirectory\\imageLabAssignment\\Week 2\\Week_2_Assignment\\rubiks_cube.png"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img_H, img_W = img.shape
kernel_size = 9
sigma = kernel_size/5
start = kernel_size//2
kernel = np.zeros((kernel_size, kernel_size))
out = np.zeros(img.shape, dtype="uint8")

weight = 1/(2*math.pi*sigma*sigma)
divider = 2*sigma*sigma

for i in range(-start, start+1):
    for j in range(-start, start + 1):
        kernel[start + i, start + j] = math.exp(-((i*i+j*j)/divider))
kernel = weight*kernel

for x in range(start, img_H-start):
    for y in range(start, img_W-start):
        sum = 0
        for u in range(-start, start+1):
            for v in range(-start, start+1):
                sum += kernel[u+start, v+start]*img[x-u, y-v]
        out[x, y] = int(sum)
        
fig = plt.figure(figsize=(20, 20))
    
fig.add_subplot(1,2,1)
plt.imshow(img, 'gray')
fig.add_subplot(1,2,2)
plt.imshow(out, 'gray')
plt.show()
