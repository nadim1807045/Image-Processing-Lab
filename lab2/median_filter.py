import numpy as np
import cv2
import matplotlib.pyplot as plt

def find_median(x, y):
    intensity_val = list()
    for i in range(-start, start+1):
        for j in range(-start, start+1):
            intensity_val.append(img[x-i, y-j])
    intensity_val.sort()
    return intensity_val[kernel_len//2]


path = "D:\\NC files\\MainDirectory\\imageLabAssignment\\Week 2\\Week_2_Assignment\\couple.jpg"

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

img_H, img_W = img.shape
kernel_size = 5
kernel_len = kernel_size*kernel_size

start = kernel_size//2
out = np.zeros(img.shape, dtype="uint8")

for i in range(start, img_H-start):
    for j in range(start, img_W - start):
        out[i, j] = find_median(i, j)


fig = plt.figure(figsize=(20, 20))
    
fig.add_subplot(1,2,1)
plt.imshow(img, 'gray')
    
fig.add_subplot(1,2,2)
plt.imshow(out, 'gray')

plt.show()
