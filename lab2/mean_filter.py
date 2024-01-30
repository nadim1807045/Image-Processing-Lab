import numpy as np
import cv2
import matplotlib.pyplot as plt

def selected_mean(x, y):
    sum = 0
    for i in range(-start, start+1):
        for j in range(-start, start + 1):
            sum += img[x+i, y+j]
    return int(sum/kernel_len)

path = "D:\\NC files\\MainDirectory\\imageLabAssignment\\Week 2\\Week_2_Assignment\\lena.png"

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img_H, img_W = img.shape
kernel_size = 3
kernel_len = kernel_size*kernel_size
start = kernel_size // 2

out = np.zeros(img.shape, dtype="uint8")

for i in range(start, img_H - start):
    for j in range(start, img_W - start):
        out[i, j] = selected_mean(i, j)

out1 = cv2.blur(img, (kernel_size, kernel_size))

fig = plt.figure(figsize=(20, 20))
    
fig.add_subplot(1,2,1)
plt.imshow(img, 'gray')
    
fig.add_subplot(1,2,2)
plt.imshow(out, 'gray')

plt.show()

