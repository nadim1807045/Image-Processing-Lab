import numpy as np
import cv2
import matplotlib.pyplot as plt

path = "D:\\NC files\\MainDirectory\\imageLabAssignment\\Week 2\\Week_2_Assignment\\lena.png"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
xKernel = np.array([[1, 0, -1],
              [1, 0, -1],
              [1, 0, -1]], np.float32)
yKernel = np.array([[1, 1, 1],
              [0, 0, 0],
              [-1, -1, -1]], np.float32)

Gx = np.zeros(img.shape, np.float32)
Gy = np.zeros(img.shape, np.float32)
mid = len(xKernel)//2

for x in range(mid, img.shape[0]-mid):
    for y in range(mid, img.shape[1] - mid):
        for i in range(-mid, mid+1):
            for j in range(-mid, mid + 1):
                Gx[x, y] += xKernel[mid+i, mid+j]*img[x-i, y-j]

for x in range(mid, img.shape[0]-mid):
    for y in range(mid, img.shape[1] - mid):
        for i in range(-mid, mid+1):
            for j in range(-mid, mid + 1):
                Gy[x, y] += yKernel[mid+i, mid+j]*img[x-i, y-j]
Gx2 = (Gx*Gx)
Gy2 = (Gy*Gy)
G = (Gx2+Gy2)
G = pow(G, 0.5)

fig = plt.figure(figsize=(20, 20)) 
fig.add_subplot(1,2,1)
plt.imshow(img, 'gray')  
fig.add_subplot(1,2,2)
plt.imshow(G, 'gray')
plt.show()