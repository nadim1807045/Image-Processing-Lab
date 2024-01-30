import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

path = "D:\\Academic\\4th Year\\2k17\\4-1\\CSE 4128 Image Lab\\lab2\\einstein.jpg"

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def normalize(x, min_, max_, Tmin, Tmax):
    if min_-max_ !=0:
        return ((Tmax - Tmin)*(x - min_))/(max_ - min_) + Tmin
    else:
        return x

#Negative Transformation
neg = np.zeros(img.shape, np.float32)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        neg[i, j] = 255 - img[i, j]

fig = plt.figure(figsize=(20, 20))
fig.add_subplot(1,2,1)
plt.imshow(img, 'gray')
fig.add_subplot(1,2,2)
plt.imshow(neg, 'gray')
plt.show()


#Log transformation
c = 1
logT = np.zeros(img.shape, np.float32)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        logT[i, j] = c*math.log(1+img[i, j])
        logT[i, j] = normalize(logT[i, j], 0, c*math.log(256), 0, 255)

fig = plt.figure(figsize=(20, 20))
fig.add_subplot(1,2,1)
plt.imshow(img, 'gray')
fig.add_subplot(1,2,2)
plt.imshow(logT, 'gray')
plt.show()


#PowerGamma Transformation
gamma = 0.4
GT = np.zeros(img.shape, np.float32)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        GT[i, j] = c*math.pow(img[i, j], gamma)
        GT[i, j] = normalize(GT[i, j], 0, c*math.pow(255, gamma), 0, 255)

fig = plt.figure(figsize=(20, 20))
fig.add_subplot(1,2,1)
plt.imshow(img, 'gray')
fig.add_subplot(1,2,2)
plt.imshow(GT, 'gray')
plt.show()


#Constrast Stretching
path = "D:\\Academic\\4th Year\\2k17\\4-1\\CSE 4128 Image Lab\\lab2\\cs.jpg"

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
s1= 0
s2 = 255
r1 = img.min()
r2 = img.max()
CS = img.copy()
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        CS[i, j] = ((s2-s1)*(img[i,j]-r1))/(r2-r1)+s1

cv2.imshow("Input", img)
cv2.imshow("Output",CS)
cv2.waitKey(0)



