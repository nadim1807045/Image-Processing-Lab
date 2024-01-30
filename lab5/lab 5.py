import numpy as np
import cv2
import matplotlib.pyplot as plt


#Skeleton Finding
path = "D:\\Academic\\4th Year\\2k17\\4-1\\CSE 4128 Image Lab\\Image Lab Test\\sample1.bmp"

img = cv2.imread(path,0)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i,j]>170:
            img[i,j] = 0
        else:
            img[i,j] = 1

out = np.zeros(img.shape,  np.uint8)
kernel = np.ones((5,5))
k = 0
i=0
while 1:
    i+=1
    prevsum = np.sum(out)
    err = cv2.erode(img , kernel, iterations = k)
    opening = cv2.morphologyEx (err , cv2.MORPH_OPEN ,kernel)
    opening = err - opening
    out = np.bitwise_or(out, opening)
    currentsum = np.sum(out)
    # plt.imshow(out, "gray");plt.show()
    if prevsum == currentsum:
        break
    k +=1

fig = plt.figure(figsize=(15,15))
fig.add_subplot(1,2,1)
plt.imshow(img, "gray");plt.title("Input image")
fig.add_subplot(1,2,2)
plt.imshow(out, "gray");plt.title("Output image")
plt.show()
