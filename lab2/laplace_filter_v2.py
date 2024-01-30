import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def getFilterCell(x, y, sigma):
    weight = -1/(math.pi*pow(sigma, 4))
    weight2 = (x*x + y*y)/(2*sigma*sigma)

    return weight*(1-weight2)*math.exp(-weight2)

def clipImage(input):
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            if input[i, j]>255:
                input[i, j] = 255
            elif input[i, j]<0:
                input[i, j] = 0
    return input.astype(dtype="float32")

def scaleImage(input):
    input_m = input - input.min()
    input_s = 255*(input_m/input_m.max())

    return input_s.astype(dtype="float32")

path = "D:\\NC files\\MainDirectory\\imageLabAssignment\\Week 2\\Week_2_Assignment\\lena.png"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))
kernel_size = 11
mid = kernel_size//2
sigma = 1
Kernel = np.zeros((kernel_size, kernel_size), dtype="float32")
out = np.zeros(img.shape, dtype="float32")

for i in range(-mid, mid+1):
    for j in range(-mid, mid+1):
        Kernel[mid+i, mid+j] = getFilterCell(i, j, sigma)

for x in range(mid, img.shape[0]-mid):
    for y in range(mid, img.shape[1]-mid):
        sum = 0
        for u in range(-mid, mid+1):
            for v in range(-mid, mid+1):
                sum += Kernel[u+mid, v+mid]*img[x-u, y-v]
        out[x, y] = sum
        
# =============================================================================
# out = clipImage(out)
# out = scaleImage(out)
# =============================================================================

cv2.imwrite("img.jpg",out)

fig = plt.figure(figsize=(20, 20)) 
fig.add_subplot(1,2,1)
plt.imshow(img, 'gray')
fig.add_subplot(1,2,2)
plt.imshow(out, 'gray')
plt.show()


# 2nd way of doing Laplace Filter
# =============================================================================
# def gaussianKernel(kSize, sigma):
#     gk = np.zeros((kernel_size, kernel_size), dtype="float32")
#     weight = 1 / (2 * math.pi * sigma * sigma)
# 
#     divider = 2 * sigma * sigma
# 
#     for i in range(-mid, mid + 1):
#         for j in range(-mid, mid + 1):
#             gk[mid + i, mid + j] = math.exp(-((i * i + j * j) / divider))
#     gk = weight * gk
#     return gk
# 
# gk1 = gaussianKernel(kernel_size, sigma=2.5)
# gk2 = gaussianKernel(kernel_size, sigma=1)
# 
# lpk = gk1 - gk2
# 
# out2nd = np.zeros(img.shape, dtype="float32")
# 
# for x in range(mid, img.shape[0]-mid):
#     for y in range(mid, img.shape[1]-mid):
#         sum = 0
#         for u in range(-mid, mid+1):
#             for v in range(-mid, mid+1):
#                 sum += lpk[u+mid, v+mid]*img[x-u, y-v]
#         out2nd[x, y] = sum
# 
# plt.imshow(out2nd, "gray")
# plt.title("After Convolution 2nd way")
# plt.show()
# 
# out2nd = clipImage(out2nd)
# plt.imshow(out, "gray")
# plt.title("After Clipping 2nd way")
# plt.show()
# 
# out2nd = scaleImage(out2nd)
# plt.imshow(out, "gray")
# plt.title("After Scaling 2nd way")
# plt.show()
# =============================================================================
