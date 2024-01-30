import numpy as np
import cv2
import matplotlib.pyplot as plt

path = "D:\\Academic\\4th Year\\2k17\\4-1\\CSE 4128 Image Lab\\lab2\\hidden.jpg"

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def histogramEqua(x, y, bins, window_size):
    
    imgs = img[x-mid:x+mid, y-mid:y+mid]
    flat = imgs.flatten()
    hist = np.zeros(bins)
    
    for pix in flat:
        hist[pix] += 1
    
    pdf = np.zeros(bins)
    
    for i in range(bins):
        pdf[i] = hist[i]/(window_size*window_size)

    cdf = np.zeros(bins)
    
    cdf[0] = pdf[0]
    
    for i in range(1, bins):
        cdf[i] = cdf[i-1]+pdf[i]

    out = np.zeros(imgs.shape)
    
    for i in range(-mid, mid):
        for j in range(-mid, mid):
            intensity = imgs[mid+i, mid+j]
            cdf_val = cdf[intensity]
            out[mid+i, mid+j] = int(cdf_val*255)

    OUT[x-mid:x+mid, y-mid:y+mid] = out

img = np.asarray(img)
flat = img.flatten()
H, W = img.shape
OUT = np.zeros(img.shape)
hist = np.zeros(256)
bins = 256
window_size = 3
mid = window_size//2


for i in range(mid, H-mid):
    for j in range(mid, W-mid):
        histogramEqua(i, j, bins, window_size)

fig = plt.figure(figsize=(20, 20))
fig.add_subplot(1,2,1)
plt.imshow(img, 'gray')
fig.add_subplot(1,2,2)
plt.imshow(OUT, 'gray')
plt.show()