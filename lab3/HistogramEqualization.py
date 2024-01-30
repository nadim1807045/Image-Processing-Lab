import numpy as np
import cv2
import matplotlib.pyplot as plt

path = "C:\\Users\\Zahim\\Pictures\\Saved Pictures\\histogram.jpg"

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def histogramEqua(img, bins):
    flat = img.flatten()
    hist = np.zeros(bins)
    H,W = img.shape
    for pix in flat:
        hist[pix] += 1

    plt.plot(hist)
    plt.show()
    
    pdf = np.zeros(bins)
    
    for i in range(bins):
        pdf[i] = hist[i]/(H*W)

    plt.plot(pdf)
    plt.show()
    
    cdf = np.zeros(bins)
    
    cdf[0] = pdf[0]
    
    for i in range(1, bins):
        cdf[i] = cdf[i-1]+pdf[i]
    
    plt.plot(cdf)
    plt.show()
    
    out = np.zeros(img.shape)
    
    for i in range(H):
        for j in range(W):
            intensity = img[i, j]
            cdf_val = cdf[intensity]
            out[i, j] = int(cdf_val*255)
            
    fig = plt.figure(figsize=(15, 15))
    
    fig.add_subplot(1,2,1)
    plt.imshow(img, 'gray')
    fig.add_subplot(1,2,2)
    plt.imshow(out, 'gray')
    plt.show()
    
    fig1 = plt.figure(figsize=(15, 5))
    out = out.flatten()
    fig1.add_subplot(1,2,1)
    plt.hist(flat, bins=256)
    fig1.add_subplot(1,2,2)
    plt.hist(out, bins=256)
    plt.show()
    
# =============================================================================
#     plt.imshow(out, "gray")
#     plt.show()
# =============================================================================


img = np.asarray(img)
bins = 256

histogramEqua(img, bins)