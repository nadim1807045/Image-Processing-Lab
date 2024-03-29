# -*- coding: utf-8 -*-
"""Class.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11lYaOKQTo7abkGhyFN2H2QlMC20ZoeQ_
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 18:49:01 2022

@author: Sunanda Das
"""

import cv2


def click_event(event, x, y, flags, params):
 
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
 
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', img)
 
    # checking for right mouse clicks    
    if event==cv2.EVENT_RBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
 
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x,y), font, 1,
                    (255, 255, 0), 2)
        cv2.imshow('image', img)
 

   
img = cv2.imread('', 0)
cv2.imshow('image', img)  
cv2.setMouseCallback('image', click_event)  
cv2.waitKey(0) 
cv2.destroyAllWindows()

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('C:\\Users\\Zahim\\Desktop\\Image Lab Test\\homo.jpg', 0)

plt.imshow(img, cmap='gray')
plt.title('Input')
plt.show()

dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
magnitude = np.log(cv2.magnitude(dft[:, :, 0], dft[:, :, 1]))

plt.imshow(magnitude, cmap='gray')
plt.title('Magnitude')
plt.show()

dft_shift = np.fft.fftshift(dft)
magnitude_specturm = np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

plt.imshow(magnitude_specturm, cmap='gray')
plt.title('Magnitude Specturm after Shift')
plt.show()

##wrong method
d0 = 25
n = 1

butter_filter = np.ones(img.shape)

center_i, center_j = img.shape[0] // 2, img.shape[1] // 2

u = [350, 250, 180, 310]
v = [310, 470, 400, 400]
u2 = []
v2 = []

for i in range(len(u)):
  u2.append(img.shape[0] - u[i])
  v2.append(img.shape[1] - v[i])

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        prod = 1
        for k in range(4):
            D1 = np.sqrt((i - u[k]) ** 2 + (j - v[k]) ** 2)
            D2 = np.sqrt((i - u2[k]) ** 2 + (j - v2[k]) ** 2)
            val = (1 / (1 + (d0 / D1) ** (2*n))) * (1 / (1 + (d0 / D2) ** (2*n)))
            prod *= val
        butter_filter[i, j] = prod

plt.imshow(butter_filter, cmap='gray')
plt.title('Notch Filer')
plt.show()

# dft_shift[:, :, 0] = dft_shift[:, :, 0] * butter_filter
dft_shift = dft_shift * butter_filter

f_ishift = np.fft.ifftshift(dft_shift)
img_back = cv2.idft(f_ishift)
img_back1 = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

plt.imshow(img_back1, cmap='gray')
plt.title('Output')
plt.show()

duv =  np.sqrt( (u[k] - i)**2 + (v[k] - j)**2 )
dmuv = np.sqrt( (img.shape[0] - u[k] - i)**2 + (img.shape[1] - v[k] - j)**2 )

def notch_reject_filter(shape, d0=9, u_k=0, v_k=0):
    P, Q = shape
    # Initialize filter with zeros
    H = np.ones((P, Q))
    
    uk = P-u_k
    vk = Q-v_k
    # Traverse through filter
    for u in range(0, P):
        for v in range(0, Q):
        
            D1 = np.sqrt((u - u_k) ** 2 + (v - v_k) ** 2)
            D2 = np.sqrt((u - uk) ** 2 + (v - vk) ** 2)
           
            if D1 < d0 or D2 < d0:
                H[u, v] = (1 / (1 + (d0 / D1) ** (2*n))) * (1 / (1 + (d0 / D2) ** (2*n)))
                
    return H

u = [350, 250, 180, 310]
v = [310, 470, 400, 400]

H = 1
for i in range(len(u)):
  tmp = notch_reject_filter(img.shape,25,u[i],v[i])
  H *= tmp
H = 255 - H
plt.imshow(H,cmap='gray')

#plt.imshow(H1,cmap='gray')
plt.imshow(H,cmap='gray')

