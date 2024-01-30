# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 03:19:29 2022

@author: ASUS
"""

import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# to plot histogram
def plot_histogram(hist_array, fig_title):
    plt.hist(hist_array.ravel(),L,[0,L-1])
    plt.title(fig_title)
    plt.show()

# to plot array
def plot_data(data_array, fig_title): 
    plt.plot(data_array)
    plt.title(fig_title)
    plt.show()
    
# to plot image
def plot_image(data_array, fig_title): 
    plt.imshow(data_array, 'gray')
    plt.title(fig_title)
    plt.show()

def histogramEqualizeErlang(erl):
    
    """
        input list containing erlang values
        returns s -> the equalized list
    """
    inpmat = erl
            
    totalnumofpixels = np.sum(inpmat)
    #list for CDF
    #probablity and output matching list    
    probablity = [0 for i in range(256)]
    s = [0 for i in range(256)]
    cuminpmat = [0 for i in range(256)]
    cuminpmat[0] = inpmat[0]
    
    probablity[0] = (cuminpmat[0] / totalnumofpixels) * 255
    s[0] = round(probablity[0])
    
    #calculating and storing CDF
    #calculating probablity
    #storing the rounding result into output intensity mapping list
    for i in range(1, 256):
        cuminpmat[i] = inpmat[i] + cuminpmat[i-1]
        
        #probablity[i] = (cuminpmat[i] / totalnumofpixels) * 255
        probablity[i] = (cuminpmat[i]) * 255
        s[i] = round(probablity[i])
        
    plot_data(inpmat, "Histogram Equalization of Erlang Distribution")
    
    return s

def histogramEqualizeImage(img):
    
    """
        Takes an image matrix as paramter
        Returns a tuple -> (histogram matrix, equalized matrix)
    """
    
    out = np.zeros([img.shape[0], img.shape[1]], "uint8")
    
    h = img.shape[0]
    w = img.shape[1]
    totalnumofpixels = h * w
    
    #input matrix where index is intensity and value is frequency
    inpmat = [0 for i in range(256)]
    #print(len(inpmat))
    
    #creating input image histogram
    for i in range(h):
        for j in range(w):
            intensity = img.item(i,j)
            inpmat[intensity] = inpmat[intensity] + 1
    
    #plotting input image histogram

    plot_histogram(img, "Input Image Histogram")
            
    #print(inpmat)
    #list for CDF
    #probablity and output matching list    
    probablity = [0 for i in range(256)]
    s = [0 for i in range(256)]
    cuminpmat = [0 for i in range(256)]
    cuminpmat[0] = inpmat[0]
    
    probablity[0] = (cuminpmat[0] / totalnumofpixels) * 255
    s[0] = round(probablity[0])
    
    #calculating and storing CDF
    #calculating probablity
    #storing the rounding result into output intensity mapping list
    for i in range(1, 256):
        cuminpmat[i] = inpmat[i] + cuminpmat[i-1]
        probablity[i] = (cuminpmat[i] / totalnumofpixels) * 255
        s[i] = round(probablity[i])
    
    
    #mapping the output intensity to result value
    for i in range(h):
        for j in range(w):
            intensity = img.item(i,j)
            result = s[intensity]
            out.itemset((i,j), result)
    
    #histogram of output image        
    plot_histogram(out, "Histogram Equalized Image")
    print(out)
    
    return (inpmat, s)

def calculateErlangValue(x, k, l):
    return ((l ** k) * (x ** (k-1)) * (math.e ** -(l * x))) / (math.factorial(k-1))

def calculateErlangValue_scale(x, k, miu):
    return ( (x ** (k-1)) * (math.e ** -(x / miu))) / ((miu **k) * math.factorial(k-1))

def createOutputMatchingImage(k1, l1):
    """
        returns the summation of two erlang functions
    """
    erl = [0 for i in range(256)]
    
    for i in range(256):
        erl[i] = calculateErlangValue(i, k1, l1)
        #erl[i] = calculateErlangValue_scale(i, k1, l1)

    print("Erlang", erl)
    
    plot_data(erl, "Erlang Distribution")
    
    return erl

def returnClosestIndex(sout, matchedInput):
    """
        returns closest index value
    """
    for i in range(1, 255):
        if matchedInput == sout[i]:
            return i
        elif matchedInput < sout[i]:
            if abs(matchedInput - sout[i]) < abs(matchedInput - sout[i-1]):
                return i
            else:
                return i-1
        
        
def performMatchingOperation(img, sinp, sout):
    
    """
    returns input image matched to the output image"
    """
    out = np.zeros([img.shape[0], img.shape[1]], "uint8")
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            intensity = img.item(i,j)
            matchedInput = sinp[intensity]
            #print(matchedInput)
            
            outputIndex = returnClosestIndex(sout, matchedInput)
                
                
            #print(intensity, matchedInput, outputIndex , sout[outputIndex])
            
            out.itemset((i,j), outputIndex)
    
    return out


L = 256


img = cv.imread("lena.jpg", 0)
#cv.imshow("input image", img)
erl = createOutputMatchingImage(115, 0.8)
#erl = createOutputMatchingImage(115, 0.8)

(histin, sin) = histogramEqualizeImage(img)
sout = histogramEqualizeErlang(erl)

print(sout)

out = performMatchingOperation(img, sin, sout)
print(out)


plot_image(img, "Input Image")
plot_image(out, "Histogram Matched Image")

plot_histogram(out, "Output Image Histogram")
