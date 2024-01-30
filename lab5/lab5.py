import cv2
import numpy as np
import matplotlib.pyplot as plt
#%%
x = 122 
y = 219

img = cv2.imread('C:/Users/Dhar_7/OneDrive/Desktop/Image/Lab/cse 4128/5.2/th_img2.jpg', 0)
img = img*1.0
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if(img[i][j] > 200.0):
            img[i][j] = 1.0
        else:
            img[i][j] = 0.0
img2 = np.zeros(img.shape)
img2[x,y] = 1.0
s = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
count = 0
res = img
plt.imshow(img,cmap='gray')
#%%
res = cv2.morphologyEx(img,cv2.MORPH_OPEN,s)
plt.imshow(res,cmap='gray')
#%%

while(1):   
    img2 = cv2.dilate(img2,s,1)
    comp = 1 - img
    r = cv2.bitwise_and(comp,img2)
    c = np.array_equal(img2, r)
    if (c==1):
        break
    else:
        img2 = r
    """c = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (r[i][j] != img2[i][j] ):
                img2 = r
                c = 1
                break
    if (c==0):
        break            
    """   
#r = 255 - r
#img = cv2.bitwise_or(img,r)
#plt.imshow(img2,cmap='gray')
#%%
img = cv2.bitwise_or(img,img2)
plt.imshow(img,cmap='gray')
#%%
for i in range(img2.shape[0]):
    for j in range(img2.shape[1]):
        if(img2[i][j] > 0.0):
            count += 1
            
#%%

for i in range(100):
    img2 = cv2.dilate(img2,s,1)
    comp = 255.0 - img
    img2 = cv2.bitwise_and(comp,img2)
    plt.imshow(img2,cmap='gray')
#%%
plt.imshow(img,cmap='gray')  
#%%
comp = 255.0 - img
plt.imshow(comp,cmap='gray')
#%%

#%%
img = img*1.0
i = cv2.bitwise_and(img,img2)
plt.imshow(i,cmap='gray')
#%%

def click_event(event, x, y, flags, params):
 
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        #i , j = x , y
        #work(x,y)
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
 

   
img = cv2.imread('C:/Users/Dhar_7/OneDrive/Desktop/Image/Lab/cse 4128/5.2/th_img2.jpg', 0)
cv2.imshow('image', img)   
cv2.setMouseCallback('image', click_event)  
cv2.waitKey(0) 
cv2.destroyAllWindows()

#%%

































