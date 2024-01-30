import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

# path = "ero.png"

# img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# img_H, img_W = img.shape

kernel_size = 7

m = kernel_size//2

kernel = np.zeros((7,7), dtype = "uint8")

for i in range(-m, m+1):
    for j in range(-m, m+1):
        if np.abs(i) + np.abs(j) == 3:
           kernel[i+m, j+m] = 1         
plt.imshow(kernel, "hot");plt.show()

# img = cv2.erode(img , kernel , 1)

# cv2.imshow("res" , img )
# cv2.imwrite("erosion_labtest_res.png", img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()