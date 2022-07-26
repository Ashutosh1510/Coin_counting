import numpy as np
import cv2
import os

CWD_PATH = os.getcwd()

#Reading the image 
img = cv2.imread('./input/coins.png',0)

#Erosion
kernel = np.ones((25,25),int)
img1 = cv2.erode(img,kernel, iterations=1)

#Dilation
kernel1 = np.ones((8,8),int)
img2 = cv2.dilate(img1,kernel1,iterations=1)

#finding the contours in the image
contours, hierarchy = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
cv2.drawContours(img, contours, -1,(255,0,0),5)

#Display of the Images
img_final = img2.copy()
cv2.putText(img_final, "Number of coins:{}".format(len(contours)), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
cv2.putText(img1,"Erosion", (60,30),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)
cv2.putText(img2,"Dilation", (60,30),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)

cv2.imwrite("./Results/Original Image.jpg", img)
cv2.imwrite("./Results/Erosion.jpg", img1)
cv2.imwrite("./Results/Dilation.jpg", img2)
cv2.imwrite("./Results/Coins Count.jpg",img_final)
cv2.waitKey(0)