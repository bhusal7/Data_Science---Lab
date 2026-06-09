              # Contours & Shape Detection
              
        # 1. finding and drawing contours 
        # contours, hierarchy = cv2.findContours(img, mode, method)                  (img=binary_img)
        # binary-IMG = black and white images,       

import cv2

img = cv2.imread('C:/Users/Acer/OneDrive/Desktop/Pradeep/Data Science/OpenCV/OIP (1).webp')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray,210,255,cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

           # Then, save or draw that img
        # cv2.drawContours(img, counter, contourIdx=,color,thickness)

draw_contours = cv2.drawContours(img, contours , -1 , (0,255,0) , 3)   

cv2.imshow('contours', draw_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()
