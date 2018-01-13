import numpy as np
import cv2 as cv


# --------------------------main-------------------------------------------

#----------------------------read and show image------------------------
cv.namedWindow('origin', cv.WINDOW_AUTOSIZE)
cv.namedWindow('defect', cv.WINDOW_AUTOSIZE)
cv.moveWindow('origin', 100, 100)
cv.moveWindow('defect', 500, 100)

img = cv.imread('./images/1.png')
cv.imshow('origin', img)

roi_img = img[50:150, 50:100]
cv.imshow("defect", roi_img)

cv.waitKey()
