import cv2 as cv
import numpy as np

img = cv.imread("images/blueCar.jpg")

h, w = img.shape[:2]
scale = 480/h

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.resize(img,(int(w*scale), int(h*scale)))

sift = cv.SIFT_create()
sift_keypoints, sift_descriptors = sift.detectAndCompute(img, None)

SIFT_featureImg = cv.drawKeypoints(img, sift_keypoints, None)



cv.imshow("Blue Car", img)
cv.imshow("SIFT Image", SIFT_featureImg)

cv.waitKey(0)
cv.destroyAllWindows()