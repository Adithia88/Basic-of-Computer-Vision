import cv2
import numpy as np

img = cv2.imread("unsri.png")
img = cv2.resize(img, (640, 640))
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows


img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows


th, img_th1 = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY) ## FIX THRESHOLD
th2, img_th2 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV) ## OTSU

cv2.imshow("img fix th", img_th1)
cv2.waitKey(0)
cv2.imshow("img otsu th", img_th2)
cv2.waitKey(0)




