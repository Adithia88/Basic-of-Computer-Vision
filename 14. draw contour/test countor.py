# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:00:27 2020

@author: Adithia Jo
"""

## countour 1 gmabar
import os
import cv2
import glob

img = cv2.imread("ASD_Ori_17.jpg")
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

img = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
imgray= cv2.drawContours(imgray, contours, -1, (0, 255, 0), 3)

cv2.imshow('Image', img)
cv2.imshow('Image GRAY', imgray)

## countour 1 folder

import cv2
import glob
import os

## path yang digunakan
root_saving = "after_countour/" ## untuk disimpan
root_testing = ("image/*.jpg") ## untuk folder yang akan diresize
file_path = glob.glob(root_testing)

## membuat folder / directory
os.makedirs(root_saving,exist_ok = True)

for path in file_path:
    
    img = cv2.imread(path)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    img = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    imgray= cv2.drawContours(imgray, contours, -1, (0, 255, 0), 3)
    
    ##save
    filename = path.split("\\")[-1]
    cv2.imwrite(root_saving + filename , img)