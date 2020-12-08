# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 23:40:32 2020

@author: Adithia Jo
"""
# import liblary

import cv2

## crop 1 gambar
img = cv2.imread("goku.jpg")
imgcrop = img[200:1128,200:800]
cv2.imshow("crop",imgcrop)