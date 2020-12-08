# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 01:05:15 2020

@author: Adithia Jo
"""


import cv2   
import glob 

## import 1 gambar

img = cv2.imread("Goku.jpg")
cv2.imshow("Goku1",img)  ## Display
cv2.imwrite("goku1.jpg",img) ## save


## import gambar 1 folder
filenames = glob.glob("Images/*.jpg")
idx=0
for filename in filenames:
    img = cv2.imread(filename)
    
    if idx == len(filenames):
        break;
    cv2.imshow(filename,img)
    idx += 1
