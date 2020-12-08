# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 01:58:21 2020

@author: Adithia Jo
"""

## RESIZE 1 GAMBAR

import cv2 
img = cv2.imread("goku.jpg")
img = cv2.resize(img, (480,480))
cv2.imshow("goku",img)

## resize 1 folder

import cv2
import glob
import os

## path yang digunakan
root_saving = "resize/" ## untuk disimpan
root_testing = ("Images/*.jpg") ## untuk folder yang akan diresize
file_path = glob.glob(root_testing)

## membuat folder / directory
os.makedirs(root_saving,exist_ok = True)

for path in file_path:
    img = cv2.imread(path)
    resize_img = cv2.resize(img,(480,480))
    
    ##save
    filename = path.split("\\")[-1]
    cv2.imwrite(root_saving + filename , resize_img)

