# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 00:02:29 2020

@author: Adithia Jo
"""

## equalization untuk 1 gambar

import cv2

img = cv2.imread("equ1.jpg",0) ## 0 grayscale 
equ = cv2.equalizeHist(img)
cv2.imshow("before_equ",img)
cv2.imshow("after_equ",equ)

## equalization 1 folder 

import cv2
import glob
import os

## path yang digunakan
root_saving = "after_equ/" ## untuk disimpan
root_testing = ("before_equ/*.jpg") ## untuk folder yang akan diresize
file_path = glob.glob(root_testing)

## membuat folder / directory
os.makedirs(root_saving,exist_ok = True)

for path in file_path:
    img = cv2.imread(path,0) ## grayscale 
    equ = cv2.equalizeHist(img)
    
    ##save
    filename = path.split("\\")[-1]
    cv2.imwrite(root_saving + filename ,equ)