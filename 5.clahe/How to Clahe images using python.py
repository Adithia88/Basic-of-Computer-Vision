# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 22:01:28 2020

@author: Adithia Jo
"""

# clahe 1 gambar
## import liblary

import cv2
import glob
import os

img = cv2.imread("equ1.jpg",0)
clahe = cv2.createCLAHE(clipLimit = 2 , tileGridSize=(8,8))
img_clahe = clahe.apply(img)
cv2.imshow("before_clahe",img)
cv2.imshow("after_clahe",img_clahe)

#Clahe 1 folder
root_path_testing = 'before_equ/*.jpg'
root_path_saving = 'Clahe Frame/'
file_path = glob.glob(root_path_testing)

#make new directory
os.makedirs(root_path_saving, exist_ok = True)

#image enchantment using clahe
clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
for path in file_path:
    image = cv2.imread(path, 0)
    image_clahe = clahe.apply(image)
    
    # split filename
    filename = path.split('\\')[-1]
    cv2.imwrite(root_path_saving + filename, image_clahe)