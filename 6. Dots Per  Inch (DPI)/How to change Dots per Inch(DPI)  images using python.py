# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 11:02:44 2020

@author: Adithia Jo
"""
## import liblary

from PIL import Image
import glob
import os

## change 1 pict

im = Image.open("test.jpg")
im.save("test720.jpg",dpi=(720,720))

## change 1 folder DPI

root_path_test = "before_DPI/*.jpg"
root_path_saving_DPI = "after_DPI/"

file_path_test = glob.glob(root_path_test)

#make new directory
os.makedirs(root_path_saving_DPI, exist_ok = True)


## change DPI
for path in file_path_test:
    image= Image.open(path)
    filename = path.split('\\')[-1]
    image.save(root_path_saving_DPI + filename, dpi=(720,720) )
