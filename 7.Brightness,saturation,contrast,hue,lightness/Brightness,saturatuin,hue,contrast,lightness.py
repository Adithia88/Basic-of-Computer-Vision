# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 11:36:34 2020

@author: Adithia Jo
"""


## import liblary

import glob 
import os
from layeris.layer_image import LayerImage

## brightness,contrast,graysclae etc
image = LayerImage.from_file("goku.jpg")
#image.grayscale()
#image.brightness(1)
#image.contrast(1)
#image.hue(1)
#image.saturation(-0.5)
image.lightness(-0.8)
image.save("goku_output.jpg",100)


## brightness,contrast,gray,etc 1 folder
## path yang digunakan 
root_path_testing = 'Before/*.jpg' ## di path mana gambar akan di blend
root_path_saving = 'After/'  ## dimana gambar akan disave
file_path = glob.glob(root_path_testing)

#make new directory
os.makedirs(root_path_saving, exist_ok = True)

for path in file_path:
    image = LayerImage.from_file(path)
    image.grayscale()
    image.brightness(1)
    image.contrast(1)
    image.hue(0.2)
    image.saturation(-0.5)
    image.lightness(-0.8)
    # split filename
    filename = path.split('\\')[-1]
    image.save(root_path_saving + filename,100) 
