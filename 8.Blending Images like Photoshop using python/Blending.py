# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 11:22:03 2020

@author: Adithia Jo
"""

## blending photoshop
## import liblary

import glob
import os
from layeris.layer_image import LayerImage

## blending 1 image
image = LayerImage.from_file("goku.jpg")
image.darken("3fe28f")
image.save("goku_out.jpg",100)

## blending 1 folder 

root_path_testing = "before_blending/*.jpg"

## mode blend 
blending = 4        ## masukan mode blend

if blending == 1:
    root_path_saving = "grayscale/" ## tempat menyimpan
    file_path = glob.glob(root_path_testing)
    
    os.makedirs(root_path_saving,exist_ok = True)
    for path in file_path:
        image = LayerImage.from_file(path)
        image.grayscale()
        
        filename = path.split("\\")[-1]
        image.save(root_path_saving + filename,100)

if blending == 2: ## darken
    root_path_saving = 'Darken/'  ## dimana gambar akan disave
    file_path = glob.glob(root_path_testing)
    #make new directory
    os.makedirs(root_path_saving, exist_ok = True)
    for path in file_path:
        image = LayerImage.from_file(path)
        image.darken("#3fe28f")
        # split filename
        filename = path.split('\\')[-1]
        image.save(root_path_saving + filename,100)

if blending == 3: ## Multiply
    root_path_saving = 'Multiply/'  ## dimana gambar akan disave
    file_path = glob.glob(root_path_testing)
    #make new directory
    os.makedirs(root_path_saving, exist_ok = True)
    for path in file_path:
        image = LayerImage.from_file(path)
        image.multiply("#3fe28f")
        # split filename
        filename = path.split('\\')[-1]
        image.save(root_path_saving + filename,100)

if blending == 4: ## Color Burn
    root_path_saving = 'Color Burn/'  ## dimana gambar akan disave
    file_path = glob.glob(root_path_testing)
    #make new directory
    os.makedirs(root_path_saving, exist_ok = True)
    for path in file_path:
        image = LayerImage.from_file(path)
        image.color_burn("#7fe3f8")
        # split filename
        filename = path.split('\\')[-1]
        image.save(root_path_saving + filename,100)

if blending == 5: ## linear burn
    root_path_saving = 'Linear Burn/'  ## dimana gambar akan disave
    file_path = glob.glob(root_path_testing)
    #make new directory
    os.makedirs(root_path_saving, exist_ok = True)
    for path in file_path:
        image = LayerImage.from_file(path)
        image.linear_burn("#e1a8ff")
        # split filename
        filename = path.split('\\')[-1]
        image.save(root_path_saving + filename,100)

if blending == 6: ## lighten
    root_path_saving = 'Lighten/'  ## dimana gambar akan disave
    file_path = glob.glob(root_path_testing)
    #make new directory
    os.makedirs(root_path_saving, exist_ok = True)
    for path in file_path:
        image = LayerImage.from_file(path)
        image.lighten("#ff3ce1")
        # split filename
        filename = path.split('\\')[-1]
        image.save(root_path_saving + filename,100)

if blending == 7: ## screen
    root_path_saving = 'Screen/'  ## dimana gambar akan disave
    file_path = glob.glob(root_path_testing)
    #make new directory
    os.makedirs(root_path_saving, exist_ok = True)
    for path in file_path:
        image = LayerImage.from_file(path)
        image.screen('#e633ba')
        # split filename
        filename = path.split('\\')[-1]
        image.save(root_path_saving + filename,100)

if blending == 8: ## color dodge
    root_path_saving = 'Color Dodge/'  ## dimana gambar akan disave
    file_path = glob.glob(root_path_testing)
    #make new directory
    os.makedirs(root_path_saving, exist_ok = True)
    for path in file_path:
        image = LayerImage.from_file(path)
        image.color_dodge('#490cc7')
        # split filename
        filename = path.split('\\')[-1]
        image.save(root_path_saving + filename,100)

if blending == 9: ## linear dodge
    root_path_saving = 'linear Dodge/'  ## dimana gambar akan disave
    file_path = glob.glob(root_path_testing)
    #make new directory
    os.makedirs(root_path_saving, exist_ok = True)
    for path in file_path:
        image = LayerImage.from_file(path)
        image.linear_dodge('#490cc7')
        # split filename
        filename = path.split('\\')[-1]
        image.save(root_path_saving + filename,100)

if blending == 10: ## overlay
    root_path_saving = 'Overlay/'  ## dimana gambar akan disave
    file_path = glob.glob(root_path_testing)
    #make new directory
    os.makedirs(root_path_saving, exist_ok = True)
    for path in file_path:
        image = LayerImage.from_file(path)
        image.overlay('#ffb956')
        # split filename
        filename = path.split('\\')[-1]
        image.save(root_path_saving + filename,100)

if blending == 11: ## Soft light
    root_path_saving = 'soft light/'  ## dimana gambar akan disave
    file_path = glob.glob(root_path_testing)
    #make new directory
    os.makedirs(root_path_saving, exist_ok = True)
    for path in file_path:
        image = LayerImage.from_file(path)
        image.soft_light('#ff3cbc')
        # split filename
        filename = path.split('\\')[-1]
        image.save(root_path_saving + filename,100)

if blending == 12: ## hard light
    root_path_saving = 'Hard light/'  ## dimana gambar akan disave
    file_path = glob.glob(root_path_testing)
    #make new directory
    os.makedirs(root_path_saving, exist_ok = True)
    for path in file_path:
        image = LayerImage.from_file(path)
        image.hard_light('#df5dff')
        # split filename
        filename = path.split('\\')[-1]
        image.save(root_path_saving + filename,100)

if blending == 13: ## vivid light
    root_path_saving = 'vivid light/'  ## dimana gambar akan disave
    file_path = glob.glob(root_path_testing)
    #make new directory
    os.makedirs(root_path_saving, exist_ok = True)
    for path in file_path:
        image = LayerImage.from_file(path)
        image.vivid_light('#ac5b7f')   
        # split filename
        filename = path.split('\\')[-1]
        image.save(root_path_saving + filename,100)

if blending == 14: ## linear light
    root_path_saving = 'Linear light/'  ## dimana gambar akan disave
    file_path = glob.glob(root_path_testing)
    #make new directory
    os.makedirs(root_path_saving, exist_ok = True)
    for path in file_path:
        image = LayerImage.from_file(path)
        image.linear_light('#9fa500')
        # split filename
        filename = path.split('\\')[-1]
        image.save(root_path_saving + filename,100)

if blending == 15: ## pin light
    root_path_saving = 'Pin light/'  ## dimana gambar akan disave
    file_path = glob.glob(root_path_testing)
    #make new directory
    os.makedirs(root_path_saving, exist_ok = True)
    for path in file_path:
        image = LayerImage.from_file(path)
        image.pin_light('#005546')
        # split filename
        filename = path.split('\\')[-1]
        image.save(root_path_saving + filename,100)

