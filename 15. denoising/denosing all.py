# -*- coding: utf-8 -*-
"""
Created on Fri May 22 23:13:20 2020

@author: Adithia Jo
"""

## brightness and contrast
#import liblary
import cv2
import glob
import numpy as np
import os 
from layeris.layer_image import LayerImage

## path yang digunakan 
root_path_testing = 'data/*.jpg' ## di path mana gambar akan di blend
root_path_saving = 'Brightness Frame/'  ## dimana gambar akan disave
file_path = glob.glob(root_path_testing)

#make new directory
os.makedirs(root_path_saving, exist_ok = True)

for path in file_path:
    image = LayerImage.from_file(path)
    image.grayscale()
    image.brightness(0.1)
    #image.lightness(-0.1)
    #image.saturation(2)
    #image.hue(2)
    image.contrast(4)
    # split filename
    filename = path.split('\\')[-1]
    image.save(root_path_saving + filename,100)
    
    
## denoising  filter 2D

import cv2
import numpy as np
from matplotlib import pyplot as plt


## path yang digunakan 
root_path_testing = 'normal1/*.jpg' ## di path mana gambar akan di blend
root_path_saving = 'filter2d/'  ## dimana gambar akan disave
file_path = glob.glob(root_path_testing)

#make new directory
os.makedirs(root_path_saving, exist_ok = True)

for path in file_path:
    img=cv2.imread(path)
    kernel = np.ones((3,3),np.float32)/9
    filt_2D = cv2.filter2D(img, - 1,kernel)
    # split filename
    filename = path.split('\\')[-1]
    cv2.imwrite(root_path_saving + filename,filt_2D)


## denoising  Blur

import cv2
import numpy as np
from matplotlib import pyplot as plt


## path yang digunakan 
root_path_testing = 'normal1/*.jpg' ## di path mana gambar akan di blend
root_path_saving = 'blurnormal1/'  ## dimana gambar akan disave
file_path = glob.glob(root_path_testing)

#make new directory
os.makedirs(root_path_saving, exist_ok = True)

for path in file_path:
    img=cv2.imread(path)
    kernel = np.ones((3,3),np.float32)/9
    blur = cv2.blur(img,(3,3))
    # split filename
    filename = path.split('\\')[-1]
    cv2.imwrite(root_path_saving + filename,blur)

## denoising Gausian Blur

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as nd


## path yang digunakan 
root_path_testing = 'normal1/*.jpg' ## di path mana gambar akan di blend
root_path_saving = 'gausianblurnomral1/'  ## dimana gambar akan disave
file_path = glob.glob(root_path_testing)

#make new directory
os.makedirs(root_path_saving, exist_ok = True)

for path in file_path:
    img=cv2.imread(path)
    kernel = np.ones((3,3),np.float32)/9
    gaussian_blur = nd.gaussian_filter(img, sigma=3)
    # split filename
    filename = path.split('\\')[-1]
    cv2.imwrite(root_path_saving + filename,gaussian_blur)


## denoising median Blur

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as nd
    


## path yang digunakan 
root_path_testing = 'normal1/*.jpg' ## di path mana gambar akan di blend
root_path_saving = 'medianblurnormal1/'  ## dimana gambar akan disave
file_path = glob.glob(root_path_testing)

#make new directory
os.makedirs(root_path_saving, exist_ok = True)

for path in file_path:
    img=cv2.imread(path)
    kernel = np.ones((3,3),np.float32)/9
    median_blur = nd.median_filter(img, size=3)
    # split filename
    filename = path.split('\\')[-1]
    cv2.imwrite(root_path_saving + filename,median_blur)

## denoising bilateral Blur

import cv2
import numpy as np
from matplotlib import pyplot as plt


## path yang digunakan 
root_path_testing = 'data/*.jpg' ## di path mana gambar akan di blend
root_path_saving = 'bilateral/'  ## dimana gambar akan disave
file_path = glob.glob(root_path_testing)

#make new directory
os.makedirs(root_path_saving, exist_ok = True)

for path in file_path:
    img=cv2.imread(path)
    kernel = np.ones((3,3),np.float32)/9
    bilateral_blur = cv2.bilateralFilter(img,9,75,75)
    # split filename
    filename = path.split('\\')[-1]
    cv2.imwrite(root_path_saving + filename,bilateral_blur)


## blending photoshop

#import liblary
import cv2
import glob
import numpy as np
import os 
from layeris.layer_image import LayerImage

## path yang digunakan 
root_path_testing = 'Brightness Frame/*.jpg' ## di path mana gambar akan di blend

## mode blend
blending=True   ## penting isikan mode
"""


"""
if blending == True: ##grayscale
    root_path_saving = 'grayscale/'  ## dimana gambar akan disave
    file_path = glob.glob(root_path_testing)
    #make new directory
    os.makedirs(root_path_saving, exist_ok = True)
    for path in file_path:
        image = LayerImage.from_file(path)
        image.grayscale()
        # split filename
        filename = path.split('\\')[-1]
        image.save(root_path_saving + filename,100)

if blending == True: ## darken
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

if blending == True: ## Multiply
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

if blending == True: ## Color Burn
    root_path_saving = 'Color Burn ABNORMAL/'  ## dimana gambar akan disave
    file_path = glob.glob(root_path_testing)
    #make new directory
    os.makedirs(root_path_saving, exist_ok = True)
    for path in file_path:
        image = LayerImage.from_file(path)
        image.color_burn("#7fe3f8")
        # split filename
        filename = path.split('\\')[-1]
        image.save(root_path_saving + filename,100)

if blending == True: ## linear burn
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

if blending == True: ## lighten
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

if blending == True: ## screen
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

if blending == True: ## color dodge
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

if blending == True: ## linear dodge
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

if blending == True: ## overlay
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

if blending == True: ## Soft light
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

if blending == True: ## hard light
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

if blending == True: ## vivid light
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

if blending == True: ## linear light
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

if blending == True: ## pin light
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


## denoising with fastNLmean

import numpy as np
import cv2 
from matplotlib import pyplot as plt

## path yang digunakan 
root_path_testing = 'data/*.jpg' ## di path mana gambar akan di blend
root_path_saving = 'fastnlmean/'  ## dimana gambar akan disave
file_path = glob.glob(root_path_testing)

#make new directory
os.makedirs(root_path_saving, exist_ok = True)

for path in file_path:
    img=cv2.imread(path)
    dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    # split filename
    filename = path.split('\\')[-1]
    cv2.imwrite(root_path_saving + filename,dst)


## Denoising filters 
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_ubyte, img_as_float
from matplotlib import pyplot as plt
from skimage import io
import numpy as np
import glob,os


## path yang digunakan 
root_path_testing = 'data/*.jpg' ## di path mana gambar akan di blend
root_path_saving = 'NLM/'  ## dimana gambar akan disave
file_path = glob.glob(root_path_testing)

#make new directory
os.makedirs(root_path_saving, exist_ok = True)

for path in file_path:
    img = img_as_float(io.imread(path))
    img=cv2.imread(path)
    sigma_est = np.mean(estimate_sigma(img, multichannel=True))
    patch_kw = dict(patch_size=5,      
                patch_distance=3,  
                multichannel=True)
    denoise_img = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=False,
                               patch_size=5, patch_distance=3, multichannel=True)
    # split filename
    filename = path.split('\\')[-1]
    cv2.imwrite(root_path_saving + filename,denoise_img)


## clahe
#import liblary
import cv2
import glob
import numpy as np
import os

#path yang akan digunakan
root_path_testing = 'normal1/*.jpg'
root_path_saving = 'Clahe/'
file_path = glob.glob(root_path_testing)

#make new directory
os.makedirs(root_path_saving, exist_ok = True)

#image enchantment using clahe
clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
for path in file_path:
    image = cv2.imread(path, 0)
    image_equ = clahe.apply(image)
    
    # split filename
    filename = path.split('\\')[-1]
    cv2.imwrite(root_path_saving + filename, image_equ)


## equalization 

import cv2
import glob
import os

## path yang digunakan
root_saving = "equalization/" ## untuk disimpan
root_testing = ("normal1/*.jpg") ## untuk folder yang akan diresize
file_path = glob.glob(root_testing)

## membuat folder / directory
os.makedirs(root_saving,exist_ok = True)

for path in file_path:
    img = cv2.imread(path,0) ## grayscale 
    equ = cv2.equalizeHist(img)
    
    ##save
    filename = path.split("\\")[-1]
    cv2.imwrite(root_saving + filename ,equ)

"""
## denoising MRI hitung PSNR
#Gaussian
from skimage import img_as_float
from skimage.metrics import peak_signal_noise_ratio
from matplotlib import pyplot as plt
from skimage import io
from scipy import ndimage as nd

noisy_img = img_as_float(io.imread("data/frame0.jpg"))
#Need to convert to float as we will be doing math on the array
#Also, most skimage functions need float numbers
ref_img = img_as_float(io.imread("Brightness Frame/frame0.jpg"))
                    
gaussian_img = nd.gaussian_filter(noisy_img, sigma=5)


noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
gaussian_cleaned_psnr = peak_signal_noise_ratio(ref_img, gaussian_img)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", gaussian_cleaned_psnr)

"""

## denosing MRI   TV SMOOTH
from skimage import img_as_float
from skimage.metrics import peak_signal_noise_ratio
from matplotlib import pyplot as plt
from skimage import io
from scipy import ndimage as nd
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
import os ,glob,cv2

## path yang digunakan 
root_path_testing = 'data/*.jpg' ## di path mana gambar akan di blend
root_path_saving = 'TV_smooth/'  ## dimana gambar akan disave
file_path = glob.glob(root_path_testing)

#make new directory
os.makedirs(root_path_saving, exist_ok = True)

for path in file_path:
    img = img_as_float(io.imread(path))
    denoise_TV = denoise_tv_chambolle(img, weight=0.3, multichannel=False)
    # split filename
    filename = path.split('\\')[-1]
    plt.imsave(root_path_saving + filename,denoise_TV)
    

"""
## wavelet smoothed 
from skimage import img_as_float
from skimage.metrics import peak_signal_noise_ratio
from matplotlib import pyplot as plt
from skimage import io
from scipy import ndimage as nd
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)

noisy_img = img_as_float(io.imread("data/frame0.jpg"))



wavelet_smoothed = denoise_wavelet(noisy_img, multichannel=False,
                           method='BayesShrink', mode='soft',
                           rescale_sigma=True)

plt.imsave("wavelet_smoothing.jpg", wavelet_smoothed, cmap='gray')




## path yang digunakan 
root_path_testing = 'data/*.jpg' ## di path mana gambar akan di blend
root_path_saving = 'wavelet/'  ## dimana gambar akan disave
file_path = glob.glob(root_path_testing)

#make new directory
os.makedirs(root_path_saving, exist_ok = True)

for path in file_path:
    img = img_as_float(io.imread(path))
    wavelet_smoothed = denoise_wavelet(img, multichannel=False,
                           method='BayesShrink', mode='soft',
                           rescale_sigma=True)
    # split filename
    filename = path.split('\\')[-1]
    plt.imsave(root_path_saving + filename,wavelet_smoothed)
    
"""



