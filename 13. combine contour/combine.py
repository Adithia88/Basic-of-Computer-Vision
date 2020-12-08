# -*- coding: utf-8 -*-
"""
Created on Sun May 10 23:34:13 2020

@author: Adithia Jo
"""

import numpy as np
import cv2 as cv
import os

def main():
    Mask_Source_Folder = "Source_Images"
    Image_Source_Folder = "Target_Images"
    Output_Folder = "Result"
    
    Mask_Files = os.listdir(Mask_Source_Folder) #get all the files from that folder
    Image_Files = os.listdir(Image_Source_Folder) #get all the files from that folder
    
    for Mask in Mask_Files: #Process for each file in the folder
        Msk_Img = cv.imread(Mask_Source_Folder + "/" + Mask,0)
        ret, thresh = cv.threshold(Msk_Img, 127, 255, 0)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        
        Img = cv.imread(Image_Source_Folder + "/" + Mask) #for this Mask and Image file name must be same.
        cv.drawContours(Img, contours, -1, (0, 0, 255), 2) #-1 means draw all contours, red color, 2 is the width of contour line
        
        if os.path.isdir(Output_Folder) is False:
            os.mkdir(Output_Folder)
        
        cv.imwrite(Output_Folder + "/" + Mask, Img)
    
   # input("Please Enter to Continue...")

if __name__ == '__main__':
    main()