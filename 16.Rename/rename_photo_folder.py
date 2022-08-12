## rename 1 folder

import cv2
import glob
import os

## path yang digunakan
root_saving = "DINA_NOPITA_SARI_Rename/" ## untuk disimpan
root_testing = ("DINA_NOPITA_SARI/*.jpg") ## untuk folder yang akan diresize
file_path = glob.glob(root_testing)

namafile = "DINA_NOPITA_SARI"

## membuat folder / directory
os.makedirs(root_saving,exist_ok = True)

currentFrame = 0

for path in file_path:
    img = cv2.imread(path)
    resize_img = cv2.resize(img,(480,480))
    
    if (currentFrame<10) :
    
        ##save
        name = namafile +str("000")+ str(currentFrame) + '.jpg'
        currentFrame = currentFrame +1 
        cv2.imwrite(root_saving + name , resize_img)
    
    elif(currentFrame>9 and currentFrame<100):
        ##save
        name = namafile +str("00")+ str(currentFrame) + '.jpg'
        currentFrame = currentFrame +1 
        cv2.imwrite(root_saving + name , resize_img)
    
 