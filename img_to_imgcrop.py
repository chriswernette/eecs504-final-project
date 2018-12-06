import cv2
import numpy as np
import matplotlib.pyplot as plt
import os 

#help function which crops images and saves the new image in same directory
img_location = 'data/Test_Images/_020_first_15/_021_first_15/frame12.jpg'
x1 = 20
x2 = 500
y1 = 40
y2 = 700

img = cv2.imread(img_location)
imgcrop = img[x1:x2,y1:y2]
newname = 'cropped_'+ os.path.basename(img_location)
newnewname = os.path.dirname(img_location) + '/' + newname
#saves to directory
cv2.imwrite(newnewname,imgcrop)
