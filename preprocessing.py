#preprocessing.py

import cv2
import numpy as np

def preprocess_data(img_location):
    '''this function will perform all the preprocessing on the input image, with
    the end output being the hough lines'''

    #read in image, blur, and grayscale
    img = cv2.imread(img_location)
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    y, x = gray.shape

    #crop the image to the upper right quadrant, add inputs
    gray_cropped = gray[0:np.floor(y/2).astype(int),np.floor(x/2).astype(int):]
    img_cropped = img[0:np.floor(y/2).astype(int),np.floor(x/2).astype(int):]

    '''get edges from Canny, try separating by color channel and getting edges 
    on those, or maybe try a different edge detector like Harris corners?'''
    edges = cv2.Canny(gray_cropped, 150, 150)

    #get Hough lines from edges image
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=35, minLineLength=100)

    return gray_cropped, img_cropped, edges, lines
