#preprocessing.py

import cv2
import numpy as np

def preprocess_data(cropped_img, canny_min = 125, canny_max = 150, 
                    hough_thresh = 30, hough_min_ll = 100, hough_max_gap = 35):
    '''this function will perform all the preprocessing on the input image, with
    the end output being the hough lines'''

    #read in image, blur, and grayscale
    img = cv2.imread(cropped_img)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    y, x = gray.shape

    #crop the image to the upper right quadrant, add inputs
    #gray_cropped = gray[0:np.floor(y/2).astype(int),np.floor(x/2).astype(int):]
    #img_cropped = img[0:np.floor(y/2).astype(int),np.floor(x/2).astype(int):]
    gray_cropped = gray[400:520,1000:1300]
    img_cropped = img[400:520,1000:1300]
    '''get edges from Canny, try separating by color channel and getting edges 
    on those, or maybe try a different edge detector like Harris corners?'''
    edges = cv2.Canny(gray_cropped, canny_min, canny_max)

    #get Hough lines from edges image
    '''don't touch first 3 arguments, fourth arg is threshold, the number of votes
    needed to be considered a line (uses binning), fifth is max line gap, if two
    lines are <35 pixels apart and approximately the same angle it will combine 
    them. Last argument is minLineLength, if lines are shorter than this they'll
    be excluded'''
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold = hough_thresh, 
                            maxLineGap = hough_max_gap, minLineLength = hough_min_ll)

    return gray_cropped, img_cropped, edges, lines
