#preprocessing.py

import cv2
import numpy as np

def preprocess_data(img, 
                    crop = 1, crop_x = np.array([400,520]), crop_y = np.array([1000,1300]), 
                    canny_auto_tune = 1, canny_min = 125, canny_max = 150, 
                    hough_thresh = 30, hough_min_ll = 100, hough_max_gap = 35):
    '''this function will perform all the preprocessing on the input image, with
    the end output being the hough lines'''

    #read in image, blur, and grayscale
    img = cv2.imread(img)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    y, x = gray.shape

    #crop image if requested
    if(crop):
        gray_cropped = gray[crop_x[0]:crop_x[1],crop_y[0]:crop_y[1]]
        img_cropped = img[crop_x[0]:crop_x[1],crop_y[0]:crop_y[1]]
    else:
        gray_cropped = gray
        img_cropped = img
    
    #trying adaptive tuning, can turn on or off
    if(canny_auto_tune):
        sigma = .33
        v = np.median(gray_cropped)
        canny_min = int(max(0, (1.0 - sigma) * v))
        canny_max = int(min(255, (1.0 + sigma) * v))
    
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
