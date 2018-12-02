import cv2
import numpy as np

def street_sign_mask(img):
    #Convert img to HSV colour
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #Create masks for traffic signs - Official RGB Defined below
    #Green - 0,153, 0; Blue - 0,0,255; 
    #Orange - 255,102,0; Yellow - 255,204,0
    #Define range for each color
    green_lower_limit = np.array([85, 255, 100]) #RGB - 0/100/0
    green_upper_limit = np.array([85,255, 180]) #RGB - 0/180/0
    blue_lower_limit = np.array([167, 255, 255]) #RGB - 0/20/255
    blue_upper_limit = np.array([170, 255, 150]) #RGB - 0/0/150
    orange_lower_limit = np.array([25,255,255]) #RGB - 255/150/0
    orange_upper_limit = np.array([11, 255, 255]) #RGB - 255/65/0
    yellow_lower_limit = np.array([18,94,140])
    yellow_upper_limit = np.array([58,255,255])

    mask_green = cv2.inRange(hsv_img, green_lower_limit, green_upper_limit)
    mask_blue = cv2.inRange(hsv_img, blue_lower_limit, blue_upper_limit)
    mask_orange = cv2.inRange(hsv_img, orange_lower_limit, orange_upper_limit)
    mask_yellow = cv2.inRange(hsv_img, yellow_lower_limit, yellow_upper_limit)

    return mask_green, mask_blue, mask_orange, mask_yellow