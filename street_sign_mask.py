import cv2
import numpy as np
#img should be RGB
def street_sign_mask(img):
    #Convert img to HSV colour
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    #Create masks for traffic signs - Official RGB Defined below
    #Green - 0,153, 0; Blue - 0,0,255; 
    #Orange - 255,102,0; Yellow - 255,204,0
    #Define range for each color
    green_lower_limit = np.array([0, 200, 80]) 
    green_upper_limit = np.array([100,255, 120]) 
    blue_lower_limit = np.array([100, 150, 100]) 
    blue_upper_limit = np.array([157, 255, 255]) 
    orange_lower_limit = np.array([0,150,230])
    orange_upper_limit = np.array([15, 255, 255])
    yellow_lower_limit = np.array([0,150,140])
    yellow_upper_limit = np.array([100,255,255])

    mask_green = cv2.inRange(hsv_img, green_lower_limit, green_upper_limit)
    mask_blue = cv2.inRange(hsv_img, blue_lower_limit, blue_upper_limit)
    mask_orange = cv2.inRange(hsv_img, orange_lower_limit, orange_upper_limit)
    mask_yellow = cv2.inRange(hsv_img, yellow_lower_limit, yellow_upper_limit)

    return mask_green, mask_blue, mask_orange, mask_yellow