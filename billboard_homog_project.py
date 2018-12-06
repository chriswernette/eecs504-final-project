import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import eta.core.image as etai

def billboard_homog_project(pts1,maskedImg):
    #pts1 is a numpy array, 4x2 which contains coordinates for bounding of image mask
    # image mask is is mask for image where contains billboard, else 0,0,0
     
    # points from mask which bound billboard
    #pts1 = np.array([[139,157],[142,213],[328,128],[319,39]])
    #maskedImg = np.load('maskedImg2.npy')
    #Predefined cooredinates we wish to project bilboard onto 
    #moe car driver perspect
    pts2 = np.array([[1460,1000],[1460,1618],[2658,1618],[2658,1000]])
    carimg = etai.read('data/IMG_20181201_151043.jpg')
    #moe GM car
    # pts2 = np.array([[327,80],[327,150],[472,150],[472,80]])
    # carimg = etai.read('data/GM-CruiseAV-800x534.jpg')
    #Find homography from original bilboard image to HUD
    H,_ = cv2.findHomography(pts1, pts2)
    #image mask in the HUD perpective
    maskedImg_perspHUD = cv2.warpPerspective(maskedImg, H,(carimg.shape[1],carimg.shape[0]))
    added_image = cv2.addWeighted(carimg,1,maskedImg_perspHUD,1,0)
  
    plt.imshow(added_image)    
    plt.show()
    return added_image

