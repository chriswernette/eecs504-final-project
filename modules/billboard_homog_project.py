import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import eta.core.image as etai

def billboard_homog_project(pts1,maskedImg,background,driverside_eye_vec,passenger_eye_vec,center_eye_vec):
    #pts1 is a numpy array, 4x2 which contains coordinates for bounding of image mask
    
    # maskedImg is is mask for image where contains billboard, else 0,0,0
    
    # background is the choice of scene, driver, passenger, center, 0,1,or 2 respectively
    
    # 3 last args is a np array x,y location for which driver is looking, center location for all 3 images points of view
    
    # points from mask which bound billboard
    #pts1 = np.array([[139,157],[142,213],[328,128],[319,39]])
    #maskedImg = np.load('maskedImg2.npy')
    #Predefined cooredinates we wish to project bilboard onto 
    #moe car driver perspect
    
    xd = driverside_eye_vec[0]
    yd = driverside_eye_vec[1]
    xp = passenger_eye_vec[0]
    yp = passenger_eye_vec[1]
    xc = center_eye_vec[0]
    yc = center_eye_vec[1]
    h = 400
    w = 1200
    
    if (background == 1):
        pts2 = np.array([[np.floor(xp-w/2),np.floor(yp-h/2)],[np.floor(xp-w/2),np.floor(yp+h/2)],[np.floor(xp+w/2),np.floor(yp+h/2)],[np.floor(xp+w/2),np.floor(yp-h/2)]])
        carimg = etai.read('data/passenger.jpg')    
    elif (background == 2):
        pts2 = np.array([[np.floor(xc-w/2),np.floor(yc-h/2)],[np.floor(xc-w/2),np.floor(yc+h/2)],[np.floor(xc+w/2),np.floor(yc+h/2)],[np.floor(xc+w/2),np.floor(yc-h/2)]])
        carimg = etai.read('data/center.jpg')
    elif (background != 1 and background !=2):
        pts2 = np.array([[np.floor(xd-w/2),np.floor(yd-h/2)],[np.floor(xd-w/2),np.floor(yd+h/2)],[np.floor(xd+w/2),np.floor(yd+h/2)],[np.floor(xd+w/2),np.floor(yd-h/2)]])
        carimg = etai.read('data/driverside.jpg')
    #moe GM car
    # pts2 = np.array([[327,80],[327,150],[472,150],[472,80]])
    # carimg = etai.read('data/GM-CruiseAV-800x534.jpg')
    #Find homography from original bilboard image to HUD
    H,_ = cv2.findHomography(pts1, pts2)
    #image mask in the HUD perpective
    maskedImg_perspHUD = cv2.warpPerspective(maskedImg, H,(carimg.shape[1],carimg.shape[0]))
    added_image = cv2.addWeighted(carimg,.6,maskedImg_perspHUD,1,0)
  
    plt.imshow(added_image)    
    plt.show()
    return added_image
