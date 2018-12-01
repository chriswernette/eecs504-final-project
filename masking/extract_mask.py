import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import eta.core.image as etai


def main():
    # Read in rectangle image    
    img = etai.read('rectangle_billboard.jpg')
    plt.figure()
    plt.imshow(img)
    plt.show()

    # Create rectangular mask
    mask = np.zeros(img.shape, dtype="uint8")
    # Coordinates correspond to 2 opposite corners of the rectangle
    cv2.rectangle(mask, (132,81), (432,173), (255,255,255), -1)
    # Combines the mask + image
    maskedImg = cv2.bitwise_and(img, mask)
    np.save('maskedImg1.npy',maskedImg)
    #Display Results
    plt.figure()
    plt.imshow(maskedImg)
    plt.show()


    # Repeat example, generalizing to all polygons    
    img = etai.read('polygon_billboard.jpg')
    plt.figure()
    plt.imshow(img)
    plt.show()
    

    mask = np.zeros(img.shape, dtype="uint8")
    # pts represents 4 corners of polygon (do not need to be rectangular)
    pts = np.array([[139,157],[142,213],[328,128],[319,39]], dtype='int32')
    # [pts] is an array of polygons, so we can make multiple masked regions per image
    cv2.fillPoly(mask, [pts], (255,255,255))
    maskedImg = cv2.bitwise_and(img, mask)
    np.save('maskedImg2.npy',maskedImg)
    plt.figure()
    plt.imshow(maskedImg)
    plt.show()


if(__name__=="__main__"):
    main()
