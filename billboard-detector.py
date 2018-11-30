'''script to read in image, determine edges and find hough lines, need to make
command line arguments in the future, for now change the img = cv.imread command
to change the image you're running on'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

#read in image
#img = cv2.imread("data/billboard1.jpg")
img = cv2.imread("data/real-billboard.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#blur to help with edge detection
blur = cv2.GaussianBlur(img, (5, 5), 0)
#plt.imshow(blur)
#plt.show()

'''gray scale so we can feed into Canny, in the future, investigate doing Canny on
each color channel and coming up with some 3xm*n edge matrix'''
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#crop to upper right quadrant
y,x = gray.shape
gray_cropped = gray[0:np.floor(y/2).astype(int),np.floor(x/2).astype(int):]
img_cropped = img[0:np.floor(y/2).astype(int),np.floor(x/2).astype(int):]
print(gray.shape)
print(gray_cropped.shape)
plt.imshow(gray_cropped,cmap='gray')
plt.show()
#plt.imshow(gray,cmap='gray')
#plt.show()

#get Canny edges, need to work on tuning lower and upper thresholds
edges = cv2.Canny(gray_cropped, 150, 150)
plt.imshow(edges)
plt.title = 'edges'
plt.show() 

'''get hough lines from Canny edges using probabilistic method to speed up 
results need to investigate using other edge detectors, using non-probabilistic
Hough lines function (takes longer but maybe more accurate), and tuning the 
thresholds more/finding some way to adaptively tune them'''
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=35, minLineLength=100)
print("Number of lines in image: " + str(len(lines)))

#put Hough lines on the cropped image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img_cropped, (x1, y1), (x2, y2), (0, 255, 0), 3)


#cv2.imshow("Edges", edges)
plt.imshow(img_cropped)
plt.title = "cropped image with hough lines"
plt.show()
#cv2.imshow("Image", img_cropped)
#cv2.waitKey(0)
#cv2.destroyAllWindows()