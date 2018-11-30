#tutorial script to find lane lines from this youtube video https://www.youtube.com/watch?v=KEYzUP7-kkU&feature=youtu.be

import cv2
import numpy as np

#load in movie as whole file
video = cv2.VideoCapture("../data/road_car_view.mp4")

#split up video frame by frame
while True:
    ret, orig_frame = video.read()

    #loop video
    if not ret:
        video = cv2.VideoCapture("road_car_view.mp4")
        continue
    
    #blur image
    frame = cv2.GaussianBlur(orig_frame, (5, 5), 0)

    '''hsv allows you to create a mask for certain colors. In this case he uses 
    a yellow lower and upper bound on the color yellow to only detect edges on 
    the yellow lane lines. In our case I think we could create an anti-mask to
    reject lines that are in the green/blue highway signs pretty easily'''
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_yellow = np.array([18, 94, 140])
    up_yellow = np.array([48, 255, 255])
    mask = cv2.inRange(hsv, low_yellow, up_yellow)
    edges = cv2.Canny(mask, 75, 150)
 
    '''Hough lines transform, can set maxLineGap to get more single lines, set min
    line length to avoid having many super short lines'''
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
 
    cv2.imshow("frame", frame)
    cv2.imshow("edges", edges)
 
    key = cv2.waitKey(25)
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()