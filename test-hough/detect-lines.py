#see tutorial here https://www.youtube.com/watch?v=KEYzUP7-kkU&feature=youtu.be
import cv2
import numpy as np

'''one annoying caveat is that cv2 imports images as BGR instead of RGB so be
careful of how to interpret the image'''
img = cv2.imread("../data/lines.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 75, 150)
 
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=250)
 
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
 
cv2.imshow("Edges", edges)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()