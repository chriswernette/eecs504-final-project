import cv2
import numpy as np
import matplotlib.pyplot as plt

#img = cv2.imread("data/billboard1.jpg")
img = cv2.imread("data/real-billboard.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
blur = cv2.GaussianBlur(img, (5, 5), 0)
#plt.imshow(blur)
#plt.show()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
y,x = gray.shape
gray_cropped = gray[0:np.floor(y/2).astype(int),np.floor(x/2).astype(int):]
img_cropped = img[0:np.floor(y/2).astype(int),np.floor(x/2).astype(int):]
print(gray.shape)
print(gray_cropped.shape)
plt.imshow(gray_cropped,cmap='gray')
plt.show()
#plt.imshow(gray,cmap='gray')
#plt.show()
edges = cv2.Canny(gray_cropped, 150, 150)
plt.imshow(edges)
plt.title = 'edges'
plt.show() 

lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=35, minLineLength=100)
print("Number of lines in image: " + str(len(lines)))


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