'''script to read in image, determine edges and find hough lines, need to make
command line arguments in the future, for now change the img = cv.imread command
to change the image you're running on'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import find_intersections as fi
from preprocessing import preprocess_data
from cluster_corners import cluster_corners

#read in image
img_location = "data/real-billboard.jpg"

gray_cropped, img_cropped, edges, lines = preprocess_data(img_location)

#plt.imshow(blur)
#plt.show()

plt.imshow(gray_cropped,cmap='gray')
plt.title('Grayscale image')
plt.show()
#plt.imshow(gray,cmap='gray')
#plt.show()

plt.imshow(edges,cmap='gray')
plt.title('Canny edges')
plt.show() 

#print("Number of lines in image: " + str(len(lines)))

#put Hough lines on the cropped image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img_cropped, (x1, y1), (x2, y2), (0, 255, 0), 3)

img_RGB = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
plt.imshow(img_RGB)
plt.title("cropped image with hough lines")
plt.show()

#find intersections of Hough lines
intersections = fi.segmented_intersections(lines)

#print(intersections.shape)
#print(intersections)

x = intersections[:,0]
y = intersections[:,1]

plt.imshow(img_RGB)
plt.scatter(x, y, c='r', s=40)
plt.title('Hough line intersections laid on top of Hough lines')
plt.show()


#apply clustering algorithm
labels, cluster_centers = cluster_corners(intersections)

x = cluster_centers[:,0]
y = cluster_centers[:,1]

#print('x,y locations of centroids')
#print(cluster_centers)

plt.imshow(img_RGB)
plt.scatter(x, y, c='r', s=40)
plt.title('Cluster Centroids laid on top of Hough lines')
plt.show()


#set up binary image of just points for the contour detection algorithm/convex Hull
clust_int = np.int0(cluster_centers)
#print(clust_int)

test_img = np.zeros_like(gray_cropped)
#print(test_img.shape)
for i in range(clust_int.shape[0]):
    x,y = clust_int[i]
    test_img[y,x] = 1

plt.imshow(test_img)
plt.show()

print(clust_int.shape)

dist = np.linalg.norm(clust_int,axis=1).reshape(clust_int.shape[0],1)
print(dist)
idx = np.argsort(dist,axis=0)
print(idx)
# clust_add_dist = np.hstack((clust_int,dist))
# print(clust_add_dist.shape)
# print(clust_add_dist)
# idx = np.argsort(clust_add_dist,axis=0)
# dist_sorted = idx[:,2]
# print(dist_sorted)

clust_dist_sorted = clust_int[idx]
print(clust_dist_sorted)
#print(clust_dist_sorted)

# cv2.imshow('draw contrours',im2)
# cv2.waitKey(0)


#defunct testing code beyond this line
################################################################################
#testing Harris Corners, goodFeaturesToTrack is what we implemented in hw3
# corners = cv2.goodFeaturesToTrack(gray_cropped,100,.1,10)
# corners = np.int0(corners)
# corners = corners.reshape((corners.shape[0],2))
# x = corners[:,0]
# y = corners[:,1]


# plt.imshow(img_RGB)
# plt.scatter(x,y,c='r',s=40)
# plt.title('Corners from cv2.goodFeaturesToTrack laid on top of hough lines')
# plt.show()