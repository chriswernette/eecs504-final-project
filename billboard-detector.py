'''script to read in image, determine edges and find hough lines, need to make
command line arguments in the future, for now change the img = cv.imread command
to change the image you're running on'''

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import find_intersections as fi
from polygon2 import polygon2
from preprocessing import preprocess_data
from cluster_corners import cluster_corners
from polygon import form_polygon, plot_mask
from skimage.morphology import convex_hull_image


#if mode  = 0, single image, if mode = 1 read in entire directory
mode = 1

#read in image
if(mode == 0):
    files = "data/real-billboard.jpg"
    num_files = 1
elif(mode == 1):
    path = '/home/chris/Documents/eecs/eecs_504/eecs504-final-project/data/Test_Images/_021_first_15/'
    files = os.listdir(path)
    files.sort()
    num_files = len(files)
    for i in range(len(files)):
        files[i] = path + files[i]
        print(files[i])


#loop through the selected files
for i in range(num_files):
    if(mode == 0):
        img_location = files
    elif(mode == 1):
        img_location = files[i]
    '''calls preprocess script to crop the image to the upper right hand quadrant,
    grayscale and blur the crop, find Canny edges and Hough lines from the cropped 
    grayscale. In the future, I'll add inputs so you can specify where the function
    looks for billboards in the image. Right now it just looks in the upper right'''
    gray_cropped, img_cropped, edges, lines = preprocess_data(img_location)
    img_cropped2 = np.copy(img_cropped)
    img_cropped2 = cv2.cvtColor(img_cropped2,cv2.COLOR_BGR2RGB)

    #show what the gray blurred image looks like
    plt.imshow(gray_cropped,cmap='gray')
    plt.title('Grayscale image')
    plt.show()

    #show what the Canny edges look like
    plt.imshow(edges,cmap='gray')
    plt.title('Canny edges')
    plt.show() 

    #put Hough lines on the cropped image one at a time just to visualize
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_cropped, (x1, y1), (x2, y2), (0, 255, 0), 3)

    #convert from BGR (openCV format) to RGB (matplotlib format) and show the lines
    img_RGB = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.title("cropped image with hough lines")
    plt.show()

    #find intersections of Hough lines
    intersections = fi.segmented_intersections(lines)

    #get (x,y) of the intersections so that we can plot them on top of image
    x = intersections[:,0]
    y = intersections[:,1]

    #show the image w/lines and then put a scatter plot of the intersections on top
    plt.imshow(img_RGB)
    plt.scatter(x, y, c='r', s=40)
    plt.title('Hough line intersections laid on top of Hough lines')
    plt.show()

    #find the clusters of the intersections - basically reduce the number of points
    labels, cluster_centers = cluster_corners(intersections)

    #get (x,y) of the cluster centers so we can plot those on top of the image next
    x = cluster_centers[:,0]
    y = cluster_centers[:,1]

    #plot the cluster locations as a scatter plot on top of the lines image
    plt.imshow(img_RGB)
    plt.scatter(x, y, c='r', s=40)
    plt.title('Cluster Centroids laid on top of Hough lines')
    plt.show()

    masked_img = polygon2(cluster_centers,img_cropped2)
    plt.imshow(masked_img)
    plt.title('The masked man approaches')
    plt.show()

    '''@TODO Peter is working on a function that will order these correctly for
    drawing a polygon. Basically the issue is the cluster points go down the left
    hand side of the billboard and then cut across to the upper right instead of the 
    lower right hand corner because the distance to the point (0,0) is less. But,
    for the polynomial mask function to work they need to be in the order that you
    would draw a box around the mask. So, Peter is working on a function that will
    get that correct order by finding the distance between each point, and reversing
    the array when we are about to go from bottom left corner -> upper right corner
    so it goes from bottom left corner -> bottom right corner'''

    ccw_clusters, corners = form_polygon(cluster_centers, img_RGB)

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

    ################################################################################
    #this was trying out some findContours/convexHull code that didn't work
    #set up binary image of just points for the contour detection algorithm/convex Hull
    #print(clust_int)

    # test_img = np.zeros_like(gray_cropped)
    # for i in range(clust_int.shape[0]):
    #     x,y = clust_int[i]
    #     test_img[y,x] = 1

    # plt.imshow(test_img)
    # plt.show()

    ################################################################################
    '''compute the distance from the upper left hand corner of the image (0,0) and
    use that to sort. The upper left hand corner should have the least distance and
    the lower right should have the largest.'''
    #dist = np.linalg.norm(clust_int,axis=1).reshape(clust_int.shape[0],1)
    #print(dist)
    #idx = np.argsort(dist,axis=0)
    #print(idx)
    #clust_dist_sorted = clust_int[idx]
    #print(clust_dist_sorted)