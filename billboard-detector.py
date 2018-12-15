'''script to read in image, determine edges and find hough lines, need to make
command line arguments in the future, for now change the img = cv.imread command
to change the image you're running on'''

import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import find_intersections as fi
from billboard_homog_project import billboard_homog_project
from polygon2 import polygon2
from preprocessing import preprocess_data
from cluster_corners import cluster_corners
from polygon import form_polygon, plot_mask
from reject import is_billboard_present
from skimage.morphology import convex_hull_image

#set tunable parameters
canny_min = 50
canny_max = 75
hough_thresh = 30
hough_min_ll = 50
hough_max_gap = 35

DEBUG = True



def detect_billboard(img_location):
    '''calls preprocess script to crop the image to the upper right hand quadrant,
    grayscale and blur the crop, find Canny edges and Hough lines from the cropped 
    grayscale. In the future, I'll add inputs so you can specify where the function
    looks for billboards in the image. Right now it just looks in the upper right'''
    img = cv2.imread(img_location)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    gray_cropped, img_cropped, edges, lines = preprocess_data(img_location,canny_min,
                                                canny_max,hough_thresh,hough_min_ll,hough_max_gap)
    img_cropped2 = np.copy(img_cropped)
    img_cropped2 = cv2.cvtColor(img_cropped2,cv2.COLOR_BGR2RGB)

    if DEBUG:
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

    if DEBUG:
	    plt.imshow(img_RGB)
	    plt.title("cropped image with hough lines")
	    plt.show()

    #find intersections of Hough lines
    intersections = fi.segmented_intersections(lines)

    #get (x,y) of the intersections so that we can plot them on top of image
    x = intersections[:,0]
    y = intersections[:,1]

    if DEBUG:
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

    if DEBUG:
	    #plot the cluster locations as a scatter plot on top of the lines image
	    plt.imshow(img_RGB)
	    plt.scatter(x, y, c='r', s=40)
	    plt.title('Cluster Centroids laid on top of Hough lines')
	    plt.show()

    #Chris' billboard mask function, uses convex hull, harris corners
    masked_img_chris, corners_final = polygon2(intersections,img_cropped2)
    if DEBUG:
	    plt.imshow(masked_img_chris)
	    plt.title('The masked man approaches')
	    plt.show()

    #Peter's billboard mask function
    ccw_corners, masked_img = form_polygon(cluster_centers, img_cropped2)

    # True if billboard detected, False otherwise
    #billboard_detected = is_billboard_present(corners)
    
    if DEBUG:
	    print('Chris corners')
	    print(corners_final)
	    print('Peter corners')
	    print(ccw_corners)

    billboard_homog_project(corners_final,masked_img_chris)

def main():
	print("Main args: ", sys.argv)
	
	#if mode  = 0, single image, if mode = 1 read in entire directory
	mode = 0

	#read in image
	if(mode == 0):
	    files = "data/frame14.jpg"
	    num_files = 1
	elif(mode == 1):
	    path = 'data/Test_Images/_021_first_15/'
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
	    detect_billboard(img_location)


if(__name__=="__main__"):
	main()