'''script to read in image, determine edges and find hough lines, need to make
command line arguments in the future, for now change the img = cv.imread command
to change the image you're running on'''

#standard libs
import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image

#our libs
import modules.find_intersections as fi
from modules.billboard_homog_project import billboard_homog_project
from modules.polygon2 import polygon2
from modules.preprocessing import preprocess_data
from modules.cluster_corners import cluster_corners
from modules.polygon import form_polygon, plot_mask
from modules.reject import is_billboard_present

loose_tune = 0
tight_tune = 1

tune = 1

#set tunable parameters
if(tune == loose_tune):
    canny_min = 40
    canny_max = 50
    hough_thresh = 10
    hough_min_ll = 5
    hough_max_gap = 15

if(tune == tight_tune):
    canny_min = 60
    canny_max = 90
    hough_thresh = 30
    hough_min_ll = 25
    hough_max_gap = 15


DEBUG = True

def no_billboard(img):
    plt.imshow(np.zeros(img.shape, dtype="uint8"))
    plt.title("No Billboard Detected")
    plt.show()



def detect_billboard(img_location, crop):
    '''calls preprocess script to crop the image to the upper right hand quadrant,
    grayscale and blur the crop, find Canny edges and Hough lines from the cropped 
    grayscale. In the future, I'll add inputs so you can specify where the function
    looks for billboards in the image. Right now it just looks in the upper right'''
    img = cv2.imread(img_location)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    gray_cropped, img_cropped, edges, lines = preprocess_data(img_location, 
                                                crop=1, crop_x=crop[2:], crop_y=crop[0:2],
                                                canny_min=canny_min, canny_max=canny_max,
                                                hough_thresh=hough_thresh, hough_min_ll=hough_min_ll, hough_max_gap=hough_max_gap)
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

    if lines is None:
        no_billboard(img_cropped2)
        return

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

    #changed to < 4, need at least 4 intersections to draw a rectangular chull
    if (len(intersections) <4):
        no_billboard(img_RGB)
        return

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

    #my algo no longer uses the cluster intersections, finds better results w/o - Chris
    if 0:
        #plot the cluster locations as a scatter plot on top of the lines image
        plt.imshow(img_RGB)
        plt.scatter(x, y, c='r', s=40)
        plt.title('Cluster Centroids laid on top of Hough lines')
        plt.show()

    #Chris' billboard mask function, uses convex hull, harris corners
    masked_img_chris, corners_final = polygon2(intersections,img_cropped2)

    if DEBUG:
        plt.imshow(masked_img_chris)
        plt.title('Masked Image, pre-overlay')
        plt.show()

    #Peter's billboard mask function
    ccw_corners, masked_img = form_polygon(cluster_centers, img_cropped2)

    # True if billboard detected, False otherwise
    billboard_detected = is_billboard_present(corners_final)
    
    if DEBUG:
        print('Chris corners')
        print(corners_final)
        print('Peter corners')
        print(ccw_corners)

    if billboard_detected:
        driverside_eye_vec = np.array([1300,1552])
        passenger_eye_vec = np.array([1850,1252])
        center_eye_vec = np.array([2000,730])
        billboard_homog_project(corners_final,masked_img_chris,2,driverside_eye_vec,passenger_eye_vec,center_eye_vec)
    else:
        no_billboard(masked_img)

    

def main():
    print("Main args: ", sys.argv)

    #if mode  = 0, single image, if mode = 1 read in entire directory
    mode = 1

	#read in image
    if(mode == 0):
        files = "data/Test_Images/_021_first_15/frame14.jpg"
        img = cv2.imread(files)
        height = img.shape[0]
        width = img.shape[1]
        crop = np.array([0,width,0,height])
        num_files = 1
    elif(mode == 1):
        #be SURE to put a / at the end so it recognizes as a directory lol
        path = 'data/Test_Images/_021_first_15/'
        files = os.listdir(path)
        files.sort()
        images = []
        for i in range(len(files)):
            root, ext = os.path.splitext(files[i])
            if(ext == '.jpg'):
                to_append = path + files[i]
                images.append(to_append)
                
        if(DEBUG):
            for i in range(len(images)):
                print(images[i])
        num_files = len(files)

    crop_name = path + 'crops.npy'
    coords = np.load(crop_name)

    #loop through the selected files
    for i in range(num_files):
        if(mode == 0):
            img_location = files
        elif(mode == 1):
            img_location = images[i]
            crop = coords[i,:]
        if(DEBUG):
            print(img_location)
        detect_billboard(img_location, crop)


if(__name__=="__main__"):
    main()