#standard libs
import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image
import eta.core.image as etai
import shutil

#our libs
import modules.find_intersections as fi
from modules.billboard_homog_project import billboard_homog_project
from modules.polygon2 import polygon2
from modules.preprocessing_crop_dims import preprocess_data
from modules.cluster_corners import cluster_corners
from modules.polygon import form_polygon, plot_mask
from modules.reject import is_billboard_present

#set tunable parameters
canny_min = 82
canny_max = 125
hough_thresh = 10
hough_min_ll = 15
hough_max_gap = 35

# crop windows [y_low, y_high, x_low, x_high]
# These windows are regions in which we expect to find billboards
# when viewed from the vantage point of the dashcam
crop_windows = [[570,690,1130,1430],[225,525,1000,1300],[425,550,740,925]]

DEBUG = False

def detect_billboard(img_location, crop_dims):
	'''calls preprocess script to crop the image to the upper right hand quadrant,
	grayscale and blur the crop, find Canny edges and Hough lines from the cropped 
	grayscale. In the future, I'll add inputs so you can specify where the function
	looks for billboards in the image. Right now it just looks in the upper right'''
	img = cv2.imread(img_location)
	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	gray_cropped, img_cropped, edges, lines = preprocess_data(img_location,crop_dims,canny_min,
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

	if lines is None:
		return False, [], []

	#put Hough lines on the cropped image one at a time just to visualize
	for line in lines:
		x1, y1, x2, y2 = line[0]
		cv2.line(img_cropped, (x1, y1), (x2, y2), (0, 255, 0), 3)

	#convert from BGR (openCV format) to RGB (matplotlib format) and show the lines
	img_RGB = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)

	if DEBUG:
		# Show the hough lines overlayed on the image
		plt.imshow(img_RGB)
		plt.title("cropped image with hough lines")
		plt.show()

	#find intersections of Hough lines
	intersections = fi.segmented_intersections(lines)

	if len(intersections) is 0:
		# There is no billboard if no hough intersections are found
		return False, [], []

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

	#Chris' billboard mask function, uses convex hull, harris corners
	masked_img, corners = polygon2(intersections,img_cropped2)

	if DEBUG:
		plt.imshow(masked_img)
		plt.title('Masked Image, pre-overlay')
		plt.show()

	if DEBUG:
		print('Corners corners')
		print(corners_final)

	# True if billboard detected, False otherwise
	billboard_detected = is_billboard_present(corners)

	return billboard_detected, corners, masked_img

def main():
	print("Main args: ", sys.argv)

	#if mode  = 0, single image, if mode = 1 read in entire directory
	mode = 1

	#read in image
	if(mode == 0):
		# Choose which image to run the algorithm on
		files = "data/Test_Images/_021_first_15/frame10.jpg"
		#files = 'data/image_set_4/_image_set_4/frame14.jpg'
		num_files = 1
	elif(mode == 1):
		# Choose which dataset to run the algorithm on
		path = 'data/Test_Images/_021_first_15/'
		#path = 'data/image_set_4/_image_set_4/'
		files = os.listdir(path)
		files.sort()
		num_files = len(files)
		for i in range(len(files)):
			files[i] = path + files[i]

	# Prepare the output directory
	shutil.rmtree('output')
	os.mkdir('output')

	#loop through the selected files
	adLock_cnt = 0
	for i in range(num_files):
		if(mode == 0):
			img_location = files
		elif(mode == 1):
			img_location = files[i]
		
		if img_location == path + "crops.npy":
			continue
		
		# Display the current image
		print(img_location)
		
		# adLock logic
		if adLock_cnt is 0:
			carimg = etai.read('data/center.jpg')
			carimg = cv2.cvtColor(carimg,cv2.COLOR_BGR2RGB)

		# Detect billboards in all listed windows
		for crop in crop_windows:
			detected, corners, mask = detect_billboard(img_location, crop_dims=crop)
			if detected:
				img = cv2.imread(img_location)
				img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
				plt.imshow(img)
				plt.title(img_location)
				plt.show()
				print("Billboard detected, projecting onto HUD:")
				driverside_eye_vec = np.array([1300,1552])
				passenger_eye_vec = np.array([1850,1252])
				center_eye_vec = np.array([2000,730])
				carimg = billboard_homog_project(corners,mask,2,driverside_eye_vec,passenger_eye_vec,center_eye_vec)
				carimg = cv2.cvtColor(carimg,cv2.COLOR_BGR2RGB)

				adLock_cnt = 4
				break

		# Write to output
		if mode is 1:
			frame_no = img_location[len(path):-4]
			carimg_path = 'output/' + frame_no + '.jpg'
			cv2.imwrite(carimg_path, carimg)

		if adLock_cnt > 0:
			adLock_cnt -= 1




if(__name__=="__main__"):
	main()