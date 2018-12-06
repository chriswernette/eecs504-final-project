import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import eta.core.image as etai

def form_polygon(clusters, img):
	# Wrapper for fitting a polygon
	# to A SINGLE BILLBOARD
	return sort_clusters(clusters, img)

def sort_clusters(clusters, img):
	# Sort clusters by euclidean distance from the origin
	idx = np.argsort(np.linalg.norm(clusters,axis=-1))
	clusters = clusters[idx]

	# Find the bottom-left and upper-right corners
	# by finding the biggest y-coordinate difference 
	# between neighboring clusters (in the sorted array)
	BL = np.argmax(abs(clusters[:,1][1:]-clusters[:,1][:-1]))
	UR = BL + 1

	# Reverse the order of the right-hand side of billboard clusters
	# then combine with the left-hand billboard clusters
	# to put points in counter-clockwise order
	ccw_clusters = np.concatenate((clusters[:UR],clusters[UR:][::-1]),axis=0)
	corners = np.array([ccw_clusters[0], ccw_clusters[BL],
						ccw_clusters[UR], ccw_clusters[-1]])

	hull = cv2.convexHull(np.array([clusters], dtype='int32'))
	plot_mask(img, ccw_clusters, 'CCW Clusters')
	plot_mask(img, corners, 'OG Corner Clusters')
	masked_img = plot_mask(img, hull, 'cv2 Convex Hull')


	return ccw_clusters, corners, masked_img

def smooth_polygon(ccw_clusters, BL):
	# Smooths polygon
	# TODO: Can remove bad points from convex hull contour
	UR = BL + 1
	UL = 0
	BR = ccw.shape[0]-1

	return


def plot_mask(img, pts, title):
	mask = np.zeros(img.shape, dtype="uint8")
	pts = np.array(pts, dtype='int32')
	cv2.fillPoly(mask, [pts], (255,255,255))
	maskedImg = cv2.bitwise_and(img, mask)
	plt.figure()
	plt.imshow(maskedImg)
	plt.title(title)
	plt.show()
	return maskedImg