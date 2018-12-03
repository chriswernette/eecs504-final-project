import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import eta.core.image as etai

def form_polygon(clusters):
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

	return ccw_clusters

def plot_mask(img, ccw_clusters):
	mask = np.zeros(img.shape, dtype="uint8")
	# pts represents 4 corners of polygon (do not need to be rectangular)
	pts = np.array(ccw_clusters, dtype='int32')
	# [pts] is an array of polygons, so we can make multiple masked regions per image
	cv2.fillPoly(mask, [pts], (255,255,255))
	maskedImg = cv2.bitwise_and(img, mask)
	plt.figure()
	plt.imshow(maskedImg)
	plt.show()