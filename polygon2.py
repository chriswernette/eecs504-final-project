import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image
from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans

DEBUG = False 

def polygon2(intersections, img_cropped):
    #need to convert cluster locations to integer so we're at valid pixel locations
    clust_int = np.int0(intersections)
    cluster_points = np.zeros((img_cropped.shape[0],img_cropped.shape[1]))

    #create an x*y*3 cluster points image for the convex hull to work on
    for i in range(clust_int.shape[0]):
        x,y = clust_int[i]
        cluster_points[y,x] = 255 

    #get convex hull from scikit, which provides a binary mask
    chull = convex_hull_image(cluster_points)
    
    test = chull[:,:].astype(np.uint8)*255
    corners = cv2.cornerHarris(test,3,3,0.09)
    plt.imshow(corners,cmap='gray')
    plt.show()

    #find only 4 corners for Moe's braindead homography
    corners_greater_zero = np.argwhere(corners>0)
    kmeans = KMeans(n_clusters=4)
    reduced_corners = kmeans.fit(corners_greater_zero)
    #reduced_corners = MeanShift(bandwidth=10).fit(corners_greater_zero)
    corners_final = np.floor(reduced_corners.cluster_centers_).astype(int)
    
    if DEBUG:
        print('corners_final locations')
        print(corners_final)

    if DEBUG:
        print('corners flipped')
        print(np.flip(corners_final,axis=1))
    
    corners_final = np.flip(corners_final,axis=1)
    
    #need to sort the corners
    x_vals = corners_final[:,0]
    idx = np.argsort(x_vals)
    
    if DEBUG:
        print(x_vals)
        print(idx)
    
    corners_sorted_x = corners_final[idx,:]
    
    if DEBUG:
        print('corner values as sorted by x')
        print(corners_sorted_x)
    
    first_two = corners_sorted_x[0:2,:]
    
    if DEBUG:
        print('first two x values')
        print(first_two)
    
    last_two = corners_sorted_x[2:,:]
    
    if DEBUG:
        print('last two x values')
        print(last_two)

    first_two_y = first_two[:,1]
    idx = np.argsort(first_two_y)
    first_two_sorted = first_two[idx,:]

    last_two_y = last_two[:,1]
    idx = np.argsort(last_two_y)[::-1]
    last_two_sorted = last_two[idx,:]

    corners_final_final = np.vstack((first_two_sorted,last_two_sorted))
    
    if DEBUG:
        print(corners_final)
        print('corners going into Moe\'s function')
        print(corners_final_final)

    if DEBUG:
        plt.imshow(corners,cmap='gray')
        plt.show()

    #to get the image mask you need to convert to uint8 and mult by 255 so the
    #channels all show up
    mask = np.zeros_like(img_cropped)
    for i in range(mask.shape[2]):
        mask[:,:,i] = chull.copy().astype(np.uint8)*255

    masked_img = cv2.bitwise_and(img_cropped, mask)
    
    return masked_img, corners_final_final