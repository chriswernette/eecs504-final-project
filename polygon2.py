import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image
from sklearn.cluster import MeanShift 

def polygon2(intersections, img_cropped):
    #need to convert cluster locations to integer so we're at valid pixel locations
    clust_int = np.int0(intersections)
    cluster_points = np.zeros((img_cropped.shape[0],img_cropped.shape[1]))
    
    #in the future, make cluster_points not MxNx3, will make it a lot faster
    #would have to

    #create an x*y*3 cluster points image for the convex hull to work on
    for i in range(clust_int.shape[0]):
        x,y = clust_int[i]
        cluster_points[y,x] = 255

    #get convex hull from scikit, which provides a binary mask
    chull = convex_hull_image(cluster_points)
    
    test = chull[:,:].astype(np.uint8)*255
    corners = cv2.cornerHarris(test,3,3,0.1)
    corners_greater_zero = np.argwhere(corners>0)
    reduced_corners = MeanShift(bandwidth=10).fit(corners_greater_zero)
    corners_final = np.floor(reduced_corners.cluster_centers_).astype(int)
    
    #print('corners flipped')
    #print(np.flip(corners_final,axis=1))
    corners_final = np.flip(corners_final,axis=1)
    
    #need to sort the corners
    x_vals = corners_final[:,0]
    idx = np.argsort(x_vals)
    #print(x_vals)
    #print(idx)
    corners_sorted_x = corners_final[idx,:]
    #print('corner values as sorted by x')
    #print(corners_sorted_x)
    first_two = corners_sorted_x[0:2,:]
    #print('first two x values')
    #print(first_two)
    last_two = corners_sorted_x[2:,:]
    #print('last two x values')
    #print(last_two)

    first_two_y = first_two[:,1]
    idx = np.argsort(first_two_y)
    first_two_sorted = first_two[idx,:]

    last_two_y = last_two[:,1]
    idx = np.argsort(last_two_y)[::-1]
    last_two_sorted = last_two[idx,:]

    corners_final_final = np.vstack((first_two_sorted,last_two_sorted))
    #print(corners_final)
    #print('corners going into Moe\'s function')
    #print(corners_final_final)

    plt.imshow(corners,cmap='gray')
    plt.show()

    #to get the image mask you need to convert to uint8 and mult by 255 so the
    #channels all show up
    mask = np.zeros_like(img_cropped)
    for i in range(mask.shape[2]):
        mask[:,:,i] = chull.copy().astype(np.uint8)*255

    masked_img = cv2.bitwise_and(img_cropped, mask)
    
    return masked_img, corners_final_final


# def polygon2(intersections, img_cropped):
#     #need to convert cluster locations to integer so we're at valid pixel locations
#     clust_int = np.int0(intersections)
#     cluster_points = np.zeros_like(img_cropped)
    
#     #in the future, make cluster_points not MxNx3, will make it a lot faster
#     #would have to

#     #create an x*y*3 cluster points image for the convex hull to work on
#     for i in range(clust_int.shape[0]):
#         x,y = clust_int[i]
#         cluster_points[y,x] = 255

#     #get convex hull from scikit, which provides a binary mask
#     chull = convex_hull_image(cluster_points)
    
#     test = chull[:,:,0].astype(np.uint8)*255

#     #to get the image mask you need to convert to uint8 and mult by 255 so the
#     #channels all show up
#     mask = chull.copy().astype(np.uint8)*255
#     masked_img = cv2.bitwise_and(img_cropped, mask)
#     corners = cv2.cornerHarris(test,3,3,0.04)
#     plt.imshow(corners)
#     plt.show()

#     return masked_img


















# plt.imshow(cluster_points, cmap = 'gray')
# plt.title('points to use for scikit learn convex hull')
# plt.show()
# plt.imshow(chull.astype(np.uint8))
# plt.title('convex hull of points from intersections')
# plt.show()
# mask = np.zeros_like(img_cropped)
# for i in range(3): 
#     mask[:,:,i] = chull.copy().astype(np.uint8) * 255
# print(mask.shape)
# print(img_RGB.shape)