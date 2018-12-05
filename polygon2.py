import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image


def polygon2(intersections, img_cropped):
    #need to convert cluster locations to integer so we're at valid pixel locations
    clust_int = np.int0(intersections)
    cluster_points = np.zeros_like(img_cropped)

    #create an x*y*3 cluster points image for the convex hull to work on
    for i in range(clust_int.shape[0]):
        x,y = clust_int[i]
        cluster_points[y,x] = 255

    #get convex hull from scikit, which provides a binary mask
    chull = convex_hull_image(cluster_points)
    
    #to get the image mask you need to convert to uint8 and mult by 255 so the
    #channels all show up
    mask = chull.copy().astype(np.uint8) * 255
    masked_img = cv2.bitwise_and(img_cropped, mask)

    return masked_img


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