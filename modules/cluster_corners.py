import numpy as np
from sklearn.cluster import MeanShift 

def cluster_corners(corner_locations):
    '''this function will take in many corner locations and find clusters 
    located around a centroid. It will involve building a graph that represents
    each cluster's adjacency to another based on the Euclidean distance, then
    separating that graph into distinct clusters that don't overlap. I think one
    of the main issues is that they'''
    
    ## of pixels away to still be considering within the cluster
    #threshold = 10
    #adjacency = build_adjaceny(corner_locations,threshold)

    clustering = MeanShift(bandwidth=10).fit(corner_locations)
    labels = clustering.labels_
    cluster_centers = clustering.cluster_centers_

    return labels, cluster_centers


def build_adjaceny(corner_locations, threshold):
    #returns adjacency matrix given corner locations and euclid distance threshold
    threshold = 10

    adjacency = zeros(corner_locations.shape[0])
    for i in range(corner_locations.shape[0]):
        for j in range(corner_locations.shape[0]):
            if(i < j):
                iRow = corner_locations[i]
                jRow = corner_locations[j]

                dist = np.linalg.norm(iRow-jRow)
                if(dist > threshold):
                    adjacency[i,j] = 1
                    adjacency[j,i] = 1

    return adjacency