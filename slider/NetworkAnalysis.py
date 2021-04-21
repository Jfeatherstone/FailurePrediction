import numpy as np

import networkx as nx
import community as community_louvain

from sklearn.neighbors import KDTree 

from slider import Settings, ImageAnalysis

def _dist(p1, p2):
    """
    Simple euclidean distance between two points
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def genUnweightedAdjacencyMatrix(pos_x, pos_y, radii, padding=2):

    adjMatrix = np.zeros([len(pos_x), len(pos_y)])
    # Instead of going over every point and every other point, we can just take nearest neighbors, since
    # only neighbor particles can be in contact
    points = np.array(list(zip(pos_x, pos_y)))
    kdTree = KDTree(points, leaf_size=10)
    
    # In 2D, 8 neighbors should be more than enough
    # +1 is so we can remove the actual point itself
    dist, ind = kdTree.query(points, k=8+1)
    
    for i in range(len(pos_x)):
        for j in range(len(ind[i])):
            if radii[i] + radii[ind[i][j]] + padding > dist[i][j]: 
                adjMatrix[i][ind[i][j]] = 1
                adjMatrix[ind[i][j]][i] = 1

    return adjMatrix


def circularMask(center, radius, imageSize):

    h, w = imageSize
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def genWeightedAdjacencyMatrix(pos_x, pos_y, radii, photoelasticImage, padding=2, contactThreshold=.1):
    # First, we generate the unweighted network
    unweightedAdjMatrix = genUnweightedAdjacencyMatrix(pos_x, pos_y, radii, padding)

    
    frame = ImageAnalysis.checkImageType(photoelasticImage)

    # Make sure that our image is grayscale
    if len(np.shape(frame)) == 3:
        frame = frame[:,:,0]

    # Now we will fill in the weights using the average particle g^2
    gSqr = ImageAnalysis.gSquared(frame)

    particleGSqrs = np.zeros(len(pos_x)) 

    for i in range(len(particleGSqrs)):
        mask = circularMask((pos_x[i], pos_y[i]), radii[i], np.shape(frame)) 
        
        particleGSqrs[i] = np.sum(np.multiply(gSqr, mask))

    weightedAdjMatrix = np.multiply(np.outer(particleGSqrs, particleGSqrs), unweightedAdjMatrix)  

    maxValue = np.max(weightedAdjMatrix)
    # Clean up the matrix
    for i in range(len(pos_x)):
        weightedAdjMatrix[i,i] = maxValue

    weightedAdjMatrix /= np.max(weightedAdjMatrix)

    weightedAdjMatrix[weightedAdjMatrix < contactThreshold] = 0

    return weightedAdjMatrix

