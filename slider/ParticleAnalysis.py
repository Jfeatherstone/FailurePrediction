import numpy as np

from slider import Settings

import cv2

from sklearn.neighbors import KDTree

"""
Particle Analysis Methods
----------------------

This file contains analysis methods that make use of tracked particle
positions and rotations

All of the methods in this file have the same signature: func(pos_x, pos_y, rot_angle, radii)
There can be other kwargs, but no other required args should be included

They should all also be marked with the attribute:

    .analysis_type = Settings.PARTICLE_ANALYSIS 

(see bottom of file for more info)
"""

def _dist(p1, p2):
    """
    Simple euclidean distance between two points
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def coordinationNumbers(pos_x, pos_y, rot_angle, radii, padding=2):
    """
    Calculate the coordination number (number of contacts) for a set of particle positions.
    Contacts are defined when the distance between the particle centers is smaller than
    the sum of the two radii, or within an amount defined by padding

    Parameters
    ----------
    
    pos_x : numpy.array
        An array of the particle x positions

    pos_y : numpy.array
        An array of the particle y positions

    rot_angle : numpy.array
        An array of the particle rotation angles, in degrees

    radii : numpy.array
        An array of the particle radii

    padding : float
        The maximum difference the particle edges can be from each other and still be considered touching (default=5)

    Returns
    -------

    numpy.ndarray : Number of contacts for each particle (same length as any of the input arrays)
    """

    points = np.array(list(zip(pos_x, pos_y)))
    kdTree = KDTree(points, leaf_size=10)
    coordinationNum = np.zeros(len(points)) 

    # In 2D, 8 neighbors should be more than enough
    # +1 is so we can remove the actual point itself
    dist, ind = kdTree.query(points, k=8+1)
    
    for i in range(len(pos_x)):
        coordinationNum[i] = sum([1 for j in range(len(ind[i])) if radii[i] + radii[ind[i][j]] + padding > dist[i][j]])

    return coordinationNum


def averageCoordinationNumber(pos_x, pos_y, rot_angle, radii, padding=2):
    """
    Calculate the average coordination number for a set of particles. Essentially a wrapper for
    coordinationNumbers that averages afterwards.
    Contacts are defined when the distance between the particle centers is smaller than
    the sum of the two radii, or within an amount defined by padding

    Parameters
    ----------
    
    pos_x : numpy.array
        An array of the particle x positions

    pos_y : numpy.array
        An array of the particle y positions

    rot_angle : numpy.array
        An array of the particle rotation angles, in degrees

    radii : numpy.array
        An array of the particle radii

    padding : float
        The maximum difference the particle edges can be from each other and still be considered touching (default=5)

    Returns
    -------
    float : The average number of contacts for all particles
    """

    return np.mean(coordinationNumbers(pos_x, pos_y, rot_angle, radii, padding=padding))


def DSquaredMin(initialParticlePos, finalParticlePos, refParticleIndex=0):
    """

    Parameters
    ----------

    initialParticlePos : numpy.ndarray or list
        An array or list of initial particle posisitions of shape (N,d) where d is the dimension of the space,
        and N is the number of particles.

    finalParticlePos : numpy.ndarray or list
        An array or list of final particle posisitions of shape (N,d) where d is the dimension of the space,
        and N is the number of particles.
    """

    numParticles, dim = np.shape(initialParticlePos)

    # The delta function
    delta = np.identity(dim)

    # Calculate the optimal epsilon (strain)
    X = np.zeros([dim, dim])
    Y = np.zeros([dim, dim])
    epsilon = np.zeros([dim, dim])

    for i in range(dim):
        for j in range(dim):
            X[i,j] = np.sum([(initialParticlePos[n][i] - initialParticlePos[refParticleIndex][i]) * (finalParticlePos[n][j] - finalParticlePos[refParticleIndex][j]) for n in range(numParticles) if n != refParticleIndex])
            Y[i,j] = np.sum([(initialParticlePos[n][i] - initialParticlePos[refParticleIndex][i]) * (initialParticlePos[n][j] - initialParticlePos[refParticleIndex][j]) for n in range(numParticles) if n != refParticleIndex])

    for i in range(dim):
        for j in range(dim):
            epsilon[i,j] = np.sum([X[i,k]/Y[j,k] for k in range(dim)]) - delta[i,j]

    # Now actually calculate the D squared min
    DSqrMin = 0
    for n in range(numParticles):
        if n == refParticleIndex:
            continue
        for j in range(dim):
            DSqrMin += (finalParticlePos[n][i] - finalParticlePos[refParticleIndex][i] - np.sum([(delta[i,j] + epsilon[i,j]) * (initialParticlePos[n][j] - initialParticlePos[refParticleIndex][j]) for j in range(dim)]))**2

    return DSqrMin


"""
Attribute Marking
-----------------

Since we have multiple types of analysis methods, that need different variables
(image, particle, network, etc.) we need to be able to differentiate them, and 
provide the correct args to each one.

This is done by assigning an identifying attribute to each function based on
what type of analysis it is (and there its method signature). The attribute will
be the same for all types of methods, but it's value will be different

The attribute for ParticleAnalysis methods is: analysis_type = Settings.PARTICLE_ANALYSIS
"""

averageCoordinationNumber.analysis_type = Settings.PARTICLE_ANALYSIS

