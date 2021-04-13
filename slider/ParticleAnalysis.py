import numpy as np

from slider import Settings

import cv2

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

def coordinationNumbers(pos_x, pos_y, rot_angle, radii, padding=5):
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

    # Not the most efficient implementation just yet, but functional
    coordinationNum = np.zeros(len(pos_x))
    for i in range(len(pos_x)):
        # Calculate the distance between each particle center
        distances = [_dist([pos_x[i], pos_y[i]], [pos_x[j], pos_y[j]]) for j in range(len(pos_x)) if j != i]
        # Calculate the sum of radii for each pair
        radiiSums = [radii[i] + radii[j] for j in range(len(pos_x)) if j != i]

        # The coordination number is the number of pairs for which the radii sum (+padding) is greater than
        # the center distance
        coordinationNum[i] = sum([1 for j in range(len(radiiSums)) if radiiSums[j] + padding > distances[j]]) 

    # Divide out the number of particles to get the average
    return coordinationNum


def averageCoordinationNumber(pos_x, pos_y, rot_angle, radii, padding=5):
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

coordinationNumbers.analysis_type = Settings.PARTICLE_ANALYSIS
averageCoordinationNumber.analysis_type = Settings.PARTICLE_ANALYSIS

