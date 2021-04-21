import numpy as np

from slider import Settings

import cv2

"""
Image Analysis Methods
----------------------

This file contains various image analysis methods that can be used (primarily)
with photoelastic data.

All of the methods in this file have the same signature: func(frame)

They should all also be marked with the attribute:

    .analysis_type = Settings.IMAGE_ANALYSIS

(see bottom of file for more info)
"""

def checkImageType(frame):
    """
    Make sure that the image is a proper image, and not a path
    """
    if isinstance(frame, str):
        # I don't want to overwrite the image itself, so create a new var for that
        newFrame = np.array(cv2.imread(frame), dtype='double')
    else:
        newFrame = frame

    return newFrame


def averageBrightness(frame):
    r"""
    Compute the average brightness of a frame.

    Parameters
    ----------

    frame : numpy.array
        The frame to compute for (or path to an image)
    """
    return np.mean(checkImageType(frame))


def varianceBrightness(frame):
    r"""
    Compute the standard deviation in brightness of a frame.

    Parameters
    ----------

    frame : numpy.array
        The frame to compute for (or path to an image)
    """
    return np.var(checkImageType(frame))


def gSquared(frame):
    
    properFrame = checkImageType(frame)
   
    # Make sure that our image is grayscale
    if len(np.shape(properFrame)) == 3:
        properFrame = properFrame[:,:,0]

    # Take the full size of the image, though know that the outermost row and
    # column of pixels will be 0
    gSquared = np.zeros([np.shape(properFrame)[0], np.shape(properFrame)[1]])
    # Iterate over every pixel
    # Regardless of whether we have a good region, G^2 needs a buffer of 1 pixel on each
    # side, so we have to crop down more
    for j in range(1, np.shape(properFrame)[0]-1):
        for k in range(1, np.shape(properFrame)[1]-1):
            # I've put a little picture of which pixels we are comparing
            # for each calculation (O is the current pixel, X are the
            # ones we are calculating)

            # - - -
            # X O X
            # - - -
            g1 = float(properFrame[j, k-1]) - float(properFrame[j, k+1])

            # - X -
            # - O -
            # - X -
            g2 = float(properFrame[j-1, k]) - float(properFrame[j+1, k])

            # - - X
            # - O -
            # X - -
            g3 = float(properFrame[j-1, k+1]) - float(properFrame[j+1, k-1])

            # X - -
            # - O -
            # - - X
            g4 = float(properFrame[j-1, k-1]) - float(properFrame[j+1, k+1])

            gSquared[j,k] = (g1*g1/4.0 + g2*g2/4.0 + g3*g3/8.0 + g4*g4/8.0)/4.0

    return gSquared


def averageGSquared(frame):
    r"""
    Compute the average local gradient squared, or $G^2$, of a frame.

    For more information, see the [DanielsLab Matlab implementation](https://github.com/DanielsNonlinearLab/Gsquared) or:

    Abed Zadeh, A., Bares, J., Brzinski, T. A., Daniels, K. E., Dijksman, J., Docquier, N., Everitt, H. O., Kollmer, J. E., Lantsoght, O., Wang, D., Workamp, M., Zhao, Y., & Zheng, H. (2019). Enlightening force chains: A review of photoelasticimetry in granular matter. Granular Matter, 21(4), 83. [10.1007/s10035-019-0942-2](https://doi.org/10.1007/s10035-019-0942-2)

    Parameters
    ----------

    frame : numpy.array
        The frame to compute the average of (or path to an image)
    """
    
    properFrame = checkImageType(frame)

    gSquared = 0
    # Iterate over every pixel
    # Regardless of whether we have a good region, G^2 needs a buffer of 1 pixel on each
    # side, so we have to crop down more
    for j in range(1, np.shape(properFrame)[0]-1):
        for k in range(1, np.shape(properFrame)[1]-1):
            # I've put a little picture of which pixels we are comparing
            # for each calculation (O is the current pixel, X are the
            # ones we are calculating)

            # - - -
            # X O X
            # - - -
            g1 = properFrame[j, k-1] - properFrame[j, k+1]

            # - X -
            # - O -
            # - X -
            g2 = properFrame[j-1, k] - properFrame[j+1, k]

            # - - X
            # - O -
            # X - -
            g3 = properFrame[j-1, k+1] - properFrame[j+1, k-1]

            # X - -
            # - O -
            # - - X
            g4 = properFrame[j-1, k-1] - properFrame[j+1, k+1]

            currGSquared = (g1*g1/4.0 + g2*g2/4.0 + g3*g3/8.0 + g4*g4/8.0)
            gSquared += currGSquared/4.0
    
    # Divide out the size of the image, since we want an average
    gSquared /= np.shape(properFrame)[0]*np.shape(properFrame)[1]
    # We take the mean because there are the 3 pixel values
    return np.mean(gSquared)


def varianceGSquared(frame):
    r"""
    Compute the variance in the local gradient squared, or $G^2$, of a frame.

    For more information, see:

    Abed Zadeh, A., Bares, J., Brzinski, T. A., Daniels, K. E., Dijksman, J., Docquier, N., Everitt, H. O., Kollmer, J. E., Lantsoght, O., Wang, D., Workamp, M., Zhao, Y., & Zheng, H. (2019). Enlightening force chains: A review of photoelasticimetry in granular matter. Granular Matter, 21(4), 83. [10.1007/s10035-019-0942-2](https://doi.org/10.1007/s10035-019-0942-2)
 
    """

    properFrame = checkImageType(frame)

    gSquared = np.zeros([np.shape(properFrame)[0]-2, np.shape(properFrame)[1]-2])
    # Iterate over every pixel
    # Regardless of whether we have a good region, G^2 needs a buffer of 1 pixel on each
    # side, so we have to crop down more
    for j in range(1, np.shape(properFrame)[0]-1):
        for k in range(1, np.shape(properFrame)[1]-1):
            # I've put a little picture of which pixels we are comparing
            # for each calculation (O is the current pixel, X are the
            # ones we are calculating)

            # - - -
            # X O X
            # - - -
            g1 = properFrame[j, k-1] - properFrame[j, k+1]

            # - X -
            # - O -
            # - X -
            g2 = properFrame[j-1, k] - properFrame[j+1, k]

            # - - X
            # - O -
            # X - -
            g3 = properFrame[j-1, k+1] - properFrame[j+1, k-1]

            # X - -
            # - O -
            # - - X
            g4 = properFrame[j-1, k-1] - properFrame[j+1, k+1]

            gSquared[j,k] = (g1*g1/4.0 + g2*g2/4.0 + g3*g3/8.0 + g4*g4/8.0)/4.0
    
    # Divide out the size of the image, since we want an average
    return np.var(gSquared)


"""
Attribute Marking
-----------------

Since we have multiple types of analysis methods, that need different variables
(image, particle, network, etc.) we need to be able to differentiate them, and 
provide the correct args to each one.

This is done by assigning an identifying attribute to each function based on
what type of analysis it is (and there its method signature). The attribute will
be the same for all types of methods, but it's value will be different

The attribute for ImageAnalysis methods is: analysis_type = Settings.IMAGE_ANALYSIS
"""

averageBrightness.analysis_type = Settings.IMAGE_ANALYSIS
varianceBrightness.analysis_type = Settings.IMAGE_ANALYSIS
averageGSquared.analysis_type = Settings.IMAGE_ANALYSIS
varianceGSquared.analysis_type = Settings.IMAGE_ANALYSIS
