import numpy as np

import matplotlib.pyplot as plt

import cv2

def visualizeTrackedParticles(trackingData, image=None, ax=None):
    r"""
    Draw tracked particles onto a figure.

    Parameters
    ----------

    trackingData : list(2) (num_particles, 4)
        List of tracking data for a single time step, where the last dimension is organized as:
        (x_pos, y_pos, rotation_angle, radius).

    image : str or np.array
        An image that will be drawn under the particles if provided. Can be provided either as
        actual image data, or as a path to an image.

    ax : matplotlib.axis
        An axis to draw the data on if provided.

    Returns
    -------

    matplotlib.axis : The axis the data is plotted on (whether one is provided or not)
    """
    if ax == None:
        fig, ax = plt.subplots()

    if image != None:
        # The image could either be the proper image data, or a path to the image
        # so we have to check to either one
        if isinstance(image, str):
            # I don't want to overwrite the image itself, so create a new var for that
            tempImage = np.array(cv2.imread(image))
        else:
            tempImage = image
        ax.imshow(tempImage)

    # Plot circles over all of the detected particles
    for i in range(len(trackingData)):
        c = plt.Circle((trackingData[i,0], trackingData[i,1]), trackingData[i,3], alpha=.5)
        ax.add_artist(c)

    return ax


def visualizeTrackedParticlesAndCommunities(trackingData, communities, image=None, colors=None, ax=None):
    r"""
    Draw tracked particles onto a figure.

    Parameters
    ----------

    trackingData : list(2) (num_particles, 4)
        List of tracking data for a single time step, where the last dimension is organized as:
        (x_pos, y_pos, rotation_angle, radius).

    image : str or np.array
        An image that will be drawn under the particles if provided. Can be provided either as
        actual image data, or as a path to an image.

    ax : matplotlib.axis
        An axis to draw the data on if provided.

    Returns
    -------

    matplotlib.axis : The axis the data is plotted on (whether one is provided or not)
    """
    if not isinstance(colors, list):
        colors = genRandomColors(max(communities)+1)

    if ax == None:
        fig, ax = plt.subplots()

    if image != None:
        # The image could either be the proper image data, or a path to the image
        # so we have to check to either one
        if isinstance(image, str):
            # I don't want to overwrite the image itself, so create a new var for that
            tempImage = np.array(cv2.imread(image))
        else:
            tempImage = image
        ax.imshow(tempImage)

    # Plot circles over all of the detected particles
    for i in range(len(trackingData)):
        c = plt.Circle((trackingData[i,0], trackingData[i,1]), trackingData[i,3], alpha=.5, color=colors[communities[i]])
        ax.add_artist(c)

    return ax

def _dist(p1, p2):
    """
    Simple euclidean distance between two points
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def markParticleContacts(trackingData, image, ax=None, padding=5, pointRadius=5):
    # Not the most efficient implementation just yet, but functional

    if ax == None:
        fig, ax = plt.subplots()

    # The image could either be the proper image data, or a path to the image
    # so we have to check to either one
    if isinstance(image, str):
        # I don't want to overwrite the image itself, so create a new var for that
        tempImage = np.array(cv2.imread(image))
    else:
        tempImage = image
    ax.imshow(tempImage)

    positions = []
    for i in range(len(trackingData)):
        # Calculate the distance between each particle center
        distances = [_dist(trackingData[i,0:2], trackingData[j,0:2]) for j in range(len(trackingData)) if j != i]
        # Calculate the sum of radii for each pair
        radiiSums = [trackingData[i,3] + trackingData[j,3] for j in range(len(trackingData)) if j != i]

        # The coordination number is the number of pairs for which the radii sum (+padding) is greater than
        # the center distance
        for j in range(len(radiiSums)):
            if radiiSums[j] + padding > distances[j]:
                # We indexed slightly differently for distances and radiiSums so we
                # have to adjust slightly to index the original trackingData list
                adJ = j + int(j >= i)
                positions.append([(trackingData[i,0]*trackingData[i,3]+trackingData[adJ,0]*trackingData[adJ,3])/radiiSums[j],
                                  (trackingData[i,1]*trackingData[i,3]+trackingData[adJ,1]*trackingData[adJ,3])/radiiSums[j]])

    # Plot circles over all of the detected particles
    for i in range(len(trackingData)):
        c = plt.Circle((trackingData[i,0], trackingData[i,1]), trackingData[i,3], alpha=.4)
        ax.add_artist(c)

    for i in range(len(positions)):
        c = plt.Circle(positions[i], pointRadius, color='red')
        ax.add_artist(c)

    return ax

def visualizePredictionSample(inputArr, outputVal, ax=None, includesTime=True, metricList=None):
    if ax == None:
        fig, ax = plt.subplots()

    if includesTime:
        xArr = inputArr[-1,:]
    else:
        xArr = np.arange(len(inputArr[0]))

    for i in range(len(inputArr) - int(includesTime)):
        subtractMean = inputArr[i,:] - np.mean(inputArr[i,:])
        if isinstance(metricList, list):
            ax.plot(xArr, subtractMean/np.max(abs(subtractMean)), label=metricList[i].__name__)
        else:
            ax.plot(xArr, subtractMean/np.max(abs(subtractMean)))

    if outputVal:
        zoneColor = 'green'
    else:
        zoneColor = 'red'

    fillX = [xArr[-1], xArr[-1] + (xArr[1] - xArr[0])]
    fillY = [-1, 1]

    ax.fill([fillX[0], fillX[0], fillX[1], fillX[1]], [fillY[0], fillY[1], fillY[1], fillY[0]], color=zoneColor, alpha=.5)

    if isinstance(metricList, list):
        ax.legend()
    ax.set_yticks([])

    return ax

def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb

def genRandomColors(size, seed=21):
    np.random.seed(seed)

    randomColors = [f"#{rgb_to_hex(tuple(np.random.choice(range(256), size=3).flatten()))}" for i in range(size)]

    return randomColors
