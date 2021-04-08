# This file contains methods for extracting features from a given movie

import cv2
import numpy as np

from toolbox import Settings, Preprocessing

# This returns an array representing the time of each frame of the video
# The majority of the information is taken from the property file, so that
# is required for this method (unlike some of the others)
# Also note that my justification for this being correct is quantatively comparing
# the average brightness to the force sensor, so it may be slightly off
def timeArr(video, runProperties):
    # The start time specified in this file seems to be the time the camera was started AFTER
    # the force sensor
    videoStartTime = runProperties["starttime"]

    # Not quite sure how this relates to the force sensor, but it seems the camera is usually
    # stopped BEFORE the force sensor 
    videoEndTime = runProperties["endtime"]

    # Now just linspace between the two with the number of frames
    videoTime = np.linspace(videoStartTime, videoEndTime, int(video.get(cv2.CAP_PROP_FRAME_COUNT)))

    return videoTime


# This is the generic function that will always be called to analyze a video
# and we simply add more functions that can be passed in `metrics` for different
# methods
def analyzeVideo(video, metrics=[], runProperties=None, subtractMean=False):

    # Instead of having a testFunc like before, you can now just not pass a 
    # metrics list and this method will simply return an array of zeros

    # Check if a file name was passed, then we open that
    if isinstance(video, str):
        video = Preprocessing.loadVideo(video)

    # Grab the total number of frames and initialize arrays
    numFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if len(metrics) == 0:
        return np.zeros([1, numFrames])

    metricValues = np.zeros([len(metrics), numFrames])

    region = None
    # If the runproperties parameter was passed, we crop to the good region
    if not runProperties == None:
        # Grab the good region of the image defined by the property file
        region = runProperties["goodarea"]

    i = 0
    while video.isOpened():

        # Read the next frame
        ret, frame = video.read()

        # Assuming there is a new frame, we grab its average brightness
        if ret:
            # Grayscale, since they are grayscale images anyway
            grayscaleFrame = frame[:,:,0]

            # Crop if we can
            if not region.any() == None:
                grayscaleFrame = grayscaleFrame[region[0]:region[1], region[2]:region[3]]

            for j in range(len(metrics)):
                metricValues[j,i] = metrics[j](grayscaleFrame)

            i += 1
        else:
            break

    # Now we reset the video so that it can be played again
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    return metricValues

def averageBrightness(frame):
    return np.mean(frame)

def standardDeviation(frame):
    return np.std(frame)

# Calculate the average local gradient squared for the image. Adapted from:
# https://github.com/DanielsNonlinearLab/Gsquared
def averageGSquared(frame):
    # Calculate the average G^2

    gSquared = 0
    # Iterate over every pixel
    # Regardless of whether we have a good region, G^2 needs a buffer of 1 pixel on each
    # side, so we have to crop down more
    for j in range(1, np.shape(frame)[0]-1):
        for k in range(1, np.shape(frame)[1]-1):
            # I've put a little picture of which pixels we are comparing
            # for each calculation (O is the current pixel, X are the
            # ones we are calculating)

            # - - -
            # X O X
            # - - -
            g1 = frame[j, k-1] - frame[j, k+1]

            # - X -
            # - O -
            # - X -
            g2 = frame[j-1, k] - frame[j+1, k]

            # - - X
            # - O -
            # X - -
            g3 = frame[j-1, k+1] - frame[j+1, k-1]

            # X - -
            # - O -
            # - - X
            g4 = frame[j-1, k-1] - frame[j+1, k+1]

            currGSquared = (g1*g1/4.0 + g2*g2/4.0 + g3*g3/8.0 + g4*g4/8.0)
            gSquared += currGSquared/4.0
    
    # Divide out the size of the image, since we want an average
    gSquared /= np.shape(frame)[0]*np.shape(frame)[1]
    return gSquared
