# This file contains methods for extracting features from a given movie

import cv2
import numpy as np

import Preprocessing
import Settings

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


# This returns the average brightness of each frame of the video
def averageBrightness(video, runProperties=None):

    # Check if a file name was passed, then we open that
    if isinstance(video, str):
        video = Preprocessing.loadVideo(video)

    # Grab the total number of frames and initialize arrays
    numFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    averageBrightnessArr = np.zeros(numFrames)

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

            averageBrightnessArr[i] = np.mean(grayscaleFrame)

            i += 1
        else:
            break

    # Now we reset the video so that it can be played again
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    return averageBrightnessArr


# Almost exactly the same as averageBrightness, except that we calculate
# the standard deviation instead of the mean
def standardDeviation(video, runProperties=None):

    # Check if a file name was passed, then we open that
    if isinstance(video, str):
        video = Preprocessing.loadVideo(video)

    # Grab the total number of frames and initialize arrays
    numFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    standardDeviationArr = np.zeros(numFrames)

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

            standardDeviationArr[i] = np.std(grayscaleFrame)

            i += 1
        else:
            break

    # Now we reset the video so that it can be played again
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    return standardDeviationArr
