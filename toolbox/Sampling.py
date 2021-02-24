# This file includes methods for sampling portions of the videos/data
# that could be used to train the neural network

import cv2
import numpy as np

from toolbox import Settings, Preprocessing, VideoAnalysis

# This is just for testing a video read in using loadVideo() (or any cv2.VideoCapture)
def playVideo(video, runProperties=None, waitTime=25, endWait=10000):
    if not video.isOpened():
        raise Exception("Video cannot be opened")


    region = None
    # If the runproperties parameter was passed, we crop to the good region
    if not runProperties == None:
        # Grab the good region of the image defined by the property file
        region = runProperties["goodarea"]
    
    i = 0
    pause = False
    while video.isOpened():

        if not pause:
            i += 1
            ret, frame = video.read()

        # So long as we have another frame
        if ret:
            # Grayscale, since they are grayscale images anyway
            grayscaleFrame = frame[:,:,0]
            # Crop if we can
            if not region.any() == None:
                grayscaleFrame = grayscaleFrame[region[0]:region[1], region[2]:region[3]]

            # This will display the image in a new window (similar to matlab figure windows)
            cv2.imshow('Frame', grayscaleFrame)

            # This just waits for a little after each image so it doens't go too fast
            # And so we can control the playback by pausing and skipping forward/backwards
            keyPress = cv2.waitKey(waitTime)
            if keyPress == 32: # Space bar
                pause = not pause

            if keyPress == 27: # Escape
                break

            if pause:

                if keyPress == 81: # Left
                    i -= 1
                    video.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = video.read()

                if keyPress == 83: # Right
                    i += 1
                    ret, frame = video.read()
                
        else:
            break

    #print(f'Played {i} frames') 
    # Wait at the end to close for 10s (default) or until a key is pressed
    cv2.waitKey(endWait)
    
    # Now we reset the video so that it can be played again
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Close the pop up window
    cv2.destroyAllWindows()

    # Return the number of frames
    return video.get(cv2.CAP_PROP_FRAME_COUNT)

# Downsample just by taking every ith element
def sliceDownsample(x, kernel):
    return x[::kernel]

# Downsample by averaging over sections, using the boxcar method
def boxcarDownsample(x, kernel):
    return [np.mean(x[i:i+kernel]) for i in range(len(x))[::kernel]]

# This one will do its best to line up two time arrays such that the
# error doesn't grow compound throughout the sequence
def timeAlignmentDownsample(highResTimeArr, highResXArr, lowResTimeArr, kernelGuess=None, deviation=5, debug=False, interpolate=True):
    """
    highResTimeArr: This is the array that will be used to sync with lowResTimeArr

    highResXArr: This is the array that we want to downsample; should be the same size as highResTimeArr

    lowResTimeArr: This defines what we want our X array to be downsampled to

    kernelGuess: This is a guess at the kernel size for the downsampling procedure. If it isn't provided,
    we will take the ratio of the two sizes of arrays. We don't do this by default since some of
    the time arrays may be offset by some amount, so this won't be as close as calculating a guess
    beforehand.

    deviation: How many indices we check on either side of the previous index + kernelGuess

    debug: If this is set to True, we will return [downsampledTimeArr, lowResXArr] instead of
    just lowResXArr

    interpolate: Whether or not we should take a weighted average of the two closest time points
    to get the downsampled X
    """

    # If a guess wasn't provided, we take the ratio of the lengths and increase deviation
    # by a little, since it may be needed
    if kernelGuess == None:
        kernelGuess = len(highResTimeArr)/len(lowResTimeArr)
        deviation = deviation*2 # Double the deviation

    # What we will eventually return
    lowResXArr = np.zeros_like(lowResTimeArr)

    # We'll keep track of this so we can compare and make sure the method works
    downsampledTimeArr = np.zeros_like(lowResTimeArr)

    # We intially guess the first time in the low res array divided by the time step
    # for the hig res array
    highResIndex = int((lowResTimeArr[0] - highResTimeArr[0])/(highResTimeArr[1] - highResTimeArr[0]))
    #print(lowResTimeArr[0] - highResTimeArr[0])

    # Until every element has been set
    for lowResIndex in range(len(lowResTimeArr)):
        # Generate 2*deviation + 1 indices (deviation on either side of indexGuess) to see
        # Make sure to only generate values between [ 0, len(highResTimeArr) )
        deviationArr = range(max(0, highResIndex-deviation), min(highResIndex+deviation+1, len(highResTimeArr)))
        
        # Take the index from the above list that gives the smallest difference between the low res array
        # The factor at the end is to convert the index of deviationArr to the actual difference between
        # the possible index and highResIndex
        # eg. deviationArr[0] would be highResIndex-deviation, but the argmin statement would give 1
        # The len thing is to handle if we are close to either boundary
        #try:
        actualIndex = highResIndex + np.argmin(abs(lowResTimeArr[lowResIndex] - highResTimeArr[deviationArr])) - (len(deviationArr) - 1 - deviation)
        #except:
            #print(highResTimeArr[1] - highResTimeArr[0])
            #return None
        actualIndex = min(actualIndex, len(highResTimeArr)-1)

        # Save the value from the high res into the low res
        downsampledTimeArr[lowResIndex] = highResTimeArr[actualIndex]

        # If we want to interpolate, we should find whether the actual time is before or after
        # the low res one, and take the next closest point (on the other side of the actual time)
        if interpolate:
            dt = lowResTimeArr[lowResIndex] - highResTimeArr[actualIndex]
            # Sometimes the the two points can line up exactly, which means we'll get an inf for a1
            # so we have to check for that. In this case, we just add one to secondClosestIndex
            # (since dt is zero, a1 will always be zero anyway
            secondClosestIndex = actualIndex + int(np.sign(dt)) + 1*(dt == 0)
            #print(actualIndex, secondClosestIndex)
            # Our value takes the form t_1*a_1 + t_2*a_2 = t_actual and a_1 + a_2 = 1
            a1 = dt / (highResTimeArr[secondClosestIndex] - highResTimeArr[actualIndex])
            a2 = 1 - a1

            #print(a1, lowResTimeArr[lowResIndex], highResTimeArr[actualIndex])

            lowResXArr[lowResIndex] = highResXArr[secondClosestIndex]*a1 + highResXArr[actualIndex]*a2
        else:
            lowResXArr[lowResIndex] = highResXArr[actualIndex]

        # Increment
        highResIndex = actualIndex + kernelGuess

    if debug:
        return [downsampledTimeArr, lowResXArr]

    return lowResXArr
