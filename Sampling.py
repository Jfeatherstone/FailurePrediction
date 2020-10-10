# This file includes methods for sampling portions of the videos/data
# that could be used to train the neural network

import cv2
import numpy as np

import Preprocessing
import VideoAnalysis
import Settings

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
def timeAlignmentDownsample(highResTimeArr, highResXArr, lowResTimeArr, kernelGuess=None, deviation=5, debug=False):
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
        actualIndex = highResIndex + np.argmin(abs(lowResTimeArr[lowResIndex] - highResTimeArr[deviationArr])) - (len(deviationArr) - 1 - deviation)
        actualIndex = min(actualIndex, len(highResTimeArr)-1)

        # Save the value from the high res into the low res
        lowResXArr[lowResIndex] = highResXArr[actualIndex]
        downsampledTimeArr[lowResIndex] = highResTimeArr[actualIndex]

        # Increment
        highResIndex = actualIndex + kernelGuess

    if debug:
        return [downsampledTimeArr, lowResXArr]

    return lowResXArr

# Randomly sample a portion of a single video
def randomSample(video, propertyFile, matFile, score, sampleLength=Settings.DEFAULT_SAMPLE_LENGTH, numSamples=1, preserveArrayType=False, downsampleMethod=timeAlignmentDownsample):
    """
    video: The video file that we want to sample from; can also be a list of video files

    propertyFile: The properties of the video that we are sampling; can also be a list of
    property files (in that case, should be the same length as video)

    matFile: The file containing the force data from the sensor for the video; can
    also be a list of mat files (in that case, should be the same length as video)

    score: The function that will be calculated for each video frame; should be of the
    form score(video, propertyFile, **kwargs).

    sampleLength: The number of frames that we want to sample

    numSamples: The number of samples we want to generate

    preserveArrayType: If numSamples is 1, should we return an array of the shape (1,x,y,z) (True) or cull the extra dimensions (x,y,z) (False)

    return type:

    We return an array of shape (numSamples, sampleLength, 3) where the 3 entries in the last dimension are:

    [0]: time
    [1]: score
    [2]: forceSensorReading

    Note that the first dimension may be collapsed if numSamples = 1; see preserveArrayType for more info
    """


    if not isinstance(video, list) and not isinstance(propertyFile, list):

        
        # Grab this and the requested score beforehand, in case we want multiple
        # samples from this file
        fullTimeArr = VideoAnalysis.timeArr(video, propertyFile)
        fullScoreArr = score(video, propertyFile)
        numFrames = len(fullTimeArr)

        # See return type explanation above for why it has this shape
        returnArr = np.zeros([numSamples, sampleLength, 3])

        for i in range(numSamples):
            # Randomly generate an interval of time for the video 
            randomStartFrame = np.random.randint(0, numFrames - sampleLength)
            endFrame = randomStartFrame + sampleLength

            # Crop the full array down to the sample
            timeArr = fullTimeArr[randomStartFrame:endFrame]
            scoreArr = fullScoreArr[randomStartFrame:endFrame]

            # Next up we need to grab the force sensor data
            # This is a little more involved than the previous steps since it
            # has a different temporal resolution than the videos
            # This means that we have to do a little bit of index magic
            
            # The force data is taken in intervals of Settings.FORCE_SENSOR_DT (.01 as of now)
            # so we have to find the index of the beginning time
            beginIndex = int(timeArr[0]/Settings.FORCE_SENSOR_DT)

            # For our kernel, we take the difference in time between frames divided
            # by the difference in time between measurements
            downsampleKernel = (timeArr[1] - timeArr[0])/Settings.FORCE_SENSOR_DT
            # We take sampleLength-1 because we already have the first point, we need
            # sampleLength-1 more
            endIndex = int(beginIndex + downsampleKernel * (sampleLength-1))

            # So I tried to make all of the downsampling methods have the same argument structure, but since
            # the time alignment one is so different, it's just not possible
            # Because of this, we have to separately evaluate the downsampling if we are using this one
            if downsampleMethod == timeAlignmentDownsample:
                # Since the time alignment method finds the t value that is closest
                # to the one in timeArr, I've found it works best to have a little bit
                # of padding on each high res array. The extraneous values won't make a 
                # difference if they aren't used, since each value is picked out, not just sliced
                padding = min(10, len(matFile["f"]))
                timeSensorArr, forceSensorArr = downsampleMethod(matFile["t"][beginIndex:endIndex+padding], matFile["f"][beginIndex:endIndex+padding], timeArr, int(downsampleKernel), debug=True)
            else:
                forceSensorArr = downsampleMethod(matFile["f"][beginIndex:endIndex], int(downsampleKernel))
                timeSensorArr = downsampleMethod(matFile["t"][beginIndex:endIndex], int(downsampleKernel))

            return [timeArr, scoreArr, timeSensorArr]
