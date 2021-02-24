import numpy as np

from toolbox import Settings, Preprocessing, VideoAnalysis, Sampling

def randomPredictionSample(video, propertyFile, matFile, scores, sampleLength=Settings.DEFAULT_SAMPLE_LENGTH, numSamples=1, preserveArrayType=False, failurePredictionIntervalModifier=1, forceEventThreshhold=0):
    """
    video: The video file that we want to sample from; can also be a list of video files

    propertyFile: The properties of the video that we are sampling; can also be a list of
    property files (in that case, should be the same length as video)

    matFile: The file containing the force data from the sensor for the video; can
    also be a list of mat files (in that case, should be the same length as video)

    scores: The function (or list of functions) that will be calculated for each video frame; each function
    should be of the form score(video, propertyFile, **kwargs).

    sampleLength: The number of frames that we want to sample

    numSamples: The number of samples we want to generate

    preserveArrayType: If numSamples is 1, should we return an array of the shape (1,x,y,z) (True) or cull the extra dimensions (x,y,z) (False)

    failurePredictionIntervalModifier: Since the force data (and therefore the failure times) have a different time resolution than the
    images, there is some ambiguity in saying if a failure event happens in the "next" timestep. To address this, we say that a failure happens
    "next" if there exists a failure in the interval [time[-1], time[-1] + failurePredictionIntervalModifier*(time[1]-time[0])]

    That is, if a failure happens before what would be the next frame of the video, times this modifier (eg. setting it to 2 would double that interval)

    return type:

    We return an array of shape (numSamples, 1+len(scores), sampleLength) where the entries in the middle dimension correspond to time
    and whatever other scores are provided

    [0]: time
    [1]: scores[0]
    [2]: scores[1]
    ...
    [n]: scores[n]

    Note that the first dimension may be collapsed if numSamples = 1; see preserveArrayType for more info
    """

    if not isinstance(video, list) and not isinstance(propertyFile, list):
        
        # Grab this and the requested score beforehand, in case we want multiple
        # samples from this file
        fullTimeArr = VideoAnalysis.timeArr(video, propertyFile)
        numFrames = len(fullTimeArr)

        # This line will probably take the longest to calculate
        fullScoreArr = np.array([s(video, propertyFile) for s in scores])

        # See return type explanation above for why it has this shape
        inputArr = np.zeros([numSamples, len(scores)+1, sampleLength])
        outputArr = np.zeros(numSamples)

        for i in range(numSamples):
            # Randomly generate an interval of time for the video 
            randomStartFrame = np.random.randint(0, numFrames - sampleLength)
            #randomStartFrame = 100 # TESTING, remove later
            endFrame = randomStartFrame + sampleLength

            # Crop the full array down to the sample
            timeArr = fullTimeArr[randomStartFrame:endFrame]
            scoreArr = fullScoreArr[:,randomStartFrame:endFrame]

            # Save the values for returning
            inputArr[i][0][:] = timeArr
            inputArr[i][1:][:] = scoreArr

            # Next, we have to determine if a failure event happens in the following interval
            failureIntervalStart = timeArr[-1] + 1e-7
            failureIntervalLength = (timeArr[1] - timeArr[0])*failurePredictionIntervalModifier

            eventArr = np.copy(matFile["time"])

            eventArr = eventArr[abs(matFile["deltaF"]) > forceEventThreshhold]
    
            # Fancy way to search for a time between our requested interval. I am not sure if this is actually
            # faster than a regular search, but it is reasonably easy to implement
            eventArr -= failureIntervalStart
            eventArr[eventArr < 0] = 0
            eventArr -= failureIntervalLength
            eventArr[eventArr > 0] = 0
            print(np.unique(eventArr))
            outputArr[i] = (len(np.unique(eventArr)) > 1)

        # Collapse the first dimension if possible (and requested)
        if numSamples == 1 and not preserveArrayType:
            return inputArr[0], outputArr[0]

        return inputArr, outputArr

        # Otherwise return the proper array

    # Do the usual vectorization method using recursion
    # We randomly choose a video in the list, and then perform the random selection for 1 sample from that
    # and then repeat
    returnArr = np.zeros([numSamples, 4, sampleLength])
    for i in range(numSamples):

        clipSelect = np.random.randint(0, len(video))
        returnArr[i] = randomPredictionSample(video[clipSelect], propertyFile[clipSelect], matFile[clipSelect], scores, sampleLength=sampleLength, numSamples=1, failurePredictionIntervalModifier=failurePredictionIntervalModifier)

    return returnArr


def fullPositivePredictionSample(video, propertyFile, matFile, scores, sampleLength=Settings.DEFAULT_SAMPLE_LENGTH, forceEventThreshhold=0):
    """
    video: The video file that we want to sample from; can also be a list of video files

    propertyFile: The properties of the video that we are sampling; can also be a list of
    property files (in that case, should be the same length as video)

    matFile: The file containing the force data from the sensor for the video; can
    also be a list of mat files (in that case, should be the same length as video)

    scores: The function (or list of functions) that will be calculated for each video frame; each function
    should be of the form score(video, propertyFile, **kwargs).

    sampleLength: The number of frames that we want to sample

    return type:

    We return an array of shape (numSamples, 1+len(scores), sampleLength) where the entries in the middle dimension correspond to time
    and whatever other scores are provided

    [0]: time
    [1]: scores[0]
    [2]: scores[1]
    ...
    [n]: scores[-1]
    """

    if not isinstance(video, list):
        
        # Grab this and the requested score beforehand, in case we want multiple
        # samples from this file
        fullTimeArr = VideoAnalysis.timeArr(video, propertyFile)
        numFrames = len(fullTimeArr)

        dt = fullTimeArr[1] - fullTimeArr[0]

        minIntervalDist = dt*(sampleLength+1)

        # This line will probably take the longest to calculate
        fullScoreArr = np.array([s(video, propertyFile) for s in scores])

        # See return type explanation above for why it has this shape
        # We use python lists instead of numpy since we are appending (1, 1, X) length arrays
        inputArr = []
        # Output array is just 1s, so we can make that at the end

        # Grab the failure events that are above the provided force threshhold
        peakTimes = np.array([matFile["time"][i] for i in range(len(matFile["time"])) if (bool(matFile["good"][i]) and abs(matFile["deltaF"][i]) > forceEventThreshhold)])

        isolatedPeakTimes = np.array([])

        # This is how we'll deal with failures intersecting each other
        peakSkip = len(peakTimes)

        # Note that we count backwards here, since we remove times that interfere with others
        for i in range(len(peakTimes)-1, -1, -1):
            #print(f"Current time: {peakTimes[i]}")
            # Skip interfering peaks; see rest of the loop
            if i > peakSkip:
                #print(f"Skippig time: {peakTimes[i]}")
                continue

            # Add this peak since we will just removed anything that interferes with it
            isolatedPeakTimes = np.append(isolatedPeakTimes, peakTimes[i])

            # Next, we check each failure, and make sure there aren't any other failures in the preceding interval
            # We don't want to modify peakTimes itself since we are iterating over it, so instead we record
            # how many entries we need to skip, and carry that out
            for j in range(1, len(peakTimes)-i+1):
                
                # We check sample length + 1 since the failure will be in the interval following the sample
                # (not the last interval)
                #print(peakTimes[i] - peakTimes[i-j])
                if peakTimes[i] - peakTimes[i-j] <  minIntervalDist:
                    # Mark the i-j peak as need to be removed
                    #print(f"Found peak intersecting current time: {peakTimes[i-j]}")
                    peakSkip = i-j-1
                else:
                    # Otherwise, we don't need to keep checking
                    break

            # Now we add this point to the data

            # Find the closest point in the time of the video
            # -1 for the same reason as the sampleLength+1 above, since the failure should be *after* the last interval
            endIndex = np.argmin(np.abs(fullTimeArr - peakTimes[i] - dt/2))
            beginIndex = endIndex - sampleLength

            currInputArr = np.zeros([1+len(scores), sampleLength])
            currInputArr[0,:] = fullTimeArr[beginIndex:endIndex]
            currInputArr[1:,:] = fullScoreArr[:,beginIndex:endIndex]

            #inputArr.append(np.reshape([fullTimeArr[beginIndex:endIndex]], [1, endIndex-beginIndex]) + np.reshape(fullScoreArr[:,beginIndex:endIndex], [len(scores), endIndex-beginIndex]))
            inputArr.append(currInputArr)

            # Now skip the peaks that interfered with this one on the next iteration

        inputArr = np.array(inputArr) 
        outputArr = np.ones(len(inputArr))

        return (inputArr, outputArr)

    # Do the usual vectorization method using recursion
    # We loop through every video and append the results to the master list
    inputArr = []
    outputArr = []
    for i in range(len(video)):

        currInputArr, currOutputArr = fullPositivePredictionSample(video[i], propertyFile[i], matFile[i], scores, sampleLength=sampleLength, forceEventThreshhold=forceEventThreshhold)
        inputArr.append(currInputArr)
        outputArr.append(currOutputArr)

    return (inputArr[0], outputArr[0])

# Method to generate training smaples where a failure does not happen in the following interval
# The majority of randomly selected samples will fit this criteria, so we use random
# generated intervals, and then make sure that there is no failure
def negativePredictionSample(video, propertyFile, matFile, scores, numSamples, sampleLength=Settings.DEFAULT_SAMPLE_LENGTH, forceEventThreshhold=0):
    """
    video: The video file that we want to sample from; can also be a list of video files

    propertyFile: The properties of the video that we are sampling; can also be a list of
    property files (in that case, should be the same length as video)

    matFile: The file containing the force data from the sensor for the video; can
    also be a list of mat files (in that case, should be the same length as video)

    scores: The function (or list of functions) that will be calculated for each video frame; each function
    should be of the form score(video, propertyFile, **kwargs).

    numSamples: The number of samples to generate

    sampleLength: The number of frames that we want to sample

    return type:

    We return an array of shape (numSamples, 1+len(scores), sampleLength) where the entries in the middle dimension correspond to time
    and whatever other scores are provided

    [0]: time
    [1]: scores[0]
    [2]: scores[1]
    ...
    [n]: scores[n]
    """

    if not isinstance(video, list):
        
        fullTimeArr = VideoAnalysis.timeArr(video, propertyFile)
        numFrames = len(fullTimeArr)

        dt = fullTimeArr[1] - fullTimeArr[0]

        minIntervalDist = dt*(sampleLength+1)

        # This line will probably take the longest to calculate
        fullScoreArr = np.array([s(video, propertyFile) for s in scores])

        # See return type explanation above for why it has this shape
        inputArr = np.zeros([numSamples, len(scores)+1, sampleLength])
        outputArr = np.zeros(numSamples)

        # Grab the failure events that are above the provided force threshhold
        peakTimes = np.array([matFile["time"][i] for i in range(len(matFile["time"])) if (bool(matFile["good"][i]) and abs(matFile["deltaF"][i]) > forceEventThreshhold)])

        i = 0
        while i < numSamples:

            # Randomly generate an interval of time for the video 
            randomStartFrame = np.random.randint(0, numFrames - sampleLength)
            #randomStartFrame = 100 # TESTING, remove later
            endFrame = randomStartFrame + sampleLength

            # Crop the full array down to the sample
            timeArr = fullTimeArr[randomStartFrame:endFrame]
            scoreArr = fullScoreArr[:,randomStartFrame:endFrame]

            # Save the values for returning
            inputArr[i][0][:] = timeArr
            inputArr[i][1:][:] = scoreArr

            # Next, we have to determine if a failure event happens anywhere in the interval
            failureIntervalStart = timeArr[0]
            failureIntervalLength = (timeArr[1] - timeArr[0])*(sampleLength+1)

            # Fancy way to search for a time between our requested interval. I am not sure if this is actually
            # faster than a regular search, but it is reasonably easy to implement
            failureInInterval = True in ((peakTimes - failureIntervalStart)[peakTimes - failureIntervalStart > 0] < failureIntervalLength)
            #print((eventArr - failureIntervalStart)[eventArr - failureIntervalStart > 0])
            #print(failureInInterval)
            
            # Assuming there is not a failure, we move on the next
            # Otherwise, the loop repeats and we replace this current one
            if not failureInInterval:
                i += 1

        outputArr = np.zeros(len(inputArr))

        return (inputArr, outputArr)

    # You can't really handle mulitple videos for this method, since it wouldn't
    # make sense to take the same amount of samples from each video
    raise Exception("negativePredictionSample should not be used for multiple videos, use fullPredictionSample instead")

def fullPredictionSample(video, propertyFile, matFile, scores, sampleLength=Settings.DEFAULT_SAMPLE_LENGTH, forceEventThreshhold=0):
    """
    video: The video file that we want to sample from; can also be a list of video files

    propertyFile: The properties of the video that we are sampling; can also be a list of
    property files (in that case, should be the same length as video)

    matFile: The file containing the force data from the sensor for the video; can
    also be a list of mat files (in that case, should be the same length as video)

    scores: The function (or list of functions) that will be calculated for each video frame; each function
    should be of the form score(video, propertyFile, **kwargs).

    sampleLength: The number of frames that we want to sample

    return type:

    We return an array of shape (numSamples, 1+len(scores), sampleLength) where the entries in the middle dimension correspond to time
    and whatever other scores are provided. numSamples is determined by the total number of positive events in the interval

    [0]: time
    [1]: scores[0]
    [2]: scores[1]
    ...
    [n]: scores[-1]
    """

    if not isinstance(video, list):

        # First, we generate all of the positive samples

        # Grab this and the requested score beforehand, in case we want multiple
        # samples from this file
        fullTimeArr = VideoAnalysis.timeArr(video, propertyFile)
        numFrames = len(fullTimeArr)

        dt = fullTimeArr[1] - fullTimeArr[0]

        minIntervalDist = dt*(sampleLength+1)

        # This line will probably take the longest to calculate
        fullScoreArr = np.array([s(video, propertyFile) for s in scores])

        # See return type explanation above for why it has this shape
        # We use python lists instead of numpy since we are appending (1, 1, X) length arrays
        posInputArr = []
        # Output array is just 1s, so we can make that at the end

        # Grab the failure events that are above the provided force threshhold
        peakTimes = np.array([matFile["time"][i] for i in range(len(matFile["time"])) if (bool(matFile["good"][i]) and abs(matFile["deltaF"][i]) > forceEventThreshhold)])

        # This is how we'll deal with failures intersecting each other
        peakSkip = len(peakTimes)

        # Note that we count backwards here, since we remove times that interfere with others
        for i in range(len(peakTimes)-1, -1, -1):
            #print(f"Current time: {peakTimes[i]}")
            # Skip interfering peaks; see rest of the loop
            if i > peakSkip:
                #print(f"Skippig time: {peakTimes[i]}")
                continue

            # Next, we check each failure, and make sure there aren't any other failures in the preceding interval
            # We don't want to modify peakTimes itself since we are iterating over it, so instead we record
            # how many entries we need to skip, and carry that out
            for j in range(1, len(peakTimes)-i+1):
                
                # We check sample length + 1 since the failure will be in the interval following the sample
                # (not the last interval)
                #print(peakTimes[i] - peakTimes[i-j])
                if peakTimes[i] - peakTimes[i-j] <  minIntervalDist:
                    # Mark the i-j peak as need to be removed
                    #print(f"Found peak intersecting current time: {peakTimes[i-j]}")
                    peakSkip = i-j-1
                else:
                    # Otherwise, we don't need to keep checking
                    break

            # Now we add this point to the data

            # Find the closest point in the time of the video
            # -1 for the same reason as the sampleLength+1 above, since the failure should be *after* the last interval
            endIndex = np.argmin(np.abs(fullTimeArr - peakTimes[i] - dt/2))
            beginIndex = endIndex - sampleLength

            currInputArr = np.zeros([1+len(scores), sampleLength])
            currInputArr[0,:] = fullTimeArr[beginIndex:endIndex]
            currInputArr[1:,:] = fullScoreArr[:,beginIndex:endIndex]

            #inputArr.append(np.reshape([fullTimeArr[beginIndex:endIndex]], [1, endIndex-beginIndex]) + np.reshape(fullScoreArr[:,beginIndex:endIndex], [len(scores), endIndex-beginIndex]))
            posInputArr.append(currInputArr)

            # Now skip the peaks that interfered with this one on the next iteration

        posInputArr = np.array(posInputArr) 
        posOutputArr = np.ones(len(posInputArr))

        # Now create an equal number of negative samples

        # See return type explanation above for why it has this shape
        negInputArr = np.zeros([len(posInputArr), len(scores)+1, sampleLength])

        i = 0
        while i < np.shape(negInputArr)[0]:

            # Randomly generate an interval of time for the video 
            randomStartFrame = np.random.randint(0, numFrames - sampleLength)
            #randomStartFrame = 100 # TESTING, remove later
            endFrame = randomStartFrame + sampleLength

            # Crop the full array down to the sample
            timeArr = fullTimeArr[randomStartFrame:endFrame]
            scoreArr = fullScoreArr[:,randomStartFrame:endFrame]

            # Save the values for returning
            negInputArr[i][0][:] = timeArr
            negInputArr[i][1:][:] = scoreArr

            # Next, we have to determine if a failure event happens anywhere in the interval
            failureIntervalStart = timeArr[0]
            failureIntervalLength = (timeArr[1] - timeArr[0])*(sampleLength+1)

            # Fancy way to search for a time between our requested interval. I am not sure if this is actually
            # faster than a regular search, but it is reasonably easy to implement
            failureInInterval = True in ((peakTimes - failureIntervalStart)[peakTimes - failureIntervalStart > 0] < failureIntervalLength)
            #print((eventArr - failureIntervalStart)[eventArr - failureIntervalStart > 0])
            #print(failureInInterval)
            
            # Assuming there is not a failure, we move on the next
            # Otherwise, the loop repeats and we replace this current one
            if not failureInInterval:
                i += 1

        negOutputArr = np.zeros(len(negInputArr))
 
        # Now concatenate the two inputs
        fullInputArr = np.zeros([np.shape(posInputArr)[0]*2, len(scores)+1, sampleLength])
        fullInputArr[:np.shape(posInputArr)[0],:,:] = posInputArr
        fullInputArr[np.shape(posInputArr)[0]:,:,:] = negInputArr
        fullOutputArr = np.append(posOutputArr, negOutputArr)

        # And randomly rearrange them
        randomOrder = np.arange(0, np.shape(fullInputArr)[0])
        np.random.shuffle(randomOrder)

        fullInputArr = fullInputArr[randomOrder]
        fullOutputArr = fullOutputArr[randomOrder]

        return (fullInputArr, fullOutputArr)

    # Do the usual vectorization method using recursion
    # We loop through every video and append the results to the master list
    inputArr = []
    outputArr = []

    for i in range(len(video)):
        currInputArr, currOutputArr = fullPredictionSample(video[i], propertyFile[i], matFile[i], scores, sampleLength=sampleLength, forceEventThreshhold=forceEventThreshhold)
        inputArr.append(currInputArr)
        outputArr.append(currOutputArr)

    totalSamples = sum([len(inputArr[i]) for i in range(len(inputArr))])
    reshapedInputArr = np.zeros([totalSamples, len(scores)+1, sampleLength])
    reshapedOutputArr = np.zeros(totalSamples)

    currIndex = 0

    for i in range(len(inputArr)):
        reshapedInputArr[currIndex:currIndex+len(inputArr[i])] = np.array(inputArr[i])
        reshapedOutputArr[currIndex:currIndex+len(inputArr[i])] = np.array(outputArr[i])
        
        currIndex += len(inputArr[i])
    return (reshapedInputArr, reshapedOutputArr)
