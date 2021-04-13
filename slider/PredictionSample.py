import numpy as np

def predictionSample(fullMetrics, failureIndices, timeArr=None, desiredOutcome=None, sampleLength=20, downsampleFactor=None, randomSeed=21):
    """
    Sample either failures or non-failures (or both) for a trial or set of trials.

    Parameters
    ----------

    fullMetrics : numpy.ndarray or list
        A list or array of values that represent a set of metrics evaluated for a complete trial,
        or a set of trials.
        eg. [[[m1(0,0), m1(0,1), ...], [m2(0,0), m2(0,1), ...]], [[m1(1,0), m1(1,1), ...], [m2(1,0), m2(1,1), ...]]]

    forcePeakIndices : list(int) or numpy.ndarray(int)
        A collection of indices that represent when failure happens, or a list of such collections.

    timeArr : list(float) or numpy.ndarray
        The time axis for the set of metrics, and that correspond to the failureIndices. If this is passed, the resultant
        array will include times as the last element.

    desiredOutcome : int or bool or None
        Whether to sample a particlular trial: samples with failure (1, True) or without failure (0, False). If sampling
        without failure, will randomly generate as many samples as there are failures in failureIndices. If None,
        will return an equal amount of both failure and non-failure samples.

    sampleLength : int
        The length of each sample, in number of frames or time steps. If downsampling (see downsampleFactor), the length of
        samples will be exactly this value, though the timesteps will be more spaced out.

    downsampleFactor : int or None
        The degree to which we should downsample the data.

    randomSeed : int
        The seed to random generation, which is required for generating the non-failure samples


    Returns
    -------

    numpy.ndarray : Array of shape (numSamples, len(fullMetrics), sampleLength) or (numSamples, len(fullMetrics)+1, sampleLength),
        depending on whether a timeArr is passed to the method. The entries in the middle dimension correspond to
        the metrics that are provided, and possibly the time. numSamples is determined by the total number of positive events in the interval.

        list(i,0,:): 0th metric for ith sample
        list(i,1,:): 1st metric for ith sample
        list(i,2,:): 2nd metric for ith sample
        ...
        list(i,-1,:): time axis for ith sample (if timeArr != None)
    """

    # If the first element of the list of indices is not a list itself, we only have
    # a single trial
    if not isinstance(failureIndices[0], np.ndarray) and not isinstance(failureIndices[0], list):
        # Handle this so we don't have to catch None later
        if downsampleFactor == None:
            downsampleFactor = 1

        # First, generate the positive samples since they are predetermined by the locations
        # of the failures

        # Make sure that there aren't any failures that are too close to each other
        minDistance = (sampleLength+1)*downsampleFactor

        # If we do have an overlap, we'll have to get rid of one of the failures
        # We want to take the one that doesn't have another failure in the middle of it, which
        # amounts to being the first one
        # We always include the first failure
        confirmedFailureIndices = [failureIndices[0]] + [failureIndices[i] for i in range(1, len(failureIndices)) if failureIndices[i]-failureIndices[i-1] > minDistance]
        # Adjust for the downsampling factor
        # For the failure indices, we just divide by the factor (and cast to int)
        confirmedFailureIndices = np.array([int(i/downsampleFactor) for i in confirmedFailureIndices])
        # For the metrics, we just use index slicing
        downsampledFullMetrics = np.array([metric[::downsampleFactor] for metric in fullMetrics])

        hasTimeArr = isinstance(timeArr, list) or isinstance(timeArr, np.ndarray)
        if hasTimeArr:
            downsampledTimeArr = timeArr[::downsampleFactor]

        failureInputArr = []
        inputArrLen = len(fullMetrics) + int(hasTimeArr)
        numTimeSteps = len(downsampledFullMetrics[0])

        # Note that we count backwards here, since we remove times that interfere with others
        for i in range(len(confirmedFailureIndices)):

            # Now we add this point to the data

            # Find the closest point in the time of the video
            # -1 for the same reason as the sampleLength+1 above, since the failure should be *after* the last interval
            endIndex = confirmedFailureIndices[i] - 1
            beginIndex = endIndex - sampleLength
            currInputArr = np.zeros([inputArrLen, sampleLength])

            for j in range(len(downsampledFullMetrics)):
                currInputArr[j,:] = downsampledFullMetrics[j,beginIndex:endIndex]

            if hasTimeArr:
                currInputArr[-1,:] = downsampledTimeArr[beginIndex:endIndex]

            failureInputArr.append(currInputArr)


        # The inputs (for our model eventually) are the metrics, and the outputs will be one
        # since a failure happens in the interval
        failureInputArr = np.array(failureInputArr) 
        failureOutputArr = np.ones(len(failureInputArr))
    
        # Return if we only wanted failure samples
        if desiredOutcome == True or desiredOutcome == 1:
            return (failureInputArr, failureOutputArr)

        # Now create an equal number of negative samples

        # See return type explanation above for why it has this shape
        nonfailureInputArr = np.zeros([len(failureInputArr), inputArrLen, sampleLength])

        i = 0
        # Seed our random generation, so we can test with consistent results
        np.random.seed(randomSeed)
        while i < np.shape(nonfailureInputArr)[0]:

            # Randomly generate an interval of time for the video 
            randomBeginIndex = np.random.randint(0, numTimeSteps - sampleLength)
            #randomStartFrame = 100 # TESTING, remove later
            endIndex = randomBeginIndex + sampleLength

            # Crop the full array down to the sample
            for j in range(len(downsampledFullMetrics)):
                nonfailureInputArr[i,j,:] = downsampledFullMetrics[j,randomBeginIndex:endIndex]

            if hasTimeArr:
                nonfailureInputArr[i,-1,:] = downsampledTimeArr[randomBeginIndex:endIndex]

            # Next, we have to determine if a failure event happens anywhere in the interval

            # Fancy way to search for a time between our requested interval. I am not sure if this is actually
            # faster than a regular search, but it is reasonably easy to implement
            failureInInterval = True in ((confirmedFailureIndices - randomBeginIndex)[confirmedFailureIndices - randomBeginIndex > 0] < sampleLength)
            #print((eventArr - failureIntervalStart)[eventArr - failureIntervalStart > 0])
            #print(failureInInterval)
            
            # Assuming there is not a failure, we move on the next
            # Otherwise, the loop repeats and we replace this current one
            if not failureInInterval:
                i += 1

        nonfailureOutputArr = np.zeros(len(nonfailureInputArr))
 
        # Return if we only wanted nonfailure samples
        if desiredOutcome == False or desiredOutcome == 0:
            return (nonfailureInputArr, nonfailureOutputArr)

        # Now concatenate the two inputs
        fullInputArr = np.zeros([np.shape(failureInputArr)[0]*2, inputArrLen, sampleLength])
        fullInputArr[:np.shape(failureInputArr)[0],:,:] = failureInputArr
        fullInputArr[np.shape(nonfailureInputArr)[0]:,:,:] = nonfailureInputArr
        fullOutputArr = np.append(failureOutputArr, nonfailureOutputArr)

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

    for i in range(len(fullMetrics)):

        # We have to catch any None type issues, since otherwise we'll
        # try to index None
        if isinstance(timeArr, list) or isinstance(timeArr, np.ndarray):
            currTimeArr = timeArr[i]
        else:
            currTimeArr = None
        currInputArr, currOutputArr = predictionSample(fullMetrics[i], failureIndices[i], timeArr=currTimeArr,
                                                       desiredOutcome=desiredOutcome, sampleLength=sampleLength, downsampleFactor=downsampleFactor,
                                                       randomSeed=randomSeed)
        inputArr.append(currInputArr)
        outputArr.append(currOutputArr)

    inputArrLen = len(fullMetrics[0]) + int(isinstance(timeArr, list))
    totalSamples = sum([len(inputArr[i]) for i in range(len(inputArr))])
    reshapedInputArr = np.zeros([totalSamples, inputArrLen, sampleLength])
    reshapedOutputArr = np.zeros(totalSamples)

    currIndex = 0

    for i in range(len(inputArr)):
        reshapedInputArr[currIndex:currIndex+len(inputArr[i])] = np.array(inputArr[i])
        reshapedOutputArr[currIndex:currIndex+len(inputArr[i])] = np.array(outputArr[i])
        
        currIndex += len(inputArr[i])
    return (reshapedInputArr, reshapedOutputArr)
