import numpy as np

from toolbox import Settings, Preprocessing, VideoAnalysis, Sampling

# Randomly sample a portion of a single video including all failure events that happen in the interval
def randomIntervalSample(video, propertyFile, matFile, score, sampleLength=Settings.DEFAULT_SAMPLE_LENGTH, numSamples=1, preserveArrayType=False, downsampleMethod=Sampling.timeAlignmentDownsample, interpolate=True, subtractForceMean=True):
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

    downsampleMethod: How we should downsample the high res force data. timeAlignmentDownsample has been shown to be significantly better than
    the other two I have implemented, so that one should almost always be used. For more info, see LaBlog around the beginning of October 2020

    interpolate: Whether we interpolate when using the timeAlignmentDownsample method. If a different downsampling method is specified, this
    parameter does nothing

    return type:

    We return an array of shape (numSamples, 4, sampleLength) where the 4 entries in the middle dimension are:

    [0]: time
    [1]: score
    [2]: forceSensorReading
    [3]: eventsInInterval

    Note that the first dimension may be collapsed if numSamples = 1; see preserveArrayType for more info
    """


    if not isinstance(video, list) and not isinstance(propertyFile, list):

        
        # Grab this and the requested score beforehand, in case we want multiple
        # samples from this file
        fullTimeArr = VideoAnalysis.timeArr(video, propertyFile)
        fullScoreArr = score(video, propertyFile)
        numFrames = len(fullTimeArr)

        # See return type explanation above for why it has this shape
        returnArr = np.zeros([numSamples, 4, sampleLength])

        for i in range(numSamples):
            # Randomly generate an interval of time for the video 
            randomStartFrame = np.random.randint(0, numFrames - sampleLength)
            #randomStartFrame = 100 # TESTING, remove later
            endFrame = randomStartFrame + sampleLength

            # Crop the full array down to the sample
            timeArr = fullTimeArr[randomStartFrame:endFrame]
            scoreArr = fullScoreArr[randomStartFrame:endFrame]

            avgTimeDiff = np.mean([timeArr[i] - timeArr[i+1] for i in range(len(timeArr)-1)])

            #print(timeArr[-1] - timeArr[0], len(timeArr), avgTimeDiff)
            # Next up we need to grab the force sensor data
            # This is a little more involved than the previous steps since it
            # has a different temporal resolution than the videos
            # This means that we have to do a little bit of index magic
            
            # The force data is taken in intervals of Settings.FORCE_SENSOR_DT (.01 as of now)
            # so we have to find the index of the beginning time
            #beginIndex = int((timeArr[0] - matFile["t"][0])/Settings.FORCE_SENSOR_DT)
            beginIndex = int(timeArr[0] / Settings.FORCE_SENSOR_DT)

            # For our kernel, we take the difference in time between frames divided
            # by the difference in time between measurements
            downsampleKernel = (timeArr[1] - timeArr[0])/Settings.FORCE_SENSOR_DT
            # We take sampleLength-1 because we already have the first point, we need
            # sampleLength-1 more
            endIndex = int(beginIndex + downsampleKernel * (sampleLength-1))

            # So I tried to make all of the downsampling methods have the same argument structure, but since
            # the time alignment one is so different, it's just not possible
            # Because of this, we have to separately evaluate the downsampling if we are using this one
            if downsampleMethod == Sampling.timeAlignmentDownsample:
                # Since the time alignment method finds the t value that is closest
                # to the one in timeArr, I've found it works best to have a little bit
                # of padding on each high res array. The extraneous values won't make a 
                # difference if they aren't used, since each value is picked out, not just sliced
                padding = min(10, len(matFile["f"]))
                timeSensorArr, forceSensorArr = downsampleMethod(matFile["t"][beginIndex:endIndex+padding], matFile["f"][beginIndex:endIndex+padding], timeArr, int(downsampleKernel), debug=True, interpolate=interpolate)
            else:
                forceSensorArr = downsampleMethod(matFile["f"][beginIndex:endIndex], int(downsampleKernel))
                timeSensorArr = downsampleMethod(matFile["t"][beginIndex:endIndex], int(downsampleKernel))

            # The next task is to get a list of detected force events that happen within our sampled interval
            # Every time in matFile["time"] should line up with a point in matFile["t"], so we just have to
            # walk through the times

            # First we combine the information across a few different arrays
            eventTimes = matFile["time"]
            eventMagnitudes = matFile["deltaF"] * matFile["good"]
    
            eventsInInterval = np.zeros(sampleLength)

            for k in range(len(eventTimes)):
                if eventTimes[k] > timeArr[0] and eventTimes[k] < timeArr[-1]:
                    # Save it in the closest time to the actual event time
                    eventsInInterval[np.argmin(abs(timeArr - eventTimes[k]))] = eventMagnitudes[k]

            if subtractForceMean:
                forceSensorArr = forceSensorArr - np.mean(forceSensorArr)

            # Save the values for returning
            returnArr[i][0][:] = timeArr[:]
            returnArr[i][1][:] = scoreArr[:]
            returnArr[i][2][:] = forceSensorArr[:]
            returnArr[i][3][:] = eventsInInterval[:]

        # Collapse the first dimension if possible (and requested)
        if numSamples == 1 and not preserveArrayType:
            return returnArr[0]

        return returnArr

        # Otherwise return the proper array

    # Do the usual vectorization method using recursion
    # We randomly choose a video in the list, and then perform the random selection for 1 sample from that
    # and then repeat
    returnArr = np.zeros([numSamples, 4, sampleLength])
    for i in range(numSamples):

        clipSelect = np.random.randint(0, len(video))
        returnArr[i] = randomSample(video[clipSelect], propertyFile[clipSelect], matFile[clipSelect], score, sampleLength=sampleLength, numSamples=1, downsampleMethod=timeAlignmentDownsample, interpolate=interpolate)

    return returnArr

# Randomly sample a portion of a single video, optimized for a large number of samples
# Compared to the previous method, we now expect an array of the scores to be passed in for
# every video for the entire length, and then we crop after that
def bulkRandomIntervalSample(propertyFiles, matFiles, scoreArr, videoTimeArr, numSamples, sampleLength=Settings.DEFAULT_SAMPLE_LENGTH, downsampleMethod=Sampling.timeAlignmentDownsample, interpolate=True, subtractForceMean=True):
    """
    scoreArr: The desired score evaluated for every video for the entire duration

    videoTimeArr: The times for each of the videos created using VideoAnalysis.timeArr

    propertyFile: The properties of the videos that we are sampling

    matFile: The file containing the force data from the sensor for the videos

    sampleLength: The number of frames that we want to sample

    numSamples: The number of samples we want to generate

    downsampleMethod: How we should downsample the high res force data. timeAlignmentDownsample has been shown to be significantly better than
    the other two I have implemented, so that one should almost always be used. For more info, see LaBlog around the beginning of October 2020

    interpolate: Whether we interpolate when using the timeAlignmentDownsample method. If a different downsampling method is specified, this
    parameter does nothing

    return type:

    We return an array of shape (numSamples, 4, sampleLength) where the 4 entries in the middle dimension are:

    [0]: time
    [1]: score
    [2]: forceSensorReading
    [3]: eventsInInterval

    Note that the first dimension may be collapsed if numSamples = 1; see preserveArrayType for more info
    """

    # Grab this and the requested score beforehand, in case we want multiple
    # samples from this file

    returnArr = np.zeros([numSamples, 4, sampleLength])
    for i in range(numSamples):

        clipSelect = np.random.randint(0, np.shape(videoTimeArr)[0])

        numFrames = len(videoTimeArr[clipSelect])

        # Randomly generate an interval of time for the video 
        randomStartFrame = np.random.randint(0, numFrames - sampleLength)
        #randomStartFrame = 100 # TESTING, remove later
        endFrame = randomStartFrame + sampleLength

        # Crop the full array down to the sample
        timeArr = videoTimeArr[clipSelect,randomStartFrame:endFrame]
        scoreArr = scoreArr[clipSelect,randomStartFrame:endFrame]

        avgTimeDiff = np.mean([timeArr[j] - timeArr[j+1] for j in range(len(timeArr)-1)])

        # Downsampling the force data 
        beginIndex = int(timeArr[0] / Settings.FORCE_SENSOR_DT)
        # For our kernel, we take the difference in time between frames divided
        # by the difference in time between measurements
        downsampleKernel = (timeArr[1] - timeArr[0])/Settings.FORCE_SENSOR_DT
        # We take sampleLength-1 because we already have the first point, we need
        # sampleLength-1 more
        endIndex = int(beginIndex + downsampleKernel * (sampleLength-1))

        # So I tried to make all of the downsampling methods have the same argument structure, but since
        # the time alignment one is so different, it's just not possible
        # Because of this, we have to separately evaluate the downsampling if we are using this one
        if downsampleMethod == Sampling.timeAlignmentDownsample:
            # Since the time alignment method finds the t value that is closest
            # to the one in timeArr, I've found it works best to have a little bit
            # of padding on each high res array. The extraneous values won't make a 
            # difference if they aren't used, since each value is picked out, not just sliced
            padding = min(10, len(matFiles[clipSelect]["f"]))
            timeSensorArr, forceSensorArr = downsampleMethod(matFiles[clipSelect]["t"][beginIndex:endIndex+padding], matFiles[clipSelect]["f"][beginIndex:endIndex+padding], timeArr, int(downsampleKernel), debug=True, interpolate=interpolate)
        else:
            forceSensorArr = downsampleMethod(matFiles[clipSelect]["f"][beginIndex:endIndex], int(downsampleKernel))
            timeSensorArr = downsampleMethod(matFiles[clipSelect]["t"][beginIndex:endIndex], int(downsampleKernel))

        # The next task is to get a list of detected force events that happen within our sampled interval
        # Every time in matFile["time"] should line up with a point in matFile["t"], so we just have to
        # walk through the times

        # First we combine the information across a few different arrays
        eventTimes = matFiles[clipSelect]["time"]
        eventMagnitudes = matFiles[clipSelect]["good"]

        eventsInInterval = np.zeros(sampleLength)

        for k in range(len(eventTimes)):
            if eventTimes[k] > timeArr[0] and eventTimes[k] < timeArr[-1]:
                # Save it in the closest time to the actual event time
                eventsInInterval[np.argmin(abs(timeArr - eventTimes[k]))] = eventMagnitudes[k]

        if subtractForceMean:
            forceSensorArr = forceSensorArr - np.mean(forceSensorArr)

        # Save the values for returning
        returnArr[i][0][:] = timeArr[:]
        returnArr[i][1][:] = scoreArr[:]
        returnArr[i][2][:] = forceSensorArr[:]
        returnArr[i][3][:] = eventsInInterval[:]

    return returnArr

