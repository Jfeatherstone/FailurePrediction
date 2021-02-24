import numpy as np
import math

from toolbox import Settings, Preprocessing, VideoAnalysis, Sampling

def generateData(videos, propertyFiles, matFiles, score, numEntries, outputFile, sampleLength, batchIncrement=100, interpolate=True):
    """
    This preprocesses a given set of videos to generate data for ML to work with.

    Since the process of evaluating scores on images takes quite a while, it is
    a good idea to generate this data beforehand, and then mess around with
    hyperparameters separately, so you don't have to regenerate data every time.
    
    This method will save the data to `outputFile` in the following format:

    <numEntries> <sampleLength> <len(videos)> <batchIncrement>
    time <score> force event
    .01 1 1 0
    .02 2 3 0
    .03 1 2 1
    etc.

    """

    # First make sure that data and videos are the same length
    if not isinstance(propertyFiles, list) or not isinstance(videos, list) or not isinstance(matFiles, list):
        raise Exception("Property files, mat files, and/or video variables are not of type 'list'")

    if len(propertyFiles) != len(videos) and len(propertyFiles) != matFiles:
        raise Exception(f"Video, property files, and mat files variables are not the same length: {len(videos)} vs. {len(propertyFiles)} vs. {len(matFiles)}")

    # If batchIncrement is set to none, just do them all in one sweep (may snack on some memory though...)
    if batchIncrement == None:
        batchIncrement = numEntries

    # We might be generating a lot of samples, so it isn't a bad idea to only generate so many at a time, so
    # we don't run out of memory (can be adjusted with batchIncrement)
    numBatches = math.ceil(numEntries / batchIncrement)

    print('Loading data...')

    scoreArr = [[] for i in range(len(videos))]
    videoTimeArr = [[] for i in range(len(videos))]
    # Produce all of the data that could be sampled from
    for i in range(len(videos)):
        videoTimeArr[i] = VideoAnalysis.timeArr(videos[i], propertyFiles[i])
        scoreArr[i] = score(videos[i], propertyFiles[i])

    print('Completed loading data, beginning sampling...')

    # First write the basic stuff at the beginning of the file
    with open(outputFile, 'w') as output:
        output.write(f'{numEntries} {sampleLength} {len(videos)} {batchIncrement}')
        output.write(f'\ntime {score.__name__} force event')

    for i in range(numBatches):
        #samples = Sampling.randomSample(videos, propertyFiles, matFiles, score, interpolate=interpolate, numSamples=batchIncrement, sampleLength=sampleLength)

        with open(outputFile, 'a') as output:
            for j in range(batchIncrement):
                for k in range(sampleLength):
                    output.write(f'\n{samples[j,0,k]} {samples[j,1,k]} {samples[j,2,k]} {samples[j,3,k]}')

    print('Sampling completed!')
    return
