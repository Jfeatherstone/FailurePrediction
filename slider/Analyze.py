import numpy as np

import cv2

import pickle
import os
import progressbar

import networkx as nx

from slider import Settings
from slider import ImageAnalysis, ParticleAnalysis, NetworkAnalysis, CommunityAnalysis

def checkImageType(frame):
    """
    Make sure that the image is a proper image, and not a path
    """
    if isinstance(frame, str):
        # I don't want to overwrite the image itself, so create a new var for that
        newFrame = np.array(cv2.imread(frame))
    else:
        newFrame = frame

    return newFrame


def analyze(photoelasticImageData, trackingData, metrics, metadata=None, loadCachedResults=True, saveCachedResults=True, progress=True, DATA_LOCATION=Settings.DATA_LOCATION, CACHE_DIR=Settings.CACHE_DIR, ANALYSIS_DIR=Settings.ANALYSIS_DIR):
    """
    A pipeline to apply a set of analysis methods (metrics) to a given trial (or set of trials).

    Parameters
    ----------

    photoelasticImageData : numpy.array or str list(numpy.array) or list(str)
        The photoelastic image data (or path to the image), or a list of images/paths

    trackingData : numpy.array or list(numpy.array)
        The particle tracking data for a trial (or trails) of the form (pos_x, pos_y, rot_angle, radius).

    metrics : func or list(func)
        A list of metrics to evaluate for the given data set (or sets). Note that every method must
        have the attribute .analysis_type to properly determine the method signature (and what data to pass).

    metadata : dict or list(dict)
        Dictionary containing information about the trial name, for use in saving/loading cached data. If
        not passed, saveCachedResults and loadCachedResults will automatically be set to False, regardless
        of kwargs value. For format, see return type of Preprocessing.loadSliderData(returnMetadata=True).

    loadCachedResults : bool
        Whether to try and load precomputed results (True) or not (False). If they do not exist, calculation
        will proceed as normal

    saveCachedResults : bool
        Whether to save the computed results (True) or not (false).

    progress : bool
        Whether to show a progress bar (True) or not (False) while computing metrics.

    DATA_LOCATION : str
        The root directory where the data is stored, for use in saving/loading cached results.

    CACHE_DIR : str
        The folder name (not full path) of the directory that may be used to save/load the cached results.

    ANALYSIS_DIR : str
        The folder name (not full path) of the directory within CACHE_DIR that may be used to save/load the
        cached results.

    Returns
    -------

    numpy.ndarray : Array of all metrics evaluated for all frames, in the same order they are passed in.
        eg. metrics = [m1, m2] -> [[m1(step[0]), m1(step[1]), ...], [m2(step[0]), m2(step[1]), ...]]

        In the case that a set of trials are passed, return will simply be a list of arrays described above.
    """

    # Make sure we have a list of metrics, even if there is only a single item
    if not isinstance(metrics, list):
        metrics = [metrics]

    # We want to make sure all of the metrics are proper
    # This means that they have a defined analysis type
    confirmedMetrics = []
    for i in range(len(metrics)):
        if not hasattr(metrics[i], 'analysis_type'):
            print(f'Invalid metric passed to analyze ({metrics[i].__name__}); does not have attribute \"analysis_type\"!')
            continue

        confirmedMetrics.append(metrics[i])
        # We want to avoid doing extra work, like reading in an image if
        # we don't actually have any metrics for images.


    # Can't load/save the results if we aren't passed metadata
    if not isinstance(metadata, list) and not isinstance(metadata, dict):
        if loadCachedResults or saveCachedResults:
            print('Warning: cannot save/load results if not passed trial metadata, skipping this step!')
        loadCachedResults = False
        saveCachedResults = False

    # Make sure the cache folder exists
    # And if it doesn't exist yet, we of course can't read any cached results
    if loadCachedResults or saveCachedResults:

        # We can't read in cached data if the folder doesn't currently exist
        if not os.path.exists(DATA_LOCATION + CACHE_DIR):
            loadCachedResults = False

            os.mkdir(DATA_LOCATION + CACHE_DIR)

        # Now make the subdirectory
        if not os.path.exists(DATA_LOCATION + CACHE_DIR + ANALYSIS_DIR):
            os.mkdir(DATA_LOCATION + CACHE_DIR + ANALYSIS_DIR)
            loadCachedResults = False

    # Now onto the actual function definition, which is called recursively later
    # If we are passed a single trial
    if isinstance(trackingData[0], np.ndarray):

        numFrames = len(photoelasticImageData)

        # The object that we will eventually return: a list of the metrics
        # evaluated for each frame
        finalMetrics = np.zeros([len(confirmedMetrics), numFrames])
        # We also need to keep track of indices for the metrics, since some may
        # be read in from a cached file, and others will have to be calculated,
        # but we want to return them in exactly the same order they were passed
        metricIndices = list(np.arange(len(confirmedMetrics)))

        if loadCachedResults:
        # Counting backwards, so we can remove metrics if need be
            for i in range(len(confirmedMetrics)-1, -1, -1):
                # First, check if we have a cached version of the data (and we want to use it)
                # The name of the file will be <trial_name>_<metric_name>_<len(trial)>.pickle
                # The reason for the length is to ensure that we don't load in a cached version
                # of the data if we have downsampled or anything, since it wouldn't reflect that
                fileName = f'{metadata["trial_name"]}_{confirmedMetrics[i].__name__}_{numFrames}.pickle'
                possibleCacheFile = DATA_LOCATION + CACHE_DIR + ANALYSIS_DIR + fileName
                
                # If the file exists and we want to read it in, do so and remove the metric
                # from our list, since we don't need to calculate it anymore
                if os.path.exists(possibleCacheFile):
                    with open(possibleCacheFile, 'rb') as cacheFile:
                        print(f'Reading analysis result from cache: {CACHE_DIR + ANALYSIS_DIR + fileName}')
                        finalMetrics[i] = pickle.load(cacheFile)

                    # Now take the metric out of our list (and our list of indices)
                    del metricIndices[i]
                    del confirmedMetrics[i]
                   
        # Now we determine which types of analysis we will have to run
        # These lines look a little scary, but really we're just splitting up our
        # metrics (and keeping track of the index) by analysis type
        imageAnalysisMetrics = [(confirmedMetrics[i], metricIndices[i]) for i in range(len(confirmedMetrics)) if confirmedMetrics[i].analysis_type == Settings.IMAGE_ANALYSIS]
        particleAnalysisMetrics = [(confirmedMetrics[i], metricIndices[i]) for i in range(len(confirmedMetrics)) if confirmedMetrics[i].analysis_type == Settings.PARTICLE_ANALYSIS]
        networkAnalysisMetrics = [(confirmedMetrics[i], metricIndices[i]) for i in range(len(confirmedMetrics)) if confirmedMetrics[i].analysis_type == Settings.NETWORK_ANALYSIS]
        communityAnalysisMetrics = [(confirmedMetrics[i], metricIndices[i]) for i in range(len(confirmedMetrics)) if confirmedMetrics[i].analysis_type == Settings.COMMUNITY_ANALYSIS]
        # Now just calculate the metrics for any that could not be loaded in
        # Iterate over all timesteps for the trial
        # It is likely more efficient to run each type of analysis separately, since
        # checking an if statement every iteration is likely more expensive than
        # running 3 separate loops
        # IMAGE
        if len(imageAnalysisMetrics) > 0:
            if progress:
                print('Performing image analysis:')
                bar = progressbar.ProgressBar(max_value=numFrames)

            for i in range(numFrames):
                # Make sure we have a proper image frame (in case we were passed a path)
                frame = checkImageType(photoelasticImageData[i])

                for j in range(len(imageAnalysisMetrics)):
                    # The first index is to make sure we are indexing correctly
                    # since some metrics may have been resolved by caching
                    finalMetrics[imageAnalysisMetrics[j][1],i] = imageAnalysisMetrics[j][0](frame)

                if progress:
                    bar.update(i)

        # PARTICLE
        if len(particleAnalysisMetrics) > 0:
            if progress:
                print('Performing particle analysis:')
                bar = progressbar.ProgressBar(max_value=numFrames)

            for i in range(numFrames):
                # Reorganize the tracking data passed in
                pos_x = trackingData[i][:,0]
                pos_y = trackingData[i][:,1]
                rot_angle = trackingData[i][:,2]
                radii = trackingData[i][:,3]

                for j in range(len(particleAnalysisMetrics)):
                    # The first index is to make sure we are indexing correctly
                    # since some metrics may have been resolved by caching
                    finalMetrics[particleAnalysisMetrics[j][1],i] = particleAnalysisMetrics[j][0](pos_x, pos_y, rot_angle, radii)

                if progress:
                    bar.update(i)

        # NETWORK
        if len(networkAnalysisMetrics) > 0:
            if progress:
                print('Performing network analysis:')
                bar = progressbar.ProgressBar(max_value=numFrames)

            for i in range(numFrames):
                # TODO: calculate the network for this frame, given the particle positions and radii
                # network = NetworkAnalysis.genNetwork(pos_x, pos_y, radii)
                pass

                if progress:
                    bar.update(i)

        # COMMUNITY
        if len(communityAnalysisMetrics) > 0:
            if progress:
                print('Performing community analysis:')
                bar = progressbar.ProgressBar(max_value=numFrames)

            for i in range(numFrames):
                pos_x = trackingData[i][:,0]
                pos_y = trackingData[i][:,1]
                radii = trackingData[i][:,3]
                frame = checkImageType(photoelasticImageData[i])

                graph = nx.from_numpy_array(NetworkAnalysis.genWeightedAdjacencyMatrix(pos_x, pos_y, radii, frame))

                # The number of detections to average over
                numDetections = 5
                for k in range(numDetections):
                    partition = CommunityAnalysis.genCommunityDetection(graph)

                    for j in range(len(communityAnalysisMetrics)):
                        finalMetrics[communityAnalysisMetrics[j][1],i] += communityAnalysisMetrics[j][0](partition)/numDetections

                if progress:
                    bar.update(i)

        # If we want to save the results we just calculated, do that
        if saveCachedResults:
            # The folders should already all exist (see above)
            numFrames = len(photoelasticImageData)
            for i in range(len(confirmedMetrics)):
                fileName = f'{metadata["trial_name"]}_{confirmedMetrics[i].__name__}_{numFrames}.pickle'
                newCacheFile = DATA_LOCATION + CACHE_DIR + ANALYSIS_DIR + fileName
                with open(newCacheFile, 'wb') as cacheFile:
                    pickle.dump(finalMetrics[metricIndices[i]], cacheFile)

        # Finally, we return the data
        return finalMetrics


    # Now for the recursive call, in case we are passed a list of trials

    finalMetrics = []
    for k in range(len(photoelasticImageData)):
        # There is a possibility that no metadata was passed, so we want
        # to make sure we don't try to index None 
        if isinstance(metadata, list):
            currMetadata = metadata[k]
        else:
            currMetadata = None
        newMetrics = analyze(photoelasticImageData[k], trackingData[k], metrics, metadata=currMetadata, loadCachedResults=loadCachedResults, progress=progress,
                             saveCachedResults=saveCachedResults, DATA_LOCATION=DATA_LOCATION, CACHE_DIR=CACHE_DIR, ANALYSIS_DIR=ANALYSIS_DIR)
        finalMetrics.append(newMetrics)

    return finalMetrics
