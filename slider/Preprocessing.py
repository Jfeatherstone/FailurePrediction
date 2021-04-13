import numpy as np
from scipy.signal import find_peaks

import os
import progressbar
import pickle

import cv2

from slider import Settings

def loadSliderData(DATA_LOCATION=Settings.DATA_LOCATION, IMAGE_DIR=Settings.IMAGE_DIR, FORCE_DIR=Settings.FORCE_DIR, TRACKING_DIR=Settings.TRACKING_DIR, progress=True, loadImages=False, returnMetadata=True, loadCachedData=False, saveCachedData=False, CACHE_DIR=Settings.CACHE_DIR):
    r"""
    Load in the slider data from a collection of folders, including the force time-series,
    the particle tracking data, and the images.

    Parameters
    ----------
    
    DATA_LOCATION : str
        The full path for the root directory that contains the folders IMAGE_DIR, FORCE_DIR, and TRACKING_DIR.

    IMAGE_DIR : str
        The folder name (not full path) of the directory that contains the image directories for each run.

    FORCE_DIR : str
        The folder name (not full path) of the directory that contains the force files for each run.

    TRACKING_DIR : str
        The folder name (not full path) of the directory that contains directories with tracking data for each run.

    progress : bool
        Whether to display a progress bar for reading in the images/tracking data (True, default) or not (False) (since it takes a while).

    loadImages : bool
        Whether to return the paths to each image (False, default) or actually load in the images (True). Due to the huge
        volume of images, it is not recommend to load them in unless you immediately plan to process them.

    returnMetadata : bool
        Whether to return metadata about the trials (True, default) or not (False)

    loadCachedData : bool
        Whether to try and load previously cached data from CACHE_DIR using pickle (True) or not (False, default). If this is
        set to True, but the cache data does not exist, the data will be read as usual from the raw files.

    saveCachedData : bool
        Whether to save the data afterwards in a cached form using pickle (True) or not (False, default). If True, the data
        will be saved inside the CACHE_DIR folder.

    CACHE_DIR : str
        The folder name (not full path) of the directory that may be used to read/save the cached data

    Returns
    -------

    list(0) : Photoelastic images (or paths; see loadImages)
    
    list(1) : White light images (or paths; see loadImages)

    list(2) : Force data

    list(3) : Tracking data

    list(4) : Trial metadata (only if returnMetadata=True)

    """
    imageFiles = os.listdir(DATA_LOCATION + IMAGE_DIR)
    # Find the index of the run by cleaning extra text from the name
    indices = [ele.replace('LabFrame', '').replace('_', '').replace('shape0', '') for ele in imageFiles]

    # Find where the force and position data should be
    predictedForceFiles = [f'shape0_{index}_FrameTimeForce.txt' for index in indices]
    actualForceFiles = os.listdir(DATA_LOCATION + FORCE_DIR)

    predictedTrackingFiles = [f'shape0_{index}' for index in indices]
    actualTrackingFiles = os.listdir(DATA_LOCATION + TRACKING_DIR)

    # Make sure the cache directories actually exist
    if saveCachedData or loadCachedData:
        if not os.path.exists(DATA_LOCATION + CACHE_DIR):
            # We can't read in cached data if the data doesn't currently exist
            loadCachedData = False

            # Next, create that folder if we are later going to save something in it
            os.mkdir(DATA_LOCATION + CACHE_DIR)

        # Now make the subdirectories
        for folder in [IMAGE_DIR, FORCE_DIR, TRACKING_DIR]:
            if not os.path.exists(DATA_LOCATION + CACHE_DIR + folder):
                os.mkdir(DATA_LOCATION + CACHE_DIR + folder)

    photoelasticImageData = []
    whiteLightImageData = []
    forceData = []
    trackingData = []

    for i in range(len(indices)):
        # Make sure the files actually exist
        if predictedForceFiles[i] in actualForceFiles and predictedTrackingFiles[i] in actualTrackingFiles:
            # If they do, we can save them as a proper run

            # The force data is very easy to read in, we don't really need to change it at all
            forceData.append(np.genfromtxt(DATA_LOCATION + FORCE_DIR + predictedForceFiles[i]))

            # This one can be a little large, so we may want to try to read in a cached version
            possibleCachedFile = DATA_LOCATION + CACHE_DIR + TRACKING_DIR + predictedTrackingFiles[i] + '.pickle'
            if loadCachedData and os.path.exists(possibleCachedFile):
                print(f'Reading tracking data from cache: {CACHE_DIR + TRACKING_DIR + predictedTrackingFiles[i]}.pickle')
                # Read in the pickle file
                with open(possibleCachedFile, 'rb') as cacheFile:
                    currTrackingData = pickle.load(cacheFile)
            else: 
                # The particle tracking data is also not too hard to read in, though we do have
                # to look at every file in the given folder
                # Make sure to sort them, so that we have the particle positions in order
                filesInFolder = np.sort(os.listdir(DATA_LOCATION + TRACKING_DIR + predictedTrackingFiles[i]))
                currTrackingData = [[] for j in range(len(filesInFolder))]
                if progress:
                    print(f'Reading tracking data from directory: {TRACKING_DIR + predictedTrackingFiles[i]}')
                    bar = progressbar.ProgressBar(max_value=len(filesInFolder))

                for j in range(len(filesInFolder)):
                    currTrackingData[j] = np.genfromtxt(DATA_LOCATION + TRACKING_DIR + predictedTrackingFiles[i] + '/' + filesInFolder[j])
                    if progress:
                        bar.update(j)
                    
            trackingData.append(currTrackingData)

            # And if we want to save the file, we do that
            if saveCachedData:
                with open(possibleCachedFile, 'wb') as cacheFile:
                    pickle.dump(currTrackingData, cacheFile)

            # Now for the image data
            # There is a ton of info here, so it may be a little heavy on memory
            # Because of this, the default option is to instead just return the path to each
            # image, and we can load them in as we actually need them
            filesInFolder = np.sort(os.listdir(DATA_LOCATION + IMAGE_DIR + imageFiles[i]))
            photoelasticFilesInFolder = [f for f in filesInFolder if 'P' in f]
            whiteLightFilesInFolder = [f for f in filesInFolder if 'N' in f]
            
            currPhotoelasticImages = [[] for j in range(len(photoelasticFilesInFolder))]
            currWhiteLightImages = [[] for j in range(len(whiteLightFilesInFolder))]

            if not loadImages:
                for j in range(len(photoelasticFilesInFolder)):
                    currPhotoelasticImages[j] = DATA_LOCATION + IMAGE_DIR + imageFiles[i] + '/' + photoelasticFilesInFolder[j]
                for j in range(len(whiteLightFilesInFolder)):
                    currWhiteLightImages[j] = DATA_LOCATION + IMAGE_DIR + imageFiles[i] + '/' + whiteLightFilesInFolder[j]

            else:
                # Warning that this will take a *very* long time to load and is not a good
                # idea if you are messing around with the data at all
                if progress:
                    print(f'Reading PE images from directory: {IMAGE_DIR + imageFiles[i]}')
                    bar = progressbar.ProgressBar(max_value=len(photoelasticFilesInFolder))

                for j in range(len(photoelasticFilesInFolder)):
                    currPhotoelasticImages[j] = np.array(cv2.imread(DATA_LOCATION + IMAGE_DIR + imageFiles[i] + '/' + photoelasticFilesInFolder[j]))
                    if progress:
                        bar.update(j)
                        
                if progress:
                    print(f'Reading WL images from directory: {IMAGE_DIR + imageFiles[i]}')
                    bar = progressbar.ProgressBar(max_value=len(whiteLightFilesInFolder))

                for j in range(len(whiteLightFilesInFolder)):
                    currWhiteLightImages[j] = np.array(cv2.imread(DATA_LOCATION + IMAGE_DIR + imageFiles[i] + '/' + whiteLightFilesInFolder[j]))
                    if progress:
                        bar.update(j)

            # Make sure the two are actually the same length (it would be weird if
            # they weren't)
            if len(currPhotoelasticImages) != len(currWhiteLightImages):
                print(f'Warning: Different number of PE and white light images for index {indices[i]}')
            photoelasticImageData.append(currPhotoelasticImages)
            whiteLightImageData.append(currWhiteLightImages)

    # If requested, we put together some information about each trial
    if returnMetadata:
        metadata = []
        for i in range(len(indices)):
            metaDict = {"trial_index": indices[i],
                        "trial_name": f'shape_{indices[i]}',
                        "image_dir": imageFiles[i],
                        "force_file": predictedForceFiles[i],
                        "time_steps": len(forceData[i])
                       }
            metadata.append(metaDict)

        return photoelasticImageData, whiteLightImageData, forceData, trackingData, metadata

    return photoelasticImageData, whiteLightImageData, forceData, trackingData

def identifyPeaks(forceData, peakSeparation=100, peakWidth=40, returnTimes=False):
    """
    Identify all of the indices of peaks for a set of force readings. Primarily just a wrapper
    for scipy's find_peaks

    Parameters
    ----------

    forceData : list(3) or list(:,3)
        A time series of (frame_num, time, force), or a list of several time series

    peakSeparation : int
        The minimum distance that can occur between detected peaks. Default (100) is chosen
        by experimentation.

    peakWidth : int
        How wide a peak has to be to be counted; helps to deal with the high resolution
        fluctuations throughout the data. Default (40) is chosen by experimentation.

    returnTimes : bool
        Whether or not to return the time of each peak (True) or simply the index of it (False, default).
    """

    if not isinstance(forceData[0][0], np.ndarray):
        # We have to take the first element of the result since it also returns
        # a dictionary with some properties (that we don't need)
        peakIndices = find_peaks(forceData[:,2], distance=peakSeparation, width=peakWidth)[0]
        
        if returnTimes:
            return np.array([forceData[j,1] for j in peakIndices])

        return peakIndices

    peakIndices = []
    for i in range(len(forceData)):
        peakIndices.append(identifyPeaks(forceData[i], peakSeparation=peakSeparation,
                                         returnTimes=returnTimes, peakWidth=peakWidth))

    return peakIndices
