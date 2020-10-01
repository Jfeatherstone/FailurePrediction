# This file contains most of the methods for loading in data from the raw data
# or the results stored in the .mat files in each directory

import scipy.io as sio # For reading mat files
import os # For list files in directory + various other stuff
import cv2 # For reading video files
import numpy as np # For array magic

import Settings # My global settings

# This will return a list of all of the folders in the data location
# If we want to include only trials that have videos (those taken in
# 2007 for the most part) then we can specify that
def listDataFiles(excludeDataWithoutMovies = False):
    rootDataDirFiles = os.listdir(Settings.DATA_LOCATION)

    # If we want all of the files, we are done
    if not excludeDataWithoutMovies:
        return rootDataDirFiles

    # Otherwise, we have to check each folder to see if a movie exists
    dataWithMovies = []
    for filePath in rootDataDirFiles:
        if os.path.exists(f'{Settings.DATA_LOCATION}{filePath}/{filePath}.bw.avi'):
            dataWithMovies.append(filePath)

    return dataWithMovies


# This returns the relevant mat files (loaded as a dict from scipy)
# for a given trial. I have designed it so that it will work
# when provided an element that is returned from listDataFiles()
# Note that dataFile can be an array, in which case it will return
# a list of dicts for each one
def loadMatFile(dataFile):

    # If we are just given a single file, it is easy enough
    if not isinstance(dataFile, list):
        return sio.loadmat(f'{Settings.DATA_LOCATION}{dataFile}/{dataFile}.quakes.mat')

    # Otherwise, we creating a list of them and return that
    matFiles = []
    for filePath in dataFile:
        matFiles.append(loadMatFile(filePath))

    return matFiles


# This returns the data from the rundata file in each directory
# It also cleans up some of the parameters that could be used later
def loadRunData(dataFile):

    if not isinstance(dataFile, list):
        keys = []
        values = []

        with open(f'{Settings.DATA_LOCATION}{dataFile}/{dataFile}.rundata') as runData:
            for line in runData.read().split('\n'):
                # We have a to first check that the line actually had an equals
                # with the isinstance so that whitespace lines are ignored
                splitLine = line.split('=')
                if isinstance(splitLine, list) and len(splitLine) == 2:
                    newKey, newValue = splitLine
                    keys.append(newKey.strip())
                    values.append(newValue.strip())

        runProperties = dict(zip(keys, values))

        # Convert the good area list (of the form "[ 1 2 3 4 ]") to an actual array
        if "goodarea" in runProperties:
            runProperties["goodarea"] = np.array([i.strip() for i in runProperties["goodarea"][1:-2].split()], dtype='int')

        # Convert various numbers to actual numbers
        # This is the list of keys whose values should actually be numbers
        numberKeys = ["velocity", "fps", "centerline", "pixpermm", "starttime", "goodtime", "endtime", "sync"]
        for key in numberKeys:
            # If we are missing the key altogether, that's an issue
            if not key in runProperties:
                #raise Exception(f'Expected key \'{key}\' in {dataFile}.rundata but didn\'t find it')
                continue

            runProperties[key] = float(runProperties[key])
        
        return runProperties

    # Otherwise, recurse over each entry
    propertyFiles = []
    for filePath in dataFile:
        propertyFiles.append(loadRunData(filePath))

    return propertyFiles


# This returns the video associated with a trial
# If the specified trial doesn't have a video, this will be
# indicated
# Again this is designed to work with the results of listDataFiles()
# and it can work with a list of data files
def loadVideo(dataFile, ignoreInvalidData = False):
    
    if not isinstance(dataFile, list):
        videoData = cv2.VideoCapture(f'{Settings.DATA_LOCATION}{dataFile}/{dataFile}.bw.avi')

        # If it opened properly, all is good
        if videoData.isOpened():
            return videoData

        # Otherwise, we'll throw an error (unless told not to)
        if ignoreInvalidData:
            return None
        else:
            raise Exception(f'Invalid data file pass to loadVideo(): {dataFile}')
        
    # Otherwise, we just recurse on each element
    videoFiles = []
    for filePath in dataFile:
        videoFiles.append(loadVideo(filePath, ignoreInvalidData=ignoreInvalidData))

    return videoFiles
 
