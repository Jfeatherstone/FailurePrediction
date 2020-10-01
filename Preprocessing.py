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
def loadMatFile(dataFile, cleanData = True):

    # If we are just given a single file, we just load that one and clean up a bit
    if not isinstance(dataFile, list):
        mat = sio.loadmat(f'{Settings.DATA_LOCATION}{dataFile}/{dataFile}.quakes.mat')
        # Unfortunately, loadmat tends to format some stuff oddly (and includes variables that we
        # don't really care about, like Matlab version), so we can clean that
        # (unless specified otherwise)
        if not cleanData:
            return mat

        # The list of variables we don't really care about
        extraneousVars = ['__header__', '__version__', '__globals__']
        for var in extraneousVars:
            if var in mat:
                mat.pop(var)


        # First, we get rid of any variables that don't actually have a value (or are [])
        # this usually removes movduration and movtime

        # We can't change a dictionary structure while iterating through it, so we record
        # which ones are bad and remove them after
        emptyItems = []
        for key in mat.keys():
            # the statement array.any() will evaluate to true if any values exist, 
            # and false otherwise
            if not mat[key].any():
                emptyItems.append(key)

        # As promised, remove the empty items
        for item in emptyItems:
            mat.pop(item)

        # Next up, we want to clean the arrays
        # In matlab, a vector is really just a matrix of shape (N,1)
        # so we'll often read one in as:
        # [[1], [2], [3], ...] so we want to strip off that second dimension to
        # get [1, 2, 3, ...]
        possibleRowVectors = ['t', 'f', 'fn', 'x']
        for var in possibleRowVectors:
            if var in mat:
                mat[var] = mat[var][:,0]
        
        # Similarly, we can also get column vectors that have shape (1.N)
        possibleColumnVectors = ['force', 'good', 'deltaF', 'duration', 'multipeak', 'position', 'slip', 'time', 'work']
        for var in possibleColumnVectors:
            if var in mat:
                mat[var] = mat[var][0,:]

        return mat

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


# This will make sure that the data exists and give some stats
# about various things. This is a good thing to run first to make
# sure you have access to the data in the first place

# It will return true if you have access to data
def testData(debug=True):
    # If debug is not enabled, this could be used just along the lines of
    # if testData(debug=False):
    #       Do something...

    # First we want to make sure that the data actually exists in the location it should
    if debug:
        print(f'Searching for data in location specified by Settings.DATA_LOCATION:\n{Settings.DATA_LOCATION}\n...')

    # Check that the location exists
    if not os.path.isdir(Settings.DATA_LOCATION):
        if debug:
            print(f'ERROR: Path {Settings.DATA_LOCATION} does not exist')

        return False

    # Now we make sure that there are files in the directory
    # We don't care whether they have movies; we'll deal with that later
    dataFiles = listDataFiles(excludeDataWithoutMovies=False)

    # If there are zero files, that's an issue
    if len(dataFiles) == 0:
        if debug:
            print(f'ERROR: Found no files in directory {Settings.DATA_LOCATION}')

        return False

    if debug:
        print(f'Found {len(dataFiles)} possible data files using listDataFiles()')


    # Now we verify that all of the rundata files exist
    runProperties = loadRunData(dataFiles)

    # We shouldn't be missing an property files
    if not len(dataFiles) == len(runProperties):
        if debug:
            print(f'ERROR: {len(dataFiles) - len(runProperties)} property files are missing')
    
        return False

    if debug:
        print(f'Verified {len(runProperties)} properties files exist')
        print(f'Example of property file keys:\n{runProperties[0].keys()}\n')
    

    # Now we check the mat data
    matData = loadMatFile(dataFiles)

    # We shouldn't be missing any
    if not len(matData) == len(dataFiles):
        if debug:
            print(f'ERROR: {len(dataFiles) - len(matData)} mat files are missing')

        return False

    if debug:
        print(f'Verified {len(matData)} mat files exist')
        print(f'Example of mat file keys:\n{matData[0].keys()}\n')

    # Now we check for movies
    movieDataFiles = listDataFiles(excludeDataWithoutMovies=True)

    if debug:
        print(f'Found {len(movieDataFiles)} data files with movies using listDataFiles(True)')

    movies = loadVideo(movieDataFiles)

    if not len(movieDataFiles) == len(movies):
        if debug:
            print(f'ERROR: {len(movieDataFiles) - len(movies)} movie files are missing')

        return False

    if debug:
        print(f'Verified {len(movies)} movies exist')

    # If we got to this point, everything looks good!
    return True
