# This file includes methods for sampling portions of the videos/data
# that could be used to train the neural network

import cv2
import numpy as np

def playVideo(video, runProperties=None, waitTime=25, endWait=10000):
    if not video.isOpened():
        raise Exception("Video cannot be opened")


    region = None
    # If the runproperties parameter was passed, we crop to the good region
    if not runProperties == None:
        # Grab the good region of the image defined by the property file
        region = runProperties["goodarea"]
    
    i = 0
    while video.isOpened():
        i += 1
        ret, frame = video.read()

        # So long as we have another frame
        if ret:
            # Grayscale, since they are grayscale images anyway
            grayscaleFrame = frame[:,:,0]
            # Crop if we can
            if not region == None:
                grayscaleFrame = grayscaleFrame[region[0]:region[1], region[2]:region[3]]

            # This will display the image in a new window (similar to matlab figure windows)
            cv2.imshow('Frame', grayscaleFrame)
            # This just waits for a little after each image so it doens't go too fast
            cv2.waitKey(waitTime)
        else:
            break

    print(f'Played {i} frames') 
    # Wait at the end to close for 10s (default) or until a key is pressed
    cv2.waitKey(endWait)

    # Close the pop up window
    cv2.destroyAllWindows()

    return
