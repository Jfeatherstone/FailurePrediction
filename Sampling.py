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
            if not region == None:
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
                    video.set(cv2.CAP_PROP_POS_FRAMES, i)
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

    return
