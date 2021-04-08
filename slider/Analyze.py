import numpy as np

from slider import Settings

def analyze(photoelasticImage, trackingData, metrics):
    """
    A pipeline to apply a set of analysis methods (metrics) to a given trial (or set of trials).

    Parameters
    ----------

    photoelasticImage : numpy.array or str list(numpy.array) or list(str)
        The photoelastic image data (or path to the image), or a list of images/paths

    trackingData : numpy.array or list(numpy.array)
        The particle tracking data for a trial (or trails) of the form (pos_x, pos_y, rot_angle, radius)

    metrics : func or list(func)
        A list of metrics to evaluate for the given data set (or sets). Note that every method must
        have the attribute .analysis_type to properly determine the method signature (and what data to pass)
    """
