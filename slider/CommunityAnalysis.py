import numpy as np

import networkx as nx
import community as community_louvain

from slider import Settings
from slider import NetworkAnalysis

def patchCommunityIndices(communities):
    """
    Take a set of community assignments and ensure that the indices are continuous
    ie. if there are N communities, the indices will be 0, 1, 2, ... N-1.

    Parameters
    ----------

    communities : numpy.ndarray or list
        Community assignments for a set of points.

    Returns
    -------

    numpy.ndarray : The same set of community assignments but with continuous indices.
    """
    uniqueCommunities = np.unique(communities)
    replaceDict = dict(zip(uniqueCommunities, range(len(uniqueCommunities))))
    communities = np.array([replaceDict[c] for c in communities])

    return communities

def genCommunityDetection(graph, resolution=1):
    partition = community_louvain.best_partition(graph, resolution=resolution, randomize=True)
    partition = patchCommunityIndices(list(partition.values()))

    return partition

def averageCommunitySize(partition):
    return len(partition)/(max(partition) + 1)

def numberOfCommunities(partition):
    return max(partition) + 1


"""
Attribute Marking
-----------------

Since we have multiple types of analysis methods, that need different variables
(image, particle, network, etc.) we need to be able to differentiate them, and 
provide the correct args to each one.

This is done by assigning an identifying attribute to each function based on
what type of analysis it is (and there its method signature). The attribute will
be the same for all types of methods, but it's value will be different

The attribute for CommunityAnalysis methods is: analysis_type = Settings.COMMUNITY_ANALYSIS
"""

averageCommunitySize.analysis_type = Settings.COMMUNITY_ANALYSIS
numberOfCommunities.analysis_type = Settings.COMMUNITY_ANALYSIS

