# This file contains several global settings used across the rest of the scripts in the style
# of a `startup.m` file in Matlab. I quite like this format, so I'll use it here as well :)

# This is the location of the data on my computer
# The data takes up about 24GB, so I store it on an
# external hard drive
# This is the default mount point for nemo (the file manager)
DATA_LOCATION = "/run/media/jack/Seagate Portable Drive/Research/geogran2/"

# I found this to be the case for several different trials, so I am assuming it
# is constant across all of them. For some reason, it was in units of kiloseconds (?)
# in the .mat files, but I converted it to regular seconds here
TIME_BETWEEN_FRAMES = 1.92
