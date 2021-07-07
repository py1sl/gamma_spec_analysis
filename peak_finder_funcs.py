#!/usr/bin/env python
# coding: utf-8

# # Peak finder functions
# - finder, plotter and get counts
# - add to Steves .py file later

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gs_analysis as ga

def sp_peak_finder(x, prominence, wlen):

    sf = ga.five_point_smooth(x)
    smooth = np.array(sf)
    sf2 = ga.five_point_smooth(smooth)
    smooth2 = np.array(sf2)
    peaks,_ = find_peaks(sf2, prominence = prominence, wlen = wlen)
    
    return(smooth2, peaks)

def peak_identifier(smooth_counts, ebins, peaks):
    
    plt.plot(ebins[peaks], smooth_counts[peaks], "xr"); plt.plot(ebins, smooth_counts)
    plt.xlabel('ebins')
    plt.ylabel('counts')
    plt.yscale('log') 
    plt.show
    

def peak_counts(peaks, index, smooth_counts, ebins):
    """ Index is the peak array index for the peak that counts is required for
        i.e [0], NOT the peak index itself i.e [3210]
        Returns the index of the peak and its calculated count
    """    
    x, y = ga.get_peak_roi(peaks[index], smooth_counts, ebins, offset=10)
    
    length = len(x)
    start_pos = x[0]
    end_pos = x[length - 1]
    start, = np.where(ebins == start_pos)
    end, = np.where(ebins == end_pos)
    
    counts = ga.net_counts(smooth_counts, start[0], end[0], m=1)
    
    return(peaks[index], counts)

