#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

#External Packages
import numpy as np
import sys

#Internal packages
from .misc import normalize
from .vis import SpikeDataPlotting

class MovingAverage:
    
    def __init__(self, install_loc, proj_loc, beta, threshold, vlf_idx):
        self.install_loc = install_loc
        self.proj_loc = proj_loc
        self.beta = beta
        self.threshold = threshold
        self.vlf_idx = vlf_idx

    def fwd_bwd_time(self, data, idx, norm=True, abs_dif=True, crop_start=0, crop_end=None):

        """
        Search for spikes outside of a forward/backword temporal moving average envalope plus tolerance.
        """

        #Define crop end
        if crop_end == None:
            crop_end = len(data[:,0])
        
        #Current data for despiking
        despike_data = data[crop_start:crop_end, idx]

        #Initiate 1D arrays
        moving_average_fwd = np.zeros((len(despike_data[:])))
        moving_average_bwd = np.zeros((len(despike_data[:])))
        despike_data_norm = np.zeros((len(despike_data[:])))

        #Potential spike indices
        spike_idx_fwd = []
        spike_idx_bwd = []

        #Compute the forward and backward moving average of basestation data
        for fwd in range(len(despike_data[:])):
            #Backwards index
            bwd = len(despike_data) - fwd - 1

            #Skip normalization
            if norm == False:
                despike_data_norm = despike_data

            #Normalize datapoints so threshold is a percent change between min/max
            elif norm == True and fwd <= bwd:
                despike_data_norm[fwd] = normalize(despike_data, fwd)
                despike_data_norm[bwd] = normalize(despike_data, bwd)

            #First iteration exception
            if fwd == 0:
                moving_average_fwd[fwd] = despike_data_norm[fwd]
                moving_average_bwd[bwd] = despike_data_norm[bwd]

            else:
                #Compute the forward moving average
                moving_average_fwd[fwd] = (1-self.beta)*moving_average_fwd[fwd-1] + \
                                           self.beta*despike_data_norm[fwd]
                #Compute the backward moving average
                moving_average_bwd[bwd] = (1-self.beta)*moving_average_bwd[bwd+1] + \
                                           self.beta*despike_data_norm[bwd]

            #Positive and negative changes considered a spike
            if abs_dif == True:
                #Return locations of forward spikes
                if abs(despike_data_norm[fwd] - moving_average_fwd[fwd]) > self.threshold:
                    spike_idx_fwd.append(fwd)

                #Return locations of backward spikes 
                if abs(despike_data_norm[bwd] - moving_average_bwd[bwd]) > self.threshold:
                    spike_idx_bwd.append(bwd)

            #Only positve changes considered a spike
            elif abs_dif == False:
                #Return locations of forward spikes
                if despike_data_norm[fwd] - moving_average_fwd[fwd] > self.threshold:
                    spike_idx_fwd.append(fwd)

                #Return locations of backward spikes 
                if despike_data_norm[bwd] - moving_average_bwd[bwd] > self.threshold:
                    spike_idx_bwd.append(bwd)

        #Spikes present in both forward and backward moving averages for current XYZ data type
        data_spikes = [x for x in spike_idx_fwd if x in spike_idx_bwd]
        
        #Make the time series plotting object
        time_series_plot = SpikeDataPlotting(self.install_loc, self.proj_loc, 
                                              [despike_data_norm, moving_average_fwd, moving_average_bwd], 
                                              [spike_idx_fwd, spike_idx_bwd, data_spikes], self.vlf_idx)
        
        #Indices of points that do not correspond to a spike
        non_spike = [x for x in range(len(data)) if x not in data_spikes]
        
        #Return data spikes
        return data_spikes, time_series_plot