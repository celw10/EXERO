#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

#External Packages
import datetime
import math
import numpy as np
import os
import time

from itertools import groupby, count
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

#Internal Packages
from .vis import SpikeDataPlotting, XYZPlotting 
from .lineid import LineID, WriteLine
from .spikeid import MovingAverage
from .misc import interp_over_zero, concat_array, haversine_dist, find_line_idx
from .output import Save

#Data QC classes and fuctions for imported raw data

class MagneticDataQC:
    
    def __init__(self, install_loc, proj_loc, fileloc, ftype, proj_bas_sr, proj_rov_sr, proj_line_space, proj_origin,
                 data_qc_report, mag_idx, vlf_idx, bas_idx_qc, rov_idx_qc, cor_idx_qc, col_rem):
        
        #Project variables for data QC Autoprocessing
        self.install_loc = install_loc
        self.proj_loc = proj_loc
        self.fileloc = fileloc
        self.ftype = ftype
        self.proj_bas_sr = proj_bas_sr
        self.proj_rov_sr = proj_rov_sr
        self.proj_line_space = proj_line_space
        self.proj_origin = proj_origin
        self.data_qc_report = data_qc_report
        self.mag_idx = mag_idx
        self.vlf_idx = vlf_idx
        self.bas_idx_qc = bas_idx_qc
        self.rov_idx_qc = rov_idx_qc
        self.cor_idx_qc = cor_idx_qc
        self.col_rem = col_rem
        
    def getfileloc(self):
        return self.fileloc
    
    def changeftype(self, ftype_new):
        self.ftype = ftype_new
        
    def indices_qc(self):
        
        """
        Setup data indices for QC module.
        """
        
        #Indices from file type
        if self.ftype == "rov":
            idx = self.rov_idx_qc
        elif self.ftype == "cor":
            idx = self.cor_idx_qc
        elif self.ftype == "bas":
            idx = self.bas_idx_qc
        elif self.ftype == "vlf":
            idx = self.vlf_idx
        elif self.ftype == "mag":
            idx = self.mag_idx
            
        #Return idx
        return idx
        
    def time_to_seconds(self):
        
        """
        Convert HHMMSS recorded time to seconds references to the first base station recording of the day.
        """
        
        #Load data
        data = np.load(self.fileloc, allow_pickle=True)
        
        #Data indices for file type
        idx = self.indices_qc()
        
        #Initiate bad time point counter and list
        counter, bad_times = 0, []

        #Base station reference time
        if self.ftype == "bas":

            #Reference time
            RefTime = datetime.datetime.strptime(str(int(data[0, idx["time"]])), "%H%M%S")

            #File containg the project reference time
            with open(self.install_loc + self.proj_loc + "/reports/project_ref_time.txt", "w+") as f:

                #Store the reference time
                f.write(str(RefTime.time()))

            #Close the reference file
            f.close()

            #Subtract Reference time from other times
            for i in range(len(data)):
                data[i, idx["time"]] = (datetime.datetime.strptime(str(int(data[i, idx["time"]])), "%H%M%S") - 
                                     RefTime).total_seconds()
                
                #Skip first index
                if i > 0:
                    #Check that the times are reasonable (always increasing)
                    if data[i,idx["time"]] <= data[i-1,idx["time"]]:
                        #Append to bad times if not increasing
                        bad_times.append(i)
                
                #Remove data points that do no exhibit an increase in recorded time
                if bad_times != []:
                    for j in reversed(bad_times):
                        [x.pop(j) for x in data]
                    #Record how many data points are removed
                    counter += len(bad_times)

        else:
            #Reference time
            with open(self.install_loc + self.proj_loc + "/reports/project_ref_time.txt", "r") as f:

                #Reference time datetime object
                RefTime = datetime.datetime.strptime(str(f.readline()), "%H:%M:%S")

            #Close the reference file
            f.close()

            #Store X-Y data for deleted points
            del_x, del_y = [], []
            #Subtract Reference time from other time
            for i in range(len(data)):
                #Initialze bad times list
                bad_times = []
                for j in range(len(data[i])):
                    data[i][j, idx["time"]] = (datetime.datetime.strptime(str(int(data[i][j, idx["time"]])), "%H%M%S") - 
                                             RefTime).total_seconds()
                    
                    #Skip first index
                    if j > 0:
                        #Check that the times are reasonable (always increasing)
                        if data[i][j,idx["time"]] <= data[i][j-1,idx["time"]]:
                            #Append to bad times if not increasing
                            bad_times.append(j)
                

                #Remove data points that do no exhibit an increase in recorded time
                if bad_times != []:
                    if self.ftype == "rov":
                        #Append longitude
                        del_x.append(data[i][bad_times, idx["long"]])
                        del_y.append(data[i][bad_times, idx["lat"]])
                    data[i] = np.delete(data[i], bad_times, axis=0)
                    #Record how many data points are removed
                    counter += len(bad_times)
                    
        #Write to data_qc_report if points are removed
        if counter != 0:
            self.data_qc_report.write(str(counter)+" data points found with measurement time decreasing. Identified data points are removed from the magnetic file type "+self.ftype+". \n")
               
        #Save dataset                       
        np.save(self.fileloc, data, allow_pickle=True)
        
        #Return long/lat for any deleted points if ftype is rov or cor
        if self.ftype == "rov":
            return [del_x, del_y]
        
    def qc_data_quality(self, min_dataq=[9,9]):
        
        """
        Ensure all data points are accompanied with a minimum desired data quality record.
        GEM DOCS: 
        first number is associated with measurement time and is a sort of gradient indicator 0=poor, 9=optimal.
        second number represents the area under signal amplitude coincident with the time of measurement, 0=poor, 9=optimal.
        """
        
        #Load data
        if self.ftype != "bas": 
            data = np.load(self.fileloc, allow_pickle=True)
        else:
            data = [np.load(self.fileloc, allow_pickle=True)]
        
        #Data indices for file type
        idx = self.indices_qc()
        
        #Initiate count and lists for deleted XY points
        del_x, del_y, count = [], [], 0

        #Loop through data for each line
        for i in range(len(data)):
            #Initiate index
            j=0
            while j < len(data[i]):
                #Ensure each component (each digit) of the data quality measurement is satisfactory
                if [int(i) for i in str(int(data[i][j, idx["dq"]]))] < min_dataq:
                    if self.ftype != "bas":
                        #Save locations of deleted points for rov/cor
                        del_x.append(data[i][j, idx["long"]])
                        del_y.append(data[i][j, idx["lat"]])
                    #Delete poor data quality measurements
                    np.delete(data[i], j, axis=0)
                    #Keep track of how many readings are low quality
                    count+=1
                    j+=1
                else:
                    j+=1
        
        #Write to data_qc_report if points are deleted
        if count != 0:
            self.data_qc_report.write("There are "+str(count)+ " magnetic data points with an insufficient data quality measurement for "+self.ftype+". \n")
                    
        #Revert base station data back to 2D array
        if self.ftype == "bas":
            data = data[0]
            
        #Save the respective tmp dataset
        np.save(self.fileloc, data, allow_pickle=True)
        
        #Return XY locations of deleted points
        return [del_x, del_y]
    
    def remove_mag_zeros(self):
    
        """
        Remove any magnetic datapoints that contain a zero reading. 
        """

        #Load data
        data = np.load(self.fileloc)

        #Data indices for file type
        idx = self.indices_qc()

        #Where are the zero entires
        zero_idx = np.where(data[:, idx["nT"]] == 0)[0]

        #Write to report if any points removed
        if len(zero_idx) != 0:
            #Write to report
            self.data_qc_report.write("Removing "+str(len(zero_idx))+" magnetic zero data points for file-type "+self.ftype+". \n")
        
        #If the file type is rov/cor
        if self.ftype == "rov":
            #Save latitude and longitude points of deleted points
            del_x = list(data[zero_idx, idx["long"]])
            del_y = list(data[zero_idx, idx["lat"]])
            
        #Delete the zero entires
        data_out = np.delete(data, zero_idx, axis=0)

        #Save data
        np.save(self.fileloc, data_out)
        
        #Return the deleted x/y points for rov/cor
        if self.ftype == "rov":
            return [del_x, del_y]
        
    def interpol_to_sr(self, line_start=[0], line_end=None):

        """
        Interpolate the dataset to its respective sample rate.
        DEPRECIATED 2021/02/24
        """
        
        #Load data
        data = np.load(self.fileloc)
        
        #Initiate line_end if ftype=bas
        if line_end == None:
            line_end = [len(data)-1]
        
        #Data indices for file type
        idx = self.indices_qc()

        #Sample rate based on file type
        sr = {"bas": self.proj_bas_sr, "rov": self.proj_rov_sr, "cor": self.proj_rov_sr}[self.ftype]

        #Initiate list of time interpolation indices
        t_interpol_idx = []
        #Initiate time interpolated dataset
        data_interpol_t = np.array(data)
        #Initiate insertion row
        row = np.zeros(len(data[0,:]))

        #Loop through identified liens
        for i in range(len(line_start)):
            #Compute time difference
            delta_t = data_interpol_t[line_end[i], idx["time"]] - data_interpol_t[line_start[i], idx["time"]]

            #Number of data points
            data_pts = len(data_interpol_t[line_start[i]:line_end[i]])

            #If there are points to interpolate over
            if delta_t/sr != data_pts:
                #Time list of known datapoints for line
                known_pt_ts = [int(x) for x in data_interpol_t[line_start[i]:line_end[i], idx["time"]]]

                #Time lines to sample rate
                sr_pt_ts = np.arange(int(data_interpol_t[line_start[i], idx["time"]]), 
                                     int(data_interpol_t[line_start[i], idx["time"]] + (delta_t/sr)), 1)

                #Return indices missing datapoints according to sample rate
                missing_ts = [j for j, x in enumerate(sr_pt_ts) if x not in known_pt_ts]
                missing_idx = [line_start[i]+j for j, x in enumerate(sr_pt_ts) if x not in known_pt_ts]

                #Loop through missing time values
                for i in range(len(missing_idx)):
                    #Insert row of zeros with missing time value
                    row[idx["time"]] = sr_pt_ts[missing_ts[i]]
                    data_interpol_t = np.insert(data_interpol_t, missing_idx[i], row, axis=0)

                line_start = [x+len(missing_idx) for x in line_start]
                line_end = [x+len(missing_idx) for x in line_end]

                #Keep record of interpolated indices
                t_interpol_idx.append(missing_idx)

            #No points to interpolate over
            else:
                t_interpol_idx.append([])

        #Flatten interpolation lists      
        interpol_idx = [item for sublist in t_interpol_idx for item in sublist]
        
        #Interpolation dictionary - interp mag reading for bas, XYZ for rov and cor
        if self.ftype == "bas":
            yax = [idx["nT"]]
        elif self.ftype == "rov" or self.ftype == "cor":
            yax = [idx["long"], idx["lat"], idx["elev"]]
        
        #Define interpolation axis
        xax = idx["time"]
                
        #Iterate through appropiate interpolations
        for idx in yax:
            #Linear interpolation over zeros
            data_interpol_t = interp_over_zero(data_interpol_t, xax, idx)

        #Return dataset with zero rows at interpolated indices and interpolated indices
        return data_interpol_t, interpol_idx
    
    #Setup to interpolate elevation for rov and cor, interpolate nT for base station
    def int_spikes_movavg(self, bas_wait=15, max_it=10, beta=0.01, threshold=0.1):
        
        """
        Interpolate base station data spikes as a function of time with forward-backward moving average spike identification.
        See BP 0917 for a good example of SPIKES. The present algorithm isn't dealing with them properly. 
        """
        
        #Load data
        data = np.load(self.fileloc)
        
        #Data indices for file type
        idx = self.indices_qc()
        
        #Initiate counters
        iteration, proc_data_spikes = 1, []

        #Spike identification object
        mov_avg = MovingAverage(self.install_loc, self.proj_loc, beta, threshold, self.vlf_idx)
        
        #Wait time for base station 
        data = data[bas_wait:data.shape[0]-bas_wait-1]
        
        #Write to report
        self.data_qc_report.write("Spike interpolation for "+self.ftype+". ")
        
        while iteration < max_it:
            #Identify data spikes
            data_spikes, spike_data_plotting = mov_avg.fwd_bwd_time(data, idx["nT"])
            
            #Indices of points that do not correspond to a spike
            non_spike = [x for x in range(len(data)) if x not in data_spikes]

            #Assert everything is accounted for
            assert len(data) - len(data_spikes) - len(non_spike) == 0

            if iteration == 0 and data_spikes == []:
                #No data spikes for interpolation
                self.data_qc_report.write(" No data spikes are identified." + "\n")

                #Plot data and save initial profile
                spike_data_plotting.plot_data_despiking([0, len(data[1:len(data)])], 
                                                     "01_data_qc_images/"+str(self.ftype)+"_dataqc")
                
                #Exit loop 
                iteration = "exit"

            #Process out data spikes if necessary
            if data_spikes != []:
                #Linear interpolation as time series
                interp = interp1d(data[non_spike, idx["time"]], data[non_spike, idx["nT"]])
                data[data_spikes, idx["nT"]] = interp(data[data_spikes, idx["time"]])
                
                #Next iteration
                iteration += 1
                
                #Save indices of processed data spikes
                proc_data_spikes += list(data_spikes)
                #Unique data spikes
                proc_data_spikes = list(set(proc_data_spikes))

            elif data_spikes == [] and iteration != 0 and iteration != "exit":
                #Change spike data plotitng object to plot all processed spikes
                spike_data_plotting.changedataspike(proc_data_spikes)
                
                #All spikes are eliminated
                self.data_qc_report.write(str(len(proc_data_spikes))+" data spikes are interpolated after "+str(iteration)+" iteration(s). \n")

                #Plot data and save final profile
                spike_data_plotting.plot_data_despiking([0, len(data[1:len(data)])], 
                                                     "01_data_qc_images/"+str(self.ftype)+"_dataqc")

                #Exit loop
                iteration = "exit"

            if iteration == max_it:
                #Change spike data plotitng object to plot all processed spikes
                spike_data_plotting.changedataspike(proc_data_spikes)
                
                #Exit as maximum iterations are exceeded
                self.data_qc_report.write(" Maximum iterations reached. "+str(len(proc_data_spikes))+" data spikes interpolated. Exiting with "+str(len(data_spikes))+" data spikes. \n")

                #Plot data and save final profile
                spike_data_plotting.plot_data_despiking([0, len(data[1:len(data)])], 
                                                     "01_data_qc_images/"+str(self.ftype)+"_dataqc")
             
            if iteration == "exit":
                #Exit loop
                iteration = max_it
                
        return data
    
    def savitzky_golay_filt(self, data, index, npts, order, smth_type="elev"):
        
        """
        Smooth a profile interpolated to sample rate using a Savitzky Golay filter. 
        """
        
        #Ensure npts not smaller than the dataset (for segments)
        if npts > data.shape[0]:
            #Redefine nps
            nptsqc = int(int(data.shape[0] / 2.0) * 2.0 - 1.0)
        else:
            #Keep original npts
            nptsqc = npts
        
        #Apply the filter
        smoothed = savgol_filter(data[:,index], nptsqc, order)
        
        #XYZ Plotting object
        xyz_plotting = XYZPlotting(self.install_loc, self.proj_loc, self.mag_idx, self.vlf_idx)
        
        #Plot before and after
        xyz_plotting.compare_two_profs(data[:,index], smoothed, "01_data_qc_images/"+self.ftype+"_smth_"+smth_type)
        
        #Replace datapoints
        data[:,index] = smoothed
        
        #Write to report
        self.data_qc_report.write("Savitzky-Golay "+smth_type+" smoothing completed for "+self.ftype+". \n")
        
        #Return data
        return data
    
    def remove_xy_spikes(self, data, line_start, line_end, interpol_idx, 
                         beta=0.1, threshold=3.0, max_it=10):

        """
        Remove temporal distance spikes corresponding to XY GPS Noise. 
        DEPRECIATED 2021/02/24
        """
        
        #Data indices for file type
        idx = self.indices_qc()
        
        #Initiate counters
        full_total, iteration = 0, 1

        #Initiate lists for deleted XY points
        del_x, del_y = [], []
        
        #Spike identification object
        mov_avg = MovingAverage(self.install_loc, self.proj_loc, beta, threshold, self.vlf_idx)
        
        #Write file type for spike elemination
        self.data_qc_report.write("GPS (X/Y) spike removal for file: " + str(self.ftype) + ". ")

        #Iterate until maximum iterations or all spikes eleminated
        while iteration < max_it:

            #Initiate point deletion counter for each iteration
            total = 0

            #Despike each individual line
            for i in range(len(line_start)):

                #Identify data spikes
                current, _ = mov_avg.fwd_bwd_time(data, idx["dist"], norm=False, abs_dif=False, 
                                                  crop_start=line_start[i]+1, crop_end=line_end[i])

                #Current points indexed to entire dataset - add one back to indices
                data_spikes = [x+line_start[i]+1 for x in current]

                #Update deletion total for iteration
                total += len(data_spikes)
                
                #If rover
                if self.ftype == "rov":
                    #Record deleted XY points
                    del_x.extend(list(data[data_spikes, idx["long"]]))
                    del_y.extend(list(data[data_spikes, idx["lat"]]))

                #Delete data spikes
                data = np.delete(data, data_spikes, axis=0)

                #Update line start/stop indices
                line_start[i+1:] = line_start[i+1:] - len(data_spikes)
                line_end[i:] = line_end[i:] - len(data_spikes)

            #Stop iterating
            if total == 0:
                #Write de-spiking results to report
                self.data_qc_report.write(str(full_total)+" distance spikes deleted are "+str(iteration-1)+" iterations. ")
                iteration = max_it
                #Remove zero distance points (includes all interpolated points)
                non_interpol_xyz = np.nonzero(data[:,idx["dist"]])[0]
                """The zero-distance points wont be displayed on the point removal plot."""
                #Write zero distance points removed to report
                self.data_qc_report.write(str(len(data)-len(non_interpol_xyz)-len(interpol_idx))+" non-interpolated zero-distance points removed. \n")
                #Remove interpolated XYZ points
                data_out = data[non_interpol_xyz]

            #Keep iterating
            else:
                iteration += 1
                full_total += total

        #Optinal returns from file type
        return data_out, [del_x, del_y]
    
    def reloc_xy(self, data, max_it=10, filt_sat=True, npts=151, order=3, beta=5e-2, threshold=2.5e-5):

        """
        Relocation of noisy GPS data points. 
        Params looking too tight for open BP survey. Good for wooded technical IAM survey though.
        """

        #Data indices for file type
        idx = self.indices_qc()

        #Initiate shifted distance and smoothing values
        shift = np.zeros((data.shape[0], 1))

        #Re-assign line start/end indices after interpolation
        line_start, line_end = find_line_idx(data, idx["line"])

        #Initiate...
        iteration, reloc_x, reloc_y, data_spikes_prev = 1, [], [], []

        #Iterate until maximum iterations or all spikes eleminated
        while iteration <= max_it:

            #Despike each individual line
            for i in range(len(line_start)):

                #Data by line
                data_line = data[line_start[i]:line_end[i]]
                
                #Ensure npts not smaller than the dataset (for segments)
                if npts > data_line.shape[0]:
                    #Redefine nps
                    nptsqc = int(int(data_line.shape[0] / 2.0) * 2.0 - 1.0)
                else:
                    #Keep original npts
                    nptsqc = npts
                
                #Smooth X/Y points
                x_smth = savgol_filter(data_line[:, idx["long"]], nptsqc, order)
                y_smth = savgol_filter(data_line[:, idx["lat"]], nptsqc, order)

                #Compute vectorized distances using pythagoras
                shift[line_start[i]:line_end[i], 0] = np.sqrt(np.square(data_line[:, idx["long"]] - x_smth) + 
                                                                   np.square(data_line[:, idx["lat"]] - y_smth))

            #Spike identification object
            mov_avg = MovingAverage(self.install_loc, self.proj_loc, beta, threshold, self.vlf_idx)

            #Identify data spikes
            data_spikes, _ = mov_avg.fwd_bwd_time(shift, 0, norm=False, abs_dif=False)

            #Algorithm is stuck
            if data_spikes == data_spikes_prev:
                #Exit
                data_spikes = []

            #If not stuck continue
            else:
                #Filtering spikes by No. sattelites
                if filt_sat == True:
                    #Define previous data spikes
                    data_spikes_prev = data_spikes

                    #Mean & stardard deviation of sattelites
                    mean_sat = np.mean(data[:,idx["sat"]])
                    sigma_sat = np.std(data[:,idx["sat"]])

                    #Number of satellites must be less than 
                    data_spikes = [x for x in data_spikes if data[x,idx["sat"]] < mean_sat-sigma_sat]

                #Indices of points that do not correspond to a spike
                non_spike = [x for x in range(len(data)) if x not in data_spikes]

            #replace spikes in data
            if data_spikes != []:
                #Save locations of (to be) relocated datapoints
                reloc_x += list(data[data_spikes, idx["long"]])
                reloc_y += list(data[data_spikes, idx["lat"]])

                #Linear interpolation of longitude spikes
                interp_x = interp1d(data[non_spike, idx["time"]], data[non_spike, idx["long"]])
                data[data_spikes, idx["long"]] = interp_x(data[data_spikes, idx["time"]])
                #Linear interpolation of latitude spikes
                interp_y = interp1d(data[non_spike, idx["time"]], data[non_spike, idx["lat"]])
                data[data_spikes, idx["lat"]] = interp_y(data[data_spikes, idx["time"]])

                #Increase iteration
                iteration += 1

                if filt_sat == False:
                    #Save last data_spikes
                    data_spikes_prev = data_spikes

            #Exit    
            elif data_spikes == []:

                #Write to report
                self.data_qc_report.write("Relocation for data points [x,y] subject to GPS noise. "+str(len(reloc_x))+" data points are relocated after "+str(iteration-1)+" iterations. \n")
                
                #Iteration to max
                iteration = max_it+1

        #Return data and relocated x/y points
        return data, [reloc_x, reloc_y]

    def rovcor_delta_dt(self):
        
        """
        Compute the changes in distance and time for each data point for rover & cor files.
        """
        
        #Upload data
        upload_data = np.load(self.fileloc, allow_pickle=True)
        
        #Data indices for file type
        idx = self.indices_qc()
        
        #Concatenate 3D object formatted array to 2D numpy array
        concat_data, line_points = concat_array(upload_data)
        
        #Avoid hard coding indices and save the number of columns
        no_cols = len(concat_data[0,:])

        #New data arraw with additional columns for line numbers, pt distance, and time difference
        data = np.zeros((np.shape(concat_data)[0], np.shape(concat_data)[1]+3))

        #Fill the new array
        data[:,0:no_cols] = concat_data
        
        #Loop to fill temporary line ID, point distances, and time differences
        for i in range(len(data)):
            #Special cases for first index
            if i == 0:
                #Time difference is zero from first point
                data[i,no_cols+2] = 0
                #Distance is SMALL for first point - remove zero distances later
                data[i,no_cols+1] = 1e-4
            else:
                #Compute time difference
                data[i,no_cols+2] = data[i,idx["time"]] - data[i-1,idx["time"]]
                #Compute XY distances in m using the haversine approximation for spherical earth
                data[i,no_cols+1] = haversine_dist(data[i-1, idx["long"]], data[i-1, idx["lat"]], 
                                                   data[i, idx["long"]], data[i, idx["lat"]])
                
        #Data to tmp
        np.save(self.fileloc, data)
        
        #Return line points
        return line_points
        
    def rovcor_lineid(self, line_points, mag_data_qc):
    
        """
        Automated line number identification using the lineid module.
        """
    
        #Line ID object from LineID class
        line_id = LineID(self.install_loc, self.proj_loc, self.proj_line_space, self.proj_origin, self.ftype, self.fileloc, 
                         self.data_qc_report, line_points, self.vlf_idx)

        #Line identification algoritm
        line_start, line_end, del_pts_line = line_id.line_id(mag_data_qc)
        
        #Return line indices and any deleted points
        return line_start, line_end, del_pts_line
        
        
    def rovcor_xyz_denoise(self):
    
        """
        Positional QC workflow for Rover and Corrected data files.
        """
            
        #Load data
        data = np.load(self.fileloc)
            
        #Data indices for file type
        idx = self.indices_qc()
        
        #Interpolate elevation data spikes
        data_elev_smth = self.savitzky_golay_filt(data, idx["elev"], 201, 3)

        #Removal of latitude/longitude spikes
        data_xy_reloc, reloc_pts_xy = self.reloc_xy(data_elev_smth, threshold=2e-5) #2-2.5 is good

        #Save processed datafile
        np.save(self.fileloc, data_xy_reloc)
                
        #Return the XYZ plotting object and deleted points
        return reloc_pts_xy
    
    def day_acqui_stats(self, inc_elev=True):

        """
        Compute acquisition-relevant statistics and write to repor.
        """

        #Define indices
        idx = self.indices_qc()
        
        #Load data
        data = np.load(self.fileloc)
        
        #Find line indices
        line_start, line_end = find_line_idx(data, idx["line"])

        #Should be equal in length
        assert len(line_start) == len(line_end)

        #Initiate total time and distance array
        tot = np.zeros((len(line_start), 2))

        #Compute total line distances & acquisition time
        for i in range(len(line_start)):
            #Define data points
            dp1 = data[line_start[i]]
            dp2 = data[line_end[i]]

            #Optionally use elevation in distance calculation
            if inc_elev == True:
                #Compute pythagoras distance along lines with elevations and great-circle distances
                tot[i,0] = haversine_dist(dp1[idx["long"]], dp1[idx["lat"]], dp2[idx["long"]], dp2[idx["lat"]], 
                                          e1=dp1[idx["elev"]], e2=dp2[idx["elev"]])

            elif inc_elev == False:
                #Compute great-circle distances along lines
                tot[i,0] = haversine_dist(dp1[idx["long"]], dp1[idx["lat"]], dp2[idx["long"]], dp2[idx["lat"]])

            #Compute acquisition times for each line
            tot[i,1] = dp2[idx["time"]] - dp1[idx["time"]] #HARD CODED TIME

        #Compute totals from line totals
        tot_sum = list(np.sum(tot, axis=0))

        #Round to meter & convert seconds to H:M:S
        tot_sum[0] = str(round(tot_sum[0],0)/1000.0)
        tot_sum[1] = time.strftime('%H:%M:%S', time.gmtime(tot_sum[1]))

        #Compute total elevation gained & total moving distance
        dist_move = 0
        elev_gain = 0

        #Loop through everything FOR NOW
        for i in range(1, len(data), 1):
            #Optionally include elvation in distance calculations
            if inc_elev == True:
                #Compute moving distance using pythagoras with elevations and great-circle distances
                dist_move += haversine_dist(data[i-1,idx["long"]], data[i-1,idx["lat"]], 
                                            data[i,idx["long"]], data[i,idx["lat"]], 
                                            e1=data[i-1,idx["elev"]], e2=data[i,idx["elev"]])

            elif inc_elev == False:
                dist_move += haversine_dist(data[i-1,idx["long"]], data[i-1,idx["lat"]], 
                                            data[i,idx["long"]], data[i,idx["lat"]])

            #Compute positive elvation gain
            if data[i,idx["elev"]] > data[i-1,idx["elev"]]:
                elev_gain += data[i,idx["elev"]] - data[i-1,idx["elev"]]

        #Report whole-number values
        elev_gain = str(int(round(elev_gain,0)))
        dist_move = str(round(dist_move,0)/1000.0)

        #Write to report
        self.data_qc_report.write("Data acquisition statistics. Line km acquired: "+tot_sum[0]+", time [H:M:S] spent acquiring data: "+tot_sum[1]+", distance [km] covered: "+dist_move+", elevation [m] gained: "+elev_gain+". \n")
                
class VLFDataQC:
    
    def __init__(self, install_loc, proj_loc, fileloc, data_qc_report, vlf_idx):
        self.install_loc = install_loc
        self.proj_loc = proj_loc
        self.fileloc = fileloc
        self.data_qc_report = data_qc_report
        self.vlf_idx = vlf_idx
        
    def vlf_sigstr_qc(self, mag_data_qc, pT_soft_cutoff=4.0, pT_hard_cutoff=3.5):

        """
        QC function for VLF data signal strength. 
        """
        
        #Data indices
        idx = mag_data_qc.indices_qc()

        #Load dataset
        data_upload = np.load(self.fileloc, allow_pickle=True)

        #Concatenate array
        data, _ = concat_array(data_upload)
        
        #Initiate list of indices below cutoff
        idx_below_soft_cutoff, idx_below_hard_cutoff, good_data = [], [], []

        #Loop each data point
        for i in range(data.shape[0]):

            #Weak signal thresholds
            if data[i,idx["pT"]] < pT_hard_cutoff: 
                #Below soft cutoff
                idx_below_hard_cutoff.append(i)
            elif data[i,idx["pT"]] < pT_soft_cutoff:
                #Below hard cutoff
                idx_below_soft_cutoff.append(i)
            else:
                #Otherwise good signal strength
                good_data.append(i)

        #Compute cubic trendline for datapoints
        x = np.polyfit(data[:,idx["time"]], data[:,idx["pT"]], 3) 

        #Function of best fit
        y = x[0]*data[:,idx["time"]]**3 + x[1]*data[:,idx["time"]]**2 + x[2]*data[:,idx["time"]] + x[3]     

        #First derivative of best fit function
        yprime = 3*x[0]*data[:,idx["time"]]**2 + 2*x[1]*data[:,idx["time"]] + x[2]

        #Investigate soft signal cutoff datapoints
        for i in idx_below_soft_cutoff:
            #Conditions invloving trendline and slope (1st derivative) of data for point deletion
            if (y[i] < pT_soft_cutoff) and (yprime[i] > 0 and any(j > i for j in good_data) == False):
                #Append poor signal quality point
                idx_below_hard_cutoff.append(i)
                #Sort list
                idx_below_hard_cutoff.sort()

        #Define new numpy array
        data_ss_qc = np.array(data)

        #Initiate deletion long/lat indices
        del_x, del_y = [], []
        
        #If no data-points are removed
        if idx_below_hard_cutoff == []:
            #Write to report
            self.data_qc_report.write("Signal strength for all VLF data points is satisfactory. \n")
            #Save VLF data to tmp
            np.save(self.fileloc, data_ss_qc)

        #If all data points are removed
        elif len(idx_below_hard_cutoff) == data.shape[0]:
            #Write to report
            self.data_qc_report.write("The signal strength of all VLF data points is inadequate. No further VLF processing. \n")
            #Save point deletion locations
            del_x = data_ss_qc[idx_below_hard_cutoff, 1]
            del_y = data_ss_qc[idx_below_hard_cutoff, 0]
            #Delete file
            os.remove(self.fileloc)

        #Some points are removed
        else:
            #Write to report
            self.data_qc_report.write("Signal strength inadequate for "+str(len(idx_below_hard_cutoff))+" of "+str(data.shape[0])+" VLF data points. \n")
            #Save point deletion locations
            del_x = data_ss_qc[idx_below_hard_cutoff, 1]
            del_y = data_ss_qc[idx_below_hard_cutoff, 0]
            #Delete any low-signal strength data points
            data_ss_qc = np.delete(data_ss_qc, idx_below_hard_cutoff, axis=0)
            #Save VLF data to tmp
            np.save(self.fileloc, data_ss_qc)

        #Plotting object
        time_series_plotting = SpikeDataPlotting(self.install_loc, self.proj_loc, data, idx_below_hard_cutoff, self.vlf_idx)

        #Plot VLF signal strength QC
        time_series_plotting.plot_vlf_sigstr(y, pT_soft_cutoff, pT_hard_cutoff, "01_data_qc_images/vlf_sig_str_QC")
        
        #Return any deleted XY points
        return [del_x, del_y]
        
    def line_id_vlf(self, mag_data_qc, magvlf_data_qc):
        
        """
        Identify line start/end indices from line_locs.txt for vlf data.
        """
        
        #Data indices for magnetic and vlf data
        idxm = mag_data_qc.indices_qc()
        idxv = magvlf_data_qc.indices_qc()
        
        #Load rover magnetic data with line ID's
        mag_data = np.load(mag_data_qc.getfileloc())
        
        #Load rover VLF data without line ID's
        vlf = np.load(self.fileloc)
        
        #Any data points remaining
        if vlf.size != 0:
        
            #VLF data with extra column
            vlf_data = np.zeros((vlf.shape[0], vlf.shape[1]+1))
            vlf_data[:,0:vlf.shape[1]] = vlf

            #Find line breaks previously defined in magnetic data
            line_start, line_end = find_line_idx(mag_data, idxm["line"])

            #Initiate line time boundary array
            line_t_bounds = np.zeros((len(line_start), 3))
            for i in range(len(line_start)):
                #Start time
                line_t_bounds[i,0] = mag_data[line_start[i], idxm["time"]]
                #End time
                line_t_bounds[i,1] = mag_data[line_end[i], idxm["time"]]
                #Assert line numbers the same for start/end
                assert mag_data[line_start[i], idxm["line"]] == mag_data[line_end[i], idxm["line"]]
                #Line number
                line_t_bounds[i,2] = mag_data[line_start[i], idxm["time"]]

            #Deletion indices if magnetic data has been removed at that time
            del_idx = []
            #Assign line numbers to VLF data based on reference time
            for i in range(len(vlf_data)):
                #Index satisfying time line boundraries
                idx = np.where(np.logical_and(np.less_equal(line_t_bounds[:,0],vlf_data[i,idxv["time"]]), 
                                              np.less_equal(vlf_data[i,idxv["time"]],line_t_bounds[:,1])))[0]

                #If idx is empty this point has been removed, delete VLF data point as well
                if idx.size == 0:
                    #Append i to deletion idx 
                    del_idx.append(i)
                else:
                    #Assign VLF data point to line
                    vlf_data[i,vlf.shape[1]] = int(line_t_bounds[idx,2])

            #Delete any VLF data points that have corresponding magnetic data points removed
            vlf_data = np.delete(vlf_data, del_idx, axis=0)

            #Save data to temp
            np.save(self.fileloc, vlf_data)