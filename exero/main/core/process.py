#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

#External Packages
import math
import numpy as np
import scipy as sp
import utm
from pathlib import Path

#Internal Packages
from .vis import DiurnalCorPlotting
from .misc import find_line_idx, gauss_window_1D, dir_cos, k_values

class MagDataProcess:
    
    def __init__(self, misc_proj, report):
        self.misc_proj = misc_proj
        self.report = report
        
    def diurnal_correction(self, xyz_plotting):
    
        """
        Complete a diurnal correction of the magnetic data. 
        Re-compute the datum as the mean of all base station records after each data upload. 
        Ensures that the mean of the data does not change. 
        """

        #Load base and rover datasets
        mag = np.load(self.misc_proj.proj_path() + "/data/preprocess/mag_rov.npy")
        bas = np.load(self.misc_proj.proj_path() + "/data/preprocess/mag_bas.npy")
        
        #Define base and rover indices
        mag_idx = self.misc_proj.indices_misc("mag")
        bas_idx = self.misc_proj.indices_misc("bas")
        
        #Preprocess data is now tmp
        np.save(self.misc_proj.proj_path() + "/data/tmp_mag_rov.npy", mag)
        
        #Plot raw data
        xyz_plotting.plot_mag_field("00Raw_Magnetic_Data", self.misc_proj, "nT")

        #Write mean to report
        self.report.write("Diurnal correction for "+str(mag.shape[0])+" magnetic data points. Rover mean pre-correction: "+str(round(np.mean(mag[:, mag_idx["nT"]]),2))+". ")

        #Compute the new datum
        datum = np.mean(bas[:, bas_idx["nT"]])

        #Identify data upload start/stop indices of base and rover data
        bas_start, bas_end = find_line_idx(bas, bas_idx["upno"])
        mag_start, mag_end = find_line_idx(mag, mag_idx["upno"])

        #Assert an equivilaent number of start/stop indices
        assert np.shape(bas_start) == np.shape(mag_start) and np.shape(bas_end) == np.shape(bas_end)

        #All bas interp to mag
        bas_interp_mag = np.zeros((mag.shape[0]))

        #For each data upload
        for i in range(len(bas_start)):
            #Define current rover and base station datasets
            mag_current = mag[mag_start[i]:mag_end[i]+1]
            bas_current = bas[bas_start[i]:bas_end[i]+1]
            
            #Interpolate the bas profile to magnetic data's time
            interp_bas = sp.interpolate.interp1d(bas_current[:, bas_idx["time"]], bas_current[:, bas_idx["nT"]])
            mag_interp_bas = interp_bas(mag_current[:, mag_idx["time"]])

            #Assign to cumulative base station data interpolated to mag
            bas_interp_mag[mag_start[i]:mag_end[i]+1] = mag_interp_bas    

        #Compute the datum
        datum = np.mean(bas_interp_mag)
        diurnal_cor_datum = np.zeros((mag.shape[0])); diurnal_cor_datum.fill(datum)

        #Compute diurnal magnetic shifts
        diurnal_shifts = bas_interp_mag - diurnal_cor_datum

        #Apply diurnal magnetic shifts to rover data
        diurnal_cor_mag = mag[:, mag_idx["nT"]] - diurnal_shifts

        #Construct corrected magnetic dataset
        mag_new = np.zeros((mag.shape[0], mag.shape[1]+1))
        mag_new[:, 0:mag.shape[1]] = mag
        mag_new[:, mag.shape[1]] = diurnal_cor_mag

        #Write mean to report
        self.report.write("Rover mean post-correction: "+str(np.mean(mag_new[:, mag_idx["dirn"]]))+". \n")
        
        #Diurnal correction plotting object
        diurnal_cor_plotting = DiurnalCorPlotting(self.misc_proj, mag_start, mag_end)
        
        #Plot project base station with datum
        diurnal_cor_plotting.plot_proj_bas(bas_interp_mag, datum, "basproj_int2mag_datum")
        
        #Plot the diurnal correction
        diurnal_cor_plotting.plot_diurnal_cor(mag_new, diurnal_shifts, "mag_diurnal_correction")
                                          
        #Save mag to tmp
        np.save(self.misc_proj.proj_path() + "/data/tmp_mag_rov.npy", mag_new)
        
        #Plot diurnal-corrected magnetic data
        xyz_plotting.plot_mag_field("01Diurnal_Corrected", self.misc_proj, "dirn")
        
    def downsamp_filter_mag(self, n, last_idx):
    
        """
        Convolve a gaussian filter of length n while downsampling the data by n. 
        """

        #Load data
        data = np.load(self.misc_proj.proj_path() + "/data/tmp_mag_rov.npy")
        
        #Define indices
        idx = self.misc_proj.indices_misc("mag")

        #Find line indices
        line_start, line_end = find_line_idx(data, idx["line"])

        #Loop through each line
        for i in range(len(line_start)): 
            #Line data
            data_line = data[line_start[i]:line_end[i]]

            #Construct convolution filter
            window_gauss = sp.signal.windows.gaussian(n, std=1) 
            data_weights = window_gauss/np.sum(window_gauss)
            index_weights = np.zeros((n))
            index_weights[int(n/2.0)] = 1.0

            #Convolve raw data
            res1 = np.apply_along_axis(lambda m: np.convolve(m, data_weights, mode='valid'), axis=0, 
                                       arr=data_line[:,0: idx["line"]])[::n,:]
            res2 = np.apply_along_axis(lambda m: np.convolve(m, index_weights, mode='valid'), axis=0, 
                                       arr=data_line[:, idx["line"]: idx["dirn"]])[::n,:]
            res3 = np.apply_along_axis(lambda m: np.convolve(m, data_weights, mode='valid'), axis=0, 
                                       arr=data_line[:, idx["dirn"]:last_idx+1])[::n,:]

            #Horizontal stack of downsampled dataset
            line_ds = np.hstack((res1, res2, res3))

            #Concatenate with other lines
            if i == 0:
                #Initiate concat
                data_ds = line_ds
            else:
                #Concatenate data & line downsample
                data_ds = np.concatenate((data_ds, line_ds), axis=0)

            #Insert first and last point for later interpolation
            data_ds = np.insert(data_ds, 0, data[0], axis=0)
            data_ds = np.insert(data_ds, data_ds.shape[0], data[data.shape[0]-1], axis=0)

        #Return the downsampled & filtered data
        return data_ds
        
    def calc_igrf(self, xyz_plotting, ds_no):

        """
        Remove the effects of the geomagnetic field from the data.
        """
        
        #Define indices
        idx = self.misc_proj.indices_misc("mag")
        
        #Downsample & filter the dataset for IGRF field computation
        data_ds = self.downsamp_filter_mag(ds_no, idx["dirn"])

        #IGRF field's path
        igrf_loc = Path(self.misc_proj.proj_path() + "/data/IGRF/proj_igrf_field.npy")
        
        #Concatenate with or make new IGRF array
        if igrf_loc.is_file() == True:
            #Load the IGRF field
            prev_igrf = np.load(igrf_loc)

            #Ensure the shape of the IGRF field is different than current data
            assert prev_igrf.shape[0] != data_ds.shape[0]

            #Indices of latest data upload
            new_idx = np.where(data_ds[:, idx["upno"]] == max(data_ds[:, idx["upno"]]))[0]

            #Compute the IGRF field
            new_igrf = self.misc_proj.compute_igrf(data_ds, new_idx)

            #Concatenate previus and new IGRF computations
            igrf = np.concatenate((prev_igrf, new_igrf), axis=0)

            #Save IGRF array
            np.save(igrf_loc, igrf)

        else:
            #Indices
            pts = np.arange(data_ds.shape[0])

            #Compute the IGRF field
            igrf = self.misc_proj.compute_igrf(data_ds, pts)

            #Save IGRF array
            np.save(igrf_loc, igrf)

        #Plot the IGRF field
        xyz_plotting.plot_igrf_field("IGRF_Geomagnetic_Field", self.misc_proj)
        
    def rem_igrf(self, xyz_plotting, mode="igrf"):
        
        #Define indices
        mag_idx = self.misc_proj.indices_misc("mag")
        igrf_idx = self.misc_proj.indices_misc("igrf")

        #Load magnetic data
        data = np.load(self.misc_proj.proj_path() + "/data/tmp_mag_rov.npy")
        
        #Load project IGRF 
        igrf = np.load(self.misc_proj.proj_path() + "/data/IGRF/proj_igrf_field.npy")

        #Add an extra column to data
        data_igrf = np.zeros((data.shape[0], data.shape[1]+1))
        data_igrf[:,0:data_igrf.shape[1]-1] = data

        #Remove the computed 13th edition of the IGRF from the data
        if mode == "igrf":
            #Linear interpolation of IGRF to data SR
            interp_t = sp.interpolate.interp1d(igrf[:, igrf_idx["time"]], igrf[:, grf_idx["tot"]])
            igrf_interp = interp_t(data[:, mag_idx["time"]])

            #Remove the IGRF field from diurnal-corrected magnetic data
            data_igrf[:, mag_idx["igrf"]] = data_igrf[:, mag_idx["dirn"]] - igrf_interp

        #Remove the mean value of the computed IGRF from the data
        elif mode == "mean":
            #Compute the mean IGRF value
            igrf_mean = np.mean(igrf[:, igrf_idx["tot"]])

            #Remove the mean IGRF value from the diurnal-corrected magnetic data
            data_igrf[:, mag_idx["igrf"]] = data_igrf[:, mag_idx["dirn"]] - igrf_mean
            
        #Save IGRF corrected magnetic data
        np.save(self.misc_proj.proj_path() + "/data/tmp_mag_rov.npy", data_igrf)

        #Plot magnetic anomaly
        xyz_plotting.plot_mag_field("02Magnetic_Anomaly", self.misc_proj, "igrf")
        
    def pos_disp_filt(self, line):

        """
        Filter the magnetic data points by line and only retain positive relative displacements.
        Ideal for generating line profiles.
        ---
        line: a single line
        ---
        returns: data points acquired only moving toward the line endpoint
        """

        #Proper data indices
        idx = self.misc_proj.indices_misc("mag")

        #Start and end points of line
        A = np.array([line[0, idx["x"]], line[0, idx["y"]]]).reshape(2,1)
        B = np.array([line[line.shape[0]-1, idx["x"]], line[line.shape[0]-1, idx["y"]]]).reshape(2,1)

        #Matrix of datapoint locations
        P = np.array([line[:, idx["x"]], line[:, idx["y"]]]).reshape(2, -1)

        #Define the vector M
        M = B-A
        #Compute running paramater through orthogonal intersection
        t = np.dot(np.subtract(P,A).T, M)/np.sum(M*M)
        #Compute orothogonal distance to intersection from point to line
        I = np.add(A, M*t.T)

        #Compute line-project point displacement
        disp = np.sqrt((I[0,:] - I[0,0])**2  + (I[1,:] - I[1,0])**2).reshape(-1,1)

        #Relative point to point displacements
        disp_rel = np.zeros((line.shape[0], 1))
        disp_rel[1:] = disp[1:] - disp[:-1]

        #Positive displacement filter
        neg_disp = np.where(disp_rel[:] < 0)[0]
        line_profile = np.delete(line, neg_disp, axis=0) 

        #Return data points
        return line_profile,  disp
    
    def bin_datapts(self, data, bin_size, max_dist):
    
        """
        Find what data indices correspond to what bin of variable size.
        ---
        data: data to bin
        bin_size: the size of each bin
        ---
        returns: 2D list of indices corresponding binned data points
        """

        #Round to ceilling for last bin
        ceiling = bin_size * int(np.ceil(max_dist/bin_size))

        #Bin based on distance filter
        idx_bin, iprev = [], 0

        for i in range(bin_size, ceiling+bin_size, bin_size):
            #Bin values in current range
            indices = list(np.where(np.logical_and(iprev <= data[:], data[:] < i))[0])

            #Append indices to idx bin
            idx_bin.append(indices)

            #Setup previous index
            iprev = i

        #Return the bin indices
        return idx_bin
    
    def filter_bins(self, data, line, idx_bin, filt_idx="igrf", last_filt_idx="igrf"):
    
        """
        Filter data bins with a 1D Gaussian filter.
        Gaussian filter accounts for point distance to the mean bin point and standard deviation of the magnetic inputs.
        ---
        data: full dataset to normalize magnetic data
        line: current line data
        idx_bin: line indices for defined data bins.
        last_filt_idx: last index to filter
        ---
        returns: filtered line
        """

        #Return appropiate indices
        idx = self.misc_proj.indices_misc("mag")

        #Shape of the filtered line will be equal to idx_bin
        line_filt = np.zeros((len(idx_bin), line.shape[1]))

        #Process each bin
        for i in range(len(idx_bin)):        
            #Compute standard deviation from normalized datapoints
            sigma = np.std(line[idx_bin[i], idx[filt_idx]])

            #Ensure sigma measurement is sufficently high
            if sigma < 0.1:
                sigma = 0.1

            #Each bin is represented by the mean data point
            mean_point = np.zeros((1,2))
            mean_point[0,0] = np.mean(line[idx_bin[i], idx["x"]])
            mean_point[0,1] = np.mean(line[idx_bin[i], idx["y"]])

            #Compute distance from the mean point, collapsing the problem to 1D
            dist = np.sqrt(np.square(mean_point[0,0] - line[idx_bin[i], idx["x"]]) + 
                           np.square(mean_point[0,1] - line[idx_bin[i], idx["y"]]))

            #Compute the 1D Gaussian window for binned datapoints, distance to mean point is x, sigma defined by mag std.
            filt = gauss_window_1D(dist, sigma)

            #Convolve data with filter (res1 & res3), return the mode for line no. & upload no.
            res1 = np.apply_along_axis(lambda m: np.convolve(m, filt, mode='valid'), axis=0, 
                                       arr=line[idx_bin[i], 0: idx["line"]])

            res2 = np.array([sp.stats.mode(line[idx_bin[i], idx["line"]])[0], 
                             sp.stats.mode(line[idx_bin[i], idx["upno"]])[0]]).reshape(1,2)

            res3 = np.apply_along_axis(lambda m: np.convolve(m, filt, mode='valid'), axis=0, 
                                       arr=line[idx_bin[i], idx["dirn"]:idx[last_filt_idx]+1])

            #Horizontal stack of downsampled dataset
            line_filt[i] = np.hstack((res1, res2, res3, mean_point))[0].reshape(1,line_filt.shape[1])

        #Return filtered line
        return line_filt
    
    def filt2d_byline(self, y_bin_size, filt_idx="igrf"):
    
        """
        Filter the 2d dataset by convolving a 1D Gaussian kernel to bins along the y-dimension of the dataset by line. 
        The data are binned by displacement, meaning the x-dimension of the bin is dependent on the distance separting \
        points of similar displacement from the line start.
        ---
        y_bin_size: the height, in meters, of the bin.
        filt_idx: the last index of the data input to filter. 
        ---
        returns: 1D line profiles with positive displacement filter (high-res).
        """

        #Define indices
        idx = self.misc_proj.indices_misc("mag")

        #Load data
        data = np.load(self.misc_proj.proj_path() + "/data/tmp_mag_rov.npy")

        #Find line indices
        line_start, line_end = find_line_idx(data, idx["line"])

        #Initiate line dist array
        line_dist = np.zeros((len(line_start), 1))

        #Initiate profiles
        line_profiles = [[] for _ in range(len(line_start))]

        #Initiate counters
        bins_wo_pts, bin_len = 0, []

        #Loop through each line
        for i in range(len(line_start)): 
            #Define line & line start/stop
            line = data[line_start[i]:line_end[i]]

            #Apply a positive displacement filter for line profiles
            line_profile, disp = self.pos_disp_filt(line)

            #Append the line profile
            line_profiles.append(line_profile)

            #Record the length of each line
            line_dist[i] = max(disp)

            #Assign data points to bins
            bin_indices = self.bin_datapts(disp, y_bin_size, max(disp))

            #Bin lengths
            bin_len.extend([len(x) for x in bin_indices])

            #Process out any empty bins
            idx_bin = [x for x in bin_indices if x != []]

            #Any bins without data points
            if bin_indices != idx_bin:
                #How many, write to report later
                bins_wo_pts += len(bin_indices) - len(idx_bin)

            #Filter each bin into a single datapoint
            line_filt = self.filter_bins(data, line, idx_bin)

            #Concatenate with other lines
            if i == 0:
                #Initiate concat
                data_filt = line_filt
            else:
                #Concatenate data & line downsample
                data_filt = np.concatenate((data_filt, line_filt), axis=0)

        #Write to report
        if bins_wo_pts == 0:
            self.report.write("All bins have at least one data point.")
            self.report.write("There are "+str(len(bin_len))+" bins.")
            self.report.write("The Mean number of data points per bin is approximately "+str(round(np.mean(np.array(bin_len)),1))+". \n")
        else:
            self.report.write(str(bins_wo_pts)+" bin(s) have no datapoints. Consider increasing the y-axis bin size.")
            self.report.write("There are "+str(len(bin_len))+" possible bins.")
            self.report.write(("The Mean number of data points per bin is approximately "+str(round(np.mean(np.array(bin_len)),1))+". \n"))

        #Save filtered dataset
        np.save(self.misc_proj.proj_path() + "/data/tmp_mag_rov.npy", data_filt)

        #Return line profiles and line distances
        return line_profiles, line_dist
    
    def grid_mag(self, y_bin_size, x_space, line_info, xyz_plotting, grid_index="igrf"):
    
        """
        Grid the magnetic data from (optionally coordinate transformed) UTM coordinates.
        ---
        y_bin_size: bin size in the y-dimension.
        line_info: array containing all line lengths.
        misc_proj: miscillaneous project object.
        xyz_plotting: gird-plotting object.
        grid_index: index of the input data to grid.
        ---
        returns: output from MiscProj construct grid function and grid_idx; nodes in x- and y-dimensions [nx, ny].
        The magnetic grid is saved to tmp.
        """

        #Setup indices
        idx = self.misc_proj.indices_misc("mag")

        #Load dataset
        data = np.load(self.misc_proj.proj_path() + "/data/tmp_mag_rov.npy")

        #Number of y nodes based off line length and distance interval
        ny = int(min(np.floor(line_info[:]/y_bin_size))/2) * 2 - 1
        #Number of x nodes from desired node spacing
        nx = int((max(data[:, idx["x"]]) - min(data[:, idx["x"]])) / (2 * x_space)) * 2 - 1
        grid_idx = [nx,ny]

        #Construct a grid from filtered
        points, xyz, min_max = self.misc_proj.construct_grid(data, "mag", ["x", "y", grid_index], npts=[nx,ny], mtd="linear")

        #Define magnetic gird
        mag_grid = np.zeros((xyz[2].shape[1], xyz[2].shape[0]))
        mag_grid = xyz[2]

        #Save to tmp
        np.save(self.misc_proj.proj_path() + "/data/tmp_mag_grid.npy", mag_grid)

        #Plot gridded data
        xyz_plotting.plot_mag_grid("03Rotated_Filtered", min_max, points=points)

        #Return output
        return points, xyz, min_max, grid_idx
        
    def new_mag_vec(self, grid_shape, min_max, xyz_plotting, ambient_field=False, rtp=True):
    
        """
        Compute the magnetic field under a new magnetization & ambient field vector. 
        ---
        grid_shape: shape of the grid before padding
        min_max: minimum and maximum x/y coordinate list as [minx, maxx, miny, maxy]
        misc_proj: miscellaneous project object.  
        xyz_plotting: grid plotting object.
        ambient_field: if True; enter the ambient field vector, if False; assuming no remnant magnetization. ONLY FALSE IMPLEMENTED
        rtp: The new magnetization & ambient field vector becomes a RTP correction when the new magnetization \
        and ambient field is 90 incl, 0 decl. ONLY TRUE IMPLEMENTED
        ---
        Saves new vec mag data to disc and saves an image of the correction. 
        """

        #Load mag grid
        mag_grid = np.load(self.misc_proj.proj_path() + "/data/tmp_mag_grid.npy")
        
        #Load project IGRF 
        igrf = np.load(self.misc_proj.proj_path() + "/data/IGRF/proj_igrf_field.npy")  

        #Setup IGRF indices
        idx = self.misc_proj.indices_misc("igrf")

        #Compute mean XYZ components of magnetic field
        m1x = np.mean(igrf[:,idx["xcmp"]])
        m1y = np.mean(igrf[:,idx["ycmp"]])
        m1z = np.mean(igrf[:,idx["zcmp"]])

        #Compute length of magnetic field
        m1_len = np.sqrt(m1x**2 + m1y**2 + m1z**2)

        #Unit vector magnetization direction
        m1 = np.array([[m1x],[m1y],[m1z]])/m1_len

        #Ambient magnetic field vector sum of magnetization and ambiant (remnant+induced?) magnitizations
        if ambient_field == False:
            f1 = m1 

        #Inclination and declination of new magnetization and ambiant (remnant+induced?) magnitizations
        if rtp == True:
            m_inc_new = 90
            m_dec_new = 0

        #Obtain directional field vectors of new field
        m2y, m2x, m2z = dir_cos(m_inc_new, m_dec_new)

        #Unit vector new magnitization direction
        m2 = np.array([[m2x],[m2y],[m2z]])

        #New ambiant magnetic field equal to magnetization
        if rtp == True:
            f2 = m2

        #Compute the 2D fourier transfrom of the input data
        fft_grid = np.fft.fft2(mag_grid)

        #Compute wavenumber coordinates
        kx, ky = k_values(grid_shape[0], grid_shape[1], mag_grid.shape[1], mag_grid.shape[0], min_max)

        #Wavenumber coordinate vector
        k = np.sqrt(np.square(kx) + np.square(ky))

        #Compute phase operators
        with np.errstate(divide='ignore', invalid='ignore'):
            theta_m1 = m1[2] + 1j*(kx*m1[0] + ky*m1[1])/k
            theta_f1 = f1[2] + 1j*(kx*f1[0] + ky*f1[1])/k
            theta_m2 = m2[2] + 1j*(kx*m2[0] + ky*m2[1])/k
            theta_f2 = f2[2] + 1j*(kx*f2[0] + ky*f2[1])/k

        #Complex phase filter
        cphase = (theta_m2 * theta_f2) / (theta_m1 * theta_f1)

        #Replace nan with zero
        nan_idx = list(map(tuple, np.where(np.isnan(cphase.real))))
        cphase[tuple(nan_idx)] = 0

        #Convolve (multiply) the phase filter with fourier-transformed grid
        cgrid = fft_grid*cphase

        #Recover the pad size
        pad_sizes = (np.shape(mag_grid) - np.array(grid_shape[::-1])) / 2
        assert pad_sizes[0] == pad_sizes[1]
        pad_size = int(pad_sizes[0])
        
        #Inverse fourier transform the result
        mag_newvec = np.fft.ifft2(cgrid)[pad_size:grid_shape[1]+pad_size, pad_size:grid_shape[0]+pad_size]
        
        #Save grid
        np.save(self.misc_proj.proj_path() + "/data/tmp_mag_grid.npy", mag_newvec.real)

        #Plot grid
        xyz_plotting.plot_mag_grid("06New_Field_Vector", min_max, points=False)
        
    def line_intersections(self, xyz_plotting, basin_size):
    
        """
        Function to find all line intersections. 
        ---
        xyz_plotting: xyz_plotting object. 
        basin_size: The maximum size of the basin for a point to be considered a local minima. 
        ---
        Returns: a 3D line-data matrix, indices corresponding to data point intersections, and \
        indices corresponding to line intersections. 
        """

        #Load data
        data = np.load(self.misc_proj.proj_path() + "/data/tmp_mag_rov.npy")
                
        #Setup mag indices
        idx = self.misc_proj.indices_misc("mag")
        
        #Find line indices
        line_start, line_end = find_line_idx(data, idx["line"])

        #Interpolate to mean number of data points - avoid low point count line segments
        mean_dpts = int(np.mean([x - y for x, y in zip(line_end, line_start)]))

        #Initiate line matrix 3D Nodpts*[Xpt*Ypt*Zpt]*Line No.
        line_matrix = np.zeros((mean_dpts, 3, len(line_start)))

        #Initiate line lenghts array
        spatial_sample = np.zeros((len(line_start),1))

        #Loop through data lines interpolated by univariate spline
        for i in range(line_matrix.shape[2]):
            #Define line
            line = data[line_start[i]:line_end[i]]

            #Mean spatial sampling distance for each interpolated line
            spatial_sample[i] = np.sum(np.sqrt(np.square(line[1:, idx["x"]] - line[:-1, idx["x"]]) + 
                                               np.square(line[1:, idx["y"]] - line[:-1, idx["y"]])))/mean_dpts


            #Construct univariate spline objects for x,y,z coordinates
            univ_spline_x = sp.interpolate.UnivariateSpline(np.arange(line.shape[0]), line[:, idx["x"]])
            univ_spline_y = sp.interpolate.UnivariateSpline(np.arange(line.shape[0]), line[:, idx["y"]])
            univ_spline_z = sp.interpolate.UnivariateSpline(np.arange(line.shape[0]), line[:, idx["igrf"]])

            #Recover interpolated x,y,z coordinates
            x_new = univ_spline_x(np.linspace(0,line.shape[0],mean_dpts))
            y_new = univ_spline_y(np.linspace(0,line.shape[0],mean_dpts))
            z_new = univ_spline_z(np.linspace(0,line.shape[0],mean_dpts))

            #Fill line matrix without interpolation & sort along y-axis
            line_matrix[:,:,i] = np.array([x_new, y_new, z_new]).T[np.array([x_new, y_new, z_new]).T[:,1].argsort()]

        #Initiate
        line_cross_idx, dist_weights, ij_loop = [], [], []

        for i in range(line_matrix.shape[2]):
            for j in range(line_matrix.shape[2]):
                #Unique combinations of indices only - speedup
                if [j,i] not in ij_loop and i != j:
                    #Construct square matrix of distances between current i & j lines
                    line_dist_diff = np.zeros((line_matrix.shape[0], line_matrix.shape[0]))

                    #Essentially a convolution testing distances between all poitns of each line
                    for k in range(line_matrix.shape[0]):
                        #Compute distance between points (least-squares so no square root of distance)
                        line_dist_diff[k,:] = np.sqrt(np.square(line_matrix[k,0,i] - line_matrix[:,0,j]) + 
                                                      np.square(line_matrix[k,1,i] - line_matrix[:,1,j]))

                    #Specify maximum theoretical distance between intersecting points
                    dist_tol = math.sqrt((spatial_sample[i]/2.0)**2 + (spatial_sample[j]/2.0)**2)

                    #Minimum of array
                    dist_min = line_dist_diff.min()

                    #Proceed if distance within tolerance
                    if dist_min < dist_tol:
                        #Initiate deletion indices
                        del_idx = []

                        #Construct array of minimum indices along axis 0
                        imin = sp.signal.argrelextrema(line_dist_diff, np.less, order=basin_size, axis=0)
                        iminarr = np.zeros((len(imin[0]), 2), dtype=np.uint16)
                        iminarr[:,0] = imin[0]
                        iminarr[:,1] = imin[1]

                        #Construct arry of minimum indices along axis 1
                        jmin = sp.signal.argrelextrema(line_dist_diff, np.less, order=basin_size, axis=1)
                        jminarr = np.zeros((len(jmin[0]), 2), dtype=np.int16)
                        jminarr[:,0] = jmin[0]
                        jminarr[:,1] = jmin[1]

                        #Initiate mask for where arrays are equal, reference axis 0
                        mask_arr = np.zeros((iminarr.shape[0], 2), dtype=np.uint8)

                        #Check each minimum along axis 0 against all minima along axis 1
                        for k in range(iminarr.shape[0]):
                            mask_arr[k] = np.any(np.all(np.equal(iminarr[k], jminarr), axis=1))

                        #Convert to mask
                        mask = np.ma.make_mask(mask_arr, shrink=False)

                        #Are there any shared minima along axes
                        if np.any(mask.all(axis=1) == True, axis=0):
                            axesmin = iminarr[mask].reshape(-1, 2)

                            #Initiate index
                            k=0
                            #Loop through minima common in axis 0 & 1
                            while k < axesmin.shape[0] - 1:
                                #Absolute difference between minima indices avoiding duplicate searches
                                minidx_diff = np.abs(np.subtract(axesmin[k+1:], axesmin[k].reshape(-1,2)))

                                #Any diagonal minima points within search range
                                if np.any(np.all(minidx_diff <= np.full((1,2), basin_size), axis=1)) == True:
                                    #Find relative indices of any diagonal minima points
                                    diag_adj_min = np.where(np.all(minidx_diff <= np.full((1, 2), basin_size), axis=1))[0]

                                    #Initiate index
                                    m = 0
                                    #Loop through any diagonally adjacent minima only retaining the smalles
                                    while m < diag_adj_min.shape[0]:
                                        #Current reference minima
                                        ref = axesmin[k]
                                        #One diagonally adjacent minimum
                                        adj_idx = axesmin[k+1+diag_adj_min[m]]

                                        #Retain the smaller minimum
                                        if line_dist_diff[ref[0], ref[1]] < line_dist_diff[adj_idx[0], adj_idx[1]]:
                                            #Append index to be removed
                                            del_idx.append(k+1+diag_adj_min[m])


                                        elif line_dist_diff[ref[0], ref[1]] > line_dist_diff[adj_idx[0], adj_idx[1]]:
                                            #Append index to be removed
                                            del_idx.append(k)
                                            #Break loop
                                            m = diag_adj_min.shape[0]

                                        #Pass if points are equal

                                        #Increase index
                                        m+=1

                                #Increase index
                                k+=1

                            #Re-enforce the distance tolerance
                            del_idx.extend(list(np.where(line_dist_diff[axesmin[:,0], axesmin[:,1]] >= dist_tol)[0]))
                            del_idx = sorted(list(set(del_idx)))

                            #Remove non-minima adjacent diagonal indices
                            axesmin = np.delete(axesmin, del_idx, axis=0)

                            #Append both indices to list for future magnetic leveling
                            line_cross_idx.append(axesmin)

                            #Append distance weights
                            dist_weights.extend(list(line_dist_diff[axesmin[:,0], axesmin[:,1]])) #Extend or append

                            #Append current IJ values to loop preventing duplicate searches
                            ij_loop.append([i,j])

        #Write to report if there are no intersections
        if not ij_loop:             
            self.report.write("No line intersections identified. \n")
    
        else:
            #Plot magnetic differences at intersections
            xyz_plotting.plot_line_ints("magvals_atline_ints", line_matrix, line_cross_idx, ij_loop)
            
            
        #Return line intersection information
        return line_matrix, line_cross_idx, ij_loop
    
    def optimize_line_int(self, tot_it=1):
    
        """
        FUNCTION IN PROGRESS.
        Optimize differences in magnetic measurments at intersections.
        ---
        tot_it: Total iterations.
        ---
        Returns: Nothing for now, this function is very much so in progress. 
        """

        #Setup indices
        flat_ij = [x for y in ij_loop for x in y] #Flatten ij_loop indices
        line_indices = list(dict.fromkeys(flat_ij)) #Unique indices
        tie_idx = [max(set(flat_ij), key = flat_ij.count)] #Indices of all tie lines as a list (one for now)
        line_indices.remove(tie_idx[0]) #Indices only corresponding to lines

        #Construct a dictionary relating B value indices to ij indices
        Bdict = {}
        for i in range(len(line_indices)):
            Bdict[line_indices[i]] = int(i)

        #Initial B-values
        B = np.ones((len(line_indices), 1))

        #Construct diagonal matrix of distance weights
        Wd = np.identity((len(dist_weights))) * (1 / np.array(dist_weights))

        #Initialize 
        iterations, tot_it = 0, 1
        val_shape = [0]
        val_shape += [x.shape[0] for x in line_cross_idx]
        M = np.zeros((sum(val_shape), 2)) #Magnetic values
        J = np.zeros((sum(val_shape), len(line_indices))) #Jacobian matrix
        R = np.zeros((sum(val_shape), 1)) #Residual vector
        W = np.identity((len(dist_weights))) * (1 / np.array(dist_weights)) #Distance weighting matrix
        WtW = np.dot(W.T, W)
        val_shape[1:] = np.array(val_shape[:-1]) + np.array(val_shape[1:])

        while iterations < tot_it: 

            #Loop through intersections
            for i in range(len(ij_loop)):
                #Intersection between two tie lines; Not sure what I want to do with this case yet...
                if set(ij_loop[i]).issubset(tie_idx):
                    print("No solution for this case yet.")

                #Intersection between line and tie -> If this case only the problem is linear. 
                elif bool(set(tie_idx) & set(ij_loop[i])):
                    #Discern tie and line indices
                    lidx = [i for i, x in enumerate(ij_loop[i]) if x not in tie_idx][0]
                    tidx = [i for i, x in enumerate(ij_loop[i]) if x in tie_idx][0]
                    #Fill first column of the magnetic matrix M as the line and the second column as the reference tie line
                    M[val_shape[i]:val_shape[i+1], 0] = line_matrix[line_cross_idx[i][:,lidx], 2, ij_loop[i][lidx]]
                    M[val_shape[i]:val_shape[i+1], 1] = line_matrix[line_cross_idx[i][:,tidx], 2, ij_loop[i][tidx]]

                    #Fill residual vector R
                    R[val_shape[i]:val_shape[i+1]] = ((B[Bdict[ij_loop[i][lidx]]] * M[val_shape[i]:val_shape[i+1], 0] - 
                                                       M[val_shape[i]:val_shape[i+1], 1])**2).reshape(-1,1)

                    #Fill Jacobain matrix J
                    J[val_shape[i]:val_shape[i+1], Bdict[ij_loop[i][lidx]]] = 2 * (B[Bdict[ij_loop[i][lidx]]] * 
                                                                                         M[val_shape[i]:val_shape[i+1], 0]**2 - 
                                                                                         M[val_shape[i]:val_shape[i+1], 0] * 
                                                                                         M[val_shape[i]:val_shape[i+1], 1])

                #Intersection between two lines -> If any of this case the problem will be non-linear? Need iterations
                elif all(ij_loop[i]) not in tie_idx:
                    print("No solution for this case yet.")

            print("Sum of objective funciton at iteration: "+str(iterations)+", ", np.sum(R))

            B = B - np.dot(np.linalg.inv(np.dot(np.dot(J.T, WtW), J)), np.dot(np.dot(J.T, WtW), R))

            iterations += 1

        #Loop through intersections
        for i in range(len(ij_loop)):

            #Fill residual vector R
            R[val_shape[i]:val_shape[i+1]] = ((B[Bdict[ij_loop[i][lidx]]] * M[val_shape[i]:val_shape[i+1], 0] - 
                                                   M[val_shape[i]:val_shape[i+1], 1])**2).reshape(-1,1)

        print("Sum of objective funciton at iteration: "+str(iterations)+", ",np.sum(R))
        
class GridTransformations:
    
    """
    1) Mean NaN Neighbours & Spline Interpol & Pad Grid into Grid Transformations???
    """
    
    def __init__(self, misc_proj, report, dtype):
        self.misc_proj = misc_proj 
        self.report = report
        self.dtype = dtype
    
    def change_report(self, report_new):
        self.report = report_new
        
    def wgs84_utm(self, mode="wgs2utm"):

        """
        Function to transfrom wgs84 lat/long coordinates to UTM coordinates or vice versa. 
        Requires external libraray utm
        ---
        dtype: "mag" or "vlf".
        mode: "wgs2utm" converts WGS84 coordinates to UTM coordinates. "utm2wgs" converts UTM coordinates back to WGS coordinates. 
        ---
        returns: x and y utm coordinates.
        """

        #Indices from data type
        idx = self.misc_proj.indices_misc(self.dtype)

        #Load data
        data = np.load(self.misc_proj.proj_path() + "/data/tmp_"+self.dtype+"_rov.npy")

        #Appropiate direction of coordinate transform
        if mode == "wgs2utm":
            #Transform lat/long coordinates to UTM
            utm_cords = utm.from_latlon(data[:, idx["lat"]], data[:, idx["long"]])
            x = utm_cords[0]
            y = utm_cords[1]
            utm_zone = utm_cords[2]
            utm_letter = utm_cords[3]
            
            #Write to report
            self.report.write("WGS84 to UTM coordinate transform with zone "+str(utm_zone)+" "+utm_letter+". \n")

        elif mode == "utm2wgs":
            #Transform UTM coordinates to lat/long
            latlong_cords = utm.to_latlon(data[:, idx["x"]], data[:, idx["y"]], utm_zone, utm_letter)
            y = latlong_cords[0]
            x = latlong_cords[1]
            
            #Write to report
            self.report.write("UTM zone "+str(utm_zone)+" "+utm_letter+" to WGS84 coordinate transform. \n")

        else:
            #Raise error
            print("error")

        #Return x & y coordinates
        return x, y
    
    def rotate_grid_yaxis(self, utm_x, utm_y):
    
        """
        Rotate a UTM grid aligning acquisition lines with the y-axis based on the reference line.
        ---
        dtype: data-type either "mag" or "vlf". 
        utm_x: utm x-coordinates.
        utm_y: utm y-coordinates.
        ---
        returns: rotation angle, x- and y-coordinate shifts.
        """

        #Load data
        data = np.load(self.misc_proj.proj_path() + "/data/tmp_"+self.dtype+"_rov.npy")

        #Process the reference line coordinates
        P1, P2 = self.misc_proj.ref_line_cords()

        #Length of the reference line
        P1P2 = np.sqrt((P2[0]-P1[0])**2 + (P2[1]-P1[1])**2)

        #Minimum y-value of reference points
        pym = min(P1[1], P2[1])
        #Corresponding x-value for shift
        if pym == P1[1]:
            pxm = P1[0]
            #Non rotated reference line from origin
            M1 = np.array([P2[0]-pxm, P2[1]-pym])
        else:
            pxm = P2[0]
            #Non rotated reference line from origin
            M1 = np.array([P1[0]-pxm, P1[1]-pym])

        #Shift datapoints
        xy_shift = np.array([utm_x - pxm, utm_y - pym]).reshape(2,-1)

        #Rotated reference line from origin
        M2 = np.array([0, P1P2])

        #Rotation angle
        phi = 2*math.pi - math.acos(np.dot(M2, M1)/(P1P2**2))

        #Rotate x and y coordinates about origin
        A = np.array([[math.cos(phi), -math.sin(phi)],[math.sin(phi), math.cos(phi)]]).reshape(2,2)
        xy_new = np.matmul(A, xy_shift)

        #Append x and y coordinate columns to the end of data
        data_xygrd = np.zeros((data.shape[0], data.shape[1]+2))

        #Formualte new dataset
        data_xygrd[:, 0:data.shape[1]] = data
        data_xygrd[:, data_xygrd.shape[1]-2] = xy_new[0]
        data_xygrd[:, data_xygrd.shape[1]-1] = xy_new[1]

        #Save new dataset to tmp
        np.save(self.misc_proj.proj_path() + "/data/tmp_"+self.dtype+"_rov.npy", data_xygrd)

        #Return the rotation angle and shift
        return phi, pxm, pym
    
    def mean_neighbours_NaN(self, min_max, plt_name, xyz_plotting):
    
        """
        Process out NaN datapoints using an iterative mean neighbours algorithm.
        ---
        min_max: minimum and maximum x/y coordinate list as [minx, maxx, miny, maxy]
        plt_name: name of the plot
        xyz_plotting: plotting object
        ---
        loads & saves the mag grid to disc.
        """

        #Load magnetic grid
        mag_grid = np.load(self.misc_proj.proj_path() + "/data/tmp_"+self.dtype+"_grid.npy")

        #Neighbours kernel
        difs = np.array([[1,1,1,0,0,-1,-1,-1], [-1,0,1,-1,1,-1,0,1]])

        #Find NaN's outside of the loop the first time
        nan_idx = list(map(tuple, np.where(np.isnan(mag_grid))))
        nan_idx_arr = np.array(nan_idx).T
        nans_ext = np.repeat(nan_idx_arr[:, :, np.newaxis], 8, axis=2)

        #Initiate counter
        iterations, orig_no_nans = 0, nans_ext.shape[0]

        #Iteratively replace NaN's with surrounding mean values
        while nan_idx_arr.size > 0:

            #Compute eight surrounding indices for each point
            adj_idx = np.add(difs, nans_ext)

            #Only want to fill indices with the maximum number of adjacent points, then iterate
            surround_vals = []

            #Loop through for now
            for i in range(adj_idx.shape[0]):    
                #Cannot be outside of grid
                y_out = list(np.where(np.logical_or(adj_idx[i][0] <= 0, adj_idx[i][0] >= mag_grid.shape[0]))[0])
                x_out = list(np.where(np.logical_or(adj_idx[i][1] <= 0, adj_idx[i][1] >= mag_grid.shape[1]))[0])
                pts_out = list(set(y_out + x_out))

                #Surrounding points within the grid
                pts_in_grid = np.delete(adj_idx[i], pts_out, axis=1)

                #Surrounding points
                surround_pts = mag_grid[pts_in_grid[0], pts_in_grid[1]]

                #Find nans again...
                surround_nans = np.where(np.isnan(surround_pts))[0]

                #Remove nans
                surround_vals.append(np.delete(surround_pts, surround_nans))

            #Adjacent points to process this iteration
            no_pts_process = max([len(x) for x in surround_vals])

            #Indices of grid points to process
            indices = [i for i, x in enumerate(surround_vals) if len(x) == no_pts_process]

            #Assert we are only replacing NaN's
            assert np.isnan(mag_grid[nan_idx_arr[indices,0],nan_idx_arr[indices,1]]).all() == True

            #Set the grid points equal to mean of surrounding data points
            mag_grid[nan_idx_arr[indices,0],nan_idx_arr[indices,1]] = [np.mean(x) for i, x in enumerate(surround_vals) if i in indices]

            #Find NaN's
            nan_idx = list(map(tuple, np.where(np.isnan(mag_grid))))
            nan_idx_arr = np.array(nan_idx).T
            nans_ext = np.repeat(nan_idx_arr[:, :, np.newaxis], 8, axis=2)

            #Increase iteration counter
            iterations += 1

        #Write to report
        self.report.write(str(orig_no_nans)+" values assigned to NaN grid points after "+str(iterations)+" iterations.")

        #Save mag grid
        np.save(self.misc_proj.proj_path() + "/data/tmp_"+self.dtype+"_grid.npy", mag_grid)

        #Plot NaN-free data
        xyz_plotting.plot_mag_grid(plt_name, min_max, points=False)
        
    def spline_interp(self, xyz, grid_idx, min_max, yspace, xspace, plt_name, xyz_plotting):
    
        """
        Increase the node-density to near 1:1 scale of bin width via rectangular bivariate spline interpolation.
        ---
        xyz: x,y, and z coordinates for grid from meshgird
        grid_idx: nodes in the x- and y-dimension as [nx, ny]
        min_max: minimum and maximum x/y coordinate list as [minx, maxx, miny, maxy]
        yspace: y-target grid spacing
        xspace: original x-grid spacing
        plt_name: name of the plot
        xyz_plotting: grid plotting object
        ---
        returns: new x/y coordinates & grid nodes as [x, y] and [nx, ny] respectfully. 
        saves grid output to disc
        """

        #Load mag grid
        mag_grid = np.load(self.misc_proj.proj_path() + "/data/tmp_"+self.dtype+"_grid.npy")

        #XY coordinate lists
        xcords = xyz[0][0,:]
        ycords = xyz[1][:,0]

        #Interpolation spline object from original grid parameters
        interp_spline = sp.interpolate.RectBivariateSpline(ycords, xcords, mag_grid, kx=3, ky=3, s=0)

        #Scaling factors for x and y -> Get x&y as close to true y-spacing (bin size) as possible
        sampx = abs(min_max[1] - min_max[0]) / int(abs(min_max[1] - min_max[0]) / yspace)
        sampy = abs(min_max[3] - min_max[2]) / int(abs(min_max[3] - min_max[2]) / yspace)

        #XY new coordinate list
        xcords_new = np.arange(min_max[0], min_max[1] + sampx, sampx)
        ycords_new = np.arange(min_max[2], min_max[3] + sampy, sampy)

        #Make sure we're not changing min/max coordinates
        assert round(min(xcords),3) == round(min(xcords_new),3) and round(max(xcords),3) == round(max(xcords_new),3)
        assert round(min(ycords),3) == round(min(ycords_new),3) and round(max(ycords),3) == round(max(ycords_new),3)

        #New nx,ny
        nxs, nys = xcords_new.shape[0], ycords_new.shape[0]

        #Scale the grid appropiately by rectangular bivariate spline interpolation
        scale_grid = np.zeros((nys, nxs))
        scale_grid = interp_spline(ycords_new, xcords_new)
        
        #save scaled grid to tmp
        np.save(self.misc_proj.proj_path() + "/data/tmp_"+self.dtype+"_grid.npy", scale_grid)

        #Plot gridded data
        xyz_plotting.plot_mag_grid(plt_name, min_max, points=False)

        #Return new grid indices
        return [xcords_new, ycords_new], [nxs, nys]
    
    def pad_grid(self):
    
        """
        Iteratively pad the grid by linearly interpolating grid edges outwards.
        ---
        min_max: minimum and maximum x/y coordinate list as [minx, maxx, miny, maxy]
        ---
        save padded grid to disc and plot padded grid. 
        """

        #Load mag grid
        mag_grid = np.load(self.misc_proj.proj_path() + "/data/tmp_"+self.dtype+"_grid.npy")
        
        #Initiate padded grid
        pad = np.zeros((mag_grid.shape[0], mag_grid.shape[1]))
        pad = mag_grid
        nxp = mag_grid.shape[1]
        nyp = mag_grid.shape[0]
        pad_size = min(mag_grid.shape[0], mag_grid.shape[1])

        #Iteration counter
        iteration = 0

        #Iterate through padding and tapering process
        while iteration < pad_size:
            #Define values of four edges
            l0 = pad[0,:]
            l1 = pad[:,nxp-1]
            l2 = pad[nyp-1,:]
            l3 = pad[:,0]

            #Linear interpolation extending the size of each edge
            l0_int = sp.interpolate.interp1d(np.arange(nxp), l0, kind="linear")
            l0_new = l0_int(np.linspace(0, nxp-1, nxp+2))
            l1_int = sp.interpolate.interp1d(np.arange(nyp), l1, kind="linear")
            l1_new = l1_int(np.linspace(0, nyp-1, nyp+2))
            l2_int = sp.interpolate.interp1d(np.arange(nxp), l2, kind="linear")
            l2_new = l2_int(np.linspace(0, nxp-1, nxp+2))
            l3_int = sp.interpolate.interp1d(np.arange(nyp), l3, kind="linear")
            l3_new = l3_int(np.linspace(0, nyp-1, nyp+2))

            #Increase padding indices
            nyp, nxp = nyp + 2, nxp + 2

            #Average overlapping indices for each corner
            l0l1 = np.mean([l0_new[nxp-1], l1_new[0]])
            l0_new[nxp-1], l1_new[0] = l0l1, l0l1
            l1l2 = np.mean([l1_new[nyp-1], l2_new[0]])
            l1_new[nyp-1], l2_new[0] = l1l2, l1l2
            l2l3 = np.mean([l2_new[nxp-1], l3_new[0]])
            l2_new[nxp-1], l3_new[0] = l2l3, l2l3
            l3l0 = np.mean([l3_new[nyp-1], l0_new[0]])
            l3_new[nyp-1], l0_new[0] = l3l0, l3l0

            #Save previous array
            pad_prev = np.zeros((pad.shape[0], pad.shape[1]))
            pad_prev = pad

            #Pad the array with zeros
            pad = np.zeros((nyp, nxp))

            #Insert previous array and edges into padded array
            pad[1:nyp-1, 1:nxp-1] = pad_prev
            pad[0,:] = l0_new
            pad[:,nxp-1] = l1_new
            pad[nyp-1,:] = l2_new
            pad[:,0] = l3_new

            #Increase iteration
            iteration += 1

        #Save padded grid to disc
        np.save(self.misc_proj.proj_path() + "/data/tmp_"+self.dtype+"_grid.npy", pad)
        
    def undo_grid_shift(self, xy_new, phi, pxm, pym):

        """
        Undo a previous coordinate shift & rotation for easier data processing & gridding.
        ---
        xy_new: new X-Y coordinate list as two 1d arrays; [xnew, ynew].
        phi: previous rotation angle.
        pxm: previous coordinate shift along the x-axis.
        pym: previous coordinate shift along the y-axis.
        ---
        returns: UTM & LatLong coordinats as [X, Y] & [Lat, Long] respectfully.
        """

        #Construct 2D coordinate grids
        xi, yi = np.meshgrid(xy_new[0], xy_new[1])

        #Flatten to 1D
        xi_flat = xi.flatten()
        yi_flat = yi.flatten()
        assert xi_flat.shape == yi_flat.shape

        #Construct coordinate array
        new_cords = np.zeros((2, xi_flat.shape[0]))
        new_cords[0, :] = xi_flat
        new_cords[1, :] = yi_flat

        #Rotate x and y coordinates back
        A = np.array([[math.cos(-phi), -math.sin(-phi)],[math.sin(-phi), math.cos(-phi)]]).reshape(2,2)
        xy_rotate = np.matmul(A, new_cords)

        #Shift datapoints back
        UTM = np.array([xy_rotate[0,:] + pxm, xy_rotate[1,:] + pym]).reshape(2,-1)

        #Convert back to lat/long coordinates
        latlong = utm.to_latlon(UTM[0,:], UTM[1,:], 21, "T") #I NEED A WAY OF GETTING UTM INFO INTO & FROM PROJECT VARIABLES

        #Return UTM & Lat/Long coordinates
        return UTM, latlong