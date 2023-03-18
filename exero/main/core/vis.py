#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

#External Packages
import numpy as np

from matplotlib import pyplot as plt
from pathlib import Path

#Internal Packages
from .misc import truncate_colormap, find_line_idx

#Setup non-interactive matplotlib backend
plt.switch_backend('Agg')

#Classes and functions for data plotting

class DiurnalCorPlotting:
    
    def __init__(self, misc_proj, start, end):
        self.misc_proj = misc_proj
        self.start = start
        self.end = end
        
    def plot_proj_bas(self, bas, datum, name):
    
        """
        Plot each of the corresponding base station profiles, interpolated to the rover data points, for the entire \
        project.
        """

        #Figure size
        plt.rcParams.update({'figure.figsize':(16,8), 'figure.dpi':200})

        #Number of colors for plot
        num_colors = len(self.start)

        #Setup plot
        fig, ax = plt.subplots()

        #Select colormap
        cm = plt.get_cmap('gnuplot')

        #Select color for plot
        ax.set_prop_cycle(color=[cm(1.*i/num_colors) for i in range(num_colors)])

        #For each data upload
        for i in range(len(self.start)):

            #Define current rover and base station datasets
            bas_current = bas[self.start[i]:self.end[i]]

            #Datum
            diurnal_cor_datum = np.zeros((max(self.end - self.start))); diurnal_cor_datum.fill(datum)

            #Plotting base_station
            ax.plot(range(len(bas_current)), bas_current, label="Base station "+str(i+1)) 

        #Plot the mean datum
        ax.plot(range(max(self.end - self.start)), diurnal_cor_datum, 'k--', label="Mean base station") 

        #Labels and legends
        plt.ylabel("Magnetic Anomaly [nT]", fontsize=16)
        plt.yticks(fontsize=12)
        plt.xlabel('Sample number', fontsize=16)
        plt.xticks(fontsize=12)
        plt.legend(loc="lower right", fontsize=10)

        #Save to reports
        plt.gcf().savefig(self.misc_proj.proj_path() + "/reports/02_data_processing_images/" + 
                          name + ".png", bbox_inches='tight', pad_inches=0.0)
        
        #Clear plot
        plt.close()
        
    def plot_diurnal_cor(self, mag, diurnal_shifts, name):

        """
        Plot the magnetic current dataset pre- and pos- diurnal correction. 
        """
        
        #Setup indices
        idx = self.misc_proj.indices_misc("mag")

        #Figure size
        plt.rcParams.update({'figure.figsize':(16,8), 'figure.dpi':200})

        #Setup axis one
        fig, ax1 = plt.subplots()
        color = 'tab:blue'
        ax1.set_xlabel("Data point", fontsize=16)
        ax1.set_ylabel("Magnetic Anomaly [nT]", color=color, fontsize=16)

        #Plot raw magnetic data
        l1 = ax1.plot(range(self.end[len(self.end)-1]+1 - self.start[len(self.start)-1]), 
                     mag[self.start[len(self.start)-1]:self.end[len(self.end)-1]+1, idx["nT"]],
                     "k", label="Raw data")

        #Plot diurnal corrected data
        l2 = ax1.plot(range(self.end[len(self.end)-1]+1 - self.start[len(self.start)-1]), 
                     mag[self.start[len(self.start)-1]:self.end[len(self.end)-1]+1, idx["dirn"]],
                     color=color, label="Diurnal corrected")

        #Setup tick parameters
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.tick_params(axis="both", labelsize=12)

        #Setup axis two
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_xlabel("Data point", fontsize=16)
        ax2.set_ylabel("Correction magnitude [nT]", color=color, fontsize=16)

        #Plot the diurnal correction
        l3 = ax2.plot(range(self.end[len(self.end)-1]+1 - self.start[len(self.start)-1]),
                      diurnal_shifts[self.start[len(self.start)-1]:self.end[len(self.end)-1]+1],
                     "r", label="Diurnal correction")

        #Setup tick parameters
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.tick_params(axis="both", labelsize=12)

        #Legend
        ls = l1+l2+l3
        labs = [l.get_label() for l in ls]
        ax1.legend(ls, labs, loc="upper left", fontsize=10) 

        #Save to reports
        plt.gcf().savefig(self.misc_proj.proj_path() + "/reports/02_data_processing_images/" + 
                          name + ".png", bbox_inches='tight', pad_inches=0.0)
        
        #Clear plot
        plt.close()

class SpikeDataPlotting:
    
    def __init__(self, install_loc, proj_loc, plot_data, spikes, vlf_idx):
        self.install_loc = install_loc
        self.proj_loc = proj_loc
        self.plot_data = plot_data
        self.spikes = spikes
        self.vlf_idx = vlf_idx
        
    def changedataspike(self, spike_new):
        self.spikes[2] = spike_new
    
    def plot_data_despiking(self, indices, name):
        #Figure size
        plt.rcParams.update({'figure.figsize':(16,8), 'figure.dpi':200})

        #Norm base station data
        plt.plot(range(indices[0], indices[1]), self.plot_data[0][indices[0]:indices[1]], 
                 "k-", label="Norm base station")
        #Forward moving average
        plt.plot(range(indices[0], indices[1]), self.plot_data[1][indices[0]:indices[1]], 
                 "b--", label="Fwd moving avg")
        #Backward moving average
        plt.plot(range(indices[0], indices[1]), self.plot_data[2][indices[0]:indices[1]], 
                 "r--", label="Bwd moving avg")

        #Identified spikes from forward moving average
        label_added = False
        for i in range(len(self.spikes[0])):
            #Crop plot
            if indices[0] < self.spikes[0][i] < indices[1]:
                #One label only 
                if label_added == False:
                    plt.plot(self.spikes[0][i], self.plot_data[0][self.spikes[0][i]], "bo", ms=1, label="Fwd spikes")
                    label_added = True
                else:
                    plt.plot(self.spikes[0][i], self.plot_data[0][self.spikes[0][i]], "bo", ms=1)

        #Identified spikes from backward moving average
        label_added = False
        for i in range(len(self.spikes[1])):
            #Crop plot
            if indices[0] < self.spikes[1][i] < indices[1]:
                #One label only
                if label_added == False:
                    plt.plot(self.spikes[1][i], self.plot_data[0][self.spikes[1][i]], "ro", ms=1, label="Bwd spikes")
                    label_added = True
                else:
                    plt.plot(self.spikes[1][i], self.plot_data[0][self.spikes[1][i]], "ro", ms=1)

        #Shared spikes from forward and backward moving averages
        label_added = False
        for i in range(len(self.spikes[2])):
            #Crop plot
            if indices[0] < self.spikes[2][i] < indices[1]:
                #One label only
                if label_added == False:
                    plt.plot(self.spikes[2][i], self.plot_data[0][self.spikes[2][i]], "go", ms=2, label="Corrected spikes")
                    label_added = True
                else:
                    plt.plot(self.spikes[2][i], self.plot_data[0][self.spikes[2][i]], "go", ms=2)

        #Labels and legends
        plt.ylabel("Normalized Data", fontsize=16)
        plt.yticks(fontsize=12)
        plt.xlabel('Measurement', fontsize=16)
        plt.xticks(fontsize=12)
        plt.legend(loc="lower right", fontsize=10)

        #Save to reports
        plt.gcf().savefig(self.install_loc + self.proj_loc + "/reports/" + 
                          name + ".png", bbox_inches='tight', pad_inches=0.0)

        #Clear plot
        plt.close()
                
    def plot_vlf_sigstr(self, y, pT_soft_cutoff, pT_hard_cutoff, name):

        """
        Plotting function for pT signal strength QC of VLF data.
        """

        #Figure size
        plt.rcParams.update({'figure.figsize':(16,8), 'figure.dpi':200})

        #Field strength cutoff arrays
        pT_soft_cutoff_array = np.zeros((self.plot_data.shape[0])); pT_soft_cutoff_array.fill(pT_soft_cutoff)
        pT_hard_cutoff_array = np.zeros((self.plot_data.shape[0])); pT_hard_cutoff_array.fill(pT_hard_cutoff)

        #Plotting self.plot_data
        plt.plot(self.plot_data[:,self.vlf_idx["time"]], self.plot_data[:,self.vlf_idx["pT"]], 
                 "bo--", label="Field Strength") 
        plt.plot(self.plot_data[:,self.vlf_idx["time"]], pT_soft_cutoff_array, 'k--', label="pT Soft Cutoff")
        plt.plot(self.plot_data[:,self.vlf_idx["time"]], pT_hard_cutoff_array, 'k-', label="pT Hard Cutoff")
        plt.plot(self.plot_data[:,self.vlf_idx["time"]], y, "g-", label="Cubic Trend")

        #Plot removed points
        label_added = False
        for i in range(len(self.spikes)):
            #One label only 
            if label_added == False:
                plt.plot(self.plot_data[self.spikes,self.vlf_idx["time"]], self.plot_data[self.spikes,self.vlf_idx["pT"]],
                         "ro", label="Removed VLF Points")
                label_added = True
            else:
                plt.plot(self.plot_data[self.spikes,self.vlf_idx["time"]], self.plot_data[self.spikes,self.vlf_idx["pT"]],
                         "ro")

        #Plot labels
        plt.ylabel("Field Strength [pT]", fontsize=16)
        plt.yticks(fontsize=12)
        plt.xlabel("Time [s]", fontsize=16)
        plt.xticks(fontsize=12)
        plt.legend(loc="upper left", fontsize=10)

        #Save to reports
        plt.gcf().savefig(self.install_loc + self.proj_loc + "/reports/" + 
                          name + ".png", bbox_inches='tight', pad_inches=0.0)

        #Clear plot
        plt.close()
        
class XYZPlotting:
    
    def __init__(self, install_loc, proj_loc, mag_idx, vlf_idx):
        self.install_loc = install_loc
        self.proj_loc = proj_loc
        self.mag_idx = mag_idx
        self.vlf_idx = vlf_idx
        
    def compare_two_profs(self, data, smooth, name):
        
        """
        Plot a profile before and after smoothing
        """
        
        #Figure size
        plt.rcParams.update({'figure.figsize':(16,8), 'figure.dpi':200})

        #Setup axis one
        fig, ax1 = plt.subplots()
        color = 'tab:blue'
        ax1.set_xlabel("Measurement", fontsize=16)
        ax1.set_ylabel("Value", color=color, fontsize=16)

        #Plot one 
        l1 = ax1.plot(range(len(data)), data, "k", lw=1, label="Raw data")

        #Plot two 
        l2 = ax1.plot(range(len(data)), smooth, c=color, lw=2, label="Smoothed data")

        #Setup tick parameters
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.tick_params(axis="both", labelsize=12)

        #Setup axis two
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_xlabel("Measurement", fontsize=16)
        ax2.set_ylabel("Difference", color=color, fontsize=16)

        #Difference
        diff = abs(data-smooth)
        
        #Plot three
        l3 = ax2.plot(range(len(data)), diff, c=color, lw=1, label="Abs Diff")

        #Legend
        ls = l1+l2+l3
        labs = [l.get_label() for l in ls]
        ax1.legend(ls, labs, loc="upper right", fontsize=10) 

        #Save to reports
        plt.gcf().savefig(self.install_loc + self.proj_loc + "/reports/" + 
                          name + ".png", bbox_inches='tight', pad_inches=0.0)
        
        #Clear plot
        plt.close()
        
    def qcd_magvlf(self, name, misc_proj, del_mag=None, reloc_mag=None, del_vlf=None):
        
        #Load tmp mag
        mag_data = np.load(self.install_loc + self.proj_loc + "/data/tmp_mag_rov.npy")
        #Load tmp vlf (if exists)
        if Path(self.install_loc + self.proj_loc + "/data/vlf_rov.npy").is_file() == True:
            vlf_data = np.load(self.install_loc + self.proj_loc + "/data/tmp_vlf_rov.npy")
                
        #Line start
        line_start, _ = find_line_idx(mag_data, self.mag_idx["line"])
        
        #Construct grid
        points, xyz, min_max = misc_proj.construct_grid(mag_data, "mag", ["long", "lat", "elev"])

        #Figure size
        plt.rcParams.update({'figure.figsize':(16,8), 'figure.dpi':200})

        #Setup fig and axis
        fig, ax = plt.subplots()

        #Custom colormap
        cmap = plt.get_cmap("terrain")
        plt_cmap = truncate_colormap(cmap, 0.25, 1.0)

        #Contour elevation
        CL = ax.contour(xyz[0], xyz[1], xyz[2], 15, linewidths=0.5, colors='k', extent=[min_max[0], min_max[1], min_max[2], min_max[3]])
        CF = ax.pcolormesh(xyz[0], xyz[1], xyz[2], cmap=plt_cmap, vmin=min(points[:,2]), vmax=max(points[:,2]))

        #Plot mag datapoints
        ax.plot(points[:,0], points[:,1], "k.", ms=5, label="Magnetic datapoint")
        #Plot vlf data points (if exists)
        if Path(self.install_loc + self.proj_loc + "/data/vlf_rov.npy").is_file() == True:
            ax.plot(vlf_data[:,self.vlf_idx["long"]], vlf_data[:,self.vlf_idx["lat"]], "bx", ms=6, label="VLF datapoint")

        if del_mag != []:
            #Plot deleted magnetic datapoints
            ax.plot(del_mag[0], del_mag[1], "r.", ms=3, label="Deleted mag. datapoint")
            
        if reloc_mag != []:
            #Plot deleted magnetic datapoints
            ax.plot(reloc_mag[0], reloc_mag[1], "m.", ms=3, label="Relocated mag. datapoint")
        
        if del_vlf != []:
            #Plot deleted vlf datapoints
            ax.plot(del_vlf[0], del_vlf[1], "rx", ms=4, label="Deleted VLF datapoint")

        #Annotate line starting locations
        for i in range(len(line_start)):
            ax.annotate("line "+str(i+1), xy=(points[line_start[i],0], points[line_start[i],1]),
                        xycoords='data')

        #Plot labels
        plt.ylabel("latitude", fontsize=16)
        plt.yticks(fontsize=12)
        plt.xlabel("longitude", fontsize=16)
        plt.xticks(fontsize=12)
        plt.legend(loc="upper right", fontsize=10)

        #Contour labels
        ax.clabel(CL, inline=True, fmt="%d", fontsize=8)

        #Colorbar
        cbar = plt.colorbar(CF)
        cbar.ax.set_ylabel('elevation [m]')

        #Save to reports
        plt.gcf().savefig(self.install_loc + self.proj_loc + "/reports/" + 
                          name + ".png", bbox_inches='tight', pad_inches=0.0)

        #Clear plot
        plt.close()
        
    def concat_magvlf(self, name, misc_proj):
        
        """
        Plot the concatenated magnetic and vlf dasets
        """
        
        #Load preprocess mag
        mag_data = np.load(self.install_loc + self.proj_loc + "/data/preprocess/mag_rov.npy")
        #Load preprocess vlf if exists
        if Path(self.install_loc + self.proj_loc + "/data/preprocess/vlf_rov.npy").is_file() == True:
            vlf_data = np.load(self.install_loc + self.proj_loc + "/data/preprocess/vlf_rov.npy")
                
        #Line start
        line_start, _ = find_line_idx(mag_data, self.mag_idx["line"])
        
        #Construct grid
        points, xyz, min_max = misc_proj.construct_grid(mag_data, "mag", ["long", "lat", "elev"])

        #Figure size
        plt.rcParams.update({'figure.figsize':(16,8), 'figure.dpi':200})

        #Setup fig and axis
        fig, ax = plt.subplots()

        #Custom colormap
        cmap = plt.get_cmap("terrain")
        plt_cmap = truncate_colormap(cmap, 0.25, 1.0)

        #Contour elevation
        CL = ax.contour(xyz[0], xyz[1], xyz[2], 15, linewidths=0.5, colors='k', extent=[min_max[0], min_max[1], min_max[2], min_max[3]])
        CF = ax.pcolormesh(xyz[0], xyz[1], xyz[2], cmap=plt_cmap, vmin=min(points[:,2]), vmax=max(points[:,2]))

        #Plot mag datapoints
        ax.plot(points[:,0], points[:,1], "k.", ms=5, label="Magnetic datapoint")
        #Plot vlf data points (if exists)
        if Path(self.install_loc + self.proj_loc + "/data/preprocess/vlf_rov.npy").is_file() == True:
            ax.plot(vlf_data[:,self.vlf_idx["long"]], vlf_data[:,self.vlf_idx["lat"]], "bx", ms=6, label="VLF datapoint")

        #Open line locs
        f = open(self.install_loc + self.proj_loc + "/data/line_locs_rov.txt", "r")
        #Strip lines
        lines = [line.rstrip("\n") for line in f]
        #Begin reading line locations for this data upload
        for l in lines:
            #Line information as float
            line_loc = [float(x) for x in l.split(None)]
            #If tie line
            if int(line_loc[0]) == 0:
                #Annotate line starting locations
                ax.annotate("tie", xy=(line_loc[1], line_loc[2]), xycoords='data')
            #For all other lines
            else:
                #Annotate line starting locations
                ax.annotate("line "+str(int(line_loc[0])), xy=(line_loc[1], line_loc[2]), xycoords='data')
        #Close line locs
        f.close()

        #Plot labels
        plt.ylabel("latitude", fontsize=16)
        plt.yticks(fontsize=12)
        plt.xlabel("longitude", fontsize=16)
        plt.xticks(fontsize=12)
        plt.legend(loc="upper right", fontsize=10)

        #Contour labels
        ax.clabel(CL, inline=True, fmt="%d", fontsize=8)

        #Colorbar
        cbar = plt.colorbar(CF)
        cbar.ax.set_ylabel('elevation [m]')

        #Save to reports
        plt.gcf().savefig(self.install_loc + self.proj_loc + "/reports/" + 
                          name + ".png", bbox_inches='tight', pad_inches=0.0)

        #Clear plot
        plt.close()
        
    def plot_mag_field(self, name, misc_proj, z):

        """
        Contour plot for an intermediate processing result of the magnetic data.
        """

        #Load magnetic data
        data = np.load(self.install_loc + self.proj_loc + "/data/tmp_mag_rov.npy")

        #Construct grid
        points, xyz, min_max = misc_proj.construct_grid(data, "mag", ["long", "lat", z], npts=[500,500])

        #Figure size
        plt.rcParams.update({'figure.figsize':(16,8), 'figure.dpi':200})

        #Setup fig and axis
        fig, ax = plt.subplots()

        #Contour magnetic field values
        CL = ax.contour(xyz[0], xyz[1], xyz[2], 6, linewidths=0.5, colors='k', extent=[min_max[0], min_max[1], min_max[2], min_max[3]])
        CF = ax.pcolormesh(xyz[0], xyz[1], xyz[2], cmap="jet", vmin=min(points[:,2]), vmax=max(points[:,2]))

        #Plot data points for IGRF computation
        ax.plot(points[:,0], points[:,1], "k.", ms=1, label="Data point")

        #Plot labels
        plt.ylabel("latitude", fontsize=16)
        plt.yticks(fontsize=12)
        plt.xlabel("longitude", fontsize=16)
        plt.xticks(fontsize=12)
        plt.legend(loc="upper right", fontsize=10)

        #Contour labels
        ax.clabel(CL, inline=True, fmt="%d", fontsize=8)

        #Colorbar
        cbar = plt.colorbar(CF)
        cbar.ax.set_ylabel('Magnetic Field [nT]')

        #Save to reports
        plt.gcf().savefig(self.install_loc + self.proj_loc + "/reports/02_data_processing_images/" + 
                          name + ".png", bbox_inches='tight', pad_inches=0.0)

        #Clear plot
        plt.close()
        
    def plot_igrf_field(self, name, misc_proj):

        """
        Contour plot the IGRF field for a current project data points. 
        """

        #Load IGRF data
        igrf = np.load(self.install_loc + self.proj_loc + "/data/IGRF/proj_igrf_field.npy")

        #Construct grid
        points, xyz, min_max = misc_proj.construct_grid(igrf, "igrf", ["long", "lat", "tot"], npts=[200,200])

        #Figure size
        plt.rcParams.update({'figure.figsize':(16,8), 'figure.dpi':200})

        #Setup fig and axis
        fig, ax = plt.subplots()

        #Contour magnetic field values
        CL = ax.contour(xyz[0], xyz[1], xyz[2], 5, linewidths=0.5, colors='k', extent=[min_max[0], min_max[1], min_max[2], min_max[3]])
        CF = ax.pcolormesh(xyz[0], xyz[1], xyz[2], cmap="jet", vmin=min(points[:,2]), vmax=max(points[:,2]))

        #Plot data points for IGRF computation
        ax.plot(points[:,0], points[:,1], "k.", ms=5, label="Data point")

        #Plot labels
        plt.ylabel("latitude", fontsize=16)
        plt.yticks(fontsize=12)
        plt.xlabel("longitude", fontsize=16)
        plt.xticks(fontsize=12)
        plt.legend(loc="upper right", fontsize=10)

        #Contour labels
        ax.clabel(CL, inline=True, fmt="%d", fontsize=8)

        #Colorbar
        cbar = plt.colorbar(CF)
        cbar.ax.set_ylabel('External Magnetic Field [nT]')

        #Save to reports
        plt.gcf().savefig(self.install_loc + self.proj_loc + "/reports/02_data_processing_images/" + 
                          name + ".png", bbox_inches='tight', pad_inches=0.0)

        #Clear plot
        plt.close()
        
    def plot_mag_grid(self, name, min_max, points=False):

        """
        Contour plot for an intermediate processing result of the magnetic data.
        """

        #Load magnetic data
        data = np.load(self.install_loc + self.proj_loc + "/data/tmp_mag_grid.npy")
        
        #Figure size
        plt.rcParams.update({'figure.figsize':(16,8), 'figure.dpi':200})

        #Setup fig and axis
        fig, ax = plt.subplots()

        #Plot magnetic grid
        im = ax.imshow(data, extent=[min_max[0], min_max[1], min_max[2], min_max[3]], cmap="jet", origin="lower")
        if type(points) != bool:
            ax.plot(points[:,0], points[:,1], 'k.', ms=1, label="Data point")

        #Plot labels
        plt.ylabel("Y", fontsize=16)
        plt.yticks(fontsize=12)
        plt.xlabel("X", fontsize=16)
        plt.xticks(fontsize=12)
        plt.legend(loc="upper right", fontsize=10)

        #Colorbar
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('magnetic field [nT]')

        #Save to reports
        plt.gcf().savefig(self.install_loc + self.proj_loc + "/reports/02_data_processing_images/" + 
                          name + ".png", bbox_inches='tight', pad_inches=0.0)

        #Clear plot
        plt.close()
        
    def plot_line_ints(self, name, line_matrix, line_cross_idx, ij_loop):
    
        """
        Plotting function to show the differences in magnetic measurements at identified line intersections.
        ---
        line_matrix: 3D matrix from line_intersections
        line_cross_idx: point indices of line intersections from line_intersections
        ij_loop: line indices of line intersections from line_intersections
        """

        #Figure size
        plt.rcParams.update({'figure.figsize':(16,8), 'figure.dpi':200})

        #Loop through each line
        for i in range(line_matrix.shape[2]):
            #Plot each line
            plt.plot(line_matrix[:,0,i], line_matrix[:,1,i], "k-", ms=1)

        #Plot identitifed intersections
        for i in range(len(ij_loop)):
            #Difference in magnetic value at each point
            deltm = abs(line_matrix[line_cross_idx[i][:,0], 2, ij_loop[i][0]] - 
                        line_matrix[line_cross_idx[i][:,1], 2, ij_loop[i][1]])

            #Plot as a scatter with z-value deltm
            sc = plt.scatter(line_matrix[line_cross_idx[i][:,0], 0, ij_loop[i][0]], 
                             line_matrix[line_cross_idx[i][:,0], 1, ij_loop[i][0]], 
                             vmin = 0, vmax = 10, c=deltm, s=100, cmap="jet")

        #Colorbar
        cb = plt.colorbar(sc, label="Magnetic Difference [nT]")
        cb.ax.tick_params(labelsize=10)
        #Axes labels
        plt.xlabel("Shifted/Rotated X [m]", fontsize=16)
        plt.xticks(fontsize=12)
        plt.ylabel("Shifted/Rotated Y [m]", fontsize=16)
        plt.yticks(fontsize=12)
        #Add a grid
        plt.grid(True)

        #Save to reports
        plt.gcf().savefig(self.install_loc + self.proj_loc + "/reports/02_data_processing_images/" + 
                          name + ".png", bbox_inches='tight', pad_inches=0.0)

        #Clear plot
        plt.close()