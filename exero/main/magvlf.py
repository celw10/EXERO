#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

#External packages
import numpy as np

from pathlib import Path

#Internal packages
from .core.input import ImportIdentify, DataOrg
from .core.vis import XYZPlotting
from .core.qc import MagneticDataQC, VLFDataQC
from .core.output import Save
from .core.process import MagDataProcess, GridTransformations
from .core.misc import join_lists
from .core.miscproj import MiscProj

#An automated rapid processing algorithm for ground magnetic and VLF data
class AutoProcessMagVLF:
    
    def __init__(self, install_loc, proj_loc, proj_bas_sr, proj_rov_sr, proj_line_space, proj_origin, 
                 mag_idx, vlf_idx, bas_idx, rov_idx_qc, rov_idx_in, cor_idx_qc, cor_idx_in, bas_idx_qc, test_idx_in,
                 col_rem, igrf_idx):
        
        #Project variables for Mag/VLF Autoprocessing
        self.install_loc = install_loc
        self.proj_loc = proj_loc
        self.proj_bas_sr = proj_bas_sr
        self.proj_rov_sr = proj_rov_sr
        self.proj_line_space = proj_line_space
        self.proj_origin = proj_origin
        self.mag_idx = mag_idx
        self.vlf_idx = vlf_idx
        self.bas_idx = bas_idx
        self.rov_idx_qc = rov_idx_qc
        self.rov_idx_in = rov_idx_in
        self.cor_idx_qc = cor_idx_qc
        self.cor_idx_in = cor_idx_in
        self.bas_idx_qc = bas_idx_qc
        self.test_idx_in = test_idx_in
        self.col_rem = col_rem
        self.igrf_idx = igrf_idx
    
    def import_data(self, filepath):
            
        #Identify import files object
        import_identify = ImportIdentify(filepath, self.install_loc, self.proj_loc)

        #Return file time from identified import file
        ftype = import_identify.identify_file_type()
        
        #Process import files object
        data_org = DataOrg(filepath, self.install_loc, self.proj_loc, ftype, self.rov_idx_in, self.cor_idx_in, 
                           self.test_idx_in)

        #Keep records of file imports
        data_org.make_import_hist_list()

        #Read in text file
        data, l = data_org.open_text()

        #Additional steps unique to rov and cor files containing both mag and vlf data
        if ftype == "rov" or ftype == "cor":

            #Separate magnetic and VLF data in each file, includes noisy data point removal
            sep_data, count = data_org.sep_magvlf_data(data, l)
            
            #Only mag data
            if [x for x in sep_data[1] if x == []]:
                #The VLF data list is empty.
                file_out = data_org.process_magvlf_data(sep_data[0], count)

                #Convert lists to an array (still in a list of [mag, vlf])
                out_data = data_org.lists_to_array(file_out)
                
                #Define fileloc
                fileloc = self.install_loc + self.proj_loc + "/data/tmp_mag_"+ftype+".npy"
                
                #Save object
                save = Save(self.install_loc, self.proj_loc, fileloc, self.col_rem)

                #Save imported mag data to tmp
                save.data_to_tmp(out_data)

            #VLF data present.
            else:
                #For mag and vlf data
                for i in range(2):
                    #Mag and VLF data are present
                    file_out = data_org.process_magvlf_data(sep_data[i], count)

                    #Convert lists to an array (still in a list of [mag, vlf])
                    out_data = data_org.lists_to_array(file_out)
                    
                    #Define dtype
                    dtype_dict = {0: "mag", 1: "vlf"}
                    dtype = dtype_dict[i]
                    
                    #Define fileloc
                    fileloc = self.install_loc + self.proj_loc + "/data/tmp_"+dtype+"_"+ftype+".npy"

                    #Save object
                    save = Save(self.install_loc, self.proj_loc, fileloc, self.col_rem)
                    
                    #Save imported and mag and vlf data to tmp
                    save.data_to_tmp(out_data)

        #Save base station data to disc    
        elif ftype == "bas":
            #Convert lists to an array
            file_out = data_org.lists_to_array(data)
            
            #Define fileloc
            fileloc = self.install_loc + self.proj_loc + "/data/tmp_mag_bas.npy"
            
            #Save object
            save = Save(self.install_loc, self.proj_loc, fileloc, self.col_rem)

            #Save imporeted and raw processed mag and vlf data to disc
            save.data_to_tmp(file_out)

        #Perform QC test with the "test" data
        else:
            #Ensure that the data decontamination passed and write the results in /reports/
            data_org.test_data(data)
            
    """
    Some things do do.
    1) Get rid of cor, we can do the correction ourselves. Importing this data only slows the process down & complicates things. \
    But I have a lot more core data than I do rov.
    """
            
    def data_qc(self):
        
        #Data QC Report
        data_qc_report = open(self.install_loc + self.proj_loc + "/reports/01_data_qc_report.txt", "w+")
        
        #Misc. project object
        misc_proj = MiscProj(self.install_loc, self.proj_loc, self.proj_bas_sr, self.proj_rov_sr, self.proj_line_space, 
                             self.proj_origin, self.mag_idx, self.vlf_idx, self.bas_idx_qc, self.rov_idx_qc, self.rov_idx_in,
                             self.cor_idx_qc, self.cor_idx_in, self.test_idx_in, self.col_rem, self.igrf_idx)
        
        #Sort tmp files for QC module
        files, vlf_file, vlf = misc_proj.sort_tmps()
                
        #QC for magnetic data
        for i in range(len(files)):
            #Define the filepath
            fileloc = self.install_loc + self.proj_loc + "/data/" + str(files[i])
           
            #Identify import files object
            identify_imports = ImportIdentify(fileloc, self.install_loc, self.proj_loc)
            
            #Save object
            save = Save(self.install_loc, self.proj_loc, fileloc, self.col_rem)

            #Return file time from identified import file
            ftype = identify_imports.identify_file_type()
            
            #Magnetic data QC object
            mag_data_qc = MagneticDataQC(self.install_loc, self.proj_loc, fileloc, ftype, self.proj_bas_sr, 
                                         self.proj_rov_sr, self.proj_line_space, self.proj_origin, data_qc_report, 
                                         self.mag_idx, self.vlf_idx, self.bas_idx_qc, self.rov_idx_qc, self.cor_idx_qc, 
                                         self.col_rem)
            
            #Data indices for file type
            index = mag_data_qc.indices_qc()
            
            #Convert HHMMSS to seconds with reference time
            mag_data_qc.time_to_seconds()
                                              
            #Ensure an appropiate data quality recording for each magnetic measurement
            del_pts_dq = mag_data_qc.qc_data_quality()
            
            #If base
            if ftype == "bas":
                #Linear interpolation of any base station magnetic spikes -> Have a look at bas spike interp tomorrow. 
                bas = mag_data_qc.int_spikes_movavg()
                
                #Save QC'd base station record to tmp
                np.save(fileloc, bas)
                
            #If rov/cor
            else:
                #Compute the change in time and distance to each datapoint
                line_points = mag_data_qc.rovcor_delta_dt()

                #Decipher line numbers
                line_start, line_end, del_pts_line = mag_data_qc.rovcor_lineid(line_points, mag_data_qc)
                                
                #Complete XYZ data positional de-noising for rov & cor data
                reloc_pts_xy = mag_data_qc.rovcor_xyz_denoise()

                #Compute pt distance, total distance, and total elevation acquisition statistics
                mag_data_qc.day_acqui_stats()
            
            #Ensure non-zero magnetic datapoints
            del_pts_0mag = mag_data_qc.remove_mag_zeros()
            
            #Finalize magnetic dataset
            save.pertinent_data("mag_" + ftype)
            
            #Write blank line to report
            data_qc_report.write("\n")
                
        #Proceed with VLF data QC if present
        if vlf == True:
            #Change ftype to "mag" for the mag_data_qc object
            mag_data_qc.changeftype("mag")
            
            #Define the filepath
            fileloc = self.install_loc + self.proj_loc + "/data/" + str(vlf_file)
            
            #Magnetic data QC object (for time correction)
            magvlf_data_qc = MagneticDataQC(self.install_loc, self.proj_loc, fileloc, "vlf", self.proj_bas_sr, 
                                            self.proj_rov_sr, self.proj_line_space, self.proj_origin, data_qc_report, 
                                            self.mag_idx, self.vlf_idx, self.bas_idx_qc, self.rov_idx_qc, self.cor_idx_qc, 
                                            self.col_rem)
            
            #Convert HHMMSS to seconds with reference time
            magvlf_data_qc.time_to_seconds()
            
            #VLF data QC object
            vlf_data_qc = VLFDataQC(self.install_loc, self.proj_loc, fileloc, data_qc_report, self.vlf_idx)
            
            #QC VLF data magnitude
            vlf_delpts = vlf_data_qc.vlf_sigstr_qc(magvlf_data_qc)
            
            #If the VLF data passed magnitude QC
            if Path(fileloc).is_file() == True:
                #VLF line ID 
                vlf_data_qc.line_id_vlf(mag_data_qc, magvlf_data_qc)

                #Save object
                save = Save(self.install_loc, self.proj_loc, fileloc, self.col_rem)

                #Finalize VLF dataset
                save.pertinent_data("vlf_rov")

            #Concatenate current upload with project data
            save.concat_magvlfbas()
            
        #Only plot for rov
        if ftype == "rov":
            #XYZ plotting object
            xyz_plotting = XYZPlotting(self.install_loc, self.proj_loc, self.mag_idx, self.vlf_idx)
            
            #Join deleted points together
            del_pts_x, del_pts_y = join_lists([del_pts_line[0], del_pts_dq[0], del_pts_0mag[0]],
                                              [del_pts_line[1], del_pts_dq[1], del_pts_0mag[1]])

            #Plot QC'd magnetic and VLF data point XYZ locations
            xyz_plotting.qcd_magvlf("01_data_qc_images/qc_magvlf_xyz", misc_proj, del_mag=[del_pts_x, del_pts_y], 
                                    reloc_mag=reloc_pts_xy, del_vlf=vlf_delpts)

            #Plot concatenated dataset
            xyz_plotting.concat_magvlf("01_data_qc_images/qc_magvlf_concat", misc_proj)
        
        #Close report file
        data_qc_report.close()
        
        """
        There is a little more to be desired here. 
        1) Get rid of Cor, but I have a lot of cor data & not rov.
        2) What about a magntiude QC? Any high datapoints known to be acquired over metal or in the vacinity of metal?
        3) I can put an object into another object as an instance, MiscProj should handle all project-variable based functions
        
        Distant goals
        
        1) The base-station de-spiking algorithm needs to be imporved. I have no idea what is noise and what is signal. \
        I'm just removing every "spike" I can get my hands on. Is there any way to incorporate space weather into this? \
        Can I devise some sort of warning system into the magnetic data when space weather spikes are potentially contamenating the data?
        
        2) Line ID and point relocation can be a little bit inconsistent. I think incorporating AI into these processes would benefit \
        them greatly. 
        """
        
    def mag_data_processing(self):
        
        #Data processing report
        report = open(self.install_loc + self.proj_loc + "/reports/02_data_processing_report.txt", "w+")
        
        #Miscellaneous project object
        misc_proj = MiscProj(self.install_loc, self.proj_loc, self.proj_bas_sr, self.proj_rov_sr, self.proj_line_space, 
                             self.proj_origin, self.mag_idx, self.vlf_idx, self.bas_idx, self.rov_idx_qc, self.rov_idx_in,
                             self.cor_idx_qc, self.cor_idx_in, self.test_idx_in, self.col_rem, self.igrf_idx)
        
        #XYZ Plotting object
        """I should be able to deal with my plotting objects better."""
        xyz_plotting = XYZPlotting(self.install_loc, self.proj_loc, self.mag_idx, self.vlf_idx)
                
        #Data corrections object
        mag_data_process = MagDataProcess(misc_proj, report)
        
        #Grid transformations object
        grid_transformations = GridTransformations(misc_proj, report, "mag")
                
        #Diurnal correction
        mag_data_process.diurnal_correction(xyz_plotting)
        
        #Compute the geomagnetic field
        """I don't need to filter data, I just need to downsample the data points to compute the IGRF field more quickly."""
        mag_data_process.calc_igrf(xyz_plotting, 9) #Downsampling number
        
        #Remove the geomagnetic field
        """RTP seems to do this as well. I'll remove the mean IGRF field before to allow for comparison before & after RTP."""
        mag_data_process.rem_igrf(xyz_plotting, mode="mean")

        #Convert Lat/Long coordiantes to UTM
        utm_x, utm_y = grid_transformations.wgs84_utm()

        #Rotate grid such that reference line aligns with y-axis
        phi, pxm, pym = grid_transformations.rotate_grid_yaxis(utm_x, utm_y)
        
        #Find grid intersections
        line_matrix, line_cross_idx, ij_loop = mag_data_process.line_intersections(xyz_plotting, 11) #Basin size for local minima

        #Level grid
        """Function in progress."""

        #Filter magnetic data
        line_profiles, line_dist = mag_data_process.filt2d_byline(3) #y_bin_size #FIGURE OUT WHAT TO DO WITH LINE PROFILES

        #Grid magnetic data
        points, xyz, min_max, grid_idx = mag_data_process.grid_mag(3, 15, line_dist, xyz_plotting) #y_bin_size & x grid spacing

        #Process out NaN points using iterative mean-neighbours
        grid_transformations.mean_neighbours_NaN(min_max, "04Mean_NaN_Neighbours", xyz_plotting)

        #Interpolate grid to bin size in the x and y dimensions
        xy_new, new_grid_idx = grid_transformations.spline_interp(xyz, grid_idx, min_max, 3, 15, "05Interpolated_GrdPts", xyz_plotting) 

        #Pad mag grid for FFT
        grid_transformations.pad_grid()

        #RTP correction
        mag_data_process.new_mag_vec(new_grid_idx, min_max, xyz_plotting)
        
        #Undo the shift/grid rotation
        UTM, latlong = grid_transformations.undo_grid_shift(xy_new, phi, pxm, pym)

        #Close data processing report
        report.close()
        
        """
        Short term goals. 
        1) Consolodate & save the processed dataset under process. 
        2) I don't need the filter when I compute the geomagnetic field. 
        3) Cleaning up object use, namely xyz_plotting. 
        
        Distant goals.
        1) The grid leveling function is nearly complete, but not fully complete. I am really waiting on a more reliant method for tie \
        line identification. I also need to consider how to deal with a tie-tie intersection. This should probably be a separte optimiation \
        from the tie-line or line-line interception optimizaiton. 
        """
        
        #Say you're
        print("done")