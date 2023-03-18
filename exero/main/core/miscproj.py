#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

#External Packages
import fnmatch
import numpy as np
import os
import pyIGRF
import utm

from matplotlib import colors as colors
from scipy.interpolate import griddata
from scipy.interpolate import interp1d

#Miscellaneous functions using any project variable

class MiscProj:
    
    """
    Class of miscellaneous functions with only project variables
    """
    
    def __init__(self, install_loc, proj_loc, proj_bas_sr, proj_rov_sr, proj_line_space, proj_origin, mag_idx, vlf_idx, 
                 bas_idx, rov_idx_qc, rov_idx_in, cor_idx_qc, cor_idx_in, test_idx_in, col_rem, igrf_idx):
        
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
        self.test_idx_in = test_idx_in
        self.col_rem = col_rem
        self.igrf_idx = igrf_idx
        
    def indices_misc(self, dtype):
        
        """
        Return the proper idnices based on data type.
        ---
        dtype: the data type
        """
        
        #Appropiate indices based on data type
        if dtype == "mag":
            idx = self.mag_idx
        elif dtype == "vlf":
            idx = self.vlf_idx
        elif dtype == "bas":
            idx = self.bas_idx
        elif dtype == "igrf":
            idx = self.igrf_idx
            
        return idx
    
    def proj_path(self):
        
        """
        Return the project path install_loc + proj_loc.
        """
        
        project_path = self.install_loc + self.proj_loc 
        
        
        return project_path
        
    def sort_tmps(self):
        
        """
        Sort the tmp files for QC module after import module.
        """
        
        #Count and sort files
        files, vlf = [], False
        for file in os.listdir(self.install_loc + self.proj_loc + "/data/"):
            if fnmatch.fnmatch(file, "*mag*"):
                #Base station data first to extract reference time
                if fnmatch.fnmatch(file, "*bas*"):
                    files.insert(0, file)
                else:
                    files.append(file)
            elif fnmatch.fnmatch(file, "*vlf*"):
                #Vlf file - only one for rov kept
                vlf_file, vlf = file, True
                
        #Return sorted files & vlf flag  
        return files, vlf_file, vlf    
    
    def construct_grid(self, data, dtype, xyz, npts=[200,200], mtd="linear"):

        """
        Construct a grid for XYZ data.
        """
        
        #Return file type indices
        idx = self.indices_misc(dtype)

        #Min/Max values for gridding/plotting
        xmin = min(data[:,idx[xyz[0]]])
        xmax = max(data[:,idx[xyz[0]]])
        ymin = min(data[:,idx[xyz[1]]])
        ymax = max(data[:,idx[xyz[1]]])

        #X/Y spacing from number of poitns
        xspace = (xmax-xmin)/float(npts[0]-1)
        yspace = (ymax-ymin)/float(npts[1]-1)

        #Grid points for elevation contour
        xarr = np.arange(xmin, xmax+xspace, xspace)[0:npts[0]]
        yarr = np.arange(ymin, ymax+yspace, yspace)[0:npts[1]]
        xi, yi = np.meshgrid(xarr, yarr)
        
        #Points
        points = np.zeros((len(data),3))
        points[:,0] = data[:,idx[xyz[0]]]
        points[:,1] = data[:,idx[xyz[1]]]
        points[:,2] = data[:,idx[xyz[2]]]

        #Elevation grid from scipy griddata
        zi = griddata(points[:,0:2], points[:,2], (xi, yi), method=mtd)
        
        return points, [xi, yi, zi], [xmin, xmax, ymin, ymax]
    
    def compute_igrf(self, data, pts):
        
        """
        Compute and save relevant IGRF field from lat/long/elev information in a magetic dataset. 
        Using pyIGRF external library, 13th generation of the International Geomagentic Reference Field.
        """
        
        #IGRF array, long/lat/elev/time/IGRFx/IGRFy/IGRFz/IGRFtot
        igrf = np.zeros((pts.shape[0], 8))

        #Compute the entire IGRF field
        for i in range(pts.shape[0]):
            #Compute the sevent-component IGRF field
            IGRF = pyIGRF.igrf_value(data[pts[i], self.mag_idx["lat"]], data[pts[i], self.mag_idx["long"]],
                                     data[pts[i], self.mag_idx["elev"]]/1000., 2020)
            #Fill the igrf array
            igrf[i,0] = data[pts[i], self.mag_idx["long"]]
            igrf[i,1] = data[pts[i], self.mag_idx["lat"]]
            igrf[i,2] = data[pts[i], self.mag_idx["elev"]]
            igrf[i,3] = data[pts[i], self.mag_idx["time"]]
            igrf[i,4] = IGRF[3]
            igrf[i,5] = IGRF[4]
            igrf[i,6] = IGRF[5]
            igrf[i,7] = IGRF[6]
            
        #Return the IGRF field
        return igrf
    
    def ref_line_cords(self, out="utm", rtype="wgs84"):
    
        """
        Convert the reference line coordinates to/from WGS84 and UTM. 
        Requires external library utm.
        ---
        out: coordinate output, "wgs84" or "utm".
        rtype: reference line coordinate type, "wgs84" or "utm". 
        ---
        returns: reference line coordinates in desired format. 
        """
        
        #Convert from wgs84 to UTM
        if rtype == "wgs84" and out == "utm":
            #For project reference line
            P1utm = utm.from_latlon(self.proj_origin[1], self.proj_origin[0])
            P2utm = utm.from_latlon(self.proj_origin[3], self.proj_origin[2])
            P1 = np.array([P1utm[0], P1utm[1]])
            P2 = np.array([P2utm[0], P2utm[1]])
        
        else:
            print("SHOULDN'T BE HERE")

        #Return reference line coordinates
        return P1, P2