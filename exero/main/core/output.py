#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

#External packages/modules
import datetime
import fnmatch
import numpy as np
import os
import re

#Classes of functions for data output

class Save:
    
    def __init__(self, install_loc, proj_loc, fileloc, col_rem):
        self.install_loc = install_loc 
        self.proj_loc = proj_loc
        self.fileloc = fileloc
        self.col_rem = col_rem
        
    def tmp_to_data(self, allow_pickle=True):
        
        """
        Load the tmp file as data.
        """
        
        #Load tmp
        data = np.load(self.fileloc, allow_pickle=allow_pickle)
        
        #Return data
        return data
        
    def data_to_tmp(self, data, allow_pickle=True):
        
        """
        Save the working data file to tmp.
        """
        
        #Save to tmp fileloc
        np.save(self.fileloc, data, allow_pickle=allow_pickle)

    def tmp_to_disc(self, dtype, ftype, savedir, allow_pickle=True):
        
        """
        Save tmp data to disc after the completion of a data processing stage.
        """
        
        #Dataset
        data = np.load(self.fileloc, allow_pickle=allow_pickle)
        
        #Initiate list of matched filenames
        match_fnames = []
        
        #Define current file name and save directory
        hist_file_dir = self.install_loc + self.proj_loc + "/data/" + savedir
        hist_file_name = re.sub('[\W]+', '', str(datetime.date.today())) + "_" + dtype + "_" + ftype + ".npy"
       
        #Properly name if multiple files are uploaded in one day
        for file in os.listdir(hist_file_dir):
            #Append potential duplicate name to list
            match_fnames.append(file)
                    
        #Append integers to the name until it is unique
        i=1
        while hist_file_name in match_fnames:
            #Change name
            hist_file_name = re.sub('[\W]+', '', str(datetime.date.today())) + "_" + dtype + "_" +  ftype + "_" + str(i) +".npy"
            #Change number
            i+=1  
    
        #Save all mag data
        if dtype == "mag":  
            
            #Save the data file
            np.save(hist_file_dir + hist_file_name, data, allow_pickle=allow_pickle)
            
        #Only save rov VLF data, cor is redundant
        if dtype == "vlf" and ftype != "cor":
            
            #Save the raw data file
            np.save(hist_file_dir + hist_file_name, data, allow_pickle=allow_pickle)
            
    def tmp_to_file(self, dtype, ftype, savedir):
        
        """
        Save magnetic or vlf data to file concatenating it with the files contents. 
        """
        
        #Dataset
        data = np.load(self.fileloc)
        
        #Initiate list of matched filenames
        match_fnames = []
        
        #Define current file name and save directory
        hist_file_dir = self.install_loc + self.proj_loc + "/data/" + savedir
        hist_file_name = dtype + "_" + ftype + ".npy"
       
        #Obtain names of files in folder
        for file in os.listdir(hist_file_dir):
            match_fnames.append(file)
                    
        #If magnetic or vlf datasets already present concatenate
        if hist_file_name in match_fnames:
            #Load dataset
            dataset = np.load(hist_file_dir + hist_file_name)
            #Concatenate with current data import
            dataset_concatenate = np.concatenate((dataset, data), axis=0)
            
            #Save concatenated dataset
            np.save(hist_file_dir + hist_file_name, dataset_concatenate)
        
        #If not present create file
        else:
            np.save(hist_file_dir + hist_file_name, data)
            
    def data_upload_no(self, data, keyword="bas"):
        
        """
        Append the data upload number to each filetype as a column prior to concatenation. 
        Assumes the keyword is uploaded each time. 
        """
        
        #Initiate counter
        counter = 0

        #Open import_list.txt
        f = open(self.install_loc + self.proj_loc + "/data/import_list.txt", 'r')

        #Read each line 
        for line in f:
            #Is the keyword in line of import list
            if keyword.lower() in line.lower():
                #Count occurrences
                counter += 1

        #Append column to data
        data_col = np.zeros((data.shape[0], data.shape[1]+1))
        data_col[:,0:data.shape[1]] = data
        
        #Set extra colum equal to the index representing the data upload
        data_col[:,data_col.shape[1]-1].fill(counter)
        
        #Return the dataset
        return data_col
            
    def pertinent_data(self, ftype):

        """
        Remove unnecessary data columns of respective file types for QC'd dataset finalization. 
        """
        
        #Load data volume
        data = np.load(self.fileloc)
        
        #Deletion columns
        del_cols = self.col_rem[ftype]
        
        #Delete columns
        data_out = np.delete(data, del_cols, axis=1)
        
        #Assign data upload number
        data_cols = self.data_upload_no(data_out)

        #Save finalized dataset to tmp
        np.save(self.fileloc, data_cols)
        
    def concat_magvlfbas(self):

        """
        Concatenate current upload of magnetic, vlf, or bas dataset with full porject dataset. 
        """

        #Concatenate diurnal corrected data with entire dataset
        for file in os.listdir(self.install_loc + self.proj_loc + "/data/"):
            #Define fileloc
            fileloc = self.install_loc + self.proj_loc + "/data/" + file

            #Identify .npy files
            if fnmatch.fnmatch(file, "*.npy"):

                #Identify rover magnetic and vlf files
                if fnmatch.fnmatch(file, "*rov*"):
                    #Save object
                    save = Save(self.install_loc, self.proj_loc, fileloc, self.col_rem)

                    #If magnetic data
                    if fnmatch.fnmatch(file, "*mag*"):
                        #Write to file or concatenate dataset
                        save.tmp_to_file("mag", "rov", "preprocess/")

                    #If vlf data
                    elif fnmatch.fnmatch(file, "*vlf*"):
                        #Write to file or concatenate dataset
                        save.tmp_to_file("vlf", "rov", "preprocess/")

                #Identify base station
                elif fnmatch.fnmatch(file, "*bas*"):
                    #Save object
                    save = Save(self.install_loc, self.proj_loc, fileloc, self.col_rem)

                    #Write to file or concatenate dataset
                    save.tmp_to_file("mag", "bas", "preprocess/")