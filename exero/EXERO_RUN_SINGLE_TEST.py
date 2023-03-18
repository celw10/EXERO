#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os, shutil, threading

#Define install location
install_loc = "/Users/christopherwilliams/Documents/1_PROFESSIONAL/1_Work/MagVLFConsult/01Codes/EXERO_V1.0"
#Define project location
proj_loc = "/tests"

#Function to move data files in and out of the watch directory
def move_txt_files(loc_dir, target_dir):
    file_names = os.listdir(loc_dir)
    for file_name in file_names:
        if file_name.endswith(".txt"):
            shutil.move(os.path.join(loc_dir, file_name), target_dir)
            
#Define the watch directory
watch_dir = install_loc + proj_loc + "/watchdir/"
#Define the sample data directory
sample_dir = install_loc + proj_loc + "/sample/"
#Define the reports directory
report_dir = install_loc + proj_loc + "/reports/"
#Define the data directory
data_dir = install_loc + proj_loc + "/data/"

#Clean the watch directory #OMIT IF YOUR TESTING MULTIPLE UPLOADS
move_txt_files(watch_dir, sample_dir)

#List of folders under data_dir where temporary data files are stored
data_clean_dirs = ["", "preprocess", "IGRF"]#

#Clean up each data directory
for d in data_clean_dirs:
    #Clean current directory of files
    for filename in os.listdir(data_dir+d):
        #Full path to file with folder dir and filename
        file_path = os.path.join(data_dir+d, filename)
        try:
            #Remove all files
            os.unlink(file_path)  
        except Exception:
            pass
            
#Clean up reports directory
for filename in os.listdir(report_dir):
    file_path = os.path.join(report_dir, filename)
    try:
        #Remove all files
        os.unlink(file_path)
    except Exception:
        pass


#Initiate data-dump in 10 seconds
threading.Timer(2.5, move_txt_files, args=[sample_dir, watch_dir]).start()
    
#Start file watch
os.system("python runner.py")