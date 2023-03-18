#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

#This is my program runner, keeps an eye on the files

#External packages
import datetime
import numpy as np
import os, os.path
import time

from watchdog.observers import Observer #External program
from watchdog.events import FileSystemEventHandler #External program

#Internal package
from main.magvlf import AutoProcessMagVLF

#Exero installation location and current project location
install_loc = "/Users/christopherwilliams/Documents/1_PROFESSIONAL/1_Work/MagVLFConsult/01Codes/EXERO_V1.0"
proj_loc = "/tests"
#Monitoring location for current project's data upload
monitoring_loc = "/watchdir/"

#Monitoring path
monitoring_path = install_loc + proj_loc + monitoring_loc

#Project specific definitions
proj_bas_sr = 4.0
proj_rov_sr = 1.0
proj_line_space = 44.0 #BP # 44.0 IAM EXT
proj_origin = np.array([-058.8382338, 047.7986949, -058.8432509, 047.8021942]) #IAMEXT np.array([-058.9706476, 047.7282913, -058.9889390, 047.7422948]) #BP
#Project data indices - Post-QC
mag_idx = {"lat": 0, "long": 1, "elev": 2, "nT": 3, "time": 4, "line": 5, "upno": 6, "dirn": 7, "igrf": 8, "x": 9, "y": 10}
vlf_idx = {"lat": 0, "long": 1, "time": 2, "freq": 3, "ip": 4, "op": 5, "h1": 6, "h2": 7, "pT": 8, "line": 9}
bas_idx = {"time": 0, "nT": 1, "upno": 2}
#Project data indices - QC
rov_idx_qc = {"lat": 0, "long": 1, "elev": 2, "nT": 3, "dq": 4, "sat": 6, "time": 7, "line": 10, "dist": 11, "tdif": 12}
rov_idx_in = {"lat": 0, "long": 1, "time": 8}
cor_idx_qc = {"dirn": 0, "lat": 1, "long": 2, "elev": 3, "nT": 4, "dq": 5, "sat": 7, "time": 8, "line": 11, "dist": 12, 
              "tdif": 13}
cor_idx_in = {"lat": 1, "long": 2, "time": 11}
bas_idx_qc = {"time": 0, "nT": 1, "dq": 2}
test_idx_in = {"nT": 2}
#Data indices to remove by pertinent data
col_rem = {"mag_bas": [2], "mag_rov": [4,5,6,8,9,11,12],
           "mag_cor": [5,6,7,9,10,12,13], "vlf_rov": [3]} 
#Data indices for IGRF data
igrf_idx = {"long": 0, "lat": 1, "elev": 2, "time": 3, "xcmp": 4, "ycmp": 5, "zcmp": 6, "tot": 7}

#https://www.michaelcho.me/article/using-pythons-watchdog-to-monitor-changes-to-a-directory

"""
THIS IS WORKING
If a bad file without on of the four keywords is imported, the runner function stops, otherwise it will always keep watch.
I should send a text to myself if that happens...

Save text output from watchdog to a file for QC

If a file is copied it is modified, make sure you MOVE data into the folder NOT COPY.
"""
 
#Waits for any events in the watched directory
class Watcher:
    DIRECTORY_TO_WATCH = monitoring_path

    def __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, monitoring_path, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Error")

        self.observer.join()

#Takes action when an event was received
class Handler(FileSystemEventHandler):
    
    @staticmethod
    def on_any_event(event):        
        if event.is_directory:
            return None
      
        elif event.event_type == 'created':
            #Take any action here when a file is first created.
            print(str(datetime.datetime.now()) + " Received created event - %s. \n" % event.src_path)
            #I MAY NEED TO WAIT FOR THE FILE TO FULLY LOAD?
            
            #Define the project setup object for the rapid mag/vlf data processing algorithm
            proj_setup = AutoProcessMagVLF(install_loc, proj_loc, proj_bas_sr, proj_rov_sr, proj_line_space, proj_origin, 
                                           mag_idx, vlf_idx, bas_idx, rov_idx_qc, rov_idx_in, cor_idx_qc, cor_idx_in, bas_idx_qc,
                                           test_idx_in, col_rem, igrf_idx)
            
            #Run the import_data module
            proj_setup.import_data(event.src_path)
            
            #Number of files in the watched directory should equal the length of import_lists.txt
            if len([name for name in os.listdir(monitoring_path) if name.endswith('.txt') and 
                    os.path.isfile(os.path.join(monitoring_path, name))]) == \
               len(open(install_loc + proj_loc + "/data/import_list.txt").readlines(  )):
            
                #Run the data_qc module
                print(str(datetime.datetime.now()) + " Beginning the data QC module. \n")
                proj_setup.data_qc()
                
                #Run the data_processing module
                #print(str(datetime.datetime.now()) + " Beginning the data processing module. \n")
                #proj_setup.mag_data_processing()
            
        elif event.event_type == 'modified':
            #Take note if a file is modified
            print(str(datetime.datetime.now()) + " Received modified event - %s. \n" % event.src_path)

        elif event.event_type == 'moved':
            #Take note if a file is moved
            print(str(datetime.datetime.now()) + " Received moved event - %s. \n" % event.src_path)
            
        elif event.event_type == 'deleted':
            #Take note if a file is deleted
            print(str(datetime.datetime.now()) + " Received deleted event - %s.\n" % event.src_path)
                
if __name__ == '__main__':
    w = Watcher()
    w.run()