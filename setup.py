#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""1) Setup tool for exero external library dependancies, i.e. watchdog"""
#Watchdog pip install watchdog

"""2) Setup tool for each projects path"""
#For example, the path for each individual processing project would be ./proj/YEAR-MONTH-LOCATION/
#Raw data imports are under ./proj/YEAR-MONTH-LOCATION/data/raw/
#QC'd data are saved under ./proj/YEAR-MONTH-LOCATION/data/qc/ 
#Processed data are saved under ./proj/YEAR-MONTH-LOCATION/data/processed/ 
#All tmp data and import history list is saved under ./proj/YEAR-MONTH-LOCATION/data/
#For processing reports under project ./proj/YEAR-MONTH-LOCATION/report/

#The project directory should be something like ./proj/YEAR-MONTH-LOCATION/
#In each project directory, I need to store imported raw data as ./proj/YEAR-MONTH-LOCATION/data/raw/hist
#tmp files are saved under raw/. and hist/.
#Historical imports restulting in the amalgamated tmp dataset are under hist/.

#THIS IS ALL I HAVE DEFINED SO FAR, FIGURE OUT HOW I WANT TO SAVE PROCESSED AND INTERMEDIATELY PROCESSED DATA