#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

#External packages/modules
import datetime
import fnmatch
import numpy as np
import operator
import os
import re
import sys
import shutil

from pathlib import Path

#Internal packages/modules
from .output import Save

#Input classes of functions for RAW data input/upload

class ImportIdentify:

    def __init__(self, filepath, install_loc, proj_loc):
        self.filepath = filepath
        self.install_loc = install_loc
        self.proj_loc = proj_loc
    
    def identify_file_type(self):
        #Dictionary of indexed file types
        ftype_dict = {0: "test", 1: "bas", 2: "rov", 3: "cor"}
        #The file type is currently unknown. 
        ftype = "unk"
        
        #Get the filename from the filepath
        filename = os.path.basename(self.filepath)
                
        #Search for each file type
        for i in range(len(ftype_dict)):
            #Is the file type keyword in the filename being imported? Case insensitive.
            if ftype_dict[i].lower() in filename.lower():
                #Properly define file type
                ftype = ftype_dict[i]

        #The file type has not been resolved
        if ftype == "unk":
            #Raise error and abort import - TEXT TO GEO WHO IS IMPORTING
            print("ERROR: A file identifier keyword is missing from: " + filename + " .Ensure an appropiate file identifier; test, bas, rov, or cor, is present in the filename, case insensitive.")
            sys.exit() #AND REBOOT?

        #Return the file type.    
        else:
            return ftype
        
class DataOrg:
    
    def __init__(self, filepath, install_loc, proj_loc, ftype, rov_idx_in, cor_idx_in, test_idx_in):
        self.filepath = filepath
        self.install_loc = install_loc
        self.proj_loc = proj_loc
        self.ftype = ftype
        self.rov_idx_in = rov_idx_in
        self.cor_idx_in = cor_idx_in
        self.test_idx_in = test_idx_in
        
    def indices_in(self):
        
        """
        Setup data indices for input module.
        """
        
        #Indices from file type
        if self.ftype == "rov":
            idx = self.rov_idx_in
        elif self.ftype == "cor":
            idx = self.cor_idx_in
        elif self.ftype == "test":
            idx = self.test_idx_in
            
        #Return indices
        return idx
        
    def make_import_hist_list(self):
        #Define the import_lists file's path
        file_loc = Path(self.install_loc + self.proj_loc + "/data/import_list.txt")
        
        #Get the filename from the filepath
        filename = os.path.basename(self.filepath)
                
        #Is there an import_list file already?
        if file_loc.is_file() == True:
            #Open previously generated import_lists file with read only permissions
            with open(self.install_loc + self.proj_loc + "/data/import_list.txt", "r") as f:
                #Check for file duplicates.
                if filename in f.read():
                    #Exit with an error message - TEXT TO GEO WHO IS IMPORTING
                    print("ERROR: Imported file is a duplicate file: " + filename + ". Each datafile must have a unique name.")
                    sys.exit() #AND REBOOT!
                    
                    #Close import_lists.txt
                    f.close()

                else:
                    #Close import_lists.txt with read only permissions
                    f.close()
                    #Open previously generated import_lists file with append only permissions
                    with open(self.install_loc + self.proj_loc + "/data/import_list.txt", "a") as f:
                        #Doccument the latest file import
                        f.write(str(datetime.datetime.now()) + ", file name: " + filename + ", file type: " + 
                                self.ftype + "\n")
                        #Close import_lists.txt with append only permissions
                        f.close()

        else:
            #Define the import number
            data_import_no = [1]
            #Make import_lists.txt file
            f = open(self.install_loc + self.proj_loc + "/data/import_list.txt", "w+")
            #Doccument the latest file import
            f.write(str(datetime.datetime.now()) + ", file name: " + filename + ", file type: " + self.ftype + "\n")
            #Close import_lists.txt
            f.close()
    
    def open_text(self):
        #Open text file
        with open(self.filepath) as f:

            #Define the input files.
            file_in, l = [], 0

            #Read the remainder of the text file as integers
            for line in f:
                
                #Skip blank lines
                if line not in ['\n', '\r\n']:

                    #Remove all non-alphanumeric charachters except decimals - using txt files output from GEMLink
                    line_AlNum = [re.sub('[^a-zA-Z0-9*-. ]+', '', i) for i in line.split(None)]
                    #If length is still one try splitting by comma for a .csv file
                    if len(line_AlNum) == 1:
                        line_AlNum = [re.sub('[^a-zA-Z0-9*-. ]+', '', i) for i in line.split(",")]
                        #Filter out empty entries
                        line_AlNum = list(filter(None, line_AlNum))

                    #Include re.sub for unsual charachters here
                    line_AlNum = [re.sub('[*]+', '0', i ) for i in line_AlNum]

                    #Properly process each row as float, string, or mixed.

                    #Check if the row is only floats, temporarly remove the decimal to run isnumeric() test.
                    if len(line_AlNum) == len([x for x in [re.sub('[\W]+', '', i) for i in line_AlNum] if x.isnumeric()]):
                        #Append floats to input data list
                        file_in.append([float(x) for x in line_AlNum])

                    #Check if the row is only strings. 
                    elif len(line_AlNum) == len([x for x in [re.sub('[\W]+', '', i) for i in line_AlNum] if x.isalpha()]):
                        #Do not append only strings to the data list
                        pass

                    #Check if the row is alphanumeric. 
                    elif len(line_AlNum) == len([x for x in [re.sub('[\W]+', '', i) for i in line_AlNum] if x.isalnum()]):
                        #Indices of numeric entires
                        num_idx = [i for i, x in enumerate([re.sub('[\W]+', '', i) for i in line_AlNum]) if x.isnumeric()]
                        #Indices of alphanumeric entires
                        alphanum_idx = [i for i, x in enumerate([re.sub('[\W]+', '', i) for i in line_AlNum]) if x.isalnum()]
                        #Exclude float or string entries from alphanumeric list
                        alpha_idx = [x for x in alphanum_idx if x not in num_idx]

                        #Check that all entires are accounted for
                        assert len(line_AlNum) == len(alpha_idx) + len(num_idx)

                        #Define line enteries properly as float or string, any alphanumeric entries are set as string
                        tmp = []
                        for i in range(len(line_AlNum)):
                            if i in alpha_idx:
                                tmp.append(str(line_AlNum[i]))
                            else:
                                tmp.append(float(line_AlNum[i]))

                        file_in.append(tmp)

                    #If a row is not alpha, numeric, or alphanumeric, raise and error and exit program
                    else:
                        #Text this error to the GEO who is importing the data
                        print("ERROR: Check line: "+str(l+1)+" of input text file for irregular data input.")
                        sys.exit() #AND REBOOT

                    #Line counter
                    l+=1

        return file_in, l
    
    def sep_magvlf_data(self, file_in, l):
        
        """
        Separate magnetic and vlf datasets from the rover file and append Lat/Long/time to VLF dataset. 
        """
        
        #Data indices
        idx = self.indices_in()
                
        #Find the indices of each line.
        line_idx = [i for i, x in enumerate(file_in) if any(y == "line" for y in x)]
        line_idx.append(l)

        #Define empty lists correspoiding to the number of lines.
        data_line_sep = [ [] for _ in range(len(line_idx)-1)]

        #Separate data types for each line
        data_vlf_sep, data_mag_sep = [ [] for _ in range(len(line_idx)-1)], [ [] for _ in range(len(line_idx)-1)]

        #Number of erroneous GPS data points
        count = 0
        
        #Working line-by-line, separate mag, vlf, and erroneous data points
        for i in range(len(line_idx)-1):
            #Separate data points by line.
            data_line_sep[i] = file_in[line_idx[i]+1:line_idx[i+1]]
            #Record all data
            total = len(data_line_sep[i])

            #Track the lengths of each line to correlate with a mag, vlf, or erroneous reading
            dtype_line_sep = [len(x) for x in data_line_sep[i]]

            #Count all line lengths
            dtype_line_count = {x:dtype_line_sep.count(x) for x in dtype_line_sep}

            #Find all data points equal in length or longer than a magnetic data point.
            data_line_sep[i] = [x for x in data_line_sep[i] if len(x) >= max(dtype_line_count.items(), 
                                                                            key=operator.itemgetter(1))[0]]

            #Remove GPS noise from and vlf-mag data points. 
            data_line_sep[i] = [x for x in data_line_sep[i] if x[idx["lat"]:idx["long"]+1] != [0,0]]

            #Exclude vlf data and cleanup.
            data_mag_sep[i] = [x[:max(dtype_line_count.items(), key=operator.itemgetter(1))[0]] for x in data_line_sep[i]]
            data_mag_sep[i] = [x for x in data_mag_sep[i] if x != []]

            #Only process rov VLF data, cor data is redundant
            if self.ftype == "rov":
                #Construct VLF dataset with magnetometer latitude, longitude and time
                t1 = [list([x[idx["lat"]]]) for x in data_line_sep[i]]
                t2 = [list([x[idx["long"]]]) for x in data_line_sep[i]]
                t3 = [list([x[idx["time"]]]) for x in data_line_sep[i]]
                #Latitude/longitude/time list
                lat_long_time = [x + y + z for x, y, z in zip(t1, t2, t3)]
                #VLF-unique data points
                t4 = [x[max(dtype_line_count.items(), key=operator.itemgetter(1))[0]:] for x in data_line_sep[i]]
                #Construct vlf dataset
                data_vlf_sep[i] = [x + y for x, y in zip(lat_long_time, t4) if y != []]

            #Count how many data points are presumed to be noise.
            count += total - len(data_mag_sep[i])
            
        #Ensure any mag/vlf list is not empty
        data_vlf_sep = [x for x in data_vlf_sep if x != []]
        data_mag_sep = [x for x in data_mag_sep if x != []]
            
        return [data_mag_sep, data_vlf_sep], count
    
    def process_magvlf_data(self, lists, count):
        
        """
        Steps to "process" imported magnetic and VLF data ensuring the dataset is free non-numerical values.
        """
        #Search through lists
        for i in reversed(range(len(lists))):
            #Enforce a minimum of two datapoints per line
            if len(lists[i]) == 1:
                lists.pop(i)
                        
        #Initialize.
        string_search = [[] for _ in range(len(lists))]
        bad_data = []

        #Full dual loop QC of data points, looking for NaN's, and non-numeric values.
        for i in range(len(lists)):
            #Complete QC line-by-line
            for j in range(len(lists[i])):
                string_search[i].append([int(type(x) == str) for x in lists[i][j]])
                                              
        #Find the most common arrangement of strings in the dataset.
        most_common = [list(x) for x in set.intersection(*[set(tuple(x) for x in y) for y in string_search])][0]

        #Save indices of most common string entires for deletion
        del_columns = [i for i, x in enumerate(most_common) if x == int(True)]
        
        #Loop through lines again, find out if any data entries that differ from the most common arrangement of strings.
        for i in range(len(lists)):

            #Are there any strings in unexpected places?
            if all(elem == most_common for elem in string_search[i]) == False:

                #Find out which data entry was problematic.
                bad_rows = [i for i, x in enumerate([elem == most_common for elem in string_search[i]]) if x == False]

                #Save each data entry that is problemation for latter deletion.
                for k in range(len(bad_rows)):
                    bad_data.append([i,bad_rows[k]])

        #Remove bad data points
        for i in reversed(range(len(bad_data))):
            lists[bad_data[i][0]].pop(bad_data[i][1])

        #Remove string columns line-by-line.
        for i in range(len(lists)):
            for j in reversed(del_columns):
                [x.pop(j) for x in lists[i]]

        return lists
    
    def lists_to_array(self, lists):
        #Setup list of arrays for 3D case.
        if self.ftype == "cor" or self.ftype == "rov":
            array = np.empty(len(lists), object)
            for i in range(len(lists)):
                array[i] = np.array(lists[i])
                
        #Setup arrays for 2D case.
        elif self.ftype == "bas":
            array = np.zeros((len(lists), len(lists[1])))
            for i in range(len(lists)):
                array[i,:] = lists[i]
            
        return array
            
    def test_data(self, data):
        
        #Data indices
        idx = self.indices_in()
        
        #Initiate list difference sums and index
        delta_mag_sum = []
        i=0

        #Ensure a test (four measurements) passed for an undetermined number of test measurements 
        while i < len(data)-3:
            #Initiate list of three test differences
            delta_mag = []

            #Append three absolute differences of the four test measurements
            for j in range(i, i+3):
                #The magnetic measurement is HARD CODED here
                delta_mag.append(abs(data[j][idx["nT"]]-data[j+1][idx["nT"]]))

            #Sum the three absolute differences
            delta_mag_sum.append(sum(delta_mag))

            #Include the next test measurement, exclude the first test measurment
            i+=1

        #Print the result of this test to text file, this is overwritten for each data import.
        with open(self.install_loc + self.proj_loc + "/reports/00_test_result.txt", "w+") as f:

            #Write the number of tests identified
            if len(data) > 4:
                f.write("There are " + str(len(data) - 3) + " possible tests identified. " + "\n")
            else:
                f.write("One test is identified. " + "\n")

            #Write the test result
            if any(x <= 5.0 for x in delta_mag_sum):
                f.write("The data decontamination test passed: " + str(round(min(delta_mag_sum),2)) + "\n")
            elif any(x <= 6.0 for x in delta_mag_sum):
                f.write("The data decontamination test result is concerning: " + str(round(min(delta_mag_sum),2)) + "\n")
            else:
                f.write("The data decontamination test did not pass. The sum of all possible tests are: " + str(delta_mag_sum) + "\n")

        #Close test_result.txt
        f.close()