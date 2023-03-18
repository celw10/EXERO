#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

#External Packages
import math
import numpy as np
import sys

from collections import Counter
from pathlib import Path

#Internal packages
from .spikeid import MovingAverage
from .misc import haversine_dist, find_line_idx, vincenty_dist, min_dist_lineproj, angle_between_vectors

class WriteLine:
    
    def __init__(self, f, line_no, A, B, t, line_id_conf):
        self.f = f
        self.line_no = line_no
        self.A = A
        self.B = B
        self.t = t
        self.line_id_conf = line_id_conf
        
    def write_line(self):
        
        """
        Write a acquisition line to a text file containing line number start/endpoints, slope, and line confidence. 
        """

        #Write the line
        self.f.write(str(self.line_no)+" "+str(self.A[0])+" "+str(self.A[1])+" "+str(self.B[0])+" "+str(self.B[1])+
                     " "+str(self.t[0])+" "+str(self.t[1])+" "+str(self.line_id_conf)+"\n")
        

class LineID:
    
    def __init__(self, install_loc, proj_loc, proj_line_space, proj_origin, ftype, fileloc, data_qc_report, line_points, vlf_idx):
        self.install_loc = install_loc
        self.proj_loc = proj_loc
        self.proj_line_space = proj_line_space
        self.proj_origin = proj_origin
        self.ftype = ftype
        self.fileloc = fileloc
        self.data_qc_report = data_qc_report
        self.line_points = line_points
        self.vlf_idx = vlf_idx
    
    def line_id(self, mag_data_qc, beta=0.1, threshold=0.025):
        
        """
        Identifies line breaks from distance/time spikes in the data and magnetometer line ID's.
        """
        
        #Load data
        data = np.load(self.fileloc)
        
        #Data indices
        didx = mag_data_qc.indices_qc()
        
        #Initiate data iteration indices: point distance and delta time
        deltaindices = [didx["dist"], didx["tdif"]]

        #Initiate data_spikes
        data_spikes = []

        #Loop through distance and time separation
        for i in deltaindices:

            #Spike identification object
            mov_avg = MovingAverage(self.install_loc, self.proj_loc, beta, threshold, self.vlf_idx)
            
            #Identify data spikes
            current, _ = mov_avg.fwd_bwd_time(data, i)

            #All temporal time and distance data spikes
            data_spikes =  data_spikes + current

        #Data spikes common to time and distance line boundraries
        dist_time_lb = [item for item, count in Counter(data_spikes).items() if count > 1]

        #Insert index zero as the start of the first line
        dist_time_lb.insert(0,0)
        #Append last index as end of final line
        dist_time_lb.append(len(data))

        #Join distance/time line breaks with magnetometer line breaks
        line_breaks = sorted(list(set(dist_time_lb+self.line_points))) # SIMPLY ADDED A LINE
        
        #Define path of line_locs.txt file
        file_loc = Path(self.install_loc + self.proj_loc + "/data/line_locs_" + str(self.ftype) + ".txt")
        
        #Initiate list of line numbers for this data upload
        current_line_nos = []

        #Check each potential line break
        for idx in range(len(line_breaks)-1):
            #Investigation points
            P1 = np.array([data[line_breaks[idx], didx["long"]], data[line_breaks[idx], didx["lat"]]])
            P2 = np.array([data[line_breaks[idx+1]-1, didx["long"]], data[line_breaks[idx+1]-1, didx["lat"]]])
            #Compute line length
            P1P2 = math.sqrt((P1[0] - P2[0])**2 + (P1[1] - P2[1])**2)
            
            #Project line length
            proj_line_len = haversine_dist(self.proj_origin[0], self.proj_origin[1], 
                                           self.proj_origin[2], self.proj_origin[3])
        
            #Is the line_locs.txt file present?
            if file_loc.is_file() == True:
                #Open line locs with append premissions
                f = open(file_loc, "r+")
                
                #Initiate line locs list
                line_locs = []
                #Reference line is closest to current line
                for line in f:
                    #Split line and append
                    line_locs.append(line.split(None))
                #Close line_locs
                f.close()

                #Convert to array with an extra column for line lengths
                line_locs_arr = np.array(line_locs)
                ll_arr = np.zeros((line_locs_arr.shape[0], line_locs_arr.shape[1]+2))
                ll_arr[:,0:line_locs_arr.shape[1]] = line_locs_arr
                
                #Can't use a previously identified tie line or line segment as a reference line
                tie_idx = np.where(np.logical_and((ll_arr[:,0] == 0), (ll_arr[:,ll_arr.shape[1]-3] == 0)))[0]
                
                #Remove any tie lines
                ll_arr_del = np.delete(ll_arr, tie_idx, axis=0)
                
                #If the only line was a tie line, go back to reference line
                if ll_arr_del.size == 0:
                    #The reference line is line one
                    refline = 1
                    #A and B points defined by the reference line
                    A = np.array([self.proj_origin[0], self.proj_origin[1]]) 
                    B = np.array([self.proj_origin[2], self.proj_origin[3]])
                                        
                #Find the reference line
                else:
                    #For each previously identified line
                    for i in range(ll_arr_del.shape[0]):
                        #Compute a and b
                        a = np.array([ll_arr_del[i,1], ll_arr_del[i,2]])
                        b = np.array([ll_arr_del[i,3], ll_arr_del[i,4]])

                        #Initiate test distances list
                        td = []
                        #For each investigation point
                        for p in [P1,P2]:
                            #Compute distance
                            td.append(min_dist_lineproj(a, b, p))
                        #Save average distance
                        ll_arr_del[i,ll_arr_del.shape[1]-2] = (td[0] + td[1]) / 2.0
                        #Save line length
                        ll_arr_del[i,ll_arr_del.shape[1]-1] = math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

                    #Reverse sort array based on distance from defined line to current
                    ll_arr_del = ll_arr_del[ll_arr_del[:,ll_arr_del.shape[1]-2].argsort()]

                    #Initiate index
                    index = 0
                    #Find the closest "reference line qualitiy" line to be the reference line
                    while index < ll_arr_del.shape[0]:
                        #Reference line vector R
                        R1 = np.array([self.proj_origin[0], self.proj_origin[1]])
                        R2 = np.array([self.proj_origin[2], self.proj_origin[3]])
                        R = abs(R2-R1)
                                   
                        #Investigation vector O 
                        O1 = np.array([ll_arr_del[index,1], ll_arr_del[index,2]])
                        O2 = np.array([ll_arr_del[index,3], ll_arr_del[index,4]])
                        O = abs(O2 - O1)
                                   
                        #Compute the angle between P and R
                        alpha = angle_between_vectors(R, O)

                        #Define the angle threshold defined by reference line length and a third of line spacing
                        threshold = (180.0 * math.atan(self.proj_line_space*(1.0/3.0) / proj_line_len) / math.pi)
                        
                        #We require a line of sufficent length and low angle compared to the project reference line
                        if  alpha <= threshold:
#ll_arr_del[index,ll_arr_del.shape[1]-1] > 0.5 * proj_line_len and
                            #Break loop
                            refidx = index
                            index = ll_arr_del.shape[0]
                            #The reference line is the closest line
                            refline = int(ll_arr_del[refidx,0])

                            #Define reference A and B points from whatever previously identified line is closest
                            A = np.array([ll_arr_del[refidx,1], ll_arr_del[refidx,2]])
                            B = np.array([ll_arr_del[refidx,3], ll_arr_del[refidx,4]])

                        #Otherwise use the project reference line again
                        else:# index == ll_arr_del.shape[0] - 1:
                            #The reference line is line one
                            refline = 1
                            #A and B points defined by the reference line
                            A = np.array([self.proj_origin[0], self.proj_origin[1]]) 
                            B = np.array([self.proj_origin[2], self.proj_origin[3]])

                        #Increase index
                        index += 1

                #Re-open line_locs.txt with append only permissions
                f = open(file_loc, "a")
                
            #First line in new project
            else:
                #The reference line is line one
                refline = 1
                #A and B points defined by the reference line
                A = np.array([self.proj_origin[0], self.proj_origin[1]]) 
                B = np.array([self.proj_origin[2], self.proj_origin[3]])
                
                #Open the line locs file with full permissions
                f = open(file_loc, "w+")

            #Initiate distances for each search line point
            d = []
            #Repeat for each line endpoint
            for p in [P1,P2]:
                #Compute distance perpendicular to line projection
                d.append(min_dist_lineproj(A, B, p))

            #Compute line number with distance to reference line and line spacing
            dp = np.array(d)
            
            #Compute the mean distance from an origin point to the reference line
            dor = np.zeros((2))
            dor[0] = min_dist_lineproj(A, B, self.proj_origin[0:2])
            dor[1] = min_dist_lineproj(A, B, self.proj_origin[2:self.proj_origin.shape[0]])
            dor_mean = np.mean(dor)
            
            #Compute the mean distance from an origin point to the investigation line
            doi = np.zeros((2))
            doi[0] = min_dist_lineproj(P1, P2, self.proj_origin[0:2])
            doi[1] = min_dist_lineproj(P1, P2, self.proj_origin[2:self.proj_origin.shape[0]])
            doi_mean = np.mean(doi)
            
            #Is this line further or closer to the origin than the refernce line
            if dor_mean <= doi_mean:
                #Add refline
                line_nos = np.rint(refline + (dp/self.proj_line_space))
            elif dor_mean > doi_mean:
                #Subtract refline
                line_nos = np.rint(refline - (dp/self.proj_line_space))
                            
            #line assignment scenarios
            if line_nos[0] == line_nos[1]:
                #Both endpoints correspond to the same line
                line_no = int(line_nos[0])
                
            #Both endpoints correspond to an adjacent line            
            elif abs(line_nos[0] - line_nos[1]) == 1:
                #Compute project reference line length
                AB = math.sqrt((self.proj_origin[0] - self.proj_origin[2])**2 + 
                               (self.proj_origin[1] - self.proj_origin[3])**2)

                #If line length is greater than the reference line length
                if P1P2 > AB:
                    #Assign as tie
                    line_no = 0

                else:
                    #Assign to line with distance closest to line spacing
                    D0 = abs(line_nos[0] - dp[0])
                    D1 = abs(line_nos[1] - dp[1])
                    if D0 < D1:
                        line_no = int(line_nos[0])
                    else:
                        line_no = int(line_nos[1])
                
            else:
                #Line zero is tie
                line_no = 0 

            #Line writing object
            write_line = WriteLine(f,  line_no,
                                   [data[line_breaks[idx], didx["long"]], data[line_breaks[idx], didx["lat"]]],
                                   [data[line_breaks[idx+1]-1, didx["long"]], data[line_breaks[idx+1]-1, didx["lat"]]], 
                                   [data[line_breaks[idx], didx["time"]], data[line_breaks[idx+1]-1, didx["time"]]], 
                                    1)

            #Write first line to line_locs.txt
            write_line.write_line()

            #Save the line numbers of this data upload
            current_line_nos.append(line_no)
            
            #Close file
            f.close()   

        #Clean-up object for line_locs.txt
        clean_line_locs = CleanLineLocs(self.install_loc, self.proj_loc, self.proj_origin, self.fileloc, self.ftype, 
                                        self.data_qc_report)
        
        #Run a cleanup of line locs looking for line duplicates and line segments, deleting line duplicates
        del_xy, current_line_nos = clean_line_locs.clean(current_line_nos, mag_data_qc)
                
        #Obtain indices of lines for this data upload
        line_start, line_end = clean_line_locs.line_assmt(current_line_nos, mag_data_qc)
        
        #Close file
        f.close()
        
        #Return start/end indices
        return line_start, line_end, del_xy
    
class CleanLineLocs:
    
    def __init__(self, install_loc, proj_loc, proj_origin, fileloc, ftype, data_qc_report):
        self.install_loc = install_loc
        self.proj_loc = proj_loc
        self.proj_origin = proj_origin
        self.fileloc = fileloc
        self.ftype = ftype
        self.data_qc_report = data_qc_report
    
    def clean(self, current_line_nos, mag_data_qc, min_line_len_pct=1.0):

        """
        Clean up the line_locs.txt file by sorting, removing duplicate lines, and identifying line segments.
        """
        
        #Data indices
        index = mag_data_qc.indices_qc()
        
        #Initiate latitude indices based on file type
        lat_dict = {"rov": 0, "cor": 1}
        lat_idx = lat_dict[self.ftype]
        
        #Initiate longitude indices based on file type
        long_dict = {"rov": 1, "cor": 2}
        long_idx = long_dict[self.ftype]
        
        #Initiate time as function of ftype
        time_dict = {"rov": 7, "cor": 8}
        time_idx = time_dict[self.ftype]
        
        #Open line_locs
        f = open(self.install_loc + self.proj_loc + "/data/line_locs_" + self.ftype + ".txt", "r")
        
        #Initiate line_locs list
        line_locs = []
        
        #Read file into list line-by-line
        for line in f:
            #Split line and append
            line_locs.append(line.split(None))
        #Close line_locs
        f.close()
        
        #Convert to array with an extra column for line lengths
        line_locs_arr = np.array(line_locs)
        ll_arr = np.zeros((line_locs_arr.shape[0], line_locs_arr.shape[1]+1))
        ll_arr[:,0:line_locs_arr.shape[1]] = line_locs_arr

        #Sort by ascending line
        ll_arr = ll_arr[ll_arr[:,0].argsort()]

        #Compute angles to reference line
        for i in range(ll_arr.shape[0]):
            #Reference line vector R
            R1 = np.array([self.proj_origin[0], self.proj_origin[1]])
            R2 = np.array([self.proj_origin[2], self.proj_origin[3]])
            R = abs(R2-R1)

            #Investigation vector O 
            O1 = np.array([ll_arr[i,1], ll_arr[i,2]])
            O2 = np.array([ll_arr[i,3], ll_arr[i,4]])
            O = abs(O2 - O1)

            #Compute the angle between P and R
            ll_arr[i,ll_arr.shape[1]-1] = angle_between_vectors(R, O)

        #List of unique line numbers
        lines = np.unique(ll_arr[:,0])
        
        #Re-open line_locs
        f = open(self.install_loc + self.proj_loc + "/data/line_locs_" + self.ftype + ".txt", "w+")
        
        #Index the lines on ll_arr
        idx = 0
        #XY points corresponding to deleted line duplicates
        del_x, del_y = [], []
        #Duplicate & segment totals
        dup_tot, seg_tot = 0, 0
        #Loop through unique line numbers
        for i in range(lines.shape[0]):
            #Count the number of times this line occurs in the numpy array
            occurrences = Counter(ll_arr[:,0])[lines[i]]
            
            #If only one occurance write line
            if occurrences == 1 or lines[i] == 0:
                #Write line object 
                write_line = WriteLine(f, int(ll_arr[idx,0]), [ll_arr[idx,1], ll_arr[idx,2]], 
                                       [ll_arr[idx,3], ll_arr[idx,4]], [ll_arr[idx,5], ll_arr[idx,6]], ll_arr[idx,7])
                #Write line
                write_line.write_line()
                
                #Increase index
                idx += 1
                
            #If more than one occurances search for erroneous lines or duplicates
            elif occurrences > 1:
                #Increase index by number of occurrences
                idx += occurrences
                
                #Return line indices of duplicates excluding confidences of 0 -> previously identified as line segments
                line_idx = np.where(np.logical_and((lines[i] == ll_arr[:,0]),(ll_arr[:,ll_arr.shape[1]-2] != 0)))[0]

                #Initiate array of potential duplicates
                dupes_arr = np.zeros((ll_arr[line_idx,:].shape[0], ll_arr[line_idx,:].shape[1]))
                dupes_arr[:,0:ll_arr[line_idx,:].shape[1]] = ll_arr[line_idx,:]

                #Sort by straightest line to the reference
                dupes_arr = dupes_arr[dupes_arr[:,dupes_arr.shape[1]-1].argsort()]
                                               
                #Define reference line points
                A = np.array([dupes_arr[0,1], dupes_arr[0,2]])
                B = np.array([dupes_arr[0,3], dupes_arr[0,4]])
                #Reference line dist
                AB = math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)

                #Initiate line duplicate or segment indices lists
                dup_idx, seg_idx = [], []

                #Loop through duplicates
                for j in range(1,dupes_arr.shape[0]):
                    #Compute current line length
                    P1P2 = math.sqrt((dupes_arr[j,1]-dupes_arr[j,3])**2 + (dupes_arr[j,2]-dupes_arr[j,4])**2)
                    #Initiate lists
                    inside_line, closest_pt = [], []

                    #Repeat for each line endpoint
                    for k in [[1,2], [3,4]]:
                        #Line point p
                        P = np.array([dupes_arr[j,k[0]], dupes_arr[j,k[1]]])

                        #Compute distance between AB projection and external point P
                        PAB = min_dist_lineproj(A, B, P, formula="pythagoras")
                        #Compute distance between A and P
                        PA = math.sqrt((P[0]-A[0])**2 + (P[1]-A[1])**2)
                        #Compute distance between B and P
                        PB = math.sqrt((P[0]-B[0])**2 + (P[1]-B[1])**2)

                        #Compute segment lengths Y1 and Y2 along AB
                        Y1 = math.sqrt(PA**2 - PAB**2)
                        Y2 = math.sqrt(PB**2 - PAB**2)

                        #If max(Y1,Y2) is less than or equal to AB
                        if max(Y1,Y2) <= AB:
                            #"Inside" the line AB
                            inside_line.append(True)

                        #If max(Y1,Y2) is greater than AB
                        elif max(Y1,Y2) > AB:
                            #"Outside" the line AB
                            inside_line.append(False)

                        #Append the distance to the closest point, PA or PB
                        closest_pt.append(min(PA,PB))

                    #If the line length is less than the minimum percent of the project reference line
                    if min_line_len_pct > (P1P2/math.sqrt((self.proj_origin[0]-self.proj_origin[2])**2 + 
                                                           (self.proj_origin[1]-self.proj_origin[3])**2)*100.0):
                        #Append as duplicate line
                        dup_idx.append(j)

                    #If both points inside reference line
                    elif all(inside_line) == True:
                        #Append as duplicate line
                        dup_idx.append(j)

                    #If one point is inside the reference line
                    elif any(inside_line) == True:
                        #How far inside the line is the point in the ref line
                        dist_inside = [x for x,y in zip(closest_pt, inside_line) if y == True][0]
                        #How far outisde the line is the point outside the ref line
                        dist_outside = [x for x,y in zip(closest_pt, inside_line) if y == False][0]
                        #Which is larger
                        if dist_inside >= dist_outside:
                            #Line is duplicate
                            dup_idx.append(j)
                        elif dist_inside < dist_outside:
                            #Line is a segment
                            seg_idx.append(j)

                    #If both points are outside the reference line
                    elif all(inside_line) == False:
                        #Append as line segment
                        seg_idx.append(j)

                    else:
                        #Shouldn't be here, SEND A TEXT
                        sys.exit() #AND REBOOT

                #Import dataset
                data = np.load(self.fileloc)
                
                #For each duplicate dataset
                for j in dup_idx:
                    #Cleanup the current line numbers
                    current_line_nos.remove(int(dupes_arr[j,0]))
                    #Start time, end time
                    line_t_bounds = [dupes_arr[j,5], dupes_arr[j,6]]

                    #Now find the data points between these two times and delete them.
                    del_idx = np.where(np.logical_and((line_t_bounds[0] <= data[:,index["time"]]),
                                                      (data[:,index["time"]] <= line_t_bounds[1])))[0]
                    
                    #Save x/y locs of deleted data for summary image
                    del_x.extend(list(data[del_idx, index["long"]]))
                    del_y.extend(list(data[del_idx, index["lat"]]))
                                        
                    #Delete line
                    data = np.delete(data, del_idx, axis=0)
                    
                #Save dataset
                np.save(self.fileloc, data)
                
                #Change line confidence of identified segments to 0
                for j in seg_idx:
                    dupes_arr[j,7] = 0
                
                #Remove duplicate lines from dupes arr
                dupes_arr = np.delete(dupes_arr, dup_idx, axis=0)
                
                #Write remaining lines to text file
                for j in range(dupes_arr.shape[0]):
                    #WriteLines object 
                    write_line = WriteLine(f, int(dupes_arr[j,0]), [dupes_arr[j,1], dupes_arr[j,2]], 
                                           [dupes_arr[j,3], dupes_arr[j,4]], [dupes_arr[j,5], dupes_arr[j,6]], 
                                           dupes_arr[j,7])
                    #Write lines
                    write_line.write_line()
                    
                #Update duplicate & segment totals
                dup_tot += len(dup_idx)
                seg_tot += len(seg_idx)
            
            else:
                #Exit shouldn't be here
                sys.exit() #AND REBOOT
                
        #Write to report
        self.data_qc_report.write(str(dup_tot)+" duplicate line(s) and "+str(seg_tot)+" line segment(s) found. "+str(len(del_x))+" data points from deleted identified duplicate/erroneous lines. \n")
                
        #Return lat/long points corresopnding to deleted line duplicates  
        return [del_x, del_y], current_line_nos
                
    
    def line_assmt(self, current_line_nos, mag_data_qc):
        
        """
        Identify line start/end indices from line_locs.txt for mag data.
        """
        
        #Load data
        data = np.load(self.fileloc)
        
        #Data indices
        index = mag_data_qc.indices_qc()
        
        #Initiate line ID index
        lineid_dict = {"rov": 10, "cor": 11}
        lineid_idx = lineid_dict[self.ftype]
        
        #Initiate latitude indices based on file type
        lat_dict = {"rov": 0, "cor": 1}
        lat_idx = lat_dict[self.ftype]
        
        #Initiate longitude indices based on file type
        long_dict = {"rov": 1, "cor": 2}
        long_idx = long_dict[self.ftype]
        
        #Initiate time as function of ftype
        time_dict = {"rov": 7, "cor": 8}
        time_idx = time_dict[self.ftype]
        
        #Line start/end indices
        line_start, line_end = [], []
        
        #Assign line numbers to the dataset
        f = open(self.install_loc + self.proj_loc + "/data/line_locs_" + str(self.ftype) + ".txt", "r")
        #Strip lines
        lines = [line.rstrip("\n") for line in f]
            
        #Begin reading line locations for this data upload
        for l in lines:
            #Line information as float
            line_loc = [float(x) for x in l.split(None)]
            
            #Look for lines in current line numbers
            if line_loc[0] in current_line_nos: 
                #Are all of line start/end lat/long points in the dataset
                if np.all(np.array([np.any(np.isin(data[:,index["long"]], line_loc[1])), 
                                    np.any(np.isin(data[:,index["lat"]], line_loc[2])), 
                                    np.any(np.isin(data[:,index["long"]], line_loc[3])), 
                                    np.any(np.isin(data[:,index["lat"]], line_loc[4]))])) == True:
                    
                    #Define indices by times as lat/long is not unique
                    start_idx = min(np.where(data[:, index["time"]] >= line_loc[5])[0])
                    end_idx = max(np.where(data[:, index["time"]] <= line_loc[6])[0])
                
                    #Record the line number for datapoints between the start and end points of the line
                    data[start_idx:end_idx, index["line"]] = line_loc[0]
                    data[end_idx, lineid_idx] = line_loc[0]
                    #Save start/end indices to list
                    line_start.append(start_idx)
                    line_end.append(end_idx)

            #Close line_locs.txt file
            f.close()
            
        #Something went terribly wrong if this fails
        assert len(line_start) == len(line_end)

        #Save data to temp
        np.save(self.fileloc, data)
        
        #Return start/end indices of each line
        return line_start, line_end