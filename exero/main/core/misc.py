#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

#External Packages
import math
import numpy as np

from matplotlib import colors as colors
from scipy.interpolate import interp1d

#Basically any function that doesn't require a project variable

def earth_radius(lat, elev=0):
    
    """
    Computation of the earth's radius with latitude and elevation
    References:
    https://planetcalc.com/7721/
    https://stackoverflow.com/questions/56420909/calculating-the-radius-of-earth-by-latitude-in-python-replicating-a-formula
    """
    
    #Angle in radians
    beta = math.radians(lat) 
    
    #Radius at poles and equator
    re = 6378137.0 
    rp = 6356752.3142
    
    #Elements for radius computation
    c = (re**2*math.cos(beta))**2
    d = (rp**2*math.sin(beta))**2
    e = (re*math.cos(beta))**2
    f = (rp*math.sin(beta))**2
    
    #Compute radius optionally adding elevation
    R = math.sqrt((c + d)/(e + f)) + elev
    
    #Return radius
    return R

def haversine_dist(x1, y1, x2, y2, e1=0, e2=0):
    
    """
    Haversine distance formula to approximate distance between two lat/long points on a sphere.
    References:
    https://www.movable-type.co.uk/scripts/latlong.html
    https://community.esri.com/t5/coordinate-reference-systems/distance-on-a-sphere-the-haversine-formula/ba-p/902128
    """
    
    #Representative latitude
    y = (y1+y2)/2.0
    
    #Radius of the earth
    R = earth_radius(y)
    
    #Radian latitude angles
    phi1 = math.radians(y1)
    phi2 = math.radians(y2)
    
    #Radian changes in phi and lambda
    delt_phi = math.radians(y2 - y1)
    delt_lambda = math.radians(x2 - x1)

    #Compute variable a
    a = math.sin(delt_phi/2.0)**2  + (math.cos(phi1) * math.cos(phi2) * math.sin(delt_lambda/2.0)**2)
    
    #Compute variable c
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
         
    #Compute distance in meters
    d = R*c
    
    #If elevation is provided
    if e1 !=0 or e2 !=0:
        #Incorporate elevation distance for SHORT distances using pythagoras
        d = math.sqrt(d**2 + (abs(e1-e2))**2)
    
    #Return distance
    return d

def vincenty_dist(x1, y1, x2, y2):
    
    """
    Computaiton of ellipsodial distance along earth using Vincenty's Formulate.
    Using WGS-84 geodetic reference oblate spheroid.
    References:
    https://www.johndcook.com/blog/2018/11/24/spheroid-distance/
    https://en.wikipedia.org/wiki/Vincenty%27s_formulae
    """
    
    #Convert Latitude and longitude to radians
    phi1, phi2 = math.radians(y1), math.radians(y2)

    #Equitorial radius
    a = 6378137.0 
    #Ellipsoid flattening
    f = 1/298.257223563
    b = (1 - f)*a 
    #Tolerance to stop iteration
    tolerance = 1e-11

    #Define variables
    U1 = math.atan((1-f)*math.tan(phi1))
    U2 = math.atan((1-f)*math.tan(phi2))
    L1, L2 = math.radians(x1), math.radians(x2)
    L = L2 - L1

    lambda_old = L + 0

    while True:
    
        t = (math.cos(U2)*math.sin(lambda_old))**2
        t += (math.cos(U1)*math.sin(U2) - math.sin(U1)*math.cos(U2)*math.cos(lambda_old))**2
        sin_sigma = t**0.5
        cos_sigma = math.sin(U1)*math.sin(U2) + math.cos(U1)*math.cos(U2)*math.cos(lambda_old)
        sigma = math.atan2(sin_sigma, cos_sigma) 
    
        sin_alpha = math.cos(U1)*math.cos(U2)*math.sin(lambda_old) / sin_sigma
        cos_sq_alpha = 1 - sin_alpha**2
        cos_2sigma_m = cos_sigma - 2*math.sin(U1)*math.sin(U2)/cos_sq_alpha
        C = f*cos_sq_alpha*(4 + f*(4-3*cos_sq_alpha))/16
    
        t = sigma + C*sin_sigma*(cos_2sigma_m + C*cos_sigma*(-1 + 2*cos_2sigma_m**2))
        lambda_new = L + (1 - C)*f*sin_alpha*t
        if abs(lambda_new - lambda_old) <= tolerance:
            break
        else:
            lambda_old = lambda_new

    u2 = cos_sq_alpha*((a**2 - b**2)/b**2)
    A = 1 + (u2/16384)*(4096 + u2*(-768+u2*(320 - 175*u2)))
    B = (u2/1024)*(256 + u2*(-128 + u2*(74 - 47*u2)))
    t = cos_2sigma_m + 0.25*B*(cos_sigma*(-1 + 2*cos_2sigma_m**2))
    t -= (B/6)*cos_2sigma_m*(-3 + 4*sin_sigma**2)*(-3 + 4*cos_2sigma_m**2)
    delta_sigma = B * sin_sigma * t
    s = b*A*(sigma - delta_sigma)

    return s

def min_dist_lineproj(A, B, P, formula="haversine"):
    
    """
    Compute the minimum distance from point (P) to the line projection of A,B.
    The formula may be "haversine" or "vincenty" for distance computation from lat/long, or pythagoras \
    for true distance computation. 
    """

    #Define the vector M
    M = B-A
    #Compute running paramater through orthogonal intersection
    t = np.dot((P-A), M)/np.dot(M,M)
    #Compute orothogonal distance to intersection from point to line
    I = A + t * M
    #Compute the distance between AB and P
    if formula == "haversine":
        D = haversine_dist(P[0], P[1], I[0], I[1])
    elif formula == "vincenty":
        D = vincenty_dist(P[0], P[1], I[0], I[1])
    elif formula == "pythagoras":
        D = math.sqrt((P[0]-I[0])**2 + (P[1]-I[1])**2)
    else:
        print("Incorrect formula value for function min_dist_lineproj. Currently accepted values are haversine, vincenty, or pythagora.")
        
    #Return orthogonal point to line distance
    return D

def angle_between_vectors(A, B):
    
    """
    Compute the angle in degrees between two vectors
    """
    
    #Compute lenghts of A and B
    AL = math.sqrt(A[0]**2 + A[1]**2)
    BL = math.sqrt(B[0]**2 + B[1]**2)
    
    #Compute the angle between P and R
    alpha = (180.0 * math.acos(np.dot(A,B) / np.dot(AL, BL)) / math.pi)
    
    #Return the angle in degrees
    return alpha
        
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    
    """
    Clip a matplot lib colormap creating a new colormap.
    Source:
    https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    """
    
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def join_lists(pts_x, pts_y):
    
    """
    Joint together equal-length lists of x/y points.
    """
    
    #Initiate lists
    join_pts_x, join_pts_y = [], []
    
    #Ensure lists are of equal length
    assert len(pts_x) == len(pts_y)
    
    #Loop through lists 
    for i in range(len(pts_x)):
        #Join x points
        join_pts_x += list(pts_x[i])
        #Join y points
        join_pts_y += list(pts_y[i])
        
    #Return joined lists of x/y points
    return join_pts_x, join_pts_y

def concat_array(obj):
    
    """
    Concatenate an object-formatted three dimensional array to a 2D numpy array.
    """
    
    #Save the number of data points per line
    line_points = []
    
    #Copy obj to preserve the original
    concat_obj = list(obj)

    #First line
    line_points.append(np.shape(obj[0])[0])

    #Concatenate the object formatted array
    for i in range(len(concat_obj)-1):
        #Number of data points in line
        line_points.append(np.shape(concat_obj[i+1])[0] + line_points[i])
        #Concatenated array of acquisition lines
        concat_obj[0] = np.concatenate((concat_obj[0], concat_obj[i+1]), axis=0)

    #Return numpy array
    return concat_obj[0], line_points

def interp_over_zero(data, xax, yax):
    
    """
    Scipy 1-dimensional linear interpolation of 2D numpy array over zeros.
    """

    #Find non-zero interpolation indices
    non_zero_idx = np.nonzero(data[:,yax])[0]

    #Linear interpolation
    interp_func = interp1d(data[non_zero_idx, xax], data[non_zero_idx, yax])
    
    #Interpolated data
    data[:, yax] = interp_func(data[:, xax]).reshape(-1) #Flattened interpolation array for base station
    
    #Return interpolated data
    return data

def normalize(data, idx, a=0, b=1):
    
    """
    Normalize a 1-D array between between a and b.
    """
    
    #Normalize between a and b
    data_norm = a + (((data[idx] - np.amin(data)) * (b - a)) / (np.amax(data) - np.amin(data)))
    
    #Return normalized data point
    return data_norm

def find_line_idx(data, idx):
    
    """
    Function to find the indices of a new line when the line number has already been assigned. 
    """
    
    #Indices that correspond to changes in line number
    line_end = np.where(data[:-1, idx] != data[1:, idx])[0]
    
    #Define line start
    line_start = np.array(line_end + 1)
    
    #Insert first and last indices
    line_start = np.insert(line_start, 0, 0)
    line_end = np.insert(line_end, line_end.shape[0], data.shape[0]-1)
    
    #Lengths should be equal
    assert len(line_start) == len(line_end)
    
    #Return line start and end indices
    return line_start, line_end

def cosine_taper(npts, nzeros):
    
    """
    Cosine taper scaled 0-1 of length npts with nzeros zero values appended afterwords. 
    """
    
    y = []
    x = np.arange(1,npts+1,1)

    #Construct the taper y
    for i in range(len(x)):
        y.append((math.cos((1/npts)*x[i]*(math.pi)) + 1) * 0.5)

    #Append additional zero pad
    y.extend(list(np.zeros((nzeros))))

    return y

def dir_cos(incl, decl, azim=0):
    
    """
    Compute three directional cosine vectors from inclination & declination. \
    Equivilaent to returning the unit vector of inc & dec. 
    ---
    incl: Inclination in degrees, positive is below the horizontal
    decl: Declination in degress, positive is east of true north
    azim: Azimuth of x axis in degrees positive is east of true north
    ---
    returns: Three-component directional cosines
    """
    
    #As radians
    xincl = incl*(math.pi/180)
    xdecl = decl*(math.pi/180)
    xazim = azim*(math.pi/180)
    
    #Compute cosine vectors
    y = math.cos(xincl) * math.cos(xdecl - xazim)
    x = math.cos(xincl) * math.sin(xdecl - xazim)
    z = math.sin(xincl)
    
    #Return cosine (unit) vectors
    return y, x, z

def k_values(nx, ny, nxp, nyp, min_max):
    
    """
    Retreive the wavenumber coordinates in radians/meter.
    ---
    nx: original grid nodes in the x (axis=1) dimension.
    ny: original grid nodes in the y (axis=0) dimension.
    nxp: padded grid nodes in the x dimension.
    nyp: padded grid nodes in the y dimension. 
    min_max: minimum and maximum x/y coordinate list as [minx, maxx, miny, maxy]
    ---
    returns: gridded x and y wavenumber coordiantes in radians/meter.
    """

    #Grid sample spacing
    dx = abs(min_max[1]-min_max[0])/(nx-1)
    dy = abs(min_max[3]-min_max[2])/(ny-1)

    #Compute wavenumber coordinates in radians/meter
    klx = 2*math.pi*np.fft.fftfreq(nxp, dx)
    kly = 2*math.pi*np.fft.fftfreq(nyp, dy)                     

    #Grid the wavenumber X and Y coordinates
    kx, ky = np.meshgrid(klx, kly)[0], np.meshgrid(klx, kly)[1]
        
    #Return wavenumber coordinates
    return kx, ky

def gauss_window_1D(x, sigma, norm=True):
    
    """
    Compute the 1-D Gaussian window.
    ---
    x: numpy array of x-coordinates.
    sigma: standard deviation.
    norm: optionally normalize the output window
    ---
    returns: 1-D Gaussian window
    """

    #Compute the normalized Gaussian kernel
    Gk = (1 / math.sqrt(2 * math.pi * sigma**2)) * np.exp(-(np.square(x)) / (2 * sigma**2))

    #Optionally normalize (the sum of all inputs equals 1)
    if norm == True:
        Gk = Gk / np.sum(Gk)

    #Return the Gaussian kernel
    return Gk