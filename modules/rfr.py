# -*- coding: utf-8 -*-
"""
@author: Josh Kemppainen
Revision 1.0
March 24, 2025
Michigan Technological University
1400 Townsend Dr.
Houghton, MI 49931

    ****************************************************************
    * Requirements:                                                *
    *   python 3.7+                                                *
    *                                                              *
    * Dependencies:                                                *
    *   python matplotlib module:                                  *
    *    - pip3 install matplotlib (if pip manager is installed)   *
    *                                                              *
    *   python numpy module:                                       *
    *    - pip3 install numpy (if pip manager is installed)        *
    *                                                              *
    *   python scipy module:                                       *
    *    - pip3 install scipy (if pip manager is installed)        *
    ****************************************************************
"""
##############################
# Import Necessary Libraries #
##############################
from scipy.signal import find_peaks
import numpy as np


def find_peaks_and_valleys(xdata, ydata, prominence=None):
    """
    Function to compute the peaks and valleys using a specified prominence.

    Parameters
    ----------
    xdata : Numpy array.
            Containing increasing independent variable data.
            
    ydata : Numpy array.
            Containing dependent variable data.
            
    prominence : Float, None, optional
                The prominence value to used for peak and valley finding.

    Returns
    -------
    xpeaks : Numpy array.
             X-values for the found peaks.
             
    ypeaks : Numpy array.
             Y-values for the found peaks.
             
    xvalleys : Numpy array.
               X-values for the found valleys.
               
    yvalleys : Numpy array.
               Y-values for the found valleys.
    """
    # Find peaks
    peaks, properties = find_peaks(ydata, prominence=prominence)
    xpeaks = xdata[peaks]; ypeaks = ydata[peaks]
    
    # Find valleys
    xvalleys, yvalleys = [], []
    if len(xpeaks) >= 2 and len(ypeaks) >= 2:
        for i in range(len(peaks)-1):
            lo = peaks[i]; hi = peaks[i+1];
            between_peaksx = xdata[lo:hi]
            between_peaksy = ydata[lo:hi]
            minimum_index = np.min(np.where(between_peaksy == between_peaksy.min())[0])
            xvalleys.append( between_peaksx[minimum_index] )
            yvalleys.append( between_peaksy[minimum_index] ) 
    xvalleys = np.array(xvalleys)
    yvalleys = np.array(yvalleys)
    return xpeaks, ypeaks, xvalleys, yvalleys


def compute_derivative(xdata, ydata):
    """
    Function to compute the 1st and 2nd order central derivatives.

    Parameters
    ----------
    xdata : Numpy array.
            Containing increasing independent variable data.
            
    ydata : Numpy array.
            Containing dependent variable data.

    Returns
    -------
    dxn : Numpy array.
          X-values for the computed derivatives.
          
    dy1 : Numpy array.
          Y-values for the computed 1st derivative.

    dy2 : Numpy array.
          Y-values for the computed 2nd derivative.
    """
    dxn = xdata[1:-1] # ignore first and last point
    dy1 = np.zeros_like(dxn)
    dy2 = np.zeros_like(dxn)
    if len(xdata) == len(ydata):
        for i in range(1, len(xdata)-1):
            dx = (xdata[i+1] - xdata[i-1])/2
            if dx == 0:
                print('WARNING finite difference dx was zero at x={}. Derivative was set to zero to avoid infinite derivative.'.format(xdata[i]))
            else:
                dy1[i-1] = (ydata[i+1] - ydata[i-1])/(2*dx)
                dy2[i-1] = (ydata[i+1] - 2*ydata[i] + ydata[i-1])/(dx*dx) 
    else: print('ERROR (compute_derivative) inconsistent number of data points between X and Y arrays')
    return dxn, dy1, dy2


def compute_fringe_slope(strain, stress, min_strain=None, max_strain=None, direction='forward'):
    """
    Function to fringe slope operation

    Parameters
    ----------
    strain : Numpy array.
             Containing increasing independent variable data.
            
    stress : Numpy array.
             Containing dependent variable data.
            
    min_strain : Float, None, optional
                The minimum strain value used for logging the fringe/slope values in.
                
    max_strain : Float, None, optional
                The maximum strain value used for logging the fringe/slope values in.
                
    direction : String, optional
                The direction which to traverse the data, with the following options:
                    'forward' -> traverse the data in the forwards direction
                    'reverse' -> traverse the data in the reverse direction

    Returns
    -------
    fringe : Numpy array.
             Containing the computed fringe response X-values.
             
    slope : Numpy array.
             Containing the computed fringe response Y-values.
    """
    # Set direction
    if direction == 'forward':
        strain = strain.copy()
        stress = stress.copy()
    elif direction == 'reverse':
        strain = np.flip(strain.copy())
        stress = np.flip(stress.copy())
    else:
        raise Exception(f'ERROR direction={direction} is not supported. Supported directions are "forward" or "reverse"')
    
    # Set defaults if min_strain or max_strain are None
    if min_strain is None: min_strain = min(strain)
    if max_strain is None: max_strain = max(strain)
    
    # Start the walked linear regression method
    slopes, fringe = [], []
    sum_xi, sum_yi, sum_xi_2, sum_yi_2, sum_xi_yi, n = 0, 0, 0, 0, 0, 0
    for x, y in zip(strain, stress):
        # Compute cumulative linear regression parameters
        sum_xi += x
        sum_yi += y
        n += 1
        sum_xi_2 += x*x
        sum_yi_2 += y*y
        sum_xi_yi += x*y
        
        # Need at least 2 points to perform linear regression
        if n <= 3: continue
        
        # Only compute outputs if x is in the desired range
        if min_strain <= x <= max_strain:
            SSxy = sum_xi_yi - (sum_xi*sum_yi/n)
            SSxx = sum_xi_2 - (sum_xi*sum_xi/n)
            b1 = SSxy/SSxx
            
            slopes.append(b1)
            fringe.append(x)
    return np.array(fringe), np.array(slopes)