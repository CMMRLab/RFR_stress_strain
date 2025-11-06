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
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np


def compute_PSD(xdata, ydata):
    """
    Function to compute the Power Spectral Density (PSD) with
    the X-values being the normalized frequencies.

    Parameters
    ----------
    xdata : Numpy array.
            Containing increasing independent variable data.
            
    ydata : Numpy array.
            Containing dependent variable data.

    Returns
    -------
    wns : Numpy array.
          X-values for the PSD which have been normalized (x_fft is the 
          non-normalized frequency components).
          
    psd : Numpy array.
          Y-values for the PSD.

    """
    # Define sampling rate and number of data points
    dx = np.mean(np.abs(np.diff(xdata)))
    if dx != 0: 
        fs = 1/dx # sampling rate
    else: fs = xdata.shape[0]/(np.max(xdata) - np.min(xdata))
    N = xdata.shape[0] # number of data points
    d = 1/fs # sampling space

    # Perform one sided FFT
    fft_response = np.fft.rfft(ydata, axis=0, norm='backward')
    x_fft = np.fft.rfftfreq(N, d=d)
    y_fft = fft_response 
    
    # Compute the final PSD and normalized cutoff frequencies
    psd = np.real( (y_fft*np.conjugate(y_fft))/N ) 
    wns = x_fft/(0.5*fs)
    return wns, psd

def compute_ESD(xdata, ydata):
    """
    Function to compute the Energy Spectral Density (ESD) with
    the X-values being the normalized frequencies.

    Parameters
    ----------
    xdata : Numpy array.
            Containing increasing independent variable data.
            
    ydata : Numpy array.
            Containing dependent variable data.

    Returns
    -------
    wns : Numpy array.
          X-values for the ESD which have been normalized (x_fft is the 
          non-normalized frequency components).
          
    psd : Numpy array.
          Y-values for the ESD.

    """
    # Define sampling rate and number of data points
    dx = np.mean(np.abs(np.diff(xdata)))
    if dx != 0: 
        fs = 1/dx # sampling rate
    else: fs = xdata.shape[0]/(np.max(xdata) - np.min(xdata))
    N = xdata.shape[0] # number of data points
    d = 1/fs # sampling space
    T = np.max(xdata) - np.min(xdata) # Duration of signal

    # Perform one sided FFT
    fft_response = np.fft.rfft(ydata, axis=0, norm='backward')
    x_fft = np.fft.rfftfreq(N, d=d)
    y_fft = fft_response 
    
    # Compute the final ESD and normalized cutoff frequencies
    esd = np.real( (y_fft*np.conjugate(y_fft))/(2*T) ) 
    wns = x_fft/(0.5*fs)
    return wns, esd


def power_to_db(power, ref_power=1):
    """
    Function to convert power to decibels (dB).
    
    Parameters
    ----------
    power: Numpy array.
           The power value (can be a single number or a NumPy array).
           
    ref_power : Float, Integer, optional
                The reference power for the conversion.
    
    Returns
    -------
    power_in_dB: Numpy array.
                 The power in decibels (dB).
    """
    power_in_dB = 10*np.log10(power/ref_power)
    return power_in_dB


def butter_lowpass_filter(xdata, ydata, wn, order, quadrant_mirror):
    """

    Parameters
    ----------
    xdata : Numpy array.
            Containing increasing independent variable data.
            
    ydata : Numpy array.
            Containing dependent variable data.
            
    wn : Float.
        The critical frequency for the low pass Butterworth filter, which
        is the point at which the gain drops to 1/sqrt(2) that of the passband
        (the “-3 dB point”).
        
    order : Integer
        The order of the filter. 
        
    quadrant_mirror : String
        String defining how to edges of the data are treated during filtering. The string format is
        'lo,hi', where lo and hi are integer values (i.e. '1,1', '2,1', ...). The purpose of 
        quadrant_mirror (qm) is to perform quadrant mirroring to reduce the transient edge effects of
        the start-up and shut-down cost of the convolution on either the forward or backwards pass.
        By mirroring the data at each end, the start-up and shut-down costs of the convolution can be
        "pushed" into data that does not matter. The data is mirrored prior to filtering and then sectioned
        back to the original data after filtering. The integer meanings for lo and hi are described below:

          +---------+-----------------------------------------------------+-----------------------------------------------------+
          | integer |                    lo X-data end                    |                    hi X-data end                    |
          +---------+-----------------------------------------------------+-----------------------------------------------------+
          |    1    | shuts off mirroring operation at "lo" X-data end    | shuts off mirroring operation at "hi" X-data end    |
          |    2    | mirrors data "head-to-head" into quadrant 2         | mirrors data "tail-to-tail" and leaves Y-data alone |
          |    3    | mirrors data "head-to-head" into quadrant 3         | mirrors data "tail-to-tail" and flips Y-data        |
          |    4    | mirrors data "head-to-tail" and leaves Y-data alone | mirrors data "tail-to-head" and leaves Y-data alone |
          +---------+-----------------------------------------------------+-----------------------------------------------------+

        When the MD stress-strain response has "slack" it is best to mirror the data into quadrant 2, and when 
        the MD stress-strain response has "no slack" it is best to mirror the data into quadrant 3 for the "lo" integer. For the "hi"
        integer majority of the time the optimal integer is 2 and will be rare to deviate to other integers. Additionally the "-p" 
        flag can be added to plot the quadrant mirroring of the data for inspection. The default is '1,1' and will be used if not
        specified. Examples:
          qm=1,2
          qm=2,2
          qm=2,3-p
          qm=1,4-p
          
        Additionally the string can be set to "msr" or "msr-p", where msr stands for minimum sum of residuals squared. The sum of residuals
        squared is determined for all permuations and combinations of lo and hi integer values; where which ever Butterworth filtered data has the
        minimum sum of residuals squared between permuations, is the "best quadrants" to use as the mirror (or to not mirror the data at all for
        quadrant 1). This is because the "edge" effects cause the low and high x-range residuals to be larger, when mirrored in the incorrect
        direction. Additionally the "-p" flag can be added to plot the optimized quadrant position and the quadrant mirroring of the data for
        inspection. Examples:
          qm=msr
          qm=msr-p

    Returns
    -------
    y : Numpy array.
        Filtered in Y-axis data.
        
    quadrant_mirror : String.
                      The quadrant mirror string used for the filter implementation.
    """
    
    #-----------------------------------------------------------------------#
    # Automagically detect which are the best quadrant mirroring locations #
    #-----------------------------------------------------------------------#
    if 'msr' in str(quadrant_mirror):
        quadrant_mirror = determine_mirroring_locations(xdata, ydata, wn, order, quadrant_mirror)
        
    #----------------------------------------------------------------------------#
    # Perform quadrant mirroring operations if two digits are in quadrant_mirror #
    #----------------------------------------------------------------------------#
    digits = [int(i) for i in str(quadrant_mirror) if i.isdigit()]
    if len(digits) == 2:
        lo, hi = digits
        xdata, ydata, lo_trim, hi_trim = data_extension(xdata, ydata, lo=lo, hi=hi)
            
        # Plot the data if the '-p' flag is at the end of quadrant_mirror
        if str(quadrant_mirror).endswith('-p'):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(xdata[0:lo_trim], ydata[0:lo_trim], 'o', ms=4, color='tab:green', label='lo data in quadrant {}'.format(lo))
            ax.plot(xdata[lo_trim:hi_trim], ydata[lo_trim:hi_trim], 'o', ms=4, color='tab:blue', label='Orignal data w/o mirror')
            ax.plot(xdata[hi_trim:-1], ydata[hi_trim:-1], 'o', ms=4, color='tab:cyan', label='hi data in quadrant {}'.format(hi))
            ax.axhline(0, ls='--', color='black', lw=1)
            ax.axvline(0, ls='--', color='black', lw=1)
            ax.legend()
    
    #------------------------------------------------#
    # Apply Butterworth filter with zero phase shift #
    #------------------------------------------------#
    sos = sp.signal.butter(order, wn, btype='low', analog=False, output='sos', fs=None)
    y = sp.signal.sosfiltfilt(sos, ydata, axis=-1, padtype=None)
    
    # Plot data if user wants
    if str(quadrant_mirror).endswith('-p') and len(digits) == 2:
        ax.plot(xdata, y, '-', lw=4, color='tab:orange', label='Filtered data')
        ax.legend()
    
    # If quadrant mirroring was used, get orginal length of data and
    if len(digits) == 2: y = y[lo_trim:hi_trim] 
    
    return y, quadrant_mirror


def determine_mirroring_locations(xdata, ydata, wn, order, quadrant_mirror):
    """
    Function to automatically determine which are the optimal directions to
    mirror the quadrants in. This is determined via a minimization of a 
    sum of residuals, where the optimal quadrant will produce a mirroring
    location that has the lowest sum of residuals.

    Parameters
    ----------
    xdata : Numpy array.
            Containing increasing independent variable data.

    ydata : Numpy array.
            Containing dependent variable data.
            
    wn : Float.
        The critical frequency for the low pass Butterworth filter, which
        is the point at which the gain drops to 1/sqrt(2) that of the passband
        (the “-3 dB point”).
        
    order : Integer
        The order of the filter. 
        
    quadrant_mirror : String
        String defining how to edges of the data are treated during filtering. 
        for this function, this string is only used to check for the "-p" flag
        so the newly constructed string can carry over the "-p" flag.
        
    Returns
    -------
    optimal_quadrant_mirror : String.
                              Optimized quadrant mirroring string.
    """
    #-------------------------------------------------#
    # Determine half_data to only check for residuals #
    # either from lo-half_data or half_data-hi        #
    #-------------------------------------------------#
    half_data = int(xdata.shape[0]/2)
    
    #------------------------------#
    # First: Optimize the "lo" end #
    #------------------------------#
    lo_quads2test = [1, 2, 3, 4]; lo_summed_residuals2 = {} # {quadrant_mirror:sum-of-residuals-squared}
    for quad in lo_quads2test: 
        quadrants = '{},{}'.format(quad, 1) # hold hi constant at 1
        ybutter, qm = butter_lowpass_filter(xdata, ydata, wn, order, quadrants)
        residuals = ydata - ybutter
        residuals = residuals[:half_data] # we only care about the first half fit
        lo_summed_residuals2[quad] = np.sum(residuals**2)
        
    # Find minimized sum of residuals squared
    lo = min(lo_summed_residuals2, key=lo_summed_residuals2.get)
    
    #-------------------------------#
    # Second: Optimize the "hi" end #
    #-------------------------------#
    hi_quads2test = [1, 2, 3, 4]; hi_summed_residuals2 = {} # {quadrant_mirror:sum-of-residuals-squared}
    for quad in hi_quads2test: 
        quadrants = '{},{}'.format(1, quad) # hold lo constant at 1
        ybutter, qm = butter_lowpass_filter(xdata, ydata, wn, order, quadrants)
        residuals = ydata - ybutter
        residuals = residuals[half_data:] # we only care about the last half fit
        hi_summed_residuals2[quad] = np.sum(residuals**2)
        
    # Find minimized sum of residuals squared
    hi = min(hi_summed_residuals2, key=hi_summed_residuals2.get)
    
    #------------------------------------#
    # Set optimal quadrant_mirror string #
    #------------------------------------#
    optimal_quadrant_mirror = '{},{}'.format(lo, hi)
    if '-p' in str(quadrant_mirror):
        optimal_quadrant_mirror = '{},{}-p'.format(lo, hi)
    return optimal_quadrant_mirror


def data_extension(xdata, ydata, lo=1, hi=1):
    """
    Function to extend data at the "lo" and "hi" end. This function assumes xdata is increasing and
    ydata is the dependent variable. This function replaces the usage of scipy's "signal.filtfilt" 
    or "signal.sosfiltfilt" padtype=<'odd' or 'even' or 'const' or None>, with greater functionality.
    
    For example the padtype=<'odd' or 'even' or 'const' or None> implementation assumes that the 
    padding on the "lo X-end" will be the same as the padding that should be used on the "hi X-end",
    however for stress-strain data, this is usually not the case.
    
    lo/hi integer meanings can be thought of as quadrant numbers for "lo" and have the same meaning 
    for "hi", but at the tail end of the data. For example:
        1 = means shut off data mirroring
        2 = means mirror data (lo=mirrored into quadrant 2)
        3 = means apply 180 degree rotation (lo=mirrored into quadrant 3)
        4 = means flip the Y-data and append at the end of the Y-data

    Parameters
    ----------
    xdata : Numpy array.
            Containing increasing independent variable data.

    ydata : Numpy array.
            Containing dependent variable data.

    lo : Integer, optional
         Integer of 1, 2, 3, or 4 to set "lo" xdata end (or start of xdata). The default is 1.
    hi : Integer, optional
         Integer of 1, 2, 3, or 4 to set "lo" xdata end (or start of xdata). The default is 1.

    Returns
    -------
    xdata : Numpy array.
            New xdata with applied mirroring indexes of "lo" and "hi".
            
    ydata : Numpy array.
            New ydata with applied mirroring indexes of "lo" and "hi".
            
    lo_trim : Integer.
              Lower location in new xdata and ydata of the original data. For example,
              the original xdata can be retrieved from the new xdata by:
                  orginal_xdata = new_xdata[lo_trim:hi_trim]
                  
    hi_trim : Integer.
              Upper location in new xdata and ydata of the original data. For example,
              the original xdata can be retrieved from the new xdata by:
                  orginal_xdata = new_xdata[lo_trim:hi_trim]
    """
    
    
    # Setup number of data points based on index mirroring
    index = 0 # mirroring index (0=means mirror at first data point at low and end last data point on hi end)
    ndata = xdata.shape[0] - (index+1)
    
    # Perform lo padding operations
    if lo == 2:
        lo_xdata = min(xdata) + xdata[index] - xdata[::-1][index+1:]
        lo_ydata = ydata[::-1][index+1:]
    elif lo == 3:
        lo_xdata = min(xdata) + xdata[index] - xdata[::-1][index+1:]
        lo_ydata = ydata[index] - ydata[::-1][index+1:]
    elif lo == 4:
        lo_xdata = min(xdata) + xdata[index] - xdata[::-1][index+1:]
        lo_ydata = ydata[index+1:] - (ydata[-(index+1)] - ydata[index]) 
    else:
        lo_xdata = np.array([])
        lo_ydata = np.array([])
    
    # Perform hi padding operations
    if hi == 2:
        hi_xdata = -min(xdata) + max(xdata) + xdata[index+1:]  
        hi_ydata = ydata[::-1][index+1:]
    elif hi == 3:
        hi_xdata = -min(xdata) + max(xdata) + xdata[index+1:]  
        hi_ydata = ydata[index] - ydata[::-1][index+1:] + 2*ydata[-(index+1)] + ydata[index]
    elif hi == 4:
        hi_xdata = -min(xdata) + max(xdata) + xdata[index+1:]  
        hi_ydata = ydata[-(index+1)] + ydata[index+1:]
    else:
        hi_xdata = np.array([])
        hi_ydata = np.array([])
        
    # Assemble data
    if lo in [2, 3, 4] and hi in [2, 3, 4]:
        xdata = np.concatenate((lo_xdata, xdata, hi_xdata), axis=0)
        ydata = np.concatenate((lo_ydata, ydata, hi_ydata), axis=0)  
        lo_trim = ndata
        hi_trim = -ndata
    elif lo in [2, 3, 4] and hi == 1:
        xdata = np.concatenate((lo_xdata, xdata), axis=0)
        ydata = np.concatenate((lo_ydata, ydata), axis=0)
        lo_trim = ndata
        hi_trim = xdata.shape[0]
    elif lo == 1 and hi in [2, 3, 4]:
        xdata = np.concatenate((xdata, hi_xdata), axis=0)
        ydata = np.concatenate((ydata, hi_ydata), axis=0)  
        lo_trim = 0
        hi_trim = -ndata
    else:
        lo_trim = 0
        hi_trim = xdata.shape[0]
    return xdata, ydata, lo_trim, hi_trim
