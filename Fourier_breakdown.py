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
    * Notes for Anaconda Spyder IDE users:                         *
    *   - If running this from the Anaconda Spyder IDE, before     *
    *     running you will have to turn on the interactive plot    *
    *     if you want the interactive plot to pop-up. You may do   *
    *     this by executing the following command in the console:  *
    *        %matplotlib qt                                        *
    *     Then to turn back on the inline plotting, where the plots*
    *     will appear in the Plots tab execute the following       *
    *     command in the console:                                  *
    *        %matplotlib inline                                    *
    ****************************************************************
"""
##############################
# Import Necessary Libraries #
##############################
import modules.read_log as read_log
import modules.signals as signals
import modules.rfr as rfr
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib as mpl  
mpl.rc('font',family='Calibri')
import numpy as np


##########
# Inputs #
##########
# logfile to read in
logfile = 'logfiles/tensile_1_EPON_862_pxld_86.8_replicate_4_FF_PCFF.log.lammps'


# Set some column keywords to find sections in logfile with thermo data.
# *NOTE: You minimally need one keyword where 'Step' is a good choice.*
keywords = ['Step']


# Set the sections of the logfile to get data from. Sections are counted
# starting from 1 and end at N-sections. The following options are available
# in example format:
#    'all'    get data for all sections from logfile
#    '1'      get data for only section 1 from log file
#    '1,3'    get data for sections 1 and 3 from log file
#    '1-3'    get data for sections 1, 2, and 3 from log file
#    '1-3,5'  get data for sections 1, 2, 3, and 5 from log file
# *NOTE if reading multiple sections from the log file and a column
#  is missing in a certain section, the get_data function will create
#  zeros.*
sections = 1


# Set strain direction of 'x' or 'y' or 'z'
strain_direction = 'x'


# Regression fringe response settings
# minxhi=<float>, where <float> is the minimum xhi strain that the linear region can be found. If <float> is zero, this is left as unbounded.
# This value can then be thought of as the "smallest linear span" to find within the stress-strain
# data. Examples:
#   minxhi=0.01
#   minxhi=0.005

# maxxhi=<float>, where <float> is the maximum xhi strain that the linear region can be found. If <float> is zero, this is left as unbounded.
# This value can then be thought of as the "largest strain to attempt finding the linear region in". Some data may have more "wiggly" linear 
# regions (particularly shear data, as shear data has more "wiggles" then tensile data), where this option allows you to control the "largest"
# linear region to search as the "wiggles" can maximize the fringe-slope response at fairly large strains. Examples:
#   maxxhi=0.03
#   maxxhi=0.05
minxhi = 0.0025
maxxhi = 0.0000


# Filter settings
# order=<int>, where <int> is the order of the low pass Butterworth filter. Examples:
#   order=2
#   order=3
#
# wn=<float or string>, where <float> is a floating point number between 0 and 1 to set critical frequency, which is the point at which the gain
# drops to 1/sqrt(2) that of the passband (the "-3 dB point"). If you want to control the "true cutoff frequency" instead of the "normalized
# cutoff frequency" you may compute "wn" in the following way:
#     wn = cutoff/nf; where "wn" is the input to the filter, "nf" is the Nyquist frequency, and "cutoff" is the cutoff frequency.
#     nf = 0.5*fs; where "fs" is the sampling frequency (1/delta-x).
#     wn = cutoff/(0.5*fs)
#
# If wn=<string>, the string can be 'op' where 'op' means optimize wn automatically by computing the Power Spectral Density (PSD) and finding where
# the power crosses the mean of the PSD. The following order values are recommend for wn='op':
#   wn='op' and order=2
# Examples:
#   wn=0.1
#   wn=0.5
#   wn=op
#
# quadrant_mirror=<string>, where <string> is in the format 'lo,hi', where lo and hi are integer values (i.e. '1,1', '2,1', ...). The purpose of 
# quadrant_mirror (qm) is to perform quadrant mirroring to reduce the transient edge effects of the start-up and shut-down cost of the convolution
# on either the forward or backwards pass. By mirroring the data at each end, the start-up and shut-down costs of the convolution can be "pushed"
# into data that does not matter. The data is mirrored prior to filtering and then sectioned back to the original data after filtering. The integer
# meanings for lo and hi are described below:
#   +---------+-----------------------------------------------------+-----------------------------------------------------+
#   | integer |                    lo X-data end                    |                    hi X-data end                    |
#   +---------+-----------------------------------------------------+-----------------------------------------------------+
#   |    1    | shuts off mirroring operation at "lo" X-data end    | shuts off mirroring operation at "hi" X-data end    |
#   |    2    | mirrors data "head-to-head" into quadrant 2         | mirrors data "tail-to-tail" and leaves Y-data alone |
#   |    3    | mirrors data "head-to-head" into quadrant 3         | mirrors data "tail-to-tail" and flips Y-data        |
#   |    4    | mirrors data "head-to-tail" and leaves Y-data alone | mirrors data "tail-to-head" and leaves Y-data alone |
#   +---------+-----------------------------------------------------+-----------------------------------------------------+
# When the MD stress-strain response has "slack" it is best to mirror the data into quadrant 2, and when the MD stress-strain response has "no
# slack" it is best to mirror the data into quadrant 3 for the "lo" integer. For the "hi" integer majority of the time the optimal integer is
# 2 and will be rare to deviate to other integers. Additionally the "-p" flag can be added to plot the quadrant mirroring of the data for
# inspection. The default is '1,1' and will be used if not specified. Examples:
#   qm=1,2
#   qm=2,2
#   qm=2,3-p
#   qm=1,4-p
# Additionally the string can be set to "msr" or "msr-p", where msr stands for minimum sum of residuals squared. The sum of residuals
# squared is determined for all permuations and combinations of lo and hi integer values; where which ever Butterworth filtered data has the
# minimum sum of residuals squared between permuations, is the "best quadrants" to use as the mirror (or to not mirror the data at all for
# quadrant 1). This is because the "edge" effects cause the low and high x-range residuals to be larger, when mirrored in the incorrect
# direction. Additionally the "-p" flag can be added to plot the optimized quadrant position and the quadrant mirroring of the data for
# inspection. Examples:
#   qm=msr
#   qm=msr-p
order = 2
wn = 'op'
quadrant_mirror = 'msr'




#############################
# Read log and show example #
#############################
if __name__ == "__main__":
    
    #-----------------------------------------------------------------------#
    # Set column variables based on strain_direction (dependent on LAMMPS   #
    # logged variables - will need to update based on user setup in LAMMPS) #
    #-----------------------------------------------------------------------#
    if strain_direction == 'x':
        stress_column = 'f_sxx_ave'  # Axial stress
        strain_column = 'v_etruex'   # Axial strain
        trans1_column = 'v_etruey'   # Transverse strain direction-1
        trans2_column = 'v_etruez'   # Transverse strain direction-2
    elif strain_direction == 'y':
        stress_column = 'f_syy_ave'  # Axial stress
        strain_column = 'v_etruey'   # Axial strain
        trans1_column = 'v_etruex'   # Transverse strain direction-1
        trans2_column = 'v_etruez'   # Transverse strain direction-2
    elif strain_direction == 'z':
        stress_column = 'f_szz_ave'  # Axial stress
        strain_column = 'v_etruez'   # Axial strain
        trans1_column = 'v_etruex'   # Transverse strain direction-1
        trans2_column = 'v_etruey'   # Transverse strain direction-2
    
    
    #-------------------------------------------#
    # Read logfile and get numpy arrays of data #
    #-------------------------------------------#
    log = read_log.file(logfile, keywords=keywords, pflag=False)
    data = log.get_data(sections, remove_duplicates=True, pflag=False) # {column-name:[list of data]}
    stress = np.array(data[stress_column])
    strain = np.array(data[strain_column])
    trans1 = np.array(data[trans1_column])
    trans2 = np.array(data[trans2_column])
    
    
    #---------------------------------------------------#
    # Filter stress, transverse 1 and 2 directions data #
    #---------------------------------------------------#
    # Compute the PSD if wn is set to 'op'
    if str(wn).startswith('op'):
        # Compute the PSD for different data sets
        wns_stress, psd_stress = signals.compute_PSD(strain, stress)
        
        # Find the mean value of the different PSD's
        mean_stress_psd = np.mean(psd_stress)

        # Find where each PSD crosses the mean
        wn_index = np.min(np.where(psd_stress < mean_stress_psd)[0])
        wn_stress = wns_stress[wn_index]

        print('{:<50} {} {}'.format('Computed normalized cutoff frequency for stress: ', wn_stress, wn_index))
        
        # Compute the corresponding power value for the crossing of the mean
        power_stress = psd_stress[np.min(np.where(wns_stress == wn_stress)[0])]
    
        # Filter the data with an optimized wn value
        filtered_stress, qm_stress = signals.butter_lowpass_filter(strain, stress, wn_stress, order, quadrant_mirror)
        
    else:
        # Filter the data with a user defined wn value
        filtered_stress, qm_stress = signals.butter_lowpass_filter(strain, stress, wn, order, quadrant_mirror)
    
    # Shift all data by the "minimum before the maximum" to remove any residual 
    # stress. The "minimum before the maximum" allows for fracture to occur where
    # the minimum stress is near zero strain and not maximum strain (in case fracture
    # creates a minimum stress lower then the residual stress).
    max_index = np.min(np.where(filtered_stress == np.max(filtered_stress))[0])
    min_stress = np.min(filtered_stress[:max_index])
    filtered_stress -= min_stress
    stress -= min_stress


    #-----------------------------------------#
    # Function to perform a Fourier breakdown #
    #-----------------------------------------#
    def FFT_breakdown(x, y, indices):
        #-----------------------------------#
        # Compute the one-sided FFT and PSD #
        #-----------------------------------#
        # Define sampling rate and number of data points
        N = x.shape[0] # number of data points
        fs = (N-1)/(np.max(x) - np.min(x)) # sampling rate
        d = 1/fs # sampling space

        # Perform one sided FFT
        X = np.fft.rfft(y, axis=0, norm='backward')
        f = np.fft.rfftfreq(N, d=d)
        
        # One sided amplitudes at each frequency
        amp = np.abs(X)/N
        amp[1:-1] *= 2
        if N % 2 == 0:
            amp[-1] /= 2

        # Set a scaling factor of 0 or 1 to cancel out (0) or leave (1) certain frequencies
        scaling_factors = np.zeros_like(amp)
        scaling_factors[indices] = 1
        psd_peaks = amp[indices]
        psd_freqs = f[indices]

        X_clean = scaling_factors*X
        y_filter = np.fft.irfft(X_clean)
        
        # Since we are performing a one sided FFT, the Nyquist freq may or may not be inlcuded
        # depending on even or odd number of data points, so append a value if Nyquist freq is
        # missing so that y_filter has the same shape as the X-data.
        if y_filter.shape != x.shape:
            y_filter = np.append(y_filter, y_filter[-1])
        return y_filter
    

    
    
    #---------------------------------#
    # Plot the results of this method #
    #---------------------------------#
    # Get the colors from colormap: https://matplotlib.org/stable/users/explain/colors/colormaps.html
    colors = plt.cm.tab20.colors

    # Color wheel defined by matplotlib:
    #   colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # However, we can construct our own color wheel to prioritize the colors we want first and we can 
    # have way more colors defined than what matplotlib defines. Below is Josh's preferred color wheel
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:red', 'tab:gray','tab:olive', 'tab:cyan', 'tab:pink', 'teal',
              'crimson', 'lime', 'tomato',  'blue', 'orange', 'green', 'purple', 'red', 'gray', 'olive', 'cyan', 'pink', 'tab:brown']


    # Function to walk around the color wheel defined by colors and get the next color,
    # if the color index is exceeds the color wheel, it will reset the color index to 0
    def walk_colors(color_index, colors):
        color = colors[color_index]
        color_index += 1
        if color_index + 1 > len(colors): color_index = 0
        return color, color_index

    # Set the default color cycle using rcParams
    plt.rcParams['axes.prop_cycle'] = cycler(color=colors)

    # Set xlimits
    delta = 0.01
    xlimits = (np.min(strain)-delta, np.max(strain)+1.5*delta)
    xlimits = (np.min(strain)-delta, np.max(strain)+1.5*delta+0.1)
    
    # Set fontsize
    fs = 12
    legend_fs_scale = 0.75
    
    # Start plotting data
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12.5, 9))

    ax1.plot(strain, stress, '.', ms=4, color='#bbbbbbff', label='LAMMPS data')
    ax1.plot(strain, filtered_stress, '-', lw=4, color='#2c7fb8ff', label='Butterworth Filtered data\nat the PSD critical frequency')
    ax1.legend(loc='upper right', bbox_to_anchor=(1, 1), fancybox=True, ncol=1, fontsize=legend_fs_scale*fs)
    ax1.set_xlabel('True Strain', fontsize=fs)
    ax1.set_ylabel('True Stress (MPa)', fontsize=fs)
    ax1.tick_params(axis='both', which='major', labelsize=fs)
    ax1.set_xlim(xlimits)
    
    if str(wn).startswith('op'):
        ax2.stem(wns_stress, psd_stress, linefmt='tab:blue', markerfmt='.', label='$|X(f)|^2/N$ for stress-strain')
        #ax2.plot(wn_stress, power_stress, 'o', ms=8, color='#ff9d3aff', label='Critical frequency\n({:.4f}, {:.4f})'.format(wn_stress, power_stress))
        ax2.axhline(mean_stress_psd, color='#ff9d3aff', ls='--', lw=2, label='Average power={:.4f}'.format(mean_stress_psd))
        ax2.axvline(wn_stress, color='#ff9d3aff', ls='--', lw=2, label='Critical frequency={:.4f}'.format(wn_stress))
        ax2.legend(loc='upper right', bbox_to_anchor=(1, 1), fancybox=True, ncol=1, fontsize=legend_fs_scale*fs)
        ax2.set_xlabel('Normalized Frequencies (unitless)', fontsize=fs)
        ax2.set_ylabel('Power Spectral Density', fontsize=fs)
        ax2.tick_params(axis='both', which='major', labelsize=fs)
        ax2.set_xlim((-0.001, 0.02)) # Comment/uncomment for xlimits
        ax2.set_ylim((-1*mean_stress_psd, 30*mean_stress_psd)) # Comment/uncomment for xlimits

    color_index = 0
    ax3.plot(strain, stress, '.', ms=4, color='#bbbbbbff', label='LAMMPS data')
    comp_wave = np.zeros_like(stress)
    lo = 0
    hi = wn_index
    for i in range(lo, hi+1):
        wave = FFT_breakdown(strain, stress, [i])
        comp_wave += wave
        label = 'PSD index: {}'.format(i)
        if i == 0: label += ' (DC-offset)'
        if i == wn_index: label += ' (Critical frequency)'
        color, color_index = walk_colors(color_index, colors)
        ax3.plot(strain, wave, '-', lw=2, color=color, label=label)
        ax2.plot(wns_stress[i], psd_stress[i], 'o', ms=8, color=color)
    ax3.plot(strain, comp_wave, '-', lw=4, color='#2c7fb8ff', label='Summed waves (indexes {}-{})'.format(lo, hi))

    ax3.legend(loc='upper right', bbox_to_anchor=(1, 1), fancybox=True, ncol=1, fontsize=legend_fs_scale*fs)
    ax3.set_xlabel('True Strain', fontsize=fs)
    ax3.set_ylabel('True Stress (MPa)', fontsize=fs)
    ax3.tick_params(axis='both', which='major', labelsize=fs)
    ax3.set_xlim(xlimits)

    ax4.plot(strain, stress, '.', ms=4, color='#bbbbbbff', label='LAMMPS data')
    comp_wave = np.zeros_like(stress)
    lo = int(wn_index+1)
    hi = int(3*wn_index)
    for i in range(lo, hi+1):
        wave = FFT_breakdown(strain, stress, [i])
        comp_wave += wave
        label = 'PSD index: {}'.format(i)
        color, color_index = walk_colors(color_index, colors)
        ax4.plot(strain, wave, '-', lw=2, color=color, label=label)
        ax2.plot(wns_stress[i], psd_stress[i], 'o', ms=8, color=color)
    ax4.plot(strain, comp_wave, '-', lw=4, color='#2c7fb8ff', label='Summed waves (indexes {}-{})'.format(lo, hi))

    ax4.legend(loc='upper right', bbox_to_anchor=(1, 1), fancybox=True, ncol=1, fontsize=legend_fs_scale*fs)
    ax4.set_xlabel('True Strain', fontsize=fs)
    ax4.set_ylabel('True Stress (MPa)', fontsize=fs)
    ax4.tick_params(axis='both', which='major', labelsize=fs)
    ax4.set_xlim(xlimits)
        
    
    fig.tight_layout()
    plt.show()