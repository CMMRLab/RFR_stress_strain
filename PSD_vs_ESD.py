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
        wns_stress, esd_stress = signals.compute_ESD(strain, stress)
        
        # Find the mean value of the different PSD and ESD
        mean_stress_psd = np.mean(psd_stress)
        mean_stress_esd = np.mean(esd_stress)

        
        # Find where each PSD crosses the mean
        wn_stress_psd = wns_stress[np.min(np.where(psd_stress < mean_stress_psd)[0])]
        wn_stress_esd = wns_stress[np.min(np.where(esd_stress < mean_stress_esd)[0])]


        
        # Compute the corresponding power value for the crossing of the mean
        power_stress_psd = psd_stress[np.min(np.where(wns_stress == wn_stress_psd)[0])]
        power_stress_esd = esd_stress[np.min(np.where(wns_stress == wn_stress_esd)[0])]

    
        # Filter the data with an optimized wn value
        filtered_stress_psd, qm_stress_psd = signals.butter_lowpass_filter(strain, stress, wn_stress_psd, order, quadrant_mirror)
        filtered_stress_esd, qm_stress_esd = signals.butter_lowpass_filter(strain, stress, wn_stress_esd, order, quadrant_mirror)

        
    else:
        # Filter the data with a user defined wn value
        filtered_stress_psd, qm_stress_psd = signals.butter_lowpass_filter(strain, stress, wn, order, quadrant_mirror)
        filtered_stress_esd, qm_stress_esd = signals.butter_lowpass_filter(strain, stress, wn, order, quadrant_mirror)

    
    #------------------------------------#
    # RFR function to use for ESD vs PSD #
    #------------------------------------#
    def RFR(strain, stress):
        # Shift all data by the "minimum before the maximum" to remove any residual 
        # stress. The "minimum before the maximum" allows for fracture to occur where
        # the minimum stress is near zero strain and not maximum strain (in case fracture
        # creates a minimum stress lower then the residual stress).
        max_index = np.min(np.where(stress == np.max(stress))[0])
        min_stress = np.min(stress[:max_index])
        stress -= min_stress
        stress -= min_stress
        
        #--------------------------------------------------------#
        # Compute the forward-backwards-forwards fringe response #
        #--------------------------------------------------------#
        # Step1: First forward response (fr1 - applying minxhi and maxxhi accordingly)
        min_strain_fr1 = min(strain); max_strain_fr1 = max(strain)
        if minxhi > 0: min_strain_fr1 = minxhi 
        if maxxhi > 0: max_strain_fr1 = maxxhi 
        fr1_fringe, fr1_slopes = rfr.compute_fringe_slope(strain, stress, min_strain=min_strain_fr1, max_strain=max_strain_fr1, direction='forward')
        fr1_max_index = np.argmax(fr1_slopes)
        fr1_max_slope = fr1_slopes[fr1_max_index]
        fr1_max_fringe = fr1_fringe[fr1_max_index]
        
        # Step2: First backwards response (br1 - applying minxhi and maxxhi accordingly)
        min_strain_br1 = min(strain)
        max_strain_br1 = fr1_max_fringe - minxhi # use minxhi to set the "span" of the smallest linear region acceptable
        fr1_max_index_absolute = np.argmin(np.abs(strain - fr1_max_fringe))
        reduced_strain = strain[0:fr1_max_index_absolute]
        reduced_stress = stress[0:fr1_max_index_absolute]
    
        br1_fringe, br1_slopes = rfr.compute_fringe_slope(reduced_strain, reduced_stress, min_strain=min_strain_br1, max_strain=max_strain_br1, direction='reverse')
        br1_max_index = np.argmax(br1_slopes)
        br1_max_slope = br1_slopes[br1_max_index]
        br1_max_fringe = br1_fringe[br1_max_index]
        br1_max_index_absolute = np.argmin(np.abs(strain - br1_max_fringe))
        
        # Step3: Second forward response (fr2 - applying minxhi and maxxhi accordingly)
        min_strain_fr2 = min_strain_fr1 + br1_max_fringe
        max_strain_fr2 = max_strain_fr1 + br1_max_fringe
        reduced_strain = strain[br1_max_index_absolute:-1]
        reduced_stress = stress[br1_max_index_absolute:-1]
        
        fr2_fringe, fr2_slopes = rfr.compute_fringe_slope(reduced_strain, reduced_stress, min_strain=min_strain_fr2, max_strain=max_strain_fr2, direction='forward')
        fr2_max_index = np.argmax(fr2_slopes)
        fr2_max_slope = fr2_slopes[fr2_max_index]
        fr2_max_fringe = fr2_fringe[fr2_max_index]
        fr2_max_index_absolute = np.argmin(np.abs(strain - fr2_max_fringe))
        
        # Set linear region bounds xlo and xhi from the 3 step FBF method
        xlo = br1_max_fringe
        xhi = fr2_max_fringe
        
        #--------------------------------------------#
        # Compute the yield point and yield strength #
        #--------------------------------------------#
        # Step1: Compute the 2nd derivative from the end of the linear region to the max strain
        reduced_fringe = fr2_fringe[fr2_max_index:-1]
        reduced_slopes = fr2_slopes[fr2_max_index:-1]
        dstrain, dslopes1, dslopes2 = rfr.compute_derivative(fr2_fringe, fr2_slopes)
        
        # Step2: Find peaks and valleys of the 2nd derivative using tuned standard deviations
        prominence = np.std(dslopes2)/3
        xpeaks, ypeaks, xvalleys, yvalleys = rfr.find_peaks_and_valleys(dstrain, dslopes2, prominence=prominence)
        
        # Step3: Use first valley with respect to strain as the yield point (if any valleys exist)
        yield_index, x_yield, y_yield, x_yield_d2, y_yield_d2 = None, None, None, None, None
        if np.any(xvalleys) and np.any(yvalleys):
            x_yield_d2 = xvalleys[0]
            y_yield_d2 = yvalleys[0]
            yield_index = np.min(np.where(strain == x_yield_d2)[0])
            x_yield = strain[yield_index]
            y_yield = stress[yield_index]
            
        return xlo, xhi, x_yield, y_yield
    
    
    #-------------------------------------------------------#
    # Compute the linear regression for the Young's modulus #
    #-------------------------------------------------------#
    # Filtered with PSD
    xlo_psd, xhi_psd, x_yield_psd, y_yield_psd = RFR(strain, filtered_stress_psd)
    xlo_index_psd = np.min(np.where(strain == xlo_psd)[0])
    xhi_index_psd = np.min(np.where(strain == xhi_psd)[0])
    youngs_modulus_coeffs_psd = np.polynomial.polynomial.polyfit(strain[xlo_index_psd:xhi_index_psd+1], filtered_stress_psd[xlo_index_psd:xhi_index_psd+1], 1)
    youngs_modulus_x_psd = np.array([xlo_psd, xhi_psd])
    youngs_modulus_y_psd = youngs_modulus_coeffs_psd[1]*youngs_modulus_x_psd + youngs_modulus_coeffs_psd[0]
    print('{:<50} {}'.format("Computed Young's modulus (PSD): ", youngs_modulus_coeffs_psd[1]))
    print('{:<50} {}'.format("Computed yield strength  (ESD): ", y_yield_psd))
    
    # Filtered with ESD
    xlo_esd, xhi_esd, x_yield_esd, y_yield_esd = RFR(strain, filtered_stress_esd)
    xlo_index_esd = np.min(np.where(strain == xlo_esd)[0])
    xhi_index_esd = np.min(np.where(strain == xhi_esd)[0])
    youngs_modulus_coeffs_esd = np.polynomial.polynomial.polyfit(strain[xlo_index_esd:xhi_index_esd+1], filtered_stress_esd[xlo_index_esd:xhi_index_esd+1], 1)
    youngs_modulus_x_esd = np.array([xlo_esd, xhi_esd])
    youngs_modulus_y_esd = youngs_modulus_coeffs_esd[1]*youngs_modulus_x_esd + youngs_modulus_coeffs_esd[0]
    print('{:<50} {}'.format("Computed Young's modulus (ESD): ", youngs_modulus_coeffs_esd[1]))
    print('{:<50} {}'.format("Computed yield strength  (ESD): ", y_yield_esd))
    
    

        
    
    
    #---------------------------------#
    # Plot the results of this method #
    #---------------------------------#
    # Set xlimits
    delta = 0.01
    xlimits = (np.min(strain)-delta, np.max(strain)+1.5*delta)
    
    # Set fontsize
    fs = 14
    label_rel_pos = (0.005, 0.99)
    
    # Start plotting data
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
    
    
    ax1.plot(strain, stress, '.', ms=4, color='#bbbbbbff', label='LAMMPS data')
    ax1.plot(strain, filtered_stress_psd, '-', lw=2, color='#2c7fb8ff', label='Filtered data\n(PSD critical frequency)')
    ax1.plot(x_yield_psd, y_yield_psd, 'o', ms=8, color='tab:purple', label='Yield point (x,y)\n({:.4f}, {:.4f})'.format(x_yield_psd, y_yield_psd))
    ax1.plot(youngs_modulus_x_psd, youngs_modulus_y_psd, '-', lw=4, color='#ff9d3aff', label="Young's modulus\n{:.4f}".format(youngs_modulus_coeffs_psd[1]))
    ax1.legend(loc='lower right', bbox_to_anchor=(1, 0), fancybox=True, ncol=1, fontsize=0.75*fs)
    ax1.set_xlabel('True Strain', fontsize=fs)
    ax1.set_ylabel('True Stress (MPa)', fontsize=fs)
    ax1.tick_params(axis='both', which='major', labelsize=fs)
    ax1.set_xlim(xlimits)
    ax1.text(*label_rel_pos, '(a)', transform=ax1.transAxes, fontsize=fs, fontweight='bold', va='top', ha='left')
    
    ax3.plot(strain, stress, '.', ms=4, color='#bbbbbbff', label='LAMMPS data')
    ax3.plot(strain, filtered_stress_esd, '-', lw=2, color='#2c7fb8ff', label='Filtered data\n(ESD critical frequency)')
    ax3.plot(x_yield_esd, y_yield_esd, 'o', ms=8, color='tab:purple', label='Yield point (x,y)\n({:.4f}, {:.4f})'.format(x_yield_esd, y_yield_esd))
    ax3.plot(youngs_modulus_x_esd, youngs_modulus_y_esd, '-', lw=4, color='#ff9d3aff', label="Young's modulus\n{:.4f}".format(youngs_modulus_coeffs_esd[1]))
    ax3.legend(loc='lower right', bbox_to_anchor=(1, 0), fancybox=True, ncol=1, fontsize=0.75*fs)
    ax3.set_xlabel('True Strain', fontsize=fs)
    ax3.set_ylabel('True Stress (MPa)', fontsize=fs)
    ax3.tick_params(axis='both', which='major', labelsize=fs)
    ax3.set_xlim(xlimits)
    ax3.text(*label_rel_pos, '(c)', transform=ax3.transAxes, fontsize=fs, fontweight='bold', va='top', ha='left')
    
    if str(wn).startswith('op'):
        ax2.stem(wns_stress, psd_stress, linefmt='tab:blue', basefmt='tab:blue', markerfmt='.', label='$|X(f)|^2/N$')
        ax2.plot(wn_stress_psd, power_stress_psd, 'o', ms=8, color='#ff9d3aff', label='{}$_c$ ({:.3f}, {:.3f})'.format(r'$\omega$', wn_stress_psd, power_stress_psd))
        ax2.axhline(mean_stress_psd, color='#ff9d3aff', ls='--', lw=2, label='Average power={:.4f}'.format(mean_stress_psd))
        ax2.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fancybox=True, ncol=1, fontsize=0.75*fs)
        ax2.set_xlabel('Normalized Frequencies, {}'.format(r'$\omega$'), fontsize=fs)
        ax2.set_ylabel('Power Spectral Density (PSD)', fontsize=fs)
        ax2.tick_params(axis='both', which='major', labelsize=fs)
        ax2.set_xlim((-0.001, 0.03)) # Comment/uncomment for xlimits
        ax2.set_ylim((-1*mean_stress_psd, 30*mean_stress_psd)) # Comment/uncomment for xlimits
        ax2.text(*label_rel_pos, '(b)', transform=ax2.transAxes, fontsize=fs, fontweight='bold', va='top', ha='left')
        

        ax4.stem(wns_stress, esd_stress, linefmt='tab:blue', basefmt='tab:blue', markerfmt='.', label='$|X(f)|^2/2T$')
        ax4.plot(wn_stress_esd, power_stress_esd, 'o', ms=8, color='#ff9d3aff', label='{}$_c$ ({:.3f}, {:.3f})'.format(r'$\omega$', wn_stress_esd, power_stress_esd))
        ax4.axhline(mean_stress_esd, color='#ff9d3aff', ls='--', lw=2, label='Average energy={:.3f}'.format(mean_stress_esd))
        ax4.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fancybox=True, ncol=1, fontsize=0.75*fs)
        ax4.set_xlabel('Normalized Frequencies, {}'.format(r'$\omega$'), fontsize=fs)
        ax4.set_ylabel('Energy Spectral Density (ESD)', fontsize=fs)
        ax4.tick_params(axis='both', which='major', labelsize=fs)
        ax4.set_xlim((-0.001, 0.03)) # Comment/uncomment for xlimits
        ax4.set_ylim((-1*mean_stress_esd, 30*mean_stress_esd)) # Comment/uncomment for xlimits
        ax4.text(*label_rel_pos, '(d)', transform=ax4.transAxes, fontsize=fs, fontweight='bold', va='top', ha='left')
        

    
    fig.tight_layout()
    #plt.show()
    basename = logfile[:logfile.rfind('.')] + '_PSD_vs_ESD'
    fig.savefig(basename+'.jpeg', dpi=1000)