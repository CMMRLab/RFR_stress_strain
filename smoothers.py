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
import modules.other_smoothers as other_smoothers
import modules.read_log as read_log
import modules.signals as signals
import modules.rfr as rfr
import matplotlib.pyplot as plt
import matplotlib as mpl  
mpl.rc('font',family='Calibri')
import numpy as np
import warnings
warnings.filterwarnings('ignore')


##########
# Inputs #
##########
# logfile to read in
logfile = 'logfiles/tensile_1_EPON_862_pxld_86.8_replicate_4_FF_PCFF.log.lammps'
strain_direction = 'x'

logfile = 'logfiles/tensile_3_PBZ_pxld_87_replicate_5_FF_PCFF.log.lammps'
strain_direction = 'z'

logfile = 'logfiles/tensile_2_AroCy_L10_pxld_97_replicate_1_FF_PCFF.log.lammps'
strain_direction = 'y'

logfile = 'logfiles/tensile_1_PEEK_pxld_90_replicate_3_FF_PCFF.log.lammps'
strain_direction = 'x'


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
        
        # Find the mean value of the different PSD and ESD
        mean_stress_psd = np.mean(psd_stress)
        
        # Find where each PSD crosses the mean
        wn_stress_psd = wns_stress[np.min(np.where(psd_stress < mean_stress_psd)[0])]

        
        # Compute the corresponding power value for the crossing of the mean
        power_stress_psd = psd_stress[np.min(np.where(wns_stress == wn_stress_psd)[0])]
        
        filtered_stress, qm_stress_psd = signals.butter_lowpass_filter(strain, stress, wn_stress_psd, order, quadrant_mirror)
        
    else:
        # Filter the data with a user defined wn value
        filtered_stress, qm_stress = signals.butter_lowpass_filter(strain, stress, wn, order, quadrant_mirror)

    
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
        stress = stress.copy() - min_stress
        xlo = min(strain)
        
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
        xhi = fr1_max_fringe
        
        # Step2: First backwards response (br1 - applying minxhi and maxxhi accordingly)
        try:
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
            xlo = br1_max_fringe
        except: pass
        
        # Step3: Second forward response (fr2 - applying minxhi and maxxhi accordingly)
        try:
            min_strain_fr2 = min_strain_fr1 + br1_max_fringe
            max_strain_fr2 = max_strain_fr1 + br1_max_fringe
            reduced_strain = strain[br1_max_index_absolute:-1]
            reduced_stress = stress[br1_max_index_absolute:-1]
            
            fr2_fringe, fr2_slopes = rfr.compute_fringe_slope(reduced_strain, reduced_stress, min_strain=min_strain_fr2, max_strain=max_strain_fr2, direction='forward')
            fr2_max_index = np.argmax(fr2_slopes)
            fr2_max_slope = fr2_slopes[fr2_max_index]
            fr2_max_fringe = fr2_fringe[fr2_max_index]
            fr2_max_index_absolute = np.argmin(np.abs(strain - fr2_max_fringe))
            xhi = fr2_max_fringe
        except: pass

        
        #--------------------------------------------#
        # Compute the yield point and yield strength #
        #--------------------------------------------#
        # Step1: Compute the 2nd derivative from the end of the linear region to the max strain
        try:
            reduced_fringe = fr2_fringe[fr2_max_index:-1]
            reduced_slopes = fr2_slopes[fr2_max_index:-1]
        except:
            reduced_fringe = fr1_fringe[fr1_max_index:-1]
            reduced_slopes = fr1_slopes[fr1_max_index:-1]
        dstrain, dslopes1, dslopes2 = rfr.compute_derivative(reduced_fringe, reduced_slopes)
        
        # Step2: Find peaks and valleys of the 2nd derivative
        dslopes2_diff = np.diff(dslopes2)
        dslopes2_abs  = np.abs(dslopes2_diff)
        prominence    = float( max(dslopes2_abs) )
        xpeaks, ypeaks, xvalleys, yvalleys = rfr.find_peaks_and_valleys(dstrain, dslopes2, prominence=prominence)
        if not np.any(xvalleys):
            prominence = None
            xpeaks, ypeaks, xvalleys, yvalleys = rfr.find_peaks_and_valleys(dstrain, dslopes2, prominence=prominence)
        
        # Step3: Use first valley with respect to strain as the yield point (if any valleys exist)
        yield_index, x_yield, y_yield, x_yield_d2, y_yield_d2 = None, None, None, None, None
        if np.any(xvalleys) and np.any(yvalleys):
            x_yield_d2 = xvalleys[0]
            y_yield_d2 = yvalleys[0]
            yield_index = np.min(np.where(strain == x_yield_d2)[0])
            x_yield = strain[yield_index]
            y_yield = stress[yield_index]
            
            
        xlo_index = np.min(np.where(strain == xlo)[0])
        xhi_index = np.min(np.where(strain == xhi)[0])
        youngs_modulus_coeffs = np.polynomial.polynomial.polyfit(strain[xlo_index:xhi_index+1], stress[xlo_index:xhi_index+1], 1)
        youngs_modulus_x = np.array([xlo, xhi])
        youngs_modulus_y = youngs_modulus_coeffs[1]*youngs_modulus_x + youngs_modulus_coeffs[0]
            
        return youngs_modulus_coeffs[1], y_yield, fr1_fringe, fr1_slopes
    
    
    

        
    
    
    #---------------------------------#
    # Plot the results of this method #
    #---------------------------------#
    # Set xlimits
    delta = 0.01
    xlimits = (np.min(strain)-delta, np.max(strain)+1.5*delta)
    
    # Set fontsize
    max_lw = 10
    fs = 14
    label_rel_pos = (0.005, 0.99)
    
    # Start plotting data
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))
    
    # Start out plotting out
    ax1.plot(strain, stress, '.', ms=4, color='#bbbbbbff', label='LAMMPS data')
    
    
    # Setup barplot data
    bar_data = {} # {'Smoother': {'modulus':value, 'strength':value, 'color':COLOR}}
    
    #-------------#
    # Butterworth #
    #-------------#
    color = 'tab:blue'   
    label = 'Butterworth (n=4, {}$_c$=From-PSD)'.format(r'$\omega$')
    modulus, strength, fr1_fringe_BW, fr1_slopes_BW = RFR(strain, filtered_stress)
    bar_data[label] = {'modulus': modulus,
                       'strength': strength,
                       'color': color}
    
    ax1.plot(strain, filtered_stress, '-', lw=max_lw, color=color, label=label)
    ax2.plot(fr1_fringe_BW, fr1_slopes_BW, '-', lw=max_lw, color=color, label=label)
    
    #--------#
    # LOWESS #
    #--------#
    frac = 0.15
    xout, yout = other_smoothers.lowess(strain, stress, fraction=frac, max_iter=10)
    
    color = 'tab:green'   
    label = 'LOWESS (fraction={})'.format(frac)
    
    modulus, strength, fr1_fringe_BW, fr1_slopes_BW = RFR(xout, yout)
    bar_data[label] = {'modulus': modulus,
                       'strength': strength,
                       'color': color}
    
    ax1.plot(xout, yout, '-', lw=0.8*max_lw, color=color, label=label)
    ax2.plot(fr1_fringe_BW, fr1_slopes_BW, '-', lw=0.8*max_lw, color=color, label=label)


    #------------------#
    # Whittaker-Eilers #
    #------------------#
    order = 2
    lmbda = 100_000_000
    xout, yout = other_smoothers.Whittaker_Eilers(strain, stress, order, lmbda)
    
    color = 'tab:purple'   
    label = 'Whittaker-Eilers (order={}, {}={})'.format(order, r'$\lambda$', lmbda)
    
    modulus, strength, fr1_fringe_BW, fr1_slopes_BW = RFR(xout, yout)
    bar_data[label] = {'modulus': modulus,
                       'strength': strength,
                       'color': color}
    
    ax1.plot(xout, yout, '-', lw=0.6*max_lw, color=color, label=label)
    ax2.plot(fr1_fringe_BW, fr1_slopes_BW, '-', lw=0.6*max_lw, color=color, label=label)
    
    
    #---------------#
    #Savitzky-Golay #
    #---------------#
    order = 1
    window = 200
    xout, yout = other_smoothers.SavitzkyGolay(strain, stress, window, order)
    
    color = 'tab:cyan'   
    label = 'Savitzky-Golay (order={}, window={})'.format(order, window)
    
    modulus, strength, fr1_fringe_BW, fr1_slopes_BW = RFR(xout, yout)
    bar_data[label] = {'modulus': modulus,
                       'strength': strength,
                       'color': color}
    
    ax1.plot(xout, yout, '-', lw=0.4*max_lw, color=color, label=label)
    ax2.plot(fr1_fringe_BW, fr1_slopes_BW, '-', lw=0.4*max_lw, color=color, label=label)
    
    
    #----------------#
    # Moving average #
    #----------------#
    window = 200
    xout, yout = other_smoothers.moving_average(strain, stress, window)
    
    color = 'tab:orange'   
    label = 'Moving average (window={})'.format(window)
    
    modulus, strength, fr1_fringe_BW, fr1_slopes_BW = RFR(xout, yout)
    bar_data[label] = {'modulus': modulus,
                       'strength': strength,
                       'color': color}
    
    ax1.plot(xout, yout, '-', lw=0.2*max_lw, color=color, label=label)
    ax2.plot(fr1_fringe_BW, fr1_slopes_BW, '-', lw=0.2*max_lw, color=color, label=label)

    
    
    
    





    #-------------------------------#
    # Plot labeling for ax1 and ax2 #
    #-------------------------------#
    ax1.legend(loc='lower right', bbox_to_anchor=(1, 0), fancybox=True, ncol=1, fontsize=0.75*fs)
    ax1.set_xlabel('True Strain', fontsize=fs)
    ax1.set_ylabel('True Stress (MPa)', fontsize=fs)
    ax1.tick_params(axis='both', which='major', labelsize=fs)
    ax1.set_xlim(xlimits)
    ax1.text(*label_rel_pos, '(a)', transform=ax1.transAxes, fontsize=fs, fontweight='bold', va='top', ha='left')
    
    
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 1), fancybox=True, ncol=1, fontsize=0.75*fs)
    ax2.set_xlabel(r'True Strain, $\epsilon$', fontsize=fs)
    ax2.set_ylabel('Fringe response (MPa)', fontsize=fs)
    ax2.tick_params(axis='both', which='major', labelsize=fs)
    ax2.set_xlim(xlimits)
    ax2.text(*label_rel_pos, '(b)', transform=ax2.transAxes, fontsize=fs, fontweight='bold', va='top', ha='left')
    
    
    #--------------#
    # Bar plotting #
    #--------------#
    labels, moduli, strengths, colors = [], [], [], []
    for label in bar_data:
        labels.append( label.split('(')[0].strip() )
        for key, value in bar_data[label].items():
            if key == 'modulus': moduli.append(value)
            if key == 'strength': strengths.append(value)
            if key == 'color': colors.append(value)

            

    bars = ax3.bar(labels, moduli, color=colors)
    ax3.set_ylabel("Young's modulus (GPa)", fontsize=fs)
    ax3.set_ylim(0, round(1.25*max(moduli), -1) )
    ax3.tick_params(axis='x', labelsize=fs, colors='black')
    ax3.tick_params(axis='y', labelsize=fs, colors='black')
    ax3.set_xticklabels(labels, rotation=45, ha='right')
    ax3.bar_label(bars, labels=['{:.2f}'.format(i) for i in moduli], padding=-12)
    ax3.axhline(np.mean(np.array(moduli)), linestyle='--', lw=1.5, zorder=0, color='#bbbbbbff', label='Average over all smoothers')
    ax3.text(*label_rel_pos, '(c)', transform=ax3.transAxes, fontsize=fs, fontweight='bold', va='top', ha='left')
    ax3.legend(loc='upper right', bbox_to_anchor=(1, 1), fancybox=True, ncol=1, fontsize=0.75*fs)
    
    
    
    bars = ax4.bar(labels, strengths, color=colors)
    ax4.set_ylabel("Yield Strength (MPa)", fontsize=fs)
    ax4.set_ylim(0, round(1.25*max(strengths), -1) )
    ax4.tick_params(axis='x', labelsize=fs, colors='black')
    ax4.tick_params(axis='y', labelsize=fs, colors='black')
    ax4.set_xticklabels(labels, rotation=45, ha='right')
    ax4.bar_label(bars, labels=['{:.2f}'.format(i) for i in strengths], padding=-12)
    ax4.axhline(np.mean(np.array(strengths)), linestyle='--', lw=1.5, zorder=0, color='#bbbbbbff', label='Average over all smoothers')
    ax4.text(*label_rel_pos, '(d)', transform=ax4.transAxes, fontsize=fs, fontweight='bold', va='top', ha='left')
    ax4.legend(loc='upper right', bbox_to_anchor=(1, 1), fancybox=True, ncol=1, fontsize=0.75*fs)

        

    
    fig.tight_layout()
    plt.show()
    basename = logfile[:logfile.rfind('.')] + '_SMOOTHERS'
    fig.savefig(basename+'.jpeg', dpi=1000)