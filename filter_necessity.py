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
import scipy as sp

# Material properties
modulus  = 4000  # Young's modulus MPa
y_strain = 0.025 # yield strain
y_stress = modulus*y_strain  

# Wave properties (simulates thermal noise)
amp = 10           # MPa (amplitude of oscillation)
wavelength = 0.003  # strain units


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
quadrant_mirror = '1,1'



# Generate strain
min_strain, max_strain, dx = 0.0, 0.05, 0.0001
strain = np.arange(min_strain, max_strain + dx, dx)

# Generate stress - Piecewise (continuous)
stress = np.where(strain <= y_strain, modulus*strain, y_stress)

# Generate wave
wave = amp*np.sin(2*np.pi*(strain)/wavelength)
thermal = stress + wave

# Compute the PSD if wn is set to 'op'
if str(wn).startswith('op'):
    # Compute the PSD for different data sets
    wns_stress, psd_stress = signals.compute_PSD(strain, thermal)
    
    # Find the mean value of the different PSD's
    mean_stress_psd = np.mean(psd_stress)

    # Find where each PSD crosses the mean
    wn_index = np.min(np.where(psd_stress < mean_stress_psd)[0])
    wn_stress = wns_stress[wn_index]/1.2
    print('{:<50} {} {}'.format('Computed normalized cutoff frequency for stress: ', wn_stress, wn_index))


    # Filter the data with an optimized wn value
    filtered, qm_stress = signals.butter_lowpass_filter(strain, thermal, wn_stress, order, quadrant_mirror)
    
else:
    # Filter the data with a user defined wn value
    filtered, qm_stress = signals.butter_lowpass_filter(strain, thermal, wn, order, quadrant_mirror)
    
    
#--------------------------------------------------#
# RFR function to use for different stress signals #
#--------------------------------------------------#
def maximized_slope(fringe, slopes, machine_precision=1e-8):
    # Find xhi and yhi, dealing with the possibility of having multiple solutions for maximized yhi. If multiply
    # solutions exists, assumed it is a machine precision issue and maximize xhi and yhi to obtain the solution.
    fringe = list(fringe)
    slopes = list(slopes)
    if slopes:
        maximized_slope = max(slopes)
        lo = maximized_slope - machine_precision
        hi = maximized_slope + machine_precision
        indexes = [i for i, y in enumerate(slopes) if lo <= y <= hi]
        maximized = max(indexes)
        xhi = fringe[maximized]
        yhi = slopes[maximized]
    else: xhi = 0; yhi = 0
    return xhi, yhi

def RFR(strain, stress):
    # Shift all data by the "minimum before the maximum" to remove any residual 
    # stress. The "minimum before the maximum" allows for fracture to occur where
    # the minimum stress is near zero strain and not maximum strain (in case fracture
    # creates a minimum stress lower then the residual stress).
    max_index = np.min(np.where(stress == np.max(stress))[0])
    min_stress = np.min(stress[:max_index])
    xlo = min(strain)
    
    #--------------------------------------------------------#
    # Compute the forward-backwards-forwards fringe response #
    #--------------------------------------------------------#
    machine_precision = 1e-4
    # Step1: First forward response (fr1 - applying minxhi and maxxhi accordingly)
    min_strain_fr1 = min(strain); max_strain_fr1 = max(strain)
    if minxhi > 0: min_strain_fr1 = minxhi 
    if maxxhi > 0: max_strain_fr1 = maxxhi 
    fr1_fringe, fr1_slopes = rfr.compute_fringe_slope(strain, stress, min_strain=min_strain_fr1, max_strain=max_strain_fr1, direction='forward')
    fr1_max_index = np.argmax(fr1_slopes)
    fr1_max_slope = fr1_slopes[fr1_max_index]
    fr1_max_fringe = fr1_fringe[fr1_max_index]
    xhi = fr1_max_fringe
    xhi, yhi = maximized_slope(fr1_fringe, fr1_slopes, machine_precision=machine_precision)
    
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
        xlo, ylo = maximized_slope(br1_fringe, br1_slopes, machine_precision=machine_precision)
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
        xhi, yhi = maximized_slope(fr2_fringe, fr2_slopes, machine_precision=machine_precision)
    except: pass

    xlo_index = np.min(np.where(strain == xlo)[0])
    xhi_index = np.min(np.where(strain == xhi)[0])
    youngs_modulus_coeffs = np.polynomial.polynomial.polyfit(strain[xlo_index:xhi_index+1], stress[xlo_index:xhi_index+1], 1)
    youngs_modulus_x = np.array([strain[xlo_index], strain[xhi_index]])
    youngs_modulus_y = youngs_modulus_coeffs[1]*youngs_modulus_x + youngs_modulus_coeffs[0]
    youngs_modulus = youngs_modulus_coeffs[1]

    
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
        
    return youngs_modulus_x, youngs_modulus_y, youngs_modulus, [x_yield, y_yield]



#---------------------#
# Start plotting data #
#---------------------#
# Set limits
xdelta, ydelta = 0.001, 20
min_strain, max_strain = min(strain), max(strain)
min_stress, max_stress = min(thermal), max(thermal)
xlimits = (min_strain - xdelta, max_strain + xdelta)
ylimits = (min_stress - ydelta, max_stress + ydelta)


# Set fontsize
plt.close('all')
fs = 12
legend_fs_scale = 1.0
label_rel_pos = (0.005, 0.98)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12.5, 9))

color_epp     = '#41b6c4ff' #'#bbbbbbff'
color_wave    = '#a1dab4ff'

color_thermal = '#bbbbbbff'
color_filter  = '#2c7fb8ff'

color_slope   = '#ff9d3aff'


# Simualted data
ax1.plot(strain, stress,       '--', lw=3, zorder=2, color=color_epp,  label='Elastic–perfectly plastic - (EPP)')
ax1.plot(strain, wave,         '--', lw=3, zorder=2, color=color_wave, label='Wave = {}*sin(2*π*strain/{})'.format(amp, wavelength))

ax1.plot(strain, thermal,      '-',  lw=4, zorder=1, color=color_thermal, label='Thermal = EPP + Wave')
ax1.plot(strain, filtered,     '-',  lw=4, zorder=1, color=color_filter,  label='Butterworth(Thermal)')

ax1.legend(loc='upper right', bbox_to_anchor=(1, 0.6), fancybox=True, ncol=1, fontsize=legend_fs_scale*fs)
ax1.set_xlabel(r'Strain, $\epsilon$', fontsize=fs)
ax1.set_ylabel(r'Stress, $\sigma$ (MPa)', fontsize=fs)
ax1.tick_params(axis='both', which='major', labelsize=fs)
ax1.set_xlim(xlimits)
ax1.set_ylim(ylimits)
ax1.text(*label_rel_pos, '(a)', transform=ax1.transAxes, fontsize=fs, fontweight='bold', va='top', ha='left')

# Running RFR on EPP
youngs_modulus_x, youngs_modulus_y, youngs_modulus, yp = RFR(strain, stress)
ax2.plot(strain, stress,                     '-', lw=4, zorder=2, color=color_epp,                label='Elastic–perfectly plastic - (EPP)')
ax2.plot(youngs_modulus_x, youngs_modulus_y, '-', lw=4, zorder=2, color=color_slope, label="RFR - Young's modulus = {:,.4f}".format(youngs_modulus))

ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.2), fancybox=True, ncol=1, fontsize=legend_fs_scale*fs)
ax2.set_xlabel(r'Strain, $\epsilon$', fontsize=fs)
ax2.set_ylabel(r'Stress, $\sigma$ (MPa)', fontsize=fs)
ax2.tick_params(axis='both', which='major', labelsize=fs)
ax2.set_xlim(xlimits)
ax2.set_ylim(ylimits)
ax2.text(*label_rel_pos, '(b)', transform=ax2.transAxes, fontsize=fs, fontweight='bold', va='top', ha='left')


# Running RFR on Thermal
youngs_modulus_x, youngs_modulus_y, youngs_modulus, yp = RFR(strain, thermal)
ax3.plot(strain, stress,       '--', lw=3, zorder=2, color=color_epp,     label='Elastic–perfectly plastic - (EPP)')
ax3.plot(strain, thermal,      '-',  lw=4, zorder=1, color=color_thermal, label='Thermal = EPP + Wave')

ax3.plot(youngs_modulus_x, youngs_modulus_y, '-', lw=4, zorder=2, color=color_slope, label="RFR - Young's modulus = {:,.4f}".format(youngs_modulus))

ax3.legend(loc='upper right', bbox_to_anchor=(1, 0.25), fancybox=True, ncol=1, fontsize=legend_fs_scale*fs)
ax3.set_xlabel(r'Strain, $\epsilon$', fontsize=fs)
ax3.set_ylabel(r'Stress, $\sigma$ (MPa)', fontsize=fs)
ax3.tick_params(axis='both', which='major', labelsize=fs)
ax3.set_xlim(xlimits)
ax3.set_ylim(ylimits)
ax3.text(*label_rel_pos, '(c)', transform=ax3.transAxes, fontsize=fs, fontweight='bold', va='top', ha='left')


# Running RFR on filtered
youngs_modulus_x, youngs_modulus_y, youngs_modulus, yp = RFR(strain, filtered)
ax4.plot(strain, stress,       '--', lw=3, zorder=2, color=color_epp,    label='Elastic–perfectly plastic - (EPP)')
ax4.plot(strain, filtered,     '-',  lw=4, zorder=1, color=color_filter, label='Butterworth(Thermal)')

ax4.plot(youngs_modulus_x, youngs_modulus_y, '-', lw=4, zorder=2, color=color_slope, label="RFR - Young's modulus = {:,.4f}".format(youngs_modulus))

ax4.legend(loc='upper right', bbox_to_anchor=(1, 0.25), fancybox=True, ncol=1, fontsize=legend_fs_scale*fs)
ax4.set_xlabel(r'Strain, $\epsilon$', fontsize=fs)
ax4.set_ylabel(r'Stress, $\sigma$ (MPa)', fontsize=fs)
ax4.tick_params(axis='both', which='major', labelsize=fs)
ax4.set_xlim(xlimits)
ax4.set_ylim(ylimits)
ax4.text(*label_rel_pos, '(d)', transform=ax4.transAxes, fontsize=fs, fontweight='bold', va='top', ha='left')


fig.tight_layout()
basename = 'filter_necessity'
fig.savefig(basename+'.jpeg', dpi=300)