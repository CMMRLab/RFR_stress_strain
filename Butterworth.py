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
import matplotlib.pyplot as plt
import matplotlib as mpl  
mpl.rc('font',family='Calibri')
import numpy as np
import scipy as sp






def butterworth_amp_response(order=2, wn=0.1):
    b, a = sp.signal.butter(order, wn, btype='low', analog=False, output='ba', fs=None)
    f, h = sp.signal.freqz(b, a, worN=4096)
    f = f / np.pi
    h = abs(h)
    return f, h

def get_ylims(h):
    tol = 0.1
    y = np.real(h)
    diff = abs(max(y) - min(y))
    lo = min(y) - tol*diff
    hi = max(y) + tol*diff
    ylo, yhi = sorted([lo, hi])
    return ylo, yhi

# Set fontsize
fs = 16
legend_fs_scale = 0.7
label_rel_pos = (0.005, 0.99)

ylo = -0.1
yhi =  1.1
mag_3db = 1/np.sqrt(2)      # ~0.707


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12.5, 9))


wn = 0.2
f, h = butterworth_amp_response(order=4, wn=wn)
ax1.plot(*butterworth_amp_response(order=2, wn=wn), lw=2, color='tab:blue', label='Butterworth (n=2)')
ax1.plot(f, h, lw=2, color='tab:purple', label='Butterworth (n=4)')
ax1.plot(*butterworth_amp_response(order=6, wn=wn), lw=2, color='tab:green', label='Butterworth (n=6)')
ax1.plot(*butterworth_amp_response(order=8, wn=wn), lw=2, color='tab:cyan', label='Butterworth (n=8)')
ax1.axvline(wn, linestyle='--', lw=1.5, color='#ff9d3aff', label='Cutoff Frequency ({}$_c$)'.format(r'$\omega$'))
ax1.axhline(mag_3db, linestyle='--', lw=1.5, color='#bbbbbbff', label='-3 dB level (~0.707)')

# Shade passband and stopband regions
ax1.fill_between(f, ylo, yhi, where=(f <= wn), alpha=0.1, color='tab:blue', label='Passband')
ax1.fill_between(f, ylo, yhi, where=(f >= wn), alpha=0.1, color='tab:orange', label='Stopband')

ax1.set_xlim((0, 1))
ax1.set_ylim((ylo, yhi))
ax1.set_xlabel('Normalized Frequency, {}'.format(r'$\omega$'), fontsize=fs)
ax1.set_ylabel('|H(jω)|', fontsize=fs)
ax1.tick_params(axis='both', which='major', labelsize=fs)
ax1.legend(loc='upper right', bbox_to_anchor=(1, 1), fancybox=True, ncol=1, fontsize=legend_fs_scale*fs)
ax1.text(*label_rel_pos, '(a)', transform=ax1.transAxes, fontsize=fs, fontweight='bold', va='top', ha='left')



wn = 0.4
f, h = butterworth_amp_response(order=4, wn=wn)
ax2.plot(*butterworth_amp_response(order=2, wn=wn), lw=2, color='tab:blue', label='Butterworth (n=2)')
ax2.plot(f, h, lw=2, color='tab:purple', label='Butterworth (n=4)')
ax2.plot(*butterworth_amp_response(order=6, wn=wn), lw=2, color='tab:green', label='Butterworth (n=6)')
ax2.plot(*butterworth_amp_response(order=8, wn=wn), lw=2, color='tab:cyan', label='Butterworth (n=8)')
ax2.axvline(wn, linestyle='--', lw=1.5, color='#ff9d3aff', label='Cutoff Frequency ({}$_c$)'.format(r'$\omega$'))
ax2.axhline(mag_3db, linestyle='--', lw=1.5, color='#bbbbbbff', label='-3 dB level (~0.707)')

# Shade passband and stopband regions
ax2.fill_between(f, ylo, yhi, where=(f <= wn), alpha=0.1, color='tab:blue', label='Passband')
ax2.fill_between(f, ylo, yhi, where=(f >= wn), alpha=0.1, color='tab:orange', label='Stopband')

ax2.set_xlim((0, 1))
ax2.set_ylim((ylo, yhi))
ax2.set_xlabel('Normalized Frequency, {}'.format(r'$\omega$'), fontsize=fs)
ax2.set_ylabel('|H(jω)|', fontsize=fs)
ax2.tick_params(axis='both', which='major', labelsize=fs)
ax2.legend(loc='upper right', bbox_to_anchor=(1, 1), fancybox=True, ncol=1, fontsize=legend_fs_scale*fs)
ax2.text(*label_rel_pos, '(b)', transform=ax2.transAxes, fontsize=fs, fontweight='bold', va='top', ha='left')



wn = 0.6
f, h = butterworth_amp_response(order=4, wn=wn)
ax3.plot(*butterworth_amp_response(order=2, wn=wn), lw=2, color='tab:blue', label='Butterworth (n=2)')
ax3.plot(f, h, lw=2, color='tab:purple', label='Butterworth (n=4)')
ax3.plot(*butterworth_amp_response(order=6, wn=wn), lw=2, color='tab:green', label='Butterworth (n=6)')
ax3.plot(*butterworth_amp_response(order=8, wn=wn), lw=2, color='tab:cyan', label='Butterworth (n=8)')
ax3.axvline(wn, linestyle='--', lw=1.5, color='#ff9d3aff', label='Cutoff Frequency ({}$_c$)'.format(r'$\omega$'))
ax3.axhline(mag_3db, linestyle='--', lw=1.5, color='#bbbbbbff', label='-3 dB level (~0.707)')

# Shade passband and stopband regions
ax3.fill_between(f, ylo, yhi, where=(f <= wn), alpha=0.1, color='tab:blue', label='Passband')
ax3.fill_between(f, ylo, yhi, where=(f >= wn), alpha=0.1, color='tab:orange', label='Stopband')

ax3.set_xlim((0, 1))
ax3.set_ylim((ylo, yhi))
ax3.set_xlabel('Normalized Frequency, {}'.format(r'$\omega$'), fontsize=fs)
ax3.set_ylabel('|H(jω)|', fontsize=fs)
ax3.tick_params(axis='both', which='major', labelsize=fs)
ax3.legend(loc='lower left', bbox_to_anchor=(0, 0), fancybox=True, ncol=1, fontsize=legend_fs_scale*fs)
ax3.text(*label_rel_pos, '(c)', transform=ax3.transAxes, fontsize=fs, fontweight='bold', va='top', ha='left')


wn = 0.8
f, h = butterworth_amp_response(order=4, wn=wn)
ax4.plot(*butterworth_amp_response(order=2, wn=wn), lw=2, color='tab:blue', label='Butterworth (n=2)')
ax4.plot(f, h, lw=2, color='tab:purple', label='Butterworth (n=4)')
ax4.plot(*butterworth_amp_response(order=6, wn=wn), lw=2, color='tab:green', label='Butterworth (n=6)')
ax4.plot(*butterworth_amp_response(order=8, wn=wn), lw=2, color='tab:cyan', label='Butterworth (n=8)')
ax4.axvline(wn, linestyle='--', lw=1.5, color='#ff9d3aff', label='Cutoff Frequency ({}$_c$)'.format(r'$\omega$'))
ax4.axhline(mag_3db, linestyle='--', lw=1.5, color='#bbbbbbff', label='-3 dB level (~0.707)')

# Shade passband and stopband regions
ax4.fill_between(f, ylo, yhi, where=(f <= wn), alpha=0.1, color='tab:blue', label='Passband')
ax4.fill_between(f, ylo, yhi, where=(f >= wn), alpha=0.1, color='tab:orange', label='Stopband')

ax4.set_xlim((0, 1))
ax4.set_ylim((ylo, yhi))
ax4.set_xlabel('Normalized Frequency, {}'.format(r'$\omega$'), fontsize=fs)
ax4.set_ylabel('|H(jω)|', fontsize=fs)
ax4.tick_params(axis='both', which='major', labelsize=fs)
ax4.legend(loc='lower left', bbox_to_anchor=(0, 0), fancybox=True, ncol=1, fontsize=legend_fs_scale*fs)
ax4.text(*label_rel_pos, '(d)', transform=ax4.transAxes, fontsize=fs, fontweight='bold', va='top', ha='left')

fig.tight_layout()
plt.show()

basename = 'Butterworth'
fig.savefig(basename+'.jpeg', dpi=300)