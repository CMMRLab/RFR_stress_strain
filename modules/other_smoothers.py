# -*- coding: utf-8 -*-
"""
@author: Josh Kemppainen
Revision 1.0
March 24, 2025
Michigan Technological University
1400 Townsend Dr.
Houghton, MI 49931
"""
##############################
# Import Necessary Libraries #
##############################
import numpy as np
import scipy as sp


#########################################
# Function implementing LOWESS smoother #
#########################################
def lowess(xdata, ydata, fraction=0.2, max_iter=10):    
    import statsmodels.api as sm
    out = sm.nonparametric.lowess(np.array(ydata), np.array(xdata), frac=fraction, it=max_iter, is_sorted=False)   
    xout = out[:, 0]
    yout = out[:, 1]
    return xout, yout

###################################
# Josh's avg/moving avg functions #
###################################
def moving_average(xdata, ydata, window):
    xout =  np.convolve(xdata, np.ones(window)/window, 'valid')
    yout =  np.convolve(ydata, np.ones(window)/window, 'valid')
    return xout, yout



############################
# Whittaker-Eiler smoother #
############################
#---------------------------------------------------------#
# Function to build D = (m-d)*N sparse difference matrix  #
# (drop-in replacement for MATLab's diff() funciton)      #
#---------------------------------------------------------#
def sparse_eye_diff(m, d, format='csc'):
    diagonals = np.zeros(2*d + 1)
    diagonals[d] = 1
    for i in range(d):
        diagonals = diagonals[:-1] - diagonals[1:]
    offsets = np.arange(d+1)
    return sp.sparse.diags(diagonals, offsets, shape=(m-d, m), format=format)


#--------------------------------------------------------------------------------------------------------------#
# Function implementing Whittaker-Eilers smoothing function w/o interopolation. The parameters are as follows: #
#    y = ydata to smooth in numpy array                                                                        #
#    d = integer order of the smoother                                                                         #
#    lmbda = lmbda float smoothing constant                                                                    #
#    compute_cve = True or False Boolean to set whether or not to compute the Cross-validation error (CVE)     #
#    cve_mode = a string of 'numpy' 'scipy' or 'fast' to set how CVE is being computed. The meaning of each:   #
#      'numpy' will compute the full h-matrix by converting the sparse C-matrix to a fully matrix and then     #
#              take the inverse.                                                                               #
#      'scipy' will use the sparse C-matrix to inverse. The 'numpy' mode is quicker for small data sets, but   #
#              uses a lot more memory as fully matrices are used. I would recommend 'scipy' as the default due #
#              to both time and space constraints.                                                             #
#      'fast'  will renormalize the y-data to be multiples of length n (set in the function) to compute the    #
#              partial h-matrix and scale that up to get the full h-matrix. Eilers default is to use fast      #
#              when the length of y-data is larger then 1000.                                                  #
#--------------------------------------------------------------------------------------------------------------#
def Whittaker_Eilers_without_interpolation(y, d, lmbda, compute_cve=False, cve_mode='scipy'):
    # If we want to compute the CVE in 'fast' mode, we need to re-normalize y to be multiples of n
    if compute_cve and cve_mode == 'fast':
        n = 50 # set the number of n-points for "smaller" h-matrix. Eiler's suggests 100.
        mn = np.floor(len(y)/n)
        mn = int(mn*n)
        y = y[:mn]
    
    # Base Whittaker Eilers smoother
    m = len(y)
    E = sp.sparse.eye(m, dtype=int, format='csc')
    D = sparse_eye_diff(m, d, format='csc')
    C = E + lmbda*D.conj().T.dot(D)
    z = sp.sparse.linalg.spsolve(C, y)
    
    # Computation of hat diagonal and cross-validation.
    if compute_cve:
        if cve_mode == 'numpy':
            C = C.todense().T
            H = np.linalg.inv(C)
            h = np.diagonal(H)
        if cve_mode == 'scipy':
            H = sp.sparse.linalg.inv(C)
            h = sp.sparse.csr_matrix.diagonal(H)
        if cve_mode == 'fast':
            E1 = sp.sparse.eye(n, dtype='int32', format='csc')
            D1 = sparse_eye_diff(n, d, format='csc')
            lambda1 = lmbda*(n/m)**(2*d)
            H1 = sp.sparse.linalg.inv(E1 + lambda1*D1.conj().T.dot(D1))
            h1 = sp.sparse.csr_matrix.diagonal(H1)
            u = np.zeros(m)
            k = int(np.floor(m/2) - 1)
            k1 = int(np.floor(n/2) - 1)
            u[k] = 1
            v = sp.sparse.linalg.spsolve(C, u)
            f = np.round( (np.arange(0,m) - 1)*(n - 1)/(m - 1)).astype(int)
            h = h1[f]*v[k]/h1[k1]
        r = (y - z)/(1 - h)
        s = np.sum(r**2)
        cve = np.sqrt(s/m)
    else: cve = None
    return z, cve

#---------------------------------------#
# Smaller standalone WE calling funtion #
#---------------------------------------#
def Whittaker_Eilers(xdata, ydata, order, lmbda):
    xout = xdata.copy()
    yout, cve = Whittaker_Eilers_without_interpolation(ydata, order, lmbda, compute_cve=False, cve_mode='fast')
    return xout, yout


#########################
# Savitzky-Golay filter #
#########################
def SavitzkyGolay(xdata, ydata, window, order):
    from scipy.signal import savgol_filter
    xout = xdata.copy()
    yout = savgol_filter(ydata, window, order)
    return xout, yout
