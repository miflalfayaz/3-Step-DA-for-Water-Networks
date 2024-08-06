# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 12:04:19 2022

@author: ibrah
"""

def Kerror(x):
    '''
    Calculates the Variance and Covariance Matrices - Sub Module of 3-step EnKF Data Assimilation for Water Distribition Networks

    Parameters
    ----------
    x : TYPE: Array 
        DESCRIPTION. Array of System States for all the ensembles to determine the ensemble mean and covariance

    Returns
    -------
    P : TYPE" Array
        DESCRIPTION.: Variance of each ensemble
    xe : TYPE: Array
        DESCRIPTION. Covariance of each ensemble

    '''
    import numpy as np
    n, m = np.shape(x)
    
    xe  = np.zeros([n, m])
    P   = np.zeros([n, n])
    
    xmiu = (np.mean(x,axis=1))
    for i in range(m):
        xe[:,i] = x[:,i] - xmiu
        
        
    P = 1 / (m-1) * np.matmul(xe, xe.T)
    
    return P, xe