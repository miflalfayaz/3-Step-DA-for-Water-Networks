# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 18:43:37 2022

@author: ibrah
"""

def Kassimilate(x, K, xobs, M, v):
    '''
    Module to carry out the data assimilation step. Takes the prior system state and updates the system state using the 
    calculated Kalman Gain, measured observations, and taking into account the error/noise in the measurement. 
    
    - Sub Module of 3-step EnKF Data Assimilation for Water Distribition Networks

    Parameters
    ----------
    x : TYPE: Array
        DESCRIPTION. Current State to be assimilated (number of items(npi/nn) x number of ensembles)
    K : TYPE: Array
        DESCRIPTION. Kalman Gain calculated for each ensemble
    xobs : TYPE: Array
        DESCRIPTION. Observation at each location
    M : TYPE: Array
        DESCRIPTION. Observed locations (Nodes/links)
    v : TYPE: Float
        DESCRIPTION. Noise/Error in the Measurement

    Returns
    -------
    xn : TYPE: Array
        DESCRIPTION. : Updated or Assimilated System State

    '''
    import numpy as np
    n, m = np.shape(x)
    xn = np.zeros([n, m])
    xobs = xobs
    for i in range(m):
        tx1 = np.matmul(M, x[:,i])[:,None]
        tx2 = xobs - tx1 - v
        tx3 = np.matmul(K , tx2)
        tx4 = x[:,i][:,None]
        yas = tx4 + tx3
        xn[:, i][:,None] = yas
    return xn
        
                    