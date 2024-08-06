# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 15:19:10 2022

@author: ibrah
"""


def  Total_Variance(var, yobs, S, ne, sim_time):
    '''
    Calculating the Total Variance for each System State - Sub Module of 3-step EnKF Data Assimilation for Water Distribition Networks
    
    Parameters
    ----------
    var : TYPE: Array
        DESCRIPTION. System State for which Total Variance is calculated
    yobs : TYPE: Array
        DESCRIPTION. Observations of that System State
    S : TYPE: Integer
        DESCRIPTION. Number of nodes or links depending on the system state
    ne : TYPE: Integer
        DESCRIPTION. Number of ensembles
    sim_time : TYPE: Integer
        DESCRIPTION. Simulation Time

    Returns
    -------
    TVA : TYPE: Array
        DESCRIPTION. Total Variance Array for all the timesteps

    '''
    import numpy as np
    # Defining array to save total variance for all timesteps
    TVA = np.zeros([sim_time, 1])
    
    # Loop to calculate Total Variance for all timesteps
    for ti in range(sim_time):
        # Variance of Hypothetical Means (VHM)        
        VHM = (1/S) * np.sum(((np.mean(var[:,:,ti], axis = 1))[:,None] - yobs[:,:,ti])**2)
        
        # Expected Value of Process Variance (EPV)
        EPV = (1/S) * np.sum(1/(ne*(ne-1)) * np.sum((var[:,:,ti] - np.mean(var[:,:,ti], axis = 1)[:,None])**2, axis = 1))
        
        # Total Variance(VHM + EPV)
        TV  = VHM + EPV
        TVA[ti, 0] = TV
    return TVA


def Total_Variance_Ca(var, yobs, S, ne, sim_time):
    '''
    Calculating the Total Variance for each System State - Sub Module of 3-step EnKF Data Assimilation for Water Distribition Networks
    
    Parameters
    ----------
    var : TYPE: Array
        DESCRIPTION. System State for which Total Variance is calculated
    yobs : TYPE: Array
        DESCRIPTION. Observations of that System State
    S : TYPE: Integer
        DESCRIPTION. Number of nodes or links depending on the system state
    ne : TYPE: Integer
        DESCRIPTION. Number of ensembles
    sim_time : TYPE: Integer
        DESCRIPTION. Simulation Time

    Returns
    -------
    TVA : TYPE: Array
        DESCRIPTION. Total Variance Array for all the timesteps

    '''
    import numpy as np
    # Defining array to save total variance for all timesteps
    TVA = np.zeros([S, sim_time])
    
    # Loop to calculate Total Variance for all timesteps
    for ti in range(sim_time):
        # Variance of Hypothetical Means (VHM)        
        VHM = (((np.mean(var[:,:,ti], axis = 1))[:,None] - yobs[:,:,ti])**2).flatten()
        
        # Expected Value of Process Variance (EPV)
        EPV = 1/(ne*(ne-1)) * np.sum((var[:,:,ti] - np.mean(var[:,:,ti], axis = 1)[:,None])**2, axis = 1)
        
        # Total Variance(VHM + EPV)
        TV  = VHM + EPV
        TVA[:, ti] = TV
    return TVA
