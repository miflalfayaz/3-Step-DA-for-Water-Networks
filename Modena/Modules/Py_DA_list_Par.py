# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 11:20:34 2022

@author: ibrah
"""
def HyMod(demand, tval_new, wdsfile):
    '''
    Module for running the Hydrauilic Model for the 3-Step EnKF Data Assimilation for Water Distribution Networks

    Parameters
    ----------
    demand : TYPE: Array 
        DESCRIPTION. Demand value for each node
    tval_new : TYPE: Integer
        DESCRIPTION. Current Time Step being Simulated in Hour
    wdsfile : TYPE: String
        DESCRIPTION. Directory of the EPANET .inp file for the waternetwork model.

    Returns
    -------
    curr_states : TYPE: List
        DESCRIPTION. Final Outputs of the Hydraulic Model

    '''
                                
    import wntr
    import csv
    import math
    import numpy as np
    import pandas as pd
    import os
    import shutil
     
    # Network model
    wdsfile          = wdsfile
    curr_states_names = {'Time [s]', 'Pressure [m]', 'Demand [l/s]','PDA State [0/1/2]', 'Qp [l/s]','Sp link status [1/0]','Level [m]','Flow [l/s]'};# 'Hp [m]', 'Energy [kWh]',}; #, 'Efficiency [%]'};
    curr_states      = [None]*8; # Store internal states of the WDN (timeseries, P, E, Eff, Qpumps, Hpumps)
    objectives_names = {'Total Energy [kWh]','Total Cost [eur]','Penalty Tanks [m]','Penalty Switches []'};
    
    # Start WNTR
    wn = wntr.network.WaterNetworkModel(wdsfile) 
      
    # Setting the model as a PDA
    wn.options.hydraulic.demand_model = 'PDD'
    reqP = 20.00    # required_pressure
    minP = 0.00     # minimum_pressure
    Pexp = 0.50     # pressure_exponent
    wn.options.hydraulic.required_pressure = reqP 
    wn.options.hydraulic.minimum_pressure  = minP 
    wn.options.hydraulic.pressure_exponent = Pexp
    ENgetcount = wn.describe(1)
    
    nnodes    = ENgetcount['Nodes']['Junctions']    # number of nodes
    nstorages = ENgetcount['Nodes']['Reservoirs']   # number of tanks and reservoirs
    nlinks    = ENgetcount['Links']['Pipes']        # number of links
    npatterns = ENgetcount['Patterns']              # number of patterns
    # ncurves   = ENgetcount(4);                    # number of curves
    # numctrls  = ENgetcount(5);                    # number of controls
    
    ndemand = nnodes - nstorages;
    
    # Set NEW base demands
    a = []                                          #list to see updated demand values
    b = []

    i = 0   
    for junction_name, node in wn.junctions():
        
        node.demand_timeseries_list[0].base_value = demand.iloc[i,0] * 0.001 # demands should be converted to cubic meters per second
        #node.demand_timeseries_list[0].base_value = 0.0005
        i += 1
        a.append(node.demand_timeseries_list[0].base_value)
        b.append(junction_name)
    
    # Saving an updated .inp file    
    # wn.write_inpfile('tmp.inp')
    # wntr.epanet.io.InpFile.write(self, filename="tmp.inp", wn=wn, units=None)
    
    it         =  0             # Current index of the variable updated only if Pressure is larger than 0.
    timeseries = []
    tstep      =  1             #[];
    E          = []             # Energy array
    P          = []             # Pressure array
    D          = []             # Demand array 
    S          = []             # PDA State array
    L          = [] 
    F          = [] 
    Qp         = []             # Discharge Pump array
    Sp         = []             # LINK status 0: closed, 1: Open
    # Hp         = [];          # Head loss Pump array
    # Ep         = [];          # Efficiency Pump array
    # FlagPmin   =  0;          # Flag for negative pressure, 0: Success, 1: Failed
    # Flagpeaktime = 0;         # Flag for peak time to update controls
       
    #if (mod(t,900) == 0)
    # Run only 1 timestep
    t = tval_new
    wn.options.time.duration = t
    
    def RunNet (wdsfile, OutPath): 
        ''' This function receives as a variable an instance of a network created with the Net() function. 
        This returns an object with all the results of the model. The queries have to be related to the instance created with this function. 
        How to do the querries, visit: https://wntr.readthedocs.io/en/latest/waternetworkmodel.html Example: Results = RunNet(MyInstance)''' 
        sim = wntr.sim.EpanetSimulator(wdsfile) 
        if OutPath=='': 
            pre=OutPath+'temp' 
                
        else: 
            pre = OutPath+'/temp' 
            sim = sim.run_sim(version= 2.2, file_prefix=pre) 
            return sim 
    
    # Folder for the Temporary Files
    DirOutput = 'Temp/'
    # Check if the directory exists
    if not os.path.exists(DirOutput):
    # If the directory doesn't exist, create it
        os.mkdir(DirOutput)
        
    # Set the directory path and name for the Temporary Files
    dir_path = DirOutput + 'tempor{}'.format(tval_new)

    # Check if the directory exists
    if not os.path.exists(dir_path):
    # If the directory doesn't exist, create it
        os.mkdir(dir_path)
        
    path_temp = DirOutput+'tempor{}'.format(tval_new) 
    
    results = RunNet(wn, path_temp)    
    #sim = wntr.sim.WNTRSimulator(wn)
    #results = sim.run_sim()
    pressure = results.node['head']
    demands = results.node['demand']
    flowrate = results.link['flowrate']
    
    Stime = np.zeros([nnodes,1]) 
    
    # retrieve the data of Pressure [11] and demand [9]. 
    # PDA state is a function from Epanet 2.2        
    Ptime = pressure.loc[[tval_new]].to_numpy()
    Ptime = Ptime.T
    Dtime = demands.loc[[tval_new]].to_numpy() * 1000 # converted to LPS
    Dtime = Dtime.T
    for i in range(len(Ptime)-nstorages):
        if Ptime[i] < minP:
            Stime[i, 0] = 0
        if Ptime[i] >= minP and Ptime[i]< reqP:
            Stime[i, 0] = 1
        if Ptime[i] >= reqP:
            Stime[i, 0] = 2
    
   # Getting tank Level and Flow
    Ltime = []      
    Ftime = []
    
    for tanks in wn.reservoirs():
        level = tanks[1].base_head   # this gets the head of the tank, for actual level its important to subtract the elevation of the tank
        flow = tanks[1].demand
        Ltime.append(level)
        Ftime.append(flow)
    
    
    Qptime = flowrate.loc[[tval_new]].to_numpy()*1000      # convert to LPS
    Sptime = np.ones(nlinks)
    
    it = it+1;
    timeseries= np.array([it-1,1]) * t # allocation is not possible because the time steps are dependent on the control
        
    P = Ptime[:-nstorages]
    D = Dtime[:-nstorages]
    S = Stime
    
    L = Ltime
    #F = np.array(Ftime) * 1000 # Convert to LPS
    
    Qp = Qptime
    Sp = Sptime
        
    # Store the variables of relevance
    curr_states[0] = timeseries             # Time
    curr_states[1] = P                      # Pressure
    curr_states[2] = D                      # Demand
    curr_states[3] = S                      # PDA State - T0 check if all nodes get sufficient pressure
    curr_states[4] = Qp                     # Flow in LPS
    curr_states[5] = Sp                     # Link Status     
    curr_states[6] = L                      # Tank Heads
    curr_states[7] = F                      # Flow from Tanks
    
    
    shutil.rmtree(dir_path)
        
    return curr_states