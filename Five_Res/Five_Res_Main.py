# -*- coding: utf-8 -*-

"""
# This script is used for running Data Assimilation in 
# Water Distribution Networks using a 3-step EnKF and WNTR.
# for Modena WDN
# Msc. Mario Castro Gama - PhD researcher UN-IHE
# Ibrahim Miflal Fayaz - Msc Student UN-IHE
# 02/12/2022
"""
import time
import numpy as np
import pandas as pd
from Modules import Py_DA_WD

# Input file directories 

wdsfile         = 'Input_Files/FivRes.inp'      # WDS Model Directory
bdemfile        = "Input_Files/FIVE_basedemands.txt"                    # Base Demands
topofile        = "Input_Files/FIVE_topology.csv"                       # Topology file
assetfile       = "Input_Files/FIVE_assets.csv"
lnkidfile       = "Input_Files/FIVE_link_ids.txt"
nodidfile       = "Input_Files/FIVE_node_ids.txt"

# Define variables 

basedemandsrd   = pd.read_csv( bdemfile,   header = None)
basedemands     = basedemandsrd.T
Aij             = pd.read_csv( topofile,   header = None)
assets          = pd.read_csv( assetfile,  header = None)
curves          = []
#link_ids        = pd.read_csv(lnkidfile ,     header = None)
#node_ids        = pd.read_csv(nodidfile,      header = None)

npi             = np.size(assets,0)                 #  number of links
nn              = len(basedemands.columns)          #  number of junctions with demand
no              = 5                                 #  number of sources

sim_time        = 24                                #  Simualtion time (hours)
ne              = 5                                 #  Ensemble size, number of simulations per tstep
Dmult           = 1.4                               #  General multiplier to all demands


# Select the Head Observations
sH              = 6                                 # number of pressure sensors
H_observ        = np.array([570,571,691,700,799,804]) # select the observation nodes HEAD
#H_observ = np.random.randint(0, nn, sH)            # random selection of the observation nodes for Head
#H_observ        = np.arange(0,935)
vzH_v           = 0.0                               # mean of the HEAD measurement errors
RzH_v           = 0.2                               # variance of HEAD measurement errors

# select the observation flows
sQ              = 5                                 # number of pressure sensors
#Q_observ = [1:sQ]                                  # First sQ pipes
Q_observ        = np.arange(npi-sQ+1,npi+1)         # Pipes linking the reservoirs
#Q_observ        = np.arange(0, 1278)
#Q_observ = np.random.randint(0, npi, sQ)           # random selection the observation links for flows
vzQ_v           = 0.0                               # mean of the FLOWS measurement errors
RzQ_v           = 2.0                               # variance of FLOWS measurement errors

# Select the Demand Observations
sq              = 25                                # number of Demand sensors
q_observ        = np.array([  28, 161, 164, 183, 206, 239, 265, 278, 279, 302,
            305, 334, 335, 341, 350, 416, 427, 457, 474, 535,
            545, 548, 608, 619, 622])                # select the observation nodes for Demand
#q_observ = np.random.randint(0, nn, sq)            # Random Selection of observation nodes for Demand
#q_observ        = np.arange(0,935)
vzq_v           = 0.0                               # mean of the Demand measurement errors
Rzq_v           = 0.2                               # variance of Demand measurement errors

Res_fname = "RESULTS/DA_{}_T{}.data".format(wdsfile[-11:-5], ne)


#%% Running the 3-Step DA (SR1)

Sim_Res = Py_DA_WD.Three_Step_EnKF(wdsfile, bdemfile, topofile, assetfile, lnkidfile, nodidfile, 
                    npi, nn, no, sim_time, ne, Dmult, Res_fname, 
                    H_observ, vzH_v, RzH_v,
                    Q_observ, vzQ_v, RzQ_v,
                    q_observ, vzq_v, Rzq_v)

#%% Running experiment with varying number of Ensembles (SR2)

# Time measurement and counter for number of runs
t = time.time()
i= 0

# Iterating through different number of ensembles
for ne in [5, 10, 15, 20, 25, 30, 40, 50, 100]:
    print('No of Ensembles set to :', ne)
    i += 1
    print('Run:', i)
    
    # Result filenemes for different simulations
    Res_fname = "RESULTS/DA_{}_E{}_T{}.data".format(wdsfile[-11:-3], ne, sim_time)           
    
    # Running the DA
    Run = Py_DA_WD.Three_Step_EnKF(wdsfile, bdemfile, topofile, assetfile, lnkidfile, nodidfile, 
                      npi, nn, no, sim_time, ne, Dmult, Res_fname, 
                      H_observ, vzH_v, RzH_v,
                      Q_observ, vzQ_v, RzQ_v,
                      q_observ, vzq_v, Rzq_v)

# Printing total computation time
total_sim_time = time.time() - t
print('Total Computation Time: {:2f}'.format(total_sim_time))


#%% Running experiment with varying Precisions (SR3)
from pathlib import Path
# List to check finished files
files = []

# Time measurement and counter for number of runs
t = time.time()
i = 0

# Looping through all possible values of Precisions
for RzH_v in [0.001, 0.1, 0.2, 0.3, 0.5, 1]:
    for RzQ_v in [0.001, 1, 2, 3, 5, 10]:
        for Rzq_v in [0.001, 0.1, 0.2, 0.3, 0.5, 1]:
            
            # Checking saved files to not re-run simulations incase of a previous interuption
            
            my_file = Path("RESULTS/DA_{}_RH{}_RQ{}_Rq{}.data".format(wdsfile[-11:-5], RzH_v, RzQ_v, Rzq_v))
            if my_file.is_file():
                print('DA_{}_RH{}_RQ{}_Rq{}.data exists'.format(wdsfile[-11:-5], RzH_v, RzQ_v, Rzq_v))
                
                # file exists
                files.append(my_file)
                count = len(files)
                print("Completed: {} / 216".format(count))
            
            # Running the simulations which have not been run
            else:
                i = i + 1
                print('Run: {} / 216'.format(i))
                print('RzH_v: {}, RzQ_v: {}, Rzq_v: {}'.format(RzH_v, RzQ_v, Rzq_v))
                Run = Py_DA_WD.Three_Step_EnKF(wdsfile, bdemfile, topofile, assetfile, lnkidfile, nodidfile, 
                                  npi, nn, no, sim_time, ne, Dmult, Res_fname, 
                                  H_observ, vzH_v, RzH_v,
                                  Q_observ, vzQ_v, RzQ_v,
                                  q_observ, vzq_v, Rzq_v)

# Printing total computation time
total_sim_time = time.time() - t
print('Total Computation Time: {:2f}'.format(total_sim_time))

#%% Running experiment with noise in state measurements (SR4)

from pathlib import Path
# List to check finished files
files = []

# Time measurement and counter for number of runs
i = 0  
t = time.time() 
files = []
ne =30
sim_time = 24 

# Noise in %
Noise = 5   

for vzH in [0.001, 0.1, 0.2, 0.3, 0.5, 1]:
    SD = (Noise/100)*vzH
    vzH_normal = np.random.normal(vzH, SD, 100)
    i = 0 
    for vzH_v in vzH_normal:
        my_file = Path("RESULTS/DA_{}_E{}_vzH{:.2f}_vzQ{:.2f}_vzq{:.2f}_{}.data".format(wdsfile[-11:-5], ne, vzH, vzQ_v, vzq_v, i))
        if my_file.is_file():
            print("DA_{}_E{}_vzH{:.2f}_vzQ{:.2f}_vzq{:.2f}_{}.data".format(wdsfile[-11:-5], ne, vzH, vzQ_v, vzq_v, i))
            # file exists
            files.append(my_file)
            count = len(files)
            print("Completed {} / 600".format(count)) 
        else:
            i = i + 1
            print('Run:', i)
            print("DA_{}_E{}_vzH{:.2f}_vzQ{:.2f}_vzq{:.2f}_{}.data".format(wdsfile[-11:-5], ne, vzH, vzQ_v, vzq_v, i))
            Res_fname = "RESULTS/DA_{}_E{}_vzH{:.2f}_vzQ{:.2f}_vzq{:.2f}_{}.data".format(wdsfile[-11:-5], ne, vzH, vzQ_v, vzq_v, i)
            Run = Py_DA_WD.Three_Step_EnKF(wdsfile, bdemfile, topofile, assetfile, lnkidfile, nodidfile, 
                              npi, nn, no, sim_time, ne, Dmult, Res_fname, 
                              H_observ, vzH_v, RzH_v,
                              Q_observ, vzQ_v, RzQ_v,
                              q_observ, vzq_v, Rzq_v)
            
# Printing total computation time
total_sim_time = time.time() - t
print('Total Computation Time: {:2f}'.format(total_sim_time))

# Time measurement and counter for number of runs
i = 0  
t = time.time() 
files = []
ne =30
sim_time = 24 

# Noise in %
Noise = 5   
           
for vzQ in [0.001, 1, 2, 3, 5, 10]:
    SD = (Noise/100)*vzQ
    vzQ_normal = np.random.normal(vzQ, SD, 100)
    i = 0 
    for vzQ_v in vzQ_normal:
        my_file = Path("RESULTS/vz/vzQ/DA_{}_E{}_vzH{:.2f}_vzQ{:.2f}_vzq{:.2f}_{}.data".format(wdsfile[-11:-5], ne, vzH_v, vzQ, vzq_v, i))
        if my_file.is_file():
            print("DA_{}_E{}_vzH{:.2f}_vzQ{:.2f}_vzq{:.2f}_{}.data".format(wdsfile[-11:-5], ne, vzH_v, vzQ, vzq_v, i))
            # file exists
            files.append(my_file)
            count = len(files)
            print("Completed {} / 600".format(count)) 
        else:
            i = i + 1
            print('Run:', i)
            print("DA_{}_E{}_vzH{:.2f}_vzQ{:.2f}_vzq{:.2f}_{}.data".format(wdsfile[-11:-5], ne, vzH_v, vzQ, vzq_v, i))
            Res_fname = "RESULTS/vz/Fiv/vzQ/DA_{}_E{}_vzH{:.2f}_vzQ{:.2f}_vzq{:.2f}_{}.data".format(wdsfile[-11:-5], ne, vzH_v, vzQ, vzq_v, i)
            Run = Py_DA_WD.Three_Step_EnKF(wdsfile, bdemfile, topofile, assetfile, lnkidfile, nodidfile, 
                              npi, nn, no, sim_time, ne, Dmult, Res_fname, 
                              H_observ, vzH_v, RzH_v,
                              Q_observ, vzQ_v, RzQ_v,
                              q_observ, vzq_v, Rzq_v)
            
# Printing total computation time
total_sim_time = time.time() - t
print('Total Computation Time: {:2f}'.format(total_sim_time))

# Time measurement and counter for number of runs
i = 0  
t = time.time() 
files = []
ne =30
sim_time = 24 

# Noise in %
Noise = 5              
            
for vzq in [0.001, 0.1, 0.2, 0.3, 0.5, 1]:
    SD = (Noise/100)*vzq
    vzq_normal = np.random.normal(vzq, SD, 100)
    i = 0 
    for vzq_v in vzq_normal:
        my_file = Path("RESULTS/vz/vzqd/DA_{}_E{}_vzH{:.2f}_vzQ{:.2f}_vzq{:.2f}_{}.data".format(wdsfile[-11:-5], ne, vzH_v, vzQ_v, vzq, i))
        if my_file.is_file():
            print("DA_{}_E{}_vzH{:.2f}_vzQ{:.2f}_vzq{:.2f}_{}.data".format(wdsfile[-11:-5], ne, vzH_v, vzQ_v, vzq, i))
            # file exists
            files.append(my_file)
            count = len(files)
            print("Completed {} / 5".format(count)) 
        # else:
        i = i + 1
        print('Run:', i)
        print("DA_{}_E{}_vzH{:.2f}_vzQ{:.2f}_vzq{:.2f}_{}.data".format(wdsfile[-11:-5], ne, vzH_v, vzQ_v, vzq, i))
        Res_fname = "RESULTS/vz/Fiv/vzqd/DA_{}_E{}_vzH{:.2f}_vzQ{:.2f}_vzq{:.2f}_{}.data".format(wdsfile[-11:-5], ne, vzH_v, vzQ_v, vzq, i)
        Run = Py_DA_WD.Three_Step_EnKF(wdsfile, bdemfile, topofile, assetfile, lnkidfile, nodidfile, 
                          npi, nn, no, sim_time, ne, Dmult, Res_fname, 
                          H_observ, vzH_v, RzH_v,
                          Q_observ, vzQ_v, RzQ_v,
                          q_observ, vzq_v, Rzq_v)
        
# Printing total computation time
total_sim_time = time.time() - t
print('Total Computation Time: {:2f}'.format(total_sim_time))
        
#%% Optimization with Greedy Algorithm (SR5)
              
import wntr
import pickle
import os
import subprocess

wn = wntr.network.WaterNetworkModel(wdsfile) 
t = time.time()
i = 0  

# Setting number of sensors to be optimized
H_sensors = 40

TV        = []
TVL       = []
nodes = wn.junction_name_list
nodes = list(map(int, nodes))
H_net = []

# Setting number of sensors to be optimized
q_sensors = 40
TVq         = []
TVqL        = []
q_net       = []

# Setting number of sensors to be optimized
Q_sensors   = 40
TVQ         = []
TVQL        = []
#links = wn.pipe_name_list
#links = list(map(int, links))
links = [*range(317)]
Q_net = []

l_time = []
i = 1
for sH in range(1, H_sensors + 1):
    try:
        with open("RESULTS/DA_{}_E{}_sH{}_test.data".format(wdsfile[-11:-5], ne, sH), 'rb') as filehandle:
            # Read the data as a binary data stream
            Sim_Res = pickle.load(filehandle)
            H_net = Sim_Res[0][0]
            print('The file exists and has been read.')
    except FileNotFoundError:
        print('The file not read')
        ls_time = time.time()
        print('No of Head sensors set to :', H_sensors)
        print('Checking Sensor No. :', sH)

        srch_spac = [item for item in nodes if item not in H_net]
        for H_sens in srch_spac:
            print('Testing node for Head:', H_sens)
            print('Search Space: ', len(srch_spac)) 
            H_net_temp = H_net[:]
            H_net_temp.append(H_sens-1)
            H_observ = np.array(H_net_temp)
            Res_fname = "RESULTS/DA_{}_E{}_sH{}_test.data".format(wdsfile[-11:-5], ne, sH)
            Sim_Res = Py_DA_WD.Three_Step_EnKF(wdsfile, bdemfile, topofile, assetfile, lnkidfile, nodidfile, 
                              npi, nn, no, sim_time, ne, Dmult, Res_fname, 
                              H_observ, vzH_v, RzH_v,
                              Q_observ, vzQ_v, RzQ_v,
                              q_observ, vzq_v, Rzq_v)
            H_net_temp.clear()
            TV.append(Sim_Res[3])
    
            lf_time = time.time() - ls_time    
            print("\n Run Time for Sensor Optimization Loop:{0:.2f} s".format(lf_time))
            i = i + 1
            print("Run No.: ", i)
        TVL.append(TV[:])
        OptH_Node = TV.index(min(TV))
        H_net.append(OptH_Node)
        TV.clear()
        l_time.append(lf_time)  
        
        Output = [[H_net, TVL], [q_net, TVqL], [Q_net, TVQL], l_time, Sim_Res[0]]
                    
        with open(Res_fname, 'wb') as filehandle:
            # Store the data as a binary data stream
            pickle.dump(Output, filehandle)
            
        if sH % 4 == 3:
            subprocess.call(["python", "Py_DA_WDN.py"])
            os._exit(0)# restart kernel

# Optimizing Demand Sensors

H_observ = np.array(H_net)
l_time = []
i = 1
for sq in range(1, q_sensors + 1):
    try:
        with open("RESULTS/DA_{}_E{}_sq{}_test.data".format(wdsfile[-11:-5], ne, sq), 'rb') as filehandle:
            # Read the data as a binary data stream
            Sim_Res = pickle.load(filehandle)
            q_net = Sim_Res[1][0]
            print('The file exists and has been read.')
    except FileNotFoundError:
        print('The file not read')
        ls_time = time.time()
        print('No of Demand sensors set to :', q_sensors)
        print('Checking Sensor No. :', sq)

        srch_spac = [item for item in nodes if item not in q_net]
        for q_sens in srch_spac:
            print('Testing node for Demand:', q_sens)
            print('Search Space: ', len(srch_spac))
            q_net_temp = q_net[:]
            q_net_temp.append(q_sens-1)
            q_observ = np.array(q_net_temp)
            Res_fname = "RESULTS/DA_{}_E{}_sq{}_test.data".format(wdsfile[-11:-5], ne, sq)
            Sim_Res = Py_DA_WD.Three_Step_EnKF(wdsfile, bdemfile, topofile, assetfile, lnkidfile, nodidfile, 
                              npi, nn, no, sim_time, ne, Dmult, Res_fname, 
                              H_observ, vzH_v, RzH_v,
                              Q_observ, vzQ_v, RzQ_v,
                              q_observ, vzq_v, Rzq_v)
            q_net_temp.clear()
            TVq.append(Sim_Res[3])
            
            lf_time = time.time() - ls_time    
            print("\n Run Time for Sensor Optimization Loop:{0:.2f} s".format(lf_time))  
            i = i + 1
            print("Run No.: ", i)
        TVqL.append(TVq[:])
        Optq_Node = TVq.index(min(TVq))
        q_net.append(Optq_Node)
        TVq.clear()
        l_time.append(lf_time)  
        
        Output = [[H_net, TVL], [q_net, TVqL], [Q_net, TVQL], l_time, Sim_Res[0]]
    
        with open(Res_fname, 'wb') as filehandle:
            # Store the data as a binary data stream
            pickle.dump(Output, filehandle)
        
# Optimizing Flow Sensors

H_observ = np.array(H_net)
q_observ = np.array(q_net)
l_time = []
i = 1

for sQ in range(1, Q_sensors + 1):
    try:
        with open("RESULTS/DA_{}_E{}_sQf{}_test.data".format(wdsfile[-11:-5], ne, sQ), 'rb') as filehandle:
            # Read the data as a binary data stream
            Sim_Res = pickle.load(filehandle)
            Q_net = Sim_Res[2][0]
            print('The file exists and has been read.')
    except FileNotFoundError:
        print('The file not read')
        ls_time = time.time()
        print('No of Flow sensors set to :', Q_sensors)
        print('Checking Sensor No. :', sQ)
    #    head_net_temp = []
        srch_spac = [item for item in links if item not in Q_net]
        for Q_sens in srch_spac:
            print('Testing link for Flow:', Q_sens)
            print('Search Space: ', len(srch_spac))
            Q_net_temp = Q_net[:]
            Q_net_temp.append(Q_sens-1)
            Q_observ = np.array(Q_net_temp)
            Res_fname = "RESULTS/DA_{}_E{}_sQf{}_test.data".format(wdsfile[-11:-5], ne, sQ)
            Sim_Res = Py_DA_WD.Three_Step_EnKF(wdsfile, bdemfile, topofile, assetfile, lnkidfile, nodidfile, 
                              npi, nn, no, sim_time, ne, Dmult, Res_fname, 
                              H_observ, vzH_v, RzH_v,
                              Q_observ, vzQ_v, RzQ_v,
                              q_observ, vzq_v, Rzq_v)
            Q_net_temp.clear()
            TVQ.append(Sim_Res[3])
            
            lf_time = time.time() - ls_time    
            print("\n Run Time for Sensor Optimization Loop:{0:.2f} s".format(lf_time))  
            i = i + 1
            print("Run No.: ", i)
    
        TVQL.append(TVQ[:])
        OptQ_Node = TVQ.index(min(TVQ))
        Q_net.append(OptQ_Node)
        TVQ.clear()
        l_time.append(lf_time)
        
        Output = [[H_net, TVL], [q_net, TVqL], [Q_net, TVQL], l_time, Sim_Res[0]]
        
        with open(Res_fname, 'wb') as filehandle:
            # Store the data as a binary data stream
            pickle.dump(Output, filehandle)
    
                    
totaltime = time.time() - t
print("\n Run Time:  {0:.2f} s" .format(totaltime))
timestr = "{0:.0f}".format(totaltime)

# Saving final output file of optimization

Output = [[H_net, TVL], [q_net, TVqL], [Q_net, TVQL], l_time, totaltime, Sim_Res[0]]
Res_fname = "RESULTS/GA_{}_sH{}_sq{}_sQ{}_TT{}.data".format(wdsfile[-11:-5], H_sensors, q_sensors, Q_sensors, timestr)

#%%
# Calibration of pipe roughness coefficient (SR6)

HWC_Range = range(100, 165, 5)
ATVR_Hqa             = np.zeros([nn , len(HWC_Range)])
ATVR_Qa              = np.zeros([npi, len(HWC_Range)])
Ki = 0
for Khaze in HWC_Range:
    Ki = Ki + 1
    print("Hazen William Coefficient: {}".format(Khaze))
    Res_fname = "RESULTS/DA_{}_E{}_T{}_K{}.data".format(wdsfile[-11:-5], ne, sim_time, Khaze)
    Sim_Res = Py_DA_WD.Three_Step_EnKF(wdsfile, bdemfile, topofile, assetfile, lnkidfile, nodidfile, 
                      npi, nn, no, sim_time, ne, Dmult, Res_fname, 
                      H_observ, vzH_v, RzH_v,
                      Q_observ, vzQ_v, RzQ_v,
                      q_observ, vzq_v, Rzq_v) 