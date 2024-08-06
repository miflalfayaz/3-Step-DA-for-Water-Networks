# -*- coding: utf-8 -*-
"""
# This script is used for reading the results of Data Assimilation with varying number of ensembles for DA.
# Ibrahim Miflal Fayaz - Msc Student UN-IHE
# 02/12/2022
"""

import pickle
import numpy as np
import pandas as pd
import plot
import plot_net
import TV
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('seaborn-colorblind')


# Lists for final results
Ens                  = []
ttime                = []
Resfile_lis          = []
ATV_H_qj             = []
ATV_H_qj_zH          = []
ATV_H_qj_zH_zQ       = []
ATV_H_qj_zH_zQ_zq    = []
  
ATV_q_qj             = []
ATV_q_qj_zH          = []
ATV_q_qj_zH_zQ       = []
ATV_q_qj_zH_zQ_zq    = []
   
ATV_Q_qj             = []
ATV_Q_qj_zH          = []
ATV_Q_qj_zH_zQ       = []
ATV_Q_qj_zH_zQ_zq    = []


for ne in [5, 10 , 15, 20, 25, 30, 40, 50, 100]:
    #resfilename = 'DA_MOD_Snell_E{}_T24.data'.format(ne)
    resfilename = 'DA_MOD_E{}.data'.format(ne)
    #resfilename = 'DA_Fiv_E{}.data'.format(ne)

    with open('RESULTS/Parellel/MOD_all/'+resfilename, 'rb') as filehandle:
        # Read the data as a binary data stream
        Sim_Res = pickle.load(filehandle)
        
    # Reading the simulation Input variables
    
    sim_time    = Sim_Res[0][0]
    ne          = Sim_Res[0][1]
    npi         = Sim_Res[0][2]
    nn          = Sim_Res[0][3]
    n0          = Sim_Res[0][4] 
    H_observ    = Sim_Res[0][5]
    q_observ    = Sim_Res[0][6]
    Q_observ    = Sim_Res[0][7]
    wdsfile     = Sim_Res[0][8]
    totaltime   = round(Sim_Res[2],2)
    
    # Reading the Result Arrays
       
    Hobs_ti             = Sim_Res[1][0 ]   
    H_qj_ti             = Sim_Res[1][1 ]   
    H_qj_zH_ti          = Sim_Res[1][2 ]    
    H_qj_zH_zQ_ti       = Sim_Res[1][3 ]  
    H_qj_zH_zQ_zq_ti    = Sim_Res[1][4 ]  
    
    qobs_ti             = Sim_Res[1][5 ]    
    q_qj_ti             = Sim_Res[1][6 ]  
    q_qj_zH_ti          = Sim_Res[1][7 ]   
    q_qj_zH_zQ_ti       = Sim_Res[1][8 ]  
    q_qj_zH_zQ_zq_ti    = Sim_Res[1][9 ]   
    
    Qobs_ti             = Sim_Res[1][10]       
    Q_qj_ti             = Sim_Res[1][11]  
    Q_qj_zH_ti          = Sim_Res[1][12]   
    Q_qj_zH_zQ_ti       = Sim_Res[1][13]  
    Q_qj_zH_zQ_zq_ti    = Sim_Res[1][14]  

#%%
    # Calculating the Total Variance
    
    TV_H_qj             = TV.Total_Variance(H_qj_ti,           Hobs_ti, nn, ne, sim_time)  
    TV_H_qj_zH          = TV.Total_Variance(H_qj_zH_ti,        Hobs_ti, nn, ne, sim_time)  #/ TV_H_qj   
    TV_H_qj_zH_zQ       = TV.Total_Variance(H_qj_zH_zQ_ti,     Hobs_ti, nn, ne, sim_time)  #/ TV_H_qj
    TV_H_qj_zH_zQ_zq    = TV.Total_Variance(H_qj_zH_zQ_zq_ti,  Hobs_ti, nn, ne, sim_time)  #/ TV_H_qj
       
    TV_q_qj             = TV.Total_Variance(q_qj_ti,           qobs_ti, nn, ne, sim_time)    
    TV_q_qj_zH          = TV.Total_Variance(q_qj_zH_ti,        qobs_ti, nn, ne, sim_time)  #/ TV_q_qj  
    TV_q_qj_zH_zQ       = TV.Total_Variance(q_qj_zH_zQ_ti,     qobs_ti, nn, ne, sim_time)  #/ TV_q_qj 
    TV_q_qj_zH_zQ_zq    = TV.Total_Variance(q_qj_zH_zQ_zq_ti,  qobs_ti, nn, ne, sim_time)  #/ TV_q_qj  
    
        
    TV_Q_qj             = TV.Total_Variance(Q_qj_ti,           Qobs_ti, npi, ne, sim_time)    
    TV_Q_qj_zH          = TV.Total_Variance(Q_qj_zH_ti,        Qobs_ti, npi, ne, sim_time)  #/ TV_Q_qj  
    TV_Q_qj_zH_zQ       = TV.Total_Variance(Q_qj_zH_zQ_ti,     Qobs_ti, npi, ne, sim_time)  #/ TV_Q_qj   
    TV_Q_qj_zH_zQ_zq    = TV.Total_Variance(Q_qj_zH_zQ_zq_ti,  Qobs_ti, npi, ne, sim_time)  #/ TV_Q_qj  

#%% 

    # Calculating Daily Average Total Variance           
    
    Resfile_lis.append(resfilename)
    Ens.append(ne)
    ttime.append(totaltime)
    ATV_H_qj.append(np.sum(TV_H_qj)/24) 
    ATV_H_qj_zH.append(np.sum(TV_H_qj_zH)/24)
    ATV_H_qj_zH_zQ.append(np.sum(TV_H_qj_zH_zQ)/24)
    ATV_H_qj_zH_zQ_zq.append(np.sum(TV_H_qj_zH_zQ_zq)/24)
       
    ATV_q_qj.append(np.sum(TV_q_qj)/24)
    ATV_q_qj_zH.append(np.sum(TV_q_qj_zH)/24) 
    ATV_q_qj_zH_zQ.append(np.sum(TV_q_qj_zH_zQ)/24)
    ATV_q_qj_zH_zQ_zq.append(np.sum(TV_q_qj_zH_zQ_zq)/24)  
    
        
    ATV_Q_qj.append(np.sum(TV_Q_qj)/24)
    ATV_Q_qj_zH.append(np.sum(TV_Q_qj_zH)/24) 
    ATV_Q_qj_zH_zQ.append(np.sum(TV_Q_qj_zH_zQ)/24) 
    ATV_Q_qj_zH_zQ_zq.append(np.sum(TV_Q_qj_zH_zQ_zq)/24) 

    Res_Dict = {'Filename':Resfile_lis, 'Ensemble No.':Ens, 
                "H|qj":ATV_H_qj, "H|qj zH":ATV_H_qj_zH, "H|qj zH zQ":ATV_H_qj_zH_zQ, "H|qj zH zQ zq":ATV_H_qj_zH_zQ_zq,
                "q|qj":ATV_q_qj, "q|qj zH":ATV_q_qj_zH, "q|qj zH zQ":ATV_q_qj_zH_zQ, "q|qj zH zQ zq":ATV_q_qj_zH_zQ_zq,
                "Q|qj":ATV_Q_qj, "Q|qj zH":ATV_Q_qj_zH, "Q|qj zH zQ":ATV_Q_qj_zH_zQ, "Q|qj zH zQ zq":ATV_Q_qj_zH_zQ_zq,}
    Results = pd.DataFrame(Res_Dict)

#%%
#Plotting Total Variance for No. Ens

x = Results.loc[:,'Ensemble No.'].tail(-1)
y1 = Results[["H|qj", "H|qj zH",  "H|qj zH zQ", "H|qj zH zQ zq"]].tail(-1)
y2 = Results[["q|qj", "q|qj zH",  "q|qj zH zQ", "q|qj zH zQ zq"]].tail(-1)
y3 = Results[["Q|qj", "Q|qj zH",  "Q|qj zH zQ", "Q|qj zH zQ zq"]].tail(-1)
y4 = ttime[0:5]

fig, ax = plt.subplots(3, figsize=(12, 8),constrained_layout=True, dpi = 300)
fig.suptitle('Average Total Variance for different number of Ensembles - Modena - HPC', fontsize=20)

ax[0].plot(x, y1, '.--', label = y1.columns.values)
# ax[0,0].set_xlabel('No. of Ensembles')  # Add an x-label to the axes.
ax[0].set_ylabel('Avg TV for Head', fontsize=16)  # Add a y-label to the axes.
#ax[0].set_title("Total variance")  # Add a title to the axes.
ax[0].legend(fontsize=14);  # Add a legend.

#plt.show()
# Plotting Total Variance for Demand
#x = np.arange(sim_time)
#fig, ax = plt.subplots()

ax[1].plot(x, y2, '.--', label = y2.columns.values)
# ax[1,0].set_xlabel('No. of Ensembles')  # Add an x-label to the axes.
ax[1].set_ylabel('Avg TV for Demand', fontsize=16)  # Add a y-label to the axes.
#ax[1].set_title("Total variance ")  # Add a title to the axes.
ax[1].legend(fontsize=14);  # Add a legend.

#plt.show()
# Plotting Total Variance for Demand
#x = np.arange(sim_time)
#fig, ax = plt.subplots()

ax[2].plot(x, y3, '.--', label = y3.columns.values)
ax[2].set_xlabel('No. of Ensembles', fontsize=16)  # Add an x-label to the axes.
ax[2].set_ylabel('Avg TV for Flow', fontsize=16)  # Add a y-label to the axes.
#ax[2].set_title("Total variance" )  # Add a title to the axes.
ax[2].legend(fontsize=14);  # Add a legend.


# ax[1,1].plot(x, y4, label = 'Simulation time')
# ax[1,1].set_xlabel('No. of Ensembles')  # Add an x-label to the axes.
# ax[1,1].set_ylabel('Time (s)')  # Add a y-label to the axes.
#ax[2].set_title("Total variance" )  # Add a title to the axes.
#ax[1].legend();  # Add a legend

plt.show()

#%%
x = Results.loc[:,'Ensemble No.']
y5 = ttime
y6 = ttime
fig, ax1 = plt.subplots(figsize=(12, 10),constrained_layout=True)
fig.suptitle('Simulation time for different number of Ensembles', fontsize=16)
ax1.plot(x, y5, label = 'Five Reservoir time')
ax1.plot(x, y6, label = 'Modena time')
ax1.set_xlabel('No. of Ensembles')  # Add an x-label to the axes.
ax1.set_ylabel('Time (s)')  # Add a y-label to the axes.
ax1.legend(fontsize=16);  # Add a legend