# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 12:25:11 2023

@author: ibrah
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
Ens5                 = []
Enssnell             = []
Ens5snell            = []
EnsPC                = []
Ens5PC               = []
ttime                = []
t5time               = []
ttimesnell           = []
t5timesnell          = []
ttimePC              = []
t5timePC             = []
Resfile_lis          = []
Res5file_lis         = []
Ressnellfile_lis     = []
Res5snellfile_lis    = []
ResPCfile_lis        = []
Res5PCfile_lis       = []

for ne in [5, 10 , 15, 20, 25, 30, 40, 50, 100]:
    resfilename = 'DA_MOD_Lap_E{}_T24.data'.format(ne)
    #resfilename = 'DA_Fiv_E{}.data'.format(ne)

    with open('RESULTS/CD/'+resfilename, 'rb') as filehandle:
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
    
    Resfile_lis.append(resfilename)    
    Ens.append(ne)   
    ttime.append(totaltime)
    
    
    
    Res_Dict = {'Filename':Resfile_lis, 'Ensemble No.':Ens, "Simulation Time":ttime}
    Results = pd.DataFrame(Res_Dict)
    
for ne in [5, 10 , 15, 20, 25, 30, 40, 50, 100]:
    resfilename = 'DA_MOD_Snell_E{}_T24.data'.format(ne)
    #resfilename = 'DA_Fiv_E{}.data'.format(ne)

    with open('RESULTS/CD/'+resfilename, 'rb') as filehandle:
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
    
    Ressnellfile_lis.append(resfilename)    
    Enssnell.append(ne)   
    ttimesnell.append(totaltime)
    
    
    
    Ressnell_Dict = {'Filename':Ressnellfile_lis, 'Ensemble No.':Enssnell, "Simulation Time":ttimesnell}
    Results_snell = pd.DataFrame(Ressnell_Dict)
    
for ne in [5, 10 , 15, 20, 25, 30, 40, 50, 100]:
    resfilename = 'DA_MOD_PC_E{}_T24.data'.format(ne)
    #resfilename = 'DA_Fiv_E{}.data'.format(ne)

    with open('RESULTS/CD/'+resfilename, 'rb') as filehandle:
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
    
    ResPCfile_lis.append(resfilename)    
    EnsPC.append(ne)   
    ttimePC.append(totaltime)
    
    ResPC_Dict = {'Filename':ResPCfile_lis, 'Ensemble No.':EnsPC, "Simulation Time":ttimePC}
    Results_PC = pd.DataFrame(ResPC_Dict) 
    
for ne in [5, 10 , 15, 20, 25, 30, 40, 50, 100]:
    resfilename = 'DA_Fiv_Lap_E{}_T24.data'.format(ne)
    #resfilename = 'DA_Fiv_E{}.data'.format(ne)

    with open('RESULTS/CD/'+resfilename, 'rb') as filehandle:
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
    
    Res5file_lis.append(resfilename)    
    Ens5.append(ne)   
    t5time.append(totaltime)
    
    
    
    Res5_Dict = {'Filename':Res5file_lis, 'Ensemble No.':Ens5, "Simulation Time":t5time}
    Results_Fiv = pd.DataFrame(Res5_Dict)   
    
for ne in [5, 10 , 15, 20, 25, 30, 40, 50, 100]:
    resfilename = 'DA_Fiv_Snell_E{}_T24.data'.format(ne)
    #resfilename = 'DA_Fiv_E{}.data'.format(ne)

    with open('RESULTS/CD/'+resfilename, 'rb') as filehandle:
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
    
    Res5snellfile_lis.append(resfilename)    
    Ens5snell.append(ne)   
    t5timesnell.append(totaltime)
    
    
    
    Res5snell_Dict = {'Filename':Res5snellfile_lis, 'Ensemble No.':Ens5snell, "Simulation Time":t5timesnell}
    Results_Fiv_snell = pd.DataFrame(Res5snell_Dict) 

for ne in [5, 10 , 15, 20, 25, 30, 40, 50, 100]:
    resfilename = 'DA_Fiv_PC_E{}_T24.data'.format(ne)
    #resfilename = 'DA_Fiv_E{}.data'.format(ne)

    with open('RESULTS/CD/'+resfilename, 'rb') as filehandle:
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
    
    Res5PCfile_lis.append(resfilename)    
    Ens5PC.append(ne)   
    t5timePC.append(totaltime)
    
    
    
    Res5PC_Dict = {'Filename':Res5PCfile_lis, 'Ensemble No.':Ens5PC, "Simulation Time":t5timePC}
    Results_Fiv_PC = pd.DataFrame(Res5PC_Dict) 
    
x = Results.loc[:,'Ensemble No.']
y1 = Results.loc[:,'Simulation Time']
y2 = Results_Fiv.loc[:,'Simulation Time']
y3 = Results_snell.loc[:,'Simulation Time']
y4 = Results_Fiv_snell.loc[:,'Simulation Time']
y5 = Results_PC.loc[:,'Simulation Time']
y6 = Results_Fiv_PC.loc[:,'Simulation Time']

fig, ax1 = plt.subplots(figsize=(10, 5.5),constrained_layout=True, dpi=300)
fig.suptitle('Simulation time for different number of Ensembles', fontsize=16)
ax1.plot(x, y1,'.-', markersize=10, label = 'System 1 Modena time')
ax1.plot(x, y2,'.-', markersize=10, label = 'System 1 Five Reservoir')
ax1.plot(x, y5,'.:', markersize=10, label = 'System 2 Modena time')
ax1.plot(x, y6,'.:', markersize=10, label = 'System 2 Five Reservoir')
ax1.plot(x, y3,'.--', markersize=10, label = 'HPC Modena time')
ax1.plot(x, y4,'.--', markersize=10, label = 'HPC Five Reservoir')
ax1.set_xlabel('No. of Ensembles', fontsize=16)  # Add an x-label to the axes.
ax1.set_ylabel('Time (s)', fontsize=16)  # Add a y-label to the axes.
ax1.legend(fontsize=12);  # Add a legend
for i in range(3, len(x)):
    ax1.annotate('{:.0f}'.format(y1[i]), [x[i]+0.5,y1[i]-10])
    ax1.annotate('{:.0f}'.format(y2[i]), [x[i]+0.5,y2[i]-10])
    ax1.annotate('{:.0f}'.format(y3[i]), [x[i]+0.5,y3[i]-5])
    ax1.annotate('{:.0f}'.format(y4[i]), [x[i]+0.5,y4[i]-10])
    ax1.annotate('{:.0f}'.format(y5[i]), [x[i]+0.5,y5[i]-30])
    ax1.annotate('{:.0f}'.format(y6[i]), [x[i]+0.5,y6[i]-10])
#%%
# y3 = y1/x
# y4 = y2/x

# fig, ax2 = plt.subplots(figsize=(10, 6),constrained_layout=True, dpi=300)
# fig.suptitle('Simulation time per Ensemble for different number of Ensembles', fontsize=16)
# ax2.plot(x, y4,'.--', markersize=10, label = 'Five Reservoir (935 Nodes, 1278 Links)')
# ax2.plot(x, y3,'.--', markersize=10, label = 'Modena time (268 Nodes, 317 Links)')
# ax2.set_xlabel('No. of Ensembles', fontsize=16)  # Add an x-label to the axes.
# ax2.set_ylabel('Time / Ensemble (s)', fontsize=16)  # Add a y-label to the axes.
# ax2.legend(fontsize=16);  # Add a legend
# for i in range(len(x)):
#     ax2.annotate('{:.2f}'.format(y3[i]), [x[i]+1,y3[i]+1])
#     ax2.annotate('{:.2f}'.format(y4[i]), [x[i]+1,y4[i]+1])
    
# #%%
# y3 = y1/268
# y4 = y2/935

# fig, ax2 = plt.subplots(figsize=(12, 10),constrained_layout=True, dpi=300)
# fig.suptitle('Simulation time per Node for different number of Ensembles', fontsize=16)
# ax2.plot(x, y4,'.--', markersize=10, label = 'Five Reservoir (935 Nodes, 1278 Links)')
# ax2.plot(x, y3,'.--', markersize=10, label = 'Modena time (268 Nodes, 317 Links)')
# ax2.set_xlabel('No. of Ensembles')  # Add an x-label to the axes.
# ax2.set_ylabel('Time / Node(s)')  # Add a y-label to the axes.
# ax2.legend(fontsize=16);  # Add a legend
# for i in range(len(x)):
#     ax2.annotate('{:.2f}'.format(y3[i]), [x[i]+0.5,y3[i]-0.1])
#     ax2.annotate('{:.2f}'.format(y4[i]), [x[i]+0.5,y4[i]-0.1])

# #%%
# y3 = y1/317
# y4 = y2/1278

# fig, ax2 = plt.subplots(figsize=(12, 10),constrained_layout=True, dpi=300)
# fig.suptitle('Simulation time per Link for different number of Ensembles', fontsize=16)
# ax2.plot(x, y4,'.--', markersize=10, label = 'Five Reservoir (935 Nodes, 1278 Links)')
# ax2.plot(x, y3,'.--', markersize=10, label = 'Modena time (268 Nodes, 317 Links)')
# ax2.set_xlabel('No. of Ensembles')  # Add an x-label to the axes.
# ax2.set_ylabel('Time / Link (s)')  # Add a y-label to the axes.
# ax2.legend();  # Add a legend
# for i in range(len(x)):
#     ax2.annotate('{:.2f}'.format(y3[i]), [x[i]+0.5,y3[i]-0.05])
#     ax2.annotate('{:.2f}'.format(y4[i]), [x[i]+0.5,y4[i]-0.05])