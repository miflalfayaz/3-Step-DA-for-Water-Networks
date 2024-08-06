# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 00:03:22 2022

@author: ibrah
"""

wdsfile     = 'FiveReservoirs_PosDir_Ver_Teq0.inp' 
nn          = 935
npi         = 1278
ne          = 20  
totaltime   = 720139  
# wdsfile     = 'MOD.inp' 
# nn          = 268
# npi         = 317
# ne          = 20  
# totaltime   = 52967

import pickle   
import numpy as np
 
#resfilename = 'GA_MOD_sH12_sq12_sQ12_TT39980.data'
# resfilename = 'GA_MOD_sH12_sq12_sQ12_TT22243.data'
# resfilename = 'GA_Fiv_sH12_sq12_sQ12_TT396324.data'
resfilename = 'GA_Fiv_sH12_sq12_sQ12_TT720139HqQ.data'

#resfilename = 'GA_MOD_sH3_sq12_sQ12_TT546.data'
#resfilename = 'GA_MOD_sH3_sq12_sQ12_TT2830.data'
#resfilename = 'GA_MOD_sH3_sq12_sQ12_TT2863.data'
#resfilename = 'GA_MOD_sH3_sq12_sQ12_TT968.data'
#resfilename = 'GA_MOD_sH1_sq40_sQ40_TT0.data'
#resfilename ='GA_MOD_sH40_sq40_sQ40_TT1.data'
#resfilename ='GA_MOD_sH12_sq12_sQ12_TT59762.data'
#resfilename ='GA_MOD_sH40_sq40_sQ40_TT52967.data'
#resfilename = 'GA_Fiv_sH40_sq40_sQ40_TT0HqQ.data'


with open('RESULTS/GA/'+resfilename, 'rb') as filehandle:
    # Read the data as a binary data stream
    Sim_Res = pickle.load(filehandle)
    
    
H_observ = np.array(Sim_Res[0][0])
q_observ = np.array(Sim_Res[1][0])
Q_observ = np.array(Sim_Res[2][0])
#Q_observ = np.array([1,3])

TVL      = Sim_Res[0][1]
#TVL      = [[x * 24 for x in inner_list] for inner_list in TVL]               # FIRST RUN OF OPTIMIZATION WITH MODENA DID NOT SAVE SIMULATION TIME
TVqL     = Sim_Res[1][1]
#TVqL      = [[x * 24 for x in inner_list] for inner_list in TVqL]
TVQL     = Sim_Res[2][1]
#TVQL      = [[x * 24 for x in inner_list] for inner_list in TVQL]


#%%
import plot_net
plot_net.monitoring(wdsfile, nn, npi, q_observ, H_observ, Q_observ)

#%%
yh = [] 
for hvalues in TVL:
#    y =  min(hvalues)
    y =  sum(hvalues)/len(hvalues)
    yh.append(y)
    
yq = [] 
for qvalues in TVqL:
#    y =  min(hvalues)
    y =  sum(qvalues)/len(qvalues)
    yq.append(y)
    
yQ = [] 
for Qvalues in TVQL:
#    y =  min(hvalues)
    y =  sum(Qvalues)/len(Qvalues)
    yQ.append(y)
    
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('seaborn-colorblind')
#style.use('tableau-colorblind10')

x = np.arange(1, len(yh)+1)
fig, ax = plt.subplots( figsize=(8,5),constrained_layout=True, dpi=300)
fig.suptitle('Combined Total Variance Ratio against no. of sensors in the Network.\n No. of ensembles: {}, Network: {}, Simulation time: {} s'.format(ne, wdsfile[0:3], totaltime), fontsize=16)

ax.plot(x, yh,'.--',label = " Optimised Head Sensors")
ax.plot(x, yq,'.:' ,label = " Optimised Demand Sensors")
ax.plot(x, yQ,'.-.',label = " Optimised Flow Sensors")
ax.set_xlabel('No. of sensors in the Network', fontsize=16)  # Add an x-label to the axes.
ax.set_ylabel('Combined Total Variance Ratio', fontsize=16)  # Add a y-label to the axes.
ax.legend()
plt.show()