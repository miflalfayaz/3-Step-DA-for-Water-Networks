# -*- coding: utf-8 -*-
"""
# This script is used for reading the results of Data Assimilation with varying variance of sensors.
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
Resfile_lis          = []
ttime                = []
vzH_vl               = [] 
vzQ_vl               = [] 
vzq_vl               = []
vzt                  = []
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

ATVR                 = []

vzH_v = 0
vzQ_v = 0
vzq_v = 0

#for vzH in [0.001, 0.1, 0.2, 0.3, 0.5, 1]:
#for vzQ in [0.001, 1, 2, 3, 5, 10]:
for vzq in [0.001, 0.1, 0.2, 0.3, 0.5, 1]:
    i = 0 
    for i in range(1, 101):
        resfilename = "DA_MOD_E30_vzH{:.2f}_vzQ{:.2f}_vzq{:.2f}_{}.data".format(vzH_v, vzQ_v, vzq, i)
        #resfilename = "DA_Fiv_E30_vzH{:.2f}_vzQ{:.2f}_vzq{:.2f}_{}.data".format(vzH_v, vzQ_v, vzq, i)
        print(resfilename)
        with open('RESULTS/vz/MOD/vzqd/'+resfilename, 'rb') as filehandle:
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
        vzH_v       = Sim_Res[0][17]
        vzQ_v       = Sim_Res[0][21]
        vzq_v       = Sim_Res[0][25]
        
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
        
        TV_H_qjr            = TV.Total_Variance(H_qj_ti,           Hobs_ti, nn, ne, sim_time)  
        TV_H_qj             = TV.Total_Variance(H_qj_ti,           Hobs_ti, nn, ne, sim_time)   
        TV_H_qj_zH          = TV.Total_Variance(H_qj_zH_ti,        Hobs_ti, nn, ne, sim_time)     
        TV_H_qj_zH_zQ       = TV.Total_Variance(H_qj_zH_zQ_ti,     Hobs_ti, nn, ne, sim_time)  
        TV_H_qj_zH_zQ_zq    = TV.Total_Variance(H_qj_zH_zQ_zq_ti,  Hobs_ti, nn, ne, sim_time)  
        
        TV_q_qjr            = TV.Total_Variance(q_qj_ti,           qobs_ti, nn, ne, sim_time)
        TV_q_qj             = TV.Total_Variance(q_qj_ti,           qobs_ti, nn, ne, sim_time)   
        TV_q_qj_zH          = TV.Total_Variance(q_qj_zH_ti,        qobs_ti, nn, ne, sim_time)  
        TV_q_qj_zH_zQ       = TV.Total_Variance(q_qj_zH_zQ_ti,     qobs_ti, nn, ne, sim_time)  
        TV_q_qj_zH_zQ_zq    = TV.Total_Variance(q_qj_zH_zQ_zq_ti,  qobs_ti, nn, ne, sim_time)  
        
            
        TV_Q_qjr            = TV.Total_Variance(Q_qj_ti,           Qobs_ti, npi, ne, sim_time)
        TV_Q_qj             = TV.Total_Variance(Q_qj_ti,           Qobs_ti, npi, ne, sim_time)  
        TV_Q_qj_zH          = TV.Total_Variance(Q_qj_zH_ti,        Qobs_ti, npi, ne, sim_time)    
        TV_Q_qj_zH_zQ       = TV.Total_Variance(Q_qj_zH_zQ_ti,     Qobs_ti, npi, ne, sim_time)   
        TV_Q_qj_zH_zQ_zq    = TV.Total_Variance(Q_qj_zH_zQ_zq_ti,  Qobs_ti, npi, ne, sim_time)  
        
        TVR_H_qjr            = TV.Total_Variance(H_qj_ti,           Hobs_ti, nn, ne, sim_time)  
        TVR_H_qj             = TV.Total_Variance(H_qj_ti,           Hobs_ti, nn, ne, sim_time)  / TVR_H_qjr 
        TVR_H_qj_zH          = TV.Total_Variance(H_qj_zH_ti,        Hobs_ti, nn, ne, sim_time)  / TVR_H_qjr   
        TVR_H_qj_zH_zQ       = TV.Total_Variance(H_qj_zH_zQ_ti,     Hobs_ti, nn, ne, sim_time)  / TVR_H_qjr
        TVR_H_qj_zH_zQ_zq    = TV.Total_Variance(H_qj_zH_zQ_zq_ti,  Hobs_ti, nn, ne, sim_time)  / TVR_H_qjr
        
        TVR_q_qjr            = TV.Total_Variance(q_qj_ti,           qobs_ti, nn, ne, sim_time)
        TVR_q_qj             = TV.Total_Variance(q_qj_ti,           qobs_ti, nn, ne, sim_time)  / TVR_q_qjr  
        TVR_q_qj_zH          = TV.Total_Variance(q_qj_zH_ti,        qobs_ti, nn, ne, sim_time)  / TVR_q_qjr  
        TVR_q_qj_zH_zQ       = TV.Total_Variance(q_qj_zH_zQ_ti,     qobs_ti, nn, ne, sim_time)  / TVR_q_qjr
        TVR_q_qj_zH_zQ_zq    = TV.Total_Variance(q_qj_zH_zQ_zq_ti,  qobs_ti, nn, ne, sim_time)  / TVR_q_qjr  
        
            
        TVR_Q_qjr            = TV.Total_Variance(Q_qj_ti,           Qobs_ti, npi, ne, sim_time)
        TVR_Q_qj             = TV.Total_Variance(Q_qj_ti,           Qobs_ti, npi, ne, sim_time)  / TVR_Q_qjr 
        TVR_Q_qj_zH          = TV.Total_Variance(Q_qj_zH_ti,        Qobs_ti, npi, ne, sim_time)  / TVR_Q_qjr  
        TVR_Q_qj_zH_zQ       = TV.Total_Variance(Q_qj_zH_zQ_ti,     Qobs_ti, npi, ne, sim_time)  / TVR_Q_qjr   
        TVR_Q_qj_zH_zQ_zq    = TV.Total_Variance(Q_qj_zH_zQ_zq_ti,  Qobs_ti, npi, ne, sim_time)  / TVR_Q_qjr  
    
    
        TVR_T =  TVR_H_qj_zH + TVR_H_qj_zH_zQ + TVR_H_qj_zH_zQ_zq + TVR_q_qj_zH + TVR_q_qj_zH_zQ + TVR_q_qj_zH_zQ_zq +  TVR_Q_qj_zH + TVR_Q_qj_zH_zQ + TVR_Q_qj_zH_zQ_zq
        TVR   = np.sum(TVR_T)/(sim_time * 9)
     
    #%% 
    
        # Calculating Daily Average Total Variance           
        
        Resfile_lis.append(resfilename)
        ttime.append(totaltime)
        vzH_vl.append(vzH_v)
        vzQ_vl.append(vzQ_v) 
        vzq_vl.append(vzq)
        vzt.append(vzq_v)
        
        ATV_H_qj.append(np.sum(TV_H_qj)/sim_time) 
        ATV_H_qj_zH.append(np.sum(TV_H_qj_zH)/sim_time)
        ATV_H_qj_zH_zQ.append(np.sum(TV_H_qj_zH_zQ)/sim_time)
        ATV_H_qj_zH_zQ_zq.append(np.sum(TV_H_qj_zH_zQ_zq)/sim_time)
           
        ATV_q_qj.append(np.sum(TV_q_qj)/sim_time)
        ATV_q_qj_zH.append(np.sum(TV_q_qj_zH)/sim_time) 
        ATV_q_qj_zH_zQ.append(np.sum(TV_q_qj_zH_zQ)/sim_time)
        ATV_q_qj_zH_zQ_zq.append(np.sum(TV_q_qj_zH_zQ_zq)/sim_time)  
        
            
        ATV_Q_qj.append(np.sum(TV_Q_qj)/sim_time)
        ATV_Q_qj_zH.append(np.sum(TV_Q_qj_zH)/sim_time) 
        ATV_Q_qj_zH_zQ.append(np.sum(TV_Q_qj_zH_zQ)/sim_time) 
        ATV_Q_qj_zH_zQ_zq.append(np.sum(TV_Q_qj_zH_zQ_zq)/sim_time)
        
        ATVR.append(TVR)
            
#%% 
Res_Dict = {'Filename':Resfile_lis, 'vzH':vzH_vl, 'vzQ':vzQ_vl, 'vzq':vzq_vl, 'TVR': ATVR, 'vzt': vzt,
            "H|qj":ATV_H_qj, "H|qj zH":ATV_H_qj_zH, "H|qj zH zQ":ATV_H_qj_zH_zQ, "H|qj zH zQ zq":ATV_H_qj_zH_zQ_zq,
            "q|qj":ATV_q_qj, "q|qj zH":ATV_q_qj_zH, "q|qj zH zQ":ATV_q_qj_zH_zQ, "q|qj zH zQ zq":ATV_q_qj_zH_zQ_zq,
            "Q|qj":ATV_Q_qj, "Q|qj zH":ATV_Q_qj_zH, "Q|qj zH zQ":ATV_Q_qj_zH_zQ, "Q|qj zH zQ zq":ATV_Q_qj_zH_zQ_zq,}

Results = pd.DataFrame(Res_Dict)

# Activate for Head
# y_max = Results.groupby('vzH')['TVR'].max()
# y_min = Results.groupby('vzH')['TVR'].min()

# ya = Results[['vzH','TVR']]

# Activate for Flow
# y_max = Results.groupby('vzQ')['TVR'].max()
# y_min = Results.groupby('vzQ')['TVR'].min()

# ya = Results[['vzQ','TVR']]

# Activate for Demand
y_max = Results.groupby('vzq')['TVR'].max()
y_min = Results.groupby('vzq')['TVR'].min()

ya = Results[['vzt','TVR']]

#%%

import matplotlib.pyplot as plt

colors = plt.get_cmap('tab10').colors

fig, ax = plt.subplots(dpi=300, facecolor='white', figsize=(5, 5))
fig.suptitle('CTVR for noise in Demand Sensors. WDN: {}'.format(wdsfile[0:3])) #, fontsize=5
ax.fill_between(y_max.index, y_min, y_max, color= 'turquoise', alpha=0.5)
ax.plot(y_max.index, y_max, color = 'darkslategray', label='Max')
ax.plot(y_min.index, y_min, color = 'darkcyan', label='Min')
ax.scatter(ya['vzt'], ya['TVR'], label='CTVR per run', s = 5, c = 'crimson', alpha=0.5, zorder=3)
ax.axhline(y=1, color='crimson', linestyle='--', linewidth = 1.5, label = 'CTVR = 1')
ax.legend()

ax.set_xlabel('Noise in Demand (vzq) / lps')
ax.set_ylabel('Combined Total Variance Ratio (CTVR)')
#plt.yscale("log")

#plt.savefig('RESULTS/vz/Fiv/vzH_plot.png', dpi=300)
plt.tight_layout()
plt.show()
