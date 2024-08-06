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
RzH_vl               = [] 
RzQ_vl               = [] 
Rzq_vl               = []
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


for RzH_v in [0.001, 0.1, 0.2, 0.3, 0.5, 1]:
    for RzQ_v in [0.001, 1, 2, 3, 5, 10]:
        for Rzq_v in [0.001, 0.1, 0.2, 0.3, 0.5, 1]:
            resfilename = 'DA_Fiv_RH{}_RQ{}_Rq{}.data'.format(RzH_v, RzQ_v, Rzq_v)
            #resfilename = 'DA_MOD_RzH{}_RzQ{}_Rzq{}.data'.format(RzH_v, RzQ_v, Rzq_v)


            with open('RESULTS/Rz/Fiv_1/'+resfilename, 'rb') as filehandle:
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
            RzH_vl.append(RzH_v)
            RzQ_vl.append(RzQ_v) 
            Rzq_vl.append(Rzq_v)
            
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
Res_Dict = {'Filename':Resfile_lis, 'RzH':RzH_vl, 'RzQ':RzQ_vl, 'Rzq':Rzq_vl, 'TVR': ATVR,
            "H|qj":ATV_H_qj, "H|qj zH":ATV_H_qj_zH, "H|qj zH zQ":ATV_H_qj_zH_zQ, "H|qj zH zQ zq":ATV_H_qj_zH_zQ_zq,
            "q|qj":ATV_q_qj, "q|qj zH":ATV_q_qj_zH, "q|qj zH zQ":ATV_q_qj_zH_zQ, "q|qj zH zQ zq":ATV_q_qj_zH_zQ_zq,
            "Q|qj":ATV_Q_qj, "Q|qj zH":ATV_Q_qj_zH, "Q|qj zH zQ":ATV_Q_qj_zH_zQ, "Q|qj zH zQ zq":ATV_Q_qj_zH_zQ_zq,}

Results = pd.DataFrame(Res_Dict)

#%%
Results_var = Results.loc[:, Results.columns != 'Filename']
Results_var = Results_var[["RzH", "Rzq", "RzQ", "TVR"]]
#Results_var = Results_var[["RzH", "Rzq", "RzQ", "H|qj", "H|qj zH",  "H|qj zH zQ", "H|qj zH zQ zq"]]
#Results_var = Results_var[["RzH", "Rzq", "RzQ", "q|qj", "q|qj zH",  "q|qj zH zQ", "q|qj zH zQ zq"]]
#Results_var = Results_var[["RzH", "Rzq", "RzQ", "Q|qj", "Q|qj zH",  "Q|qj zH zQ", "Q|qj zH zQ zq"]]

RzH_va = [0.001, 0.1, 0.2, 0.3, 0.5, 1 ]
RzQ_va = [0.001,   1,   2,   3,   5, 10]
Rzq_va = [0.001, 0.1, 0.2, 0.3, 0.5, 1 ]

x_var     = Rzq_va
x_var_str = 'Rzq'

cons_var1 = RzH_va
cons_str1 = 'RzH'

cons_var2 = RzQ_va
cons_str2 = 'RzQ'
style = ['.--', '.:', '.-.', '^-']

fig, ax = plt.subplots(6,6, figsize=(18, 12), dpi = 300)
if x_var_str == 'Rzq':
    x_str     = 'Precision of Demand sensor (lps)'
    fig.suptitle('Average Total Variance ratio for varying Precision of Demand sensors. WDN: {}'.format(wdsfile[0:3]), fontsize=16)
    
if x_var_str == 'RzH':
    x_str     = 'Precision of Head sensor (m)'
    fig.suptitle('Average Total Variance ratio for varying Precision of Head sensors. WDN: {}'.format(wdsfile[0:3]), fontsize=16)
    
if x_var_str == 'RzQ':
    x_str     = 'Precision of Flow sensor (lps)'
    fig.suptitle('Average Total Variance ratio for varying Precision of Flow sensors. WDN: {}'.format(wdsfile[0:3]), fontsize=16)

y1  = Results_var.query(cons_str1 +' ==' + str(cons_var1[0]) +'and ' + cons_str2 + ' ==' + str(cons_var2[0])).iloc[: , 3:]
y2  = Results_var.query(cons_str1 +' ==' + str(cons_var1[0]) +'and ' + cons_str2 + ' ==' + str(cons_var2[1])).iloc[: , 3:]
y3  = Results_var.query(cons_str1 +' ==' + str(cons_var1[0]) +'and ' + cons_str2 + ' ==' + str(cons_var2[2])).iloc[: , 3:]
y4  = Results_var.query(cons_str1 +' ==' + str(cons_var1[0]) +'and ' + cons_str2 + ' ==' + str(cons_var2[3])).iloc[: , 3:]
y5  = Results_var.query(cons_str1 +' ==' + str(cons_var1[0]) +'and ' + cons_str2 + ' ==' + str(cons_var2[4])).iloc[: , 3:]
y6  = Results_var.query(cons_str1 +' ==' + str(cons_var1[0]) +'and ' + cons_str2 + ' ==' + str(cons_var2[5])).iloc[: , 3:]
y7  = Results_var.query(cons_str1 +' ==' + str(cons_var1[1]) +'and ' + cons_str2 + ' ==' + str(cons_var2[0])).iloc[: , 3:]
y8  = Results_var.query(cons_str1 +' ==' + str(cons_var1[1]) +'and ' + cons_str2 + ' ==' + str(cons_var2[1])).iloc[: , 3:]
y9  = Results_var.query(cons_str1 +' ==' + str(cons_var1[1]) +'and ' + cons_str2 + ' ==' + str(cons_var2[2])).iloc[: , 3:]
y10 = Results_var.query(cons_str1 +' ==' + str(cons_var1[1]) +'and ' + cons_str2 + ' ==' + str(cons_var2[3])).iloc[: , 3:]
y11 = Results_var.query(cons_str1 +' ==' + str(cons_var1[1]) +'and ' + cons_str2 + ' ==' + str(cons_var2[4])).iloc[: , 3:]
y12 = Results_var.query(cons_str1 +' ==' + str(cons_var1[1]) +'and ' + cons_str2 + ' ==' + str(cons_var2[5])).iloc[: , 3:]
y13 = Results_var.query(cons_str1 +' ==' + str(cons_var1[2]) +'and ' + cons_str2 + ' ==' + str(cons_var2[0])).iloc[: , 3:]
y14 = Results_var.query(cons_str1 +' ==' + str(cons_var1[2]) +'and ' + cons_str2 + ' ==' + str(cons_var2[1])).iloc[: , 3:]
y15 = Results_var.query(cons_str1 +' ==' + str(cons_var1[2]) +'and ' + cons_str2 + ' ==' + str(cons_var2[2])).iloc[: , 3:]
y16 = Results_var.query(cons_str1 +' ==' + str(cons_var1[2]) +'and ' + cons_str2 + ' ==' + str(cons_var2[3])).iloc[: , 3:]
y17 = Results_var.query(cons_str1 +' ==' + str(cons_var1[2]) +'and ' + cons_str2 + ' ==' + str(cons_var2[4])).iloc[: , 3:]
y18 = Results_var.query(cons_str1 +' ==' + str(cons_var1[2]) +'and ' + cons_str2 + ' ==' + str(cons_var2[5])).iloc[: , 3:]
y19 = Results_var.query(cons_str1 +' ==' + str(cons_var1[3]) +'and ' + cons_str2 + ' ==' + str(cons_var2[0])).iloc[: , 3:]
y20 = Results_var.query(cons_str1 +' ==' + str(cons_var1[3]) +'and ' + cons_str2 + ' ==' + str(cons_var2[1])).iloc[: , 3:]
y21 = Results_var.query(cons_str1 +' ==' + str(cons_var1[3]) +'and ' + cons_str2 + ' ==' + str(cons_var2[2])).iloc[: , 3:]
y22 = Results_var.query(cons_str1 +' ==' + str(cons_var1[3]) +'and ' + cons_str2 + ' ==' + str(cons_var2[3])).iloc[: , 3:]
y23 = Results_var.query(cons_str1 +' ==' + str(cons_var1[3]) +'and ' + cons_str2 + ' ==' + str(cons_var2[4])).iloc[: , 3:]
y24 = Results_var.query(cons_str1 +' ==' + str(cons_var1[3]) +'and ' + cons_str2 + ' ==' + str(cons_var2[5])).iloc[: , 3:]
y25 = Results_var.query(cons_str1 +' ==' + str(cons_var1[4]) +'and ' + cons_str2 + ' ==' + str(cons_var2[0])).iloc[: , 3:]
y26 = Results_var.query(cons_str1 +' ==' + str(cons_var1[4]) +'and ' + cons_str2 + ' ==' + str(cons_var2[1])).iloc[: , 3:]
y27 = Results_var.query(cons_str1 +' ==' + str(cons_var1[4]) +'and ' + cons_str2 + ' ==' + str(cons_var2[2])).iloc[: , 3:]
y28 = Results_var.query(cons_str1 +' ==' + str(cons_var1[4]) +'and ' + cons_str2 + ' ==' + str(cons_var2[3])).iloc[: , 3:]
y29 = Results_var.query(cons_str1 +' ==' + str(cons_var1[4]) +'and ' + cons_str2 + ' ==' + str(cons_var2[4])).iloc[: , 3:]
y30 = Results_var.query(cons_str1 +' ==' + str(cons_var1[4]) +'and ' + cons_str2 + ' ==' + str(cons_var2[5])).iloc[: , 3:]
y31 = Results_var.query(cons_str1 +' ==' + str(cons_var1[5]) +'and ' + cons_str2 + ' ==' + str(cons_var2[0])).iloc[: , 3:]
y32 = Results_var.query(cons_str1 +' ==' + str(cons_var1[5]) +'and ' + cons_str2 + ' ==' + str(cons_var2[1])).iloc[: , 3:]
y33 = Results_var.query(cons_str1 +' ==' + str(cons_var1[5]) +'and ' + cons_str2 + ' ==' + str(cons_var2[2])).iloc[: , 3:]
y34 = Results_var.query(cons_str1 +' ==' + str(cons_var1[5]) +'and ' + cons_str2 + ' ==' + str(cons_var2[3])).iloc[: , 3:]
y35 = Results_var.query(cons_str1 +' ==' + str(cons_var1[5]) +'and ' + cons_str2 + ' ==' + str(cons_var2[4])).iloc[: , 3:]
y36 = Results_var.query(cons_str1 +' ==' + str(cons_var1[5]) +'and ' + cons_str2 + ' ==' + str(cons_var2[5])).iloc[: , 3:]


ax[0,0].plot(x_var, y1, '.--')
ax[0,1].plot(x_var, y2, '.--')
ax[0,2].plot(x_var, y3, '.--')
ax[0,3].plot(x_var, y4, '.--')
ax[0,4].plot(x_var, y5, '.--')
ax[0,5].plot(x_var, y6, '.--')
ax[1,0].plot(x_var, y7, '.--')
ax[1,1].plot(x_var, y8, '.--')
ax[1,2].plot(x_var, y9, '.--')
ax[1,3].plot(x_var, y10, '.--')
ax[1,4].plot(x_var, y11, '.--')
ax[1,5].plot(x_var, y12, '.--')
ax[2,0].plot(x_var, y13, '.--')
ax[2,1].plot(x_var, y14, '.--')
ax[2,2].plot(x_var, y15, '.--')
ax[2,3].plot(x_var, y16, '.--')
ax[2,4].plot(x_var, y17, '.--')
ax[2,5].plot(x_var, y18, '.--')
ax[3,0].plot(x_var, y19, '.--')
ax[3,1].plot(x_var, y20, '.--')
ax[3,2].plot(x_var, y21, '.--')
ax[3,3].plot(x_var, y22, '.--')
ax[3,4].plot(x_var, y23, '.--')
ax[3,5].plot(x_var, y24, '.--')
ax[4,0].plot(x_var, y25, '.--')
ax[4,1].plot(x_var, y26, '.--')
ax[4,2].plot(x_var, y27, '.--')
ax[4,3].plot(x_var, y28, '.--')
ax[4,4].plot(x_var, y29, '.--')
ax[4,5].plot(x_var, y30, '.--')
ax[5,0].plot(x_var, y31, '.--')
ax[5,1].plot(x_var, y32, '.--')
ax[5,2].plot(x_var, y33, '.--')
ax[5,3].plot(x_var, y34, '.--')
ax[5,4].plot(x_var, y35, '.--')
ax[5,5].plot(x_var, y36, '.--')

ax[5,0].set_xlabel(x_str)
ax[5,1].set_xlabel(x_str)
ax[5,2].set_xlabel(x_str)
ax[5,3].set_xlabel(x_str)
ax[5,4].set_xlabel(x_str)
ax[5,5].set_xlabel(x_str)

ax[0,0].set_ylabel('Average Total Variance')
ax[1,0].set_ylabel('Average Total Variance')
ax[2,0].set_ylabel('Average Total Variance')
ax[3,0].set_ylabel('Average Total Variance')
ax[4,0].set_ylabel('Average Total Variance')
ax[5,0].set_ylabel('Average Total Variance')


ax[0,0].legend(title=cons_str1 +': ' + str(0) +' and '+ cons_str2 +': ' + str(0))
ax[0,1].legend(title=cons_str1 +': ' + str(0) +' and '+ cons_str2 +': ' + str(cons_var2[1]))
ax[0,2].legend(title=cons_str1 +': ' + str(0) +' and '+ cons_str2 +': ' + str(cons_var2[2]))
ax[0,3].legend(title=cons_str1 +': ' + str(0) +' and '+ cons_str2 +': ' + str(cons_var2[3]))
ax[0,4].legend(title=cons_str1 +': ' + str(0) +' and '+ cons_str2 +': ' + str(cons_var2[4]))
ax[0,5].legend(title=cons_str1 +': ' + str(0) +' and '+ cons_str2 +': ' + str(cons_var2[5]))
ax[1,0].legend(title=cons_str1 +': ' + str(cons_var1[1]) +' and '+ cons_str2 +': ' + str(0))
ax[1,1].legend(title=cons_str1 +': ' + str(cons_var1[1]) +' and '+ cons_str2 +': ' + str(cons_var2[1]))
ax[1,2].legend(title=cons_str1 +': ' + str(cons_var1[1]) +' and '+ cons_str2 +': ' + str(cons_var2[2]))
ax[1,3].legend(title=cons_str1 +': ' + str(cons_var1[1]) +' and '+ cons_str2 +': ' + str(cons_var2[3]))
ax[1,4].legend(title=cons_str1 +': ' + str(cons_var1[1]) +' and '+ cons_str2 +': ' + str(cons_var2[4]))
ax[1,5].legend(title=cons_str1 +': ' + str(cons_var1[1]) +' and '+ cons_str2 +': ' + str(cons_var2[5]))
ax[2,0].legend(title=cons_str1 +': ' + str(cons_var1[2]) +' and '+ cons_str2 +': ' + str(0))
ax[2,1].legend(title=cons_str1 +': ' + str(cons_var1[2]) +' and '+ cons_str2 +': ' + str(cons_var2[1]))
ax[2,2].legend(title=cons_str1 +': ' + str(cons_var1[2]) +' and '+ cons_str2 +': ' + str(cons_var2[2]))
ax[2,3].legend(title=cons_str1 +': ' + str(cons_var1[2]) +' and '+ cons_str2 +': ' + str(cons_var2[3]))
ax[2,4].legend(title=cons_str1 +': ' + str(cons_var1[2]) +' and '+ cons_str2 +': ' + str(cons_var2[4]))
ax[2,5].legend(title=cons_str1 +': ' + str(cons_var1[2]) +' and '+ cons_str2 +': ' + str(cons_var2[5]))
ax[3,0].legend(title=cons_str1 +': ' + str(cons_var1[3]) +' and '+ cons_str2 +': ' + str(0))
ax[3,1].legend(title=cons_str1 +': ' + str(cons_var1[3]) +' and '+ cons_str2 +': ' + str(cons_var2[1]))
ax[3,2].legend(title=cons_str1 +': ' + str(cons_var1[3]) +' and '+ cons_str2 +': ' + str(cons_var2[2]))
ax[3,3].legend(title=cons_str1 +': ' + str(cons_var1[3]) +' and '+ cons_str2 +': ' + str(cons_var2[3]))
ax[3,4].legend(title=cons_str1 +': ' + str(cons_var1[3]) +' and '+ cons_str2 +': ' + str(cons_var2[4]))
ax[3,5].legend(title=cons_str1 +': ' + str(cons_var1[3]) +' and '+ cons_str2 +': ' + str(cons_var2[5]))
ax[4,0].legend(title=cons_str1 +': ' + str(cons_var1[4]) +' and '+ cons_str2 +': ' + str(0))
ax[4,1].legend(title=cons_str1 +': ' + str(cons_var1[4]) +' and '+ cons_str2 +': ' + str(cons_var2[1]))
ax[4,2].legend(title=cons_str1 +': ' + str(cons_var1[4]) +' and '+ cons_str2 +': ' + str(cons_var2[2]))
ax[4,3].legend(title=cons_str1 +': ' + str(cons_var1[4]) +' and '+ cons_str2 +': ' + str(cons_var2[3]))
ax[4,4].legend(title=cons_str1 +': ' + str(cons_var1[4]) +' and '+ cons_str2 +': ' + str(cons_var2[4]))
ax[4,5].legend(title=cons_str1 +': ' + str(cons_var1[4]) +' and '+ cons_str2 +': ' + str(cons_var2[5]))
ax[5,0].legend(title=cons_str1 +': ' + str(cons_var1[5]) +' and '+ cons_str2 +': ' + str(0))
ax[5,1].legend(title=cons_str1 +': ' + str(cons_var1[5]) +' and '+ cons_str2 +': ' + str(cons_var2[1]))
ax[5,2].legend(title=cons_str1 +': ' + str(cons_var1[5]) +' and '+ cons_str2 +': ' + str(cons_var2[2]))
ax[5,3].legend(title=cons_str1 +': ' + str(cons_var1[5]) +' and '+ cons_str2 +': ' + str(cons_var2[3]))
ax[5,4].legend(title=cons_str1 +': ' + str(cons_var1[5]) +' and '+ cons_str2 +': ' + str(cons_var2[4]))
ax[5,5].legend(title=cons_str1 +': ' + str(cons_var1[5]) +' and '+ cons_str2 +': ' + str(cons_var2[5]))

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

fig.legend( y1,     # The line objects
           loc="upper center",   # Position of legend
           borderaxespad=2.5,    # Small spacing around legend box
           title ="Average total variance",  # Title for the legend
           ncol = 12)
plt.tight_layout()
fig.subplots_adjust(top=0.92)

plt.show()
figname = "DA_{}_X{}_Y1{}_Y2{}.png".format(wdsfile[0:3], x_var, cons_str1, cons_str2)


H_observ = Sim_Res[0][5]
q_observ = Sim_Res[0][6]
Q_observ = Sim_Res[0][7]
#%%
qname= 'network'
plot_net.monitoring(wdsfile, nn, npi, q_observ, H_observ, Q_observ)

#plt.savefig('FIGURES/' + figname, dpi='figure', format= 'jpeg')
#plt.yscale("log") 
#%%
qname = 'Five Reservoirs, Monitoring Network of Head Sensors'

plot_net.only_monitoring(wdsfile, nn, npi, H_observ, qname)

qname = 'Five Reservoirs, Monitoring Network of Demand Sensors'

plot_net.only_monitoring(wdsfile, nn, npi, q_observ, qname)

qname = 'Five Reservoirs, Monitoring Network of Flow Sensors'
plot_net.linkonly_monitoring(wdsfile, nn, npi, Q_observ, qname)

#%%
fig, ax1 = plt.subplots(figsize=(12, 8), dpi = 300)
if x_var_str == 'Rzq':
    x_str     = 'Precision of Demand sensor (lps)'
   # fig.suptitle('Average Total Variance ratio for varying precision of Demand sensors. WDN: {}'.format(wdsfile[0:3]), fontsize=12)
    
if x_var_str == 'RzH':
    x_str     = 'Precision of Head sensor (m)'
    #fig.suptitle('Average Total Variance ratio for varying precision of Head sensors. WDN: {}'.format(wdsfile[0:3]), fontsize=12)
    
if x_var_str == 'RzQ':
    x_str     = 'Precision of Flow sensor (lps)'
    #fig.suptitle('Average Total Variance ratio for varying precision of Flow sensors. WDN: {}'.format(wdsfile[0:3]), fontsize=12)


y100  = Results_var.query(cons_str1 +' ==' + str(cons_var1[1]) +'and ' + cons_str2 + ' ==' + str(cons_var2[1])).iloc[: , 3:]
ax1.plot(x_var, y100, '.--', linewidth=3, markersize=15)
ax1.set_xlabel(x_str, fontsize=16)
ax1.set_ylabel('Average Total Variance', fontsize=16)
ax1.legend(title=cons_str1 +': ' + str(cons_var1[1]) +' and '+ cons_str2 +': ' + str(cons_var2[1]), fontsize=16)
fig.legend( y1,     # The line objects
           loc="upper center",   # Position of legend
           borderaxespad=0.5,    # Small spacing around legend box
           title ="Average total variance",  # Title for the legend
           ncol = 12)
plt.tight_layout()
fig.subplots_adjust(top=0.92)


plt.show()