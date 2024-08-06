# -*- coding: utf-8 -*-
"""
# This script is used for reading the results of Data Assimilation
# Ibrahim Miflal Fayaz - Msc Student UN-IHE
# 02/12/2022
"""

import pickle
import numpy as np
import plot
import plot_net
import TV
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('seaborn-colorblind')

#resfilename = 'DA_MOD_RT5_RH1.5_RQ1.5_Rq1.5.data'
# Ensemble number experiment Results
#resfilename = 'DA_MOD_E5_T24.data'      # 5   Ensembles
#resfilename = 'DA_MOD_E10_T24.data'    # 10  Ensembles
#resfilename = 'DA_MOD_E20_T24.data'    # 20  Ensembles
#resfilename = 'DA_MOD_E30_T24.data'    # 30  Ensembles
#resfilename = 'DA_MOD_E50_T24.data'    # 50  Ensembles
#resfilename = 'DA_MOD_E100_T24.data'   # 100 Ensembles

#resfilename = 'DA_MOD_E100_T24.data'

#resfilename = 'DA_Fiv_E5_T24.data'      # 5   Ensembles
#resfilename = 'DA_Fiv_E10_T24.data'    # 10  Ensembles
#resfilename = 'DA_Fiv_E20_T24.data'    # 20  Ensembles
#resfilename = 'DA_Fiv_E30_T24.data'    # 30  Ensembles
#resfilename = 'DA_Fiv_E50_T24.data'    # 50  Ensembles
#resfilename = 'DA_Fiv_E100_T24.data'   # 100 Ensembles

#resfilename = 'DA_MOD_E30_ALLSENSE.data'



# Ensemble number experiment Results
#resfilename = 'DA_MOD_E5.data'      # 5   Ensembles
#resfilename = 'DA_MOD_E10.data'    # 10  Ensembles
#resfilename = 'DA_MOD_E20.data'    # 20  Ensembles
#resfilename = 'DA_MOD_E30.data'    # 30  Ensembles
#resfilename = 'DA_MOD_E50.data'    # 50  Ensembles
resfilename = 'DA_MOD_E100.data'   # 100 Ensembles

with open('RESULTS/Ens/MOD/'+resfilename, 'rb') as filehandle:
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

# Plotting Total Variance for Head
x = np.arange(sim_time)
fig, ax = plt.subplots(3, figsize=(14, 10),constrained_layout=True, dpi=300)
fig.suptitle('Total Variance - No. of ensembles: {}, Network: {}, Simulation time: {}s'.format(ne, wdsfile[0:3], totaltime), fontsize=16)

ax[0].plot(x, TV_H_qj,         '.:',  color='darkred', label = "H|qj ( AVG: {:#.2g} )".format(np.sum(TV_H_qj)/24))
ax[0].plot(x, TV_H_qj_zH,      '.--', color='crimson', label = "H|qj zH ( AVG: {:#.2g} )".format(np.sum(TV_H_qj_zH)/24))
ax[0].plot(x, TV_H_qj_zH_zQ,   '.-.', color='orange',  label = "H|qj zH zQ ( AVG: {:#.2g} )".format(np.sum(TV_H_qj_zH_zQ)/24))
ax[0].plot(x, TV_H_qj_zH_zQ_zq,'.-',  color='green',   label = "H|qj zH zQ zq ( AVG: {:#.2g} )".format(np.sum(TV_H_qj_zH_zQ_zq)/24))
ax[0].set_xlabel('Time(Hrs)', fontsize=16)  # Add an x-label to the axes.
ax[0].set_ylabel('Total Variance for Head', fontsize=16)  # Add a y-label to the axes.
#ax[0].set_title("Total variance")  # Add a title to the axes.
ax[0].legend(fontsize=10);  # Add a legend.

#plt.show()
# Plotting Total Variance for Demand
#x = np.arange(sim_time)
#fig, ax = plt.subplots()

ax[1].plot(x, TV_q_qj,         '.:',  color='darkred', label = "q|qj ( AVG: {:#.2g} )".format(np.sum(TV_q_qj)/24))
ax[1].plot(x, TV_q_qj_zH,      '.--', color='crimson', label = "q|qj zH ( AVG: {:#.2g} )".format(np.sum(TV_q_qj_zH)/24))
ax[1].plot(x, TV_q_qj_zH_zQ,   '.-.', color='orange',  label = "q|qj zH zQ ( AVG: {:#.2g} )".format(np.sum(TV_q_qj_zH_zQ)/24))
ax[1].plot(x, TV_q_qj_zH_zQ_zq,'.-',  color='green',   label = "q|qj zH zQ zq ( AVG: {:#.2g} )".format(np.sum(TV_q_qj_zH_zQ_zq)/24))
ax[1].set_xlabel('Time(Hrs)', fontsize=16)  # Add an x-label to the axes.
ax[1].set_ylabel('Total Variance for Demand', fontsize=16)  # Add a y-label to the axes.
#ax[1].set_title("Total variance ")  # Add a title to the axes.
ax[1].legend(fontsize=10);  # Add a legend.

#plt.show()
# Plotting Total Variance for Demand
#x = np.arange(sim_time)
#fig, ax = plt.subplots()

ax[2].plot(x, TV_Q_qj,         '.:',  color='darkred', label = "Q|qj ( AVG: {:#.2g} )".format(np.sum(TV_Q_qj)/24))
ax[2].plot(x, TV_Q_qj_zH,      '.--', color='crimson', label = "Q|qj zH ( AVG: {:#.2g} )".format(np.sum(TV_Q_qj_zH)/24))
ax[2].plot(x, TV_Q_qj_zH_zQ,   '.-.', color='orange',  label = "Q|qj zH zQ ( AVG: {:#.2g} )".format(np.sum(TV_Q_qj_zH_zQ)/24))
ax[2].plot(x, TV_Q_qj_zH_zQ_zq,'.-',  color='green',   label = "Q|qj zH zQ zq ( AVG: {:#.2g} )".format(np.sum(TV_Q_qj_zH_zQ_zq)/24))
ax[2].set_xlabel('Time(Hrs)', fontsize=16)  # Add an x-label to the axes.
ax[2].set_ylabel('Total Variance for Flow', fontsize=16)  # Add a y-label to the axes.
#ax[2].set_title("Total variance" )  # Add a title to the axes.
ax[2].legend(fontsize=10);  # Add a legend.

plt.show()
#%% 
qname = 'Modena, Monitoring Network of Head Sensors'

plot_net.only_monitoring(wdsfile, nn, npi, H_observ, qname)

qname = 'Modena, Monitoring Network of Flow Sensors'
plot_net.linkonly_monitoring(wdsfile, nn, npi, Q_observ, qname)

qname = 'Modena, Monitoring Network of Demand Sensors'

plot_net.only_monitoring(wdsfile, nn, npi, H_observ, qname)
# for ti in range(0, sim_time, 24):
#     qname   = 'q|qj,zH,zQ, zq error, No. of ensembles: {}, Timestep: {}'.format(ne, ti)
#     qrange  = [None, None]
#     plot_net.node_error(q_qj_zH_zQ_zq_ti[:,:,ti], qobs_ti[:,:,ti], qname, qrange, wdsfile)

#%% 
# from bokeh.plotting import figure, show
# from bokeh.layouts import column

# for no in range(100,102):
#     Node = no
#     x = np.arange(24)
#     yh1 = np.max(H_qj_zH_zQ_zq_ti[Node,:,:], axis = 0)
#     yl1 = np.min(H_qj_zH_zQ_zq_ti[Node,:,:], axis = 0)
#     Yobs = Hobs_ti[Node, 0, :]
    
#     f1 = figure(title = 'H|qj, zH, zQ, zq error for Node: {}, No. of ensembles: {}'.format(Node, ne), x_axis_label=' Time (hr) ', y_axis_label= 'H|qj, zH, zQ, zq error', height=800, width = 2440)
#     f1.varea(x=x, y1=yh1, y2=yl1 ,fill_color="red", fill_alpha=0.5)
#     f1.line(x, yl1, legend_label="Maximum Error.", color="crimson", line_width=0.2)
#     f1.line(x, yh1, legend_label="Minimum Error.", color="darkred", line_width=0.2)
#     f1.legend.location = "bottom_right"
#     f1.circle(x, Yobs, color='blue', size = 5)
#     show(column(f1))

#%% 
H_qj_zH_zQ_zq_avg = np.mean(H_qj_zH_zQ_zq_ti, axis = 2)
Hobs_avg = np.mean(Hobs_ti, axis = 2)

Q_qj_zH_zQ_zq_avg = np.mean(Q_qj_zH_zQ_zq_ti, axis = 2)
Qobs_avg = np.mean(Qobs_ti, axis = 2)

q_qj_zH_zQ_zq_avg = np.mean(q_qj_zH_zQ_zq_ti, axis = 2)
qobs_avg = np.mean(qobs_ti, axis = 2)

#%%
for ti in range(1):
    qname   = ''
    qrange  = [None, None]
    plot_net.node_error(H_qj_zH_zQ_zq_avg, Hobs_avg, qname, qrange, wdsfile)

#%%

for ti in range(1):
    qname   = 'Modena, 24 Hour Average Ensemble error Q|qj,zH,zQ,zq, No. of ensembles: {}'.format(ne)
    qrange  = [None, None]
    plot_net.node_error(Q_qj_zH_zQ_zq_avg, Qobs_avg, qname, qrange, wdsfile)

for ti in range(1):
    qname   = 'Modena, 24 Hour Average Ensemble error q|qj,zH,zQ,zq, No. of ensembles: {}'.format(ne)
    qrange  = [None, None]
    plot_net.node_error(q_qj_zH_zQ_zq_avg, qobs_avg, qname, qrange, wdsfile)
# #%%     
# for ti in range(0, sim_time, 7):
#     qname   = 'q|qj,zH,zQ, zq error, No. of ensembles: {}, Timestep: {}'.format(ne, ti)
#     qrange  = [None, None]
#     plot_net.node_error(q_qj_zH_zQ_zq_ti[:,:,ti], qobs_ti[:,:,ti], qname, qrange, wdsfile)


# #%% 
# for ti in range(0, sim_time, 7):
#     qname   = 'Q|qj,zH,zQ, zq error, No. of ensembles: {}, Timestep: {}'.format(ne, ti)
#     qrange  = [None, None]
#     plot_net.link_error(Q_qj_zH_zQ_zq_ti[:,:,ti], Qobs_ti[:,:,ti], qname, qrange, wdsfile)

#%% 
for ti in range(12):
    Hnames = ['H|qj', 'H|qj,zH', 'H|qj,zH,zQ', 'H|qj,zH,zQ,zq']
    Hrange = [0, 0];
    plot.ensembles_errors(H_qj_ti[:,:,ti], H_qj_zH_ti[:,:,ti], H_qj_zH_zQ_ti[:,:,ti], H_qj_zH_zQ_zq_ti[:,:,ti], Hnames, Hrange, Hobs_ti[:,:,ti].flatten(), ti, 130)
    
#%% 
for ti in range(1):
    Hnames = ['q|qj', 'q|qj,zH', 'q|qj,zH,zQ', 'q|qj,zH,zQ,zq']
    Hrange = [0, 0];
    plot.ensembles_errors(q_qj_ti[:,:,ti], q_qj_zH_ti[:,:,ti], q_qj_zH_zQ_ti[:,:,ti], q_qj_zH_zQ_zq_ti[:,:,ti], Hnames, Hrange, qobs_ti[:,:,ti].flatten(), ti, 130)
    
#%% 
for ti in range(1):
    Hnames = ['Q|qj', 'Q|qj,zH', 'Q|qj,zH,zQ', 'Q|qj,zH,zQ,zq']
    Hrange = [0, 0];
    plot.ensembles_errors(Q_qj_ti[:,:,ti], Q_qj_zH_ti[:,:,ti], Q_qj_zH_zQ_ti[:,:,ti], Q_qj_zH_zQ_zq_ti[:,:,ti], Hnames, Hrange, Qobs_ti[:,:,ti].flatten(), ti, 130)

#%%

#Plotting Total Variance for No. Ens
#ensembles_errors(y1,y2,y3,y4,ynames,yrange,yobs)

y1 = H_qj_ti[:,:,ti]
y2 = H_qj_zH_ti[:,:,ti]
y3 = H_qj_zH_zQ_ti[:,:,ti]
y4 = H_qj_zH_zQ_zq_ti[:,:,ti]
ynames = Hnames
yrange = (-1.5, 1.5)
yobs = Hobs_ti[:,:,ti].flatten()


# fig.legend( y1,     # The line objects
#            loc="upper center",   # Position of legend
#            borderaxespad=2.5,    # Small spacing around legend box
#            title ="Average total variance",  # Title for the legend
#            ncol = 12)

#plt.show()
    
# ax[0].plot(xx, y1, '.--', label = y1.columns.values)
# # ax[0,0].set_xlabel('No. of Ensembles')  # Add an x-label to the axes.
# ax[0].set_ylabel('H|qj')  # Add a y-label to the axes.
# #ax[0].set_title("Total variance")  # Add a title to the axes.
# ax[0].legend();  # Add a legend.

# #plt.show()
# # Plotting Total Variance for Demand
# #x = np.arange(sim_time)
# #fig, ax = plt.subplots()

# ax[1].plot(x, y2, '.--', label = y2.columns.values)
# # ax[1,0].set_xlabel('No. of Ensembles')  # Add an x-label to the axes.
# ax[1].set_ylabel('Total Variance for Demand')  # Add a y-label to the axes.
# #ax[1].set_title("Total variance ")  # Add a title to the axes.
# ax[1].legend();  # Add a legend.

# #plt.show()
# # Plotting Total Variance for Demand
# #x = np.arange(sim_time)
# #fig, ax = plt.subplots()

# ax[2].plot(x, y3, '.--', label = y3.columns.values)
# ax[2].set_xlabel('No. of Ensembles')  # Add an x-label to the axes.
# ax[2].set_ylabel('Total Variance for Flow')  # Add a y-label to the axes.
# #ax[2].set_title("Total variance" )  # Add a title to the axes.
# ax[2].legend();  # Add a legend.

# # ax[1,1].plot(x, y4, label = 'Simulation time')
# # ax[1,1].set_xlabel('No. of Ensembles')  # Add an x-label to the axes.
# # ax[1,1].set_ylabel('Time (s)')  # Add a y-label to the axes.
# #ax[2].set_title("Total variance" )  # Add a title to the axes.
# #ax[1].legend();  # Add a legend

# plt.show()
# #%% 
# for ti in range(1):
#     Hnames = ['Q{q{j}}', 'Q|{q{j},z{H}}', 'Q|{q{j},z{H},z{Q}}', 'Q|{q{j},z{H},z{Q},z{q}}']
#     Hrange = [+30.0, +80.0];
#     plot.ensembles_errors(Q_qj_ti[:,:,ti], Q_qj_zH_ti[:,:,ti], Q_qj_zH_zQ_ti[:,:,ti], Q_qj_zH_zQ_zq_ti[:,:,ti], Hnames, Hrange, Qobs_ti[:,:,ti].flatten())

# #%% 
# for ti in range(2):
#     Hnames = ['q{q{j}}', 'q|{q{j},z{H}}', 'q|{q{j},z{H},z{Q}}', 'q|{q{j},z{H},z{Q},z{q}}']
#     Hrange = [+30.0, +80.0];
#     plot.ensembles_errors(q_qj_ti[:,:,ti], q_qj_zH_ti[:,:,ti], q_qj_zH_zQ_ti[:,:,ti], q_qj_zH_zQ_zq_ti[:,:,ti], Hnames, Hrange, qobs_ti[:,:,ti].flatten())

# #%% 
# for ti in range(sim_time):
#     Hnames = ['H_{q_{j}}', 'H|_{q_{j},z_{H}}', 'H|_{q_{j},z_{H},z_{Q}}', 'H|_{q_{j},z_{H},z_{Q},z_{q}}']
#     Hrange_err = [-5.0, +5.0];
#     plot.ensembles_errors_all(H_qj_ti[:,:,ti], H_qj_zH_ti[:,:,ti], H_qj_zH_zQ_ti[:,:,ti], H_qj_zH_zQ_zq_ti[:,:,ti], Hnames, Hrange_err, Hobs_ti[:,:,ti].flatten())


#%%
from bokeh.plotting import figure, show
from bokeh.layouts import column

#for no in range(9,11):
no = 55    
Node = no
x = np.arange(24)
yh1 = np.max(H_qj_zH_zQ_zq_ti[Node,:,:], axis = 0)
yl1 = np.min(H_qj_zH_zQ_zq_ti[Node,:,:], axis = 0)
Yobs = Hobs_ti[Node, 0, :]

f1 = figure(title = 'q|qj, zH, zQ, zq error for Node: {}, No. of ensembles: {}'.format(Node, ne), x_axis_label=' Time (hr) ', y_axis_label= 'H|qj, zH, zQ, zq error', height=800, width = 2440)
f1.varea(x=x, y1=yh1, y2=yl1 ,fill_color="red", fill_alpha=0.5)
f1.line(x, yl1, legend_label="Maximum Error.", color="crimson", line_width=0.2)
f1.line(x, yh1, legend_label="Minimum Error.", color="darkred", line_width=0.2)
f1.legend.location = "bottom_right"
f1.circle(x, Yobs, color='blue', size = 5)

show(f1)
    
