# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 12:54:55 2023

@author: ibrah
"""

import wntr
import csv
import math
import numpy as np
import pandas as pd
 
wdsfile         = "MOD.inp"    
bdemfile        = "MOD_base_demands.txt"            # Base Demands
# Network model
wdsfile          = wdsfile
nn = 268

basedemandsrd   = pd.read_csv( bdemfile,   header = None)
basedemands     = basedemandsrd.T

wn = wntr.network.WaterNetworkModel(wdsfile) 

qbase = (1.2*basedemands*(1 + 0.02 * np.random.randn(1,nn))).T
demand =qbase
i = 0
a=[]
b=[]
for junction_name, node in wn.junctions():
    
    node.demand_timeseries_list[0].base_value = demand.iloc[i,0] * 0.001 # demands should be converted to cubic meters per second
    #node.demand_timeseries_list[0].base_value = 0.0005
    i += 1
    a.append(node.demand_timeseries_list[0].base_value)
    b.append(junction_name)


import wntr.graphics
import matplotlib.pyplot as plt
import matplotlib.style as style
 

# Calculate the number of links in each set
num_links = wn.num_links
set_size = num_links // 5
last_set_size = set_size + num_links % 5

# Divide links into 5 sets
n_sets = 5
link_sets = [[] for _ in range(n_sets)]
i = 0
j = 0
for pipe_name, link in wn.links():
    link_sets[j].append(link)
    i += 1
    if i >= set_size and j < n_sets-1:
        j += 1
        set_size = last_set_size if j == n_sets-1 else set_size
        i = 0

# Set different roughness coefficient for each set
roughness_coefficients = [110.0, 120.0, 130.0, 140.0, 150.0]
data = []
for i, link_set in enumerate(link_sets):
    for link in link_set:
        link_name = link._link_name
        roughness = roughness_coefficients[i]
        link.roughness = roughness
        data.append((link_name, roughness))

# Create dataframe
df = pd.DataFrame(data, columns=['Link Name', 'Roughness Coefficient'])

# Plot water network with roughness coefficient
fig, ax = plt.subplots(dpi=300)
wntr.graphics.plot_network(wn, ax=ax, link_attribute='roughness', node_size = 2,  link_cmap=['midnightblue', 'dodgerblue', 'green', 'lime', 'gold'])
plt.title('Water Network with Roughness Coefficient')
plt.show()

#Saving an updated .inp file    
wn.write_inpfile('MOD_CA.inp')



#%%

Pip_Grp = pd.read_csv('MOD_Ca_Grp.csv')


# Loop through each row in the DataFrame
for index, row in Pip_Grp.iterrows():
    # Get the pipe ID and group number for this row
    pipe_id = row['Pipe ID']
    group_number = row['Group ID']
    
    # Find the pipe with this ID in the water network model
    pipe = wn.get_link(pipe_id)
    
    # Set the roughness based on the group number
    if group_number == 1:
        pipe.roughness = 100
    elif group_number == 2:
        pipe.roughness = 110
    elif group_number == 3:
        pipe.roughness = 120
    elif group_number == 4:
        pipe.roughness = 130
    elif group_number == 5:
        pipe.roughness = 140
    else:
        print(f"Invalid group number {group_number} for pipe {pipe_id}")
        
# Create a pandas DataFrame to store the results
results_df = pd.DataFrame(columns=['Pipe ID', 'Roughness Coefficient'])

# Loop through each pipe in the water network model and add the roughness to the DataFrame
for link in wn.links.values():
    results_df = results_df.append({'Pipe ID': link._link_name, 'Roughness Coefficient': link.roughness}, ignore_index=True)

# Plot water network with roughness coefficient
fig, ax = plt.subplots(dpi=300)
wntr.graphics.plot_network(wn, ax=ax, link_attribute='roughness', node_size = 2,  link_cmap=['midnightblue', 'dodgerblue', 'green', 'lime', 'gold'])
plt.title('Water Network Divided by Roughness Coefficient Groups')
plt.show()


#Saving an updated .inp file    
wn.write_inpfile('MOD_CAG2.inp')

# import wntr.graphics
# import matplotlib.pyplot as plt


# # Calculate the number of links in each set
# num_links = wn.num_links
# set_size = num_links // 5
# last_set_size = set_size + num_links % 5

# # Divide links into 5 sets
# n_sets = 5
# link_sets = [[] for _ in range(n_sets)]
# i = 0
# for pipe_name, link in wn.links():
#     link_sets[i].append(link)
#     i = (i + 1) % n_sets

# # Set different roughness coefficient for each set
# roughness_coefficients = [110.0, 120.0, 130.0, 140.0, 150.0]
# data = []
# for i, link_set in enumerate(link_sets):
#     for link in link_set:
#         link_name = link._link_name
#         roughness = roughness_coefficients[i]
#         link.roughness = roughness
#         data.append((link_name, roughness))

# # Create dataframe
# df = pd.DataFrame(data, columns=['Link Name', 'Roughness Coefficient'])

# # Plot water network with roughness coefficient
# fig, ax = plt.subplots(dpi=300)
# wntr.graphics.plot_network(wn, ax=ax, link_attribute='roughness', node_size = 2)
# plt.title('Water Network with Roughness Coefficient')
# plt.show()

# #Saving an updated .inp file    
# wn.write_inpfile('tmp_cA.inp')
#wntr.epanet.io.InpFile.write(self, filename="tmp.inp", wn=wn, units=None)
    
    #wn.set_link_attribute(link_name, 'roughness_coefficient', 130)