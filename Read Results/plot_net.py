# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 12:41:52 2022

@author: ibrah
"""
def node_error(y1, yobs, qname, qrange, wdsfile):

    import numpy as np
    import pandas as pd
    import wntr
    import matplotlib.pyplot as plt
    import matplotlib.style as style
    style.use('seaborn-colorblind')
    
    fig, ax1 = plt.subplots(figsize=(5, 5), constrained_layout=True, dpi=300)                                
    
    #Importing Variables
    wdsfile      =      wdsfile                                        # Import WDS file
    wn = wntr.network.WaterNetworkModel(wdsfile) 
    
    ylx = np.min(y1, axis = 1) 
    yl1 = ylx[:,None] - yobs
    yhx = np.max(y1, axis = 1)
    yh1 = yhx[:,None] - yobs
    yerror = np.subtract(yh1, yl1)
    
    #D Variable Shape Correction
    n, m = np.shape(yerror)                                           # Getting Shape of variable
    
    ylen = len(wn.node_name_list)                                     # Shape of wds nodes (requirement to plot)
    jlen = len(wn.junction_name_list)                                 # Finding variable shape (missing nodes to be added)
    lastrows = np.zeros([ylen-jlen, m])                               # Missing nodes to be added as zeros
    
    plot_ens = np.vstack((yerror, lastrows))                          # Array Stacked
    
    # Plotting each ensemble seperately
    
    plot_ens = pd.DataFrame(plot_ens[:,0])                            # Isolating each ensemble values
    plot_ens_lis = plot_ens.values.T.tolist()
    # Plotting
    plot_var = dict(zip(wn.node_name_list, plot_ens_lis[0])) # Setting Variable to required format as a dictionary for WNTR
    ax = wntr.graphics.network.plot_network(wn, node_attribute = plot_var, 
                                                    #node_attribute_name = qname,
                                                    node_size= 30,
                                                    node_range = qrange,
                                                    title= qname, ax = ax1,
                                                    add_colorbar=False)    # Plotting
    
    return ax


def link_error(y1, yobs, qname, qrange, wdsfile):

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import wntr
    
                                 
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True, dpi=300)
    #Importing Variables
    wdsfile      =      wdsfile                                        # Import WDS file
    wn = wntr.network.WaterNetworkModel(wdsfile) 
    
    ylx = np.min(y1, axis = 1) 
    yl1 = ylx[:,None] - yobs
    yhx = np.max(y1, axis = 1)
    yh1 = yhx[:,None] - yobs
    yerror = np.subtract(yh1, yl1)
    
    #D Variable Shape Correction
    n, m = np.shape(yerror)                                           # Getting Shape of variable
    
    ylen = len(wn.link_name_list)                                     # Shape of wds nodes (requirement to plot)
    jlen = len(wn.pipe_name_list)                                     # Finding variable shape (missing nodes to be added)
    lastrows = np.zeros([ylen-jlen, m])                               # Missing nodes to be added as zeros
    
    plot_ens = np.vstack((yerror, lastrows))                          # Array Stacked
    
    # Plotting each ensemble seperately
    
    plot_ens = pd.DataFrame(plot_ens[:,0])                            # Isolating each ensemble values
    plot_ens_lis = plot_ens.values.T.tolist()
    # Plotting
    plot_var = dict(zip(wn.link_name_list, plot_ens_lis[0])) # Setting Variable to required format as a dictionary for WNTR
    ax = wntr.graphics.network.plot_network(wn, link_attribute = plot_var, link_width =3, title= qname, node_size = 5, node_alpha = 0.5, ax= ax)    # Plotting
    
    return ax

def link_TV(y1, qname, wdsfile):

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import wntr
    
                                 
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True, dpi=300)
    #Importing Variables
    wdsfile      =      wdsfile                                        # Import WDS file
    wn = wntr.network.WaterNetworkModel(wdsfile) 
    
    yerror = y1
    
    #D Variable Shape Correction
    n, m = np.shape(yerror)                                           # Getting Shape of variable
    
    ylen = len(wn.link_name_list)                                     # Shape of wds nodes (requirement to plot)
    jlen = len(wn.pipe_name_list)                                     # Finding variable shape (missing nodes to be added)
    lastrows = np.zeros([ylen-jlen, m])                               # Missing nodes to be added as zeros
    
    plot_ens = np.vstack((yerror, lastrows))                          # Array Stacked
    
    # Plotting each ensemble seperately
    
    plot_ens = pd.DataFrame(plot_ens[:,0])                            # Isolating each ensemble values
    plot_ens_lis = plot_ens.values.T.tolist()
    # Plotting
    plot_var = dict(zip(wn.link_name_list, plot_ens_lis[0])) # Setting Variable to required format as a dictionary for WNTR
    ax = wntr.graphics.network.plot_network(wn, link_attribute = plot_var, link_width =3, title= qname, node_size = 5, node_alpha = 0.5, ax= ax)    # Plotting
    
    return ax

def monitoring(wdsfile, nn, npi, q_observ, H_observ, Q_observ):
    import numpy as np
    import pandas as pd
    import wntr   
    import matplotlib.pyplot as plt
    import matplotlib.style as style
    style.use('seaborn-colorblind')                      
    
    fig, ax1 = plt.subplots(nrows = 2, ncols=2,  figsize=(15, 12), constrained_layout=True, dpi = 300)
    #Importing Variables
    wdsfile      =      wdsfile                                        # Import WDS file
    wn = wntr.network.WaterNetworkModel(wdsfile) 
    
    # Demand Network
    qsize = np.eye(nn)
    qnet = qsize[q_observ-1,:]
    qnet = np.sum(qnet, axis = 0)[:,None]
    
    #D Variable Shape Correction
    n, m = np.shape(qnet)                                             # Getting Shape of variable
    yqlen = len(wn.node_name_list)                                     # Shape of wds nodes (requirement to plot)
    jqlen = len(wn.junction_name_list)                                 # Finding variable shape (missing nodes to be added)
    lastrows = np.zeros([yqlen-jqlen, m])                               # Missing nodes to be added as zeros
    plot_q = np.vstack((qnet, lastrows))                            # Array Stacked
    plot_q = pd.DataFrame(plot_q)
     
    # Head Network
    Hsize = np.eye(nn)
    Hnet = Hsize[H_observ-1,:]
    Hnet = np.sum(Hnet, axis = 0)[:,None]
    
    #D Variable Shape Correction
    n, m = np.shape(Hnet)                                             # Getting Shape of variable
    yHlen = len(wn.node_name_list)                                     # Shape of wds nodes (requirement to plot)
    jHlen = len(wn.junction_name_list)                                 # Finding variable shape (missing nodes to be added)
    lastrows = np.zeros([yHlen-jHlen, m])                               # Missing nodes to be added as zeros
    plot_H = np.vstack((Hnet, lastrows))                            # Array Stacked
    plot_H = pd.DataFrame(plot_H) 
    
    # Flow Network
    Qsize = np.eye(npi)
    Qnet = Qsize[Q_observ-1,:]
    Qnet = np.sum(Qnet, axis = 0)[:,None]
    
    #D Variable Shape Correction
    n, m = np.shape(Qnet)                                             # Getting Shape of variable
    yQlen = len(wn.link_name_list)                                     # Shape of wds nodes (requirement to plot)
    jQlen = len(wn.pipe_name_list)                                 # Finding variable shape (missing nodes to be added)
    lastrows = np.zeros([yQlen-jQlen, m])                               # Missing nodes to be added as zeros
    plot_Q = np.vstack((Qnet, lastrows))                            # Array Stacked
    plot_Q = pd.DataFrame(plot_Q) 
    
    # Combined Network for head and demand
    plot_C = plot_q + plot_H
    
    
    # Plotting Monitoring Network
    plot_H_lis = plot_H.values.T.tolist()
    plot_q_lis = plot_q.values.T.tolist()
    plot_C_lis = plot_C.values.T.tolist()
    plot_lin_lis = plot_Q.values.T.tolist()
    # Plotting
    plot_nodH_var = dict(zip(wn.node_name_list, plot_H_lis[0]))
    plot_nodq_var = dict(zip(wn.node_name_list, plot_q_lis[0])) # Setting Variable to required format as a dictionary for WNTR
    plot_nodC_var = dict(zip(wn.node_name_list, plot_C_lis[0]))
    plot_lin_var = dict(zip(wn.link_name_list, plot_lin_lis[0]))
    ax = wntr.graphics.network.plot_network(wn, node_attribute = plot_nodH_var,
                                                title= "Monitoring Network for Head",
                                                node_cmap =['k', 'dodgerblue'],
                                                node_size = 40, 
                                                add_colorbar=False, ax=ax1[0,0])
    ax = wntr.graphics.network.plot_network(wn, node_attribute = plot_nodq_var,
                                                title= "Monitoring Network for Demand",
                                                node_cmap =['k', 'dodgerblue'], 
                                                node_size = 40,
                                                add_colorbar=False, ax=ax1[0,1])
    ax = wntr.graphics.network.plot_network(wn, link_attribute = plot_lin_var, 
                                                title= "Monitoring Network for Flow",
                                                link_width =3,
                                                node_cmap =['k', 'blue'],
                                                node_size = 40, 
                                                link_cmap = ['grey', 'dodgerblue'], 
                                                add_colorbar=False, ax=ax1[1,0])
    ax = wntr.graphics.network.plot_network(wn, node_attribute = plot_nodC_var, 
                                                link_attribute = plot_lin_var, 
                                                title= "Combined Monitoring Network",
                                                link_width =3,
                                                node_cmap =['k','dodgerblue' ],#,'darkorange'],
                                                node_size = 40, 
                                                link_cmap = ['dimgrey', 'dodgerblue'],
                                                add_colorbar=False, ax=ax1[1,1])
    return ax

def combined_monitoring(wdsfile, nn, npi, q_observ, H_observ, Q_observ, qname):
    import numpy as np
    import pandas as pd
    import wntr   
    import matplotlib.pyplot as plt
    import matplotlib.style as style
    style.use('seaborn-colorblind')                      
    
    fig, ax1 = plt.subplots(figsize=(10, 10), constrained_layout=True, dpi=300)
    #Importing Variables
    wdsfile      =      wdsfile                                        # Import WDS file
    wn = wntr.network.WaterNetworkModel(wdsfile) 
    
    # Demand Network
    qsize = np.eye(nn)
    qnet = qsize[q_observ-1,:]
    qnet = np.sum(qnet, axis = 0)[:,None]
    
    #D Variable Shape Correction
    n, m = np.shape(qnet)                                             # Getting Shape of variable
    yqlen = len(wn.node_name_list)                                     # Shape of wds nodes (requirement to plot)
    jqlen = len(wn.junction_name_list)                                 # Finding variable shape (missing nodes to be added)
    lastrows = np.zeros([yqlen-jqlen, m])                               # Missing nodes to be added as zeros
    plot_q = np.vstack((qnet, lastrows))                            # Array Stacked
    plot_q = pd.DataFrame(plot_q)
     
    # Head Network
    Hsize = np.eye(nn)
    Hnet = Hsize[H_observ-1,:]
    Hnet = np.sum(Hnet, axis = 0)[:,None]
    
    #D Variable Shape Correction
    n, m = np.shape(Hnet)                                             # Getting Shape of variable
    yHlen = len(wn.node_name_list)                                     # Shape of wds nodes (requirement to plot)
    jHlen = len(wn.junction_name_list)                                 # Finding variable shape (missing nodes to be added)
    lastrows = np.zeros([yHlen-jHlen, m])                               # Missing nodes to be added as zeros
    plot_H = np.vstack((Hnet, lastrows))                            # Array Stacked
    plot_H = pd.DataFrame(plot_H) 
    
    # Flow Network
    Qsize = np.eye(npi)
    Qnet = Qsize[Q_observ-1,:]
    Qnet = np.sum(Qnet, axis = 0)[:,None]
    
    #D Variable Shape Correction
    n, m = np.shape(Qnet)                                             # Getting Shape of variable
    yQlen = len(wn.link_name_list)                                     # Shape of wds nodes (requirement to plot)
    jQlen = len(wn.pipe_name_list)                                 # Finding variable shape (missing nodes to be added)
    lastrows = np.zeros([yQlen-jQlen, m])                               # Missing nodes to be added as zeros
    plot_Q = np.vstack((Qnet, lastrows))                            # Array Stacked
    plot_Q = pd.DataFrame(plot_Q) 
    
    # Combined Network for head and demand
    plot_C = plot_q + plot_H
    
    # Plotting Monitoring Network
    plot_H_lis = plot_H.values.T.tolist()
    plot_q_lis = plot_q.values.T.tolist()
    plot_C_lis = plot_C.values.T.tolist()
    plot_lin_lis = plot_Q.values.T.tolist()
    # Plotting
    plot_nodH_var = dict(zip(wn.node_name_list, plot_H_lis[0]))
    plot_nodq_var = dict(zip(wn.node_name_list, plot_q_lis[0])) # Setting Variable to required format as a dictionary for WNTR
    plot_nodC_var = dict(zip(wn.node_name_list, plot_C_lis[0]))
    plot_lin_var = dict(zip(wn.link_name_list, plot_lin_lis[0]))
    
    ax = wntr.graphics.network.plot_network(wn, node_attribute = plot_nodC_var, 
                                                link_attribute = plot_lin_var, 
                                                title= qname,
                                                link_width =3,
                                                node_cmap =['k','blue'],#,'red'],
                                                node_size = 50, 
                                                link_cmap = ['dimgrey', 'blue'],
                                                add_colorbar=False, ax= ax1)

    return ax

def only_monitoring(wdsfile, nn, npi, q_observ, qname):
    import numpy as np
    import pandas as pd
    import wntr   
    import matplotlib.pyplot as plt
    import matplotlib.style as style
    style.use('seaborn-colorblind')                      
    
    fig, ax1 = plt.subplots(figsize=(5, 5), constrained_layout=True, dpi=300)
    #Importing Variables
    wdsfile      =      wdsfile                                        # Import WDS file
    wn = wntr.network.WaterNetworkModel(wdsfile) 
    
    # Demand Network
    qsize = np.eye(nn)
    qnet = qsize[q_observ-1,:]
    qnet = np.sum(qnet, axis = 0)[:,None]
    
    #D Variable Shape Correction
    n, m = np.shape(qnet)                                             # Getting Shape of variable
    yqlen = len(wn.node_name_list)                                     # Shape of wds nodes (requirement to plot)
    jqlen = len(wn.junction_name_list)                                 # Finding variable shape (missing nodes to be added)
    lastrows = np.zeros([yqlen-jqlen, m])                               # Missing nodes to be added as zeros
    plot_q = np.vstack((qnet, lastrows))                            # Array Stacked
    plot_q = pd.DataFrame(plot_q)
     
    plot_q_lis = plot_q.values.T.tolist()
    
    # Plotting
    import random
    plot_nodq_var = dict(zip(wn.node_name_list, plot_q_lis[0]))
    plot_zeros_list = [random.random() for _ in range(317)]
    plot_link = dict(zip(wn.link_name_list, plot_zeros_list))
    ax = wntr.graphics.network.plot_network(wn, node_attribute = plot_nodq_var, 
                                                title= qname,
                                                link_width =1,
                                                link_attribute = plot_link,
                                                link_cmap = ['white', 'white'],
                                                node_cmap =['white','dodgerblue'],#,'red'],
                                                node_size = 30, 
                                                add_colorbar=False, ax= ax1)
    
    return ax

def linkonly_monitoring(wdsfile, nn, npi, Q_observ, qname):
    import numpy as np
    import pandas as pd
    import wntr   
    import matplotlib.pyplot as plt
    import matplotlib.style as style
    style.use('seaborn-colorblind')                      
    
    fig, ax1 = plt.subplots(figsize=(5, 5), constrained_layout=True, dpi=300)
    #Importing Variables
    wdsfile      =      wdsfile                                        # Import WDS file
    wn = wntr.network.WaterNetworkModel(wdsfile) 
    
    # Flow Network
    Qsize = np.eye(npi)
    Qnet = Qsize[Q_observ-1,:]
    Qnet = np.sum(Qnet, axis = 0)[:,None]
    
    #D Variable Shape Correction
    n, m = np.shape(Qnet)                                             # Getting Shape of variable
    yQlen = len(wn.link_name_list)                                     # Shape of wds nodes (requirement to plot)
    jQlen = len(wn.pipe_name_list)                                 # Finding variable shape (missing nodes to be added)
    lastrows = np.zeros([yQlen-jQlen, m])                               # Missing nodes to be added as zeros
    plot_Q = np.vstack((Qnet, lastrows))                            # Array Stacked
    plot_Q = pd.DataFrame(plot_Q) 
       
    plot_lin_lis = plot_Q.values.T.tolist()
    
    # Plotting
    plot_lin_var = dict(zip(wn.link_name_list, plot_lin_lis[0]))

    ax = wntr.graphics.network.plot_network(wn, link_attribute = plot_lin_var,
                                                title= qname,
                                                link_width =3,
                                                link_cmap = ['white', 'teal'],
                                                node_cmap =['white'],#,'red'],
                                                node_size = 0.001, 
                                                add_colorbar=False, ax= ax1)
    
    return ax