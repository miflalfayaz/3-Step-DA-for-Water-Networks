# -*- coding: utf-8 -*-

"""
# This script is used for Data Assimilation in 
#Water Distribution Networks using EnKF and WNTR.
# Msc. Mario Castro Gama - PhD UN-IHE
# Ibrahim Miflal Fayaz - Msc UN-IHE
# 20/04/2023
"""

def Three_Step_EnKF(wdsfile, bdemfile, topofile, assetfile, lnkidfile, nodidfile, 
                    npi, nn, no, sim_time, ne, Dmult, Res_fname, 
                    H_observ, vzH_v, RzH_v,
                    Q_observ, vzQ_v, RzQ_v,
                    q_observ, vzq_v, Rzq_v):
    '''
        This script is a Python implementation of an algorithm for data assimilation in a 
        water distribution network (WDN). The algorithm uses a 3-step Ensemble Kalman Filter (EnKF) 
        to estimate the state of the WDN based on a set of observations. 
        
        Parameters
        ----------
        wdsfile : TYPE: String
            DESCRIPTION. Filename including relative directory of the EPANET .inp of the WDN file. 
        bdemfile : TYPE: String
            DESCRIPTION. Filename including relative directory of the BDEMfile for Basedemands.
        topofile : TYPE: String
            DESCRIPTION. Filename including relative directory of the Topofile for Topology of the WDN.
        assetfile : TYPE String
            DESCRIPTION. Filename including relative directory of the assetfile for Assets of the WDN.
        lnkidfile : TYPE String
            DESCRIPTION. Filename including relative directory of the linkidfile for lnkidfile of the WDN.
        nodidfile : TYPE String
            DESCRIPTION. Filename including relative directory of the nodidfile for nodidfile of the WDN.
        npi : TYPE: Integer
            DESCRIPTION. Number of Pipes
        nn : TYPE: Integer
            DESCRIPTION. Number of Nodes
        no : TYPE: Integer
            DESCRIPTION. Number of Sources
        sim_time : TYPE: Integer
            DESCRIPTION. Integer for defining the simulation time for the algorithm in hours.
        ne : TYPE: Integer
            DESCRIPTION. Integer for defining the number of ensembles for the EnKF.
        Dmult : TYPE float
            DESCRIPTION. Float to define the General multiplier to all demands.
        Res_fname : TYPE: String
            DESCRIPTION. Filename including relative directory for the results file
        H_observ : TYPE: Array
            DESCRIPTION. Array defining the location of pressure sensors.
        vzH_v : TYPE: Float
            DESCRIPTION. Noise in Pressure Measurement
        RzH_v : TYPE: Float
            DESCRIPTION. Precision of Pressure Sensors
        Q_observ : TYPE: Array
            DESCRIPTION. Array defining the location of Flow sensors.
        vzQ_v : TYPE: Float
            DESCRIPTION. Noise in Flow Measurement
        RzQ_v : TYPE: Float
            DESCRIPTION. Precision of Flow Sensors
        q_observ : TYPE: Array
            DESCRIPTION. Array defining the location of demand sensors.
        vzq_v : TYPE: Float
            DESCRIPTION. Noise in Demand Measurement
        Rzq_v : TYPE: Float
            DESCRIPTION. Precision of Demand Sensors
    
        Returns
        -------
        The result files of the Three_Step_EnKF Algorithm to the directory defined as per Res_fname.
        The Output file includes Input Variables, Simulation Results and Total Variance.
    
    '''
   
    # Importing Necessary Libraries 
    
    import numpy as np                              # Importing Numpy Library for Mathemtical Operations
    import pandas as pd                             # Importing Pandas for reading files and Mathematical Operations
    from Modules import Py_DA_list_Par              # Importing Module for running Hydraulic Model with EPAnet Simulator
  # from Modules import Py_DA_list                  # Importing Module for running Hydraulic Model with WNTR Simulator
    from Modules import K_HazenWilliams             # Importing Module for calculating Hazen William Coefficient
    from Modules import Kerror                      # Importing Module for calculating Covariance
    from Modules import Kassimilate                 # Importing Module for Assimilation
    from Modules import TV                          # Importing Module for Calculating Total Variance 
    import time                                     # Importing Library to measure time
    import wntr                                     # Importing WNTR for Hydraulic Modelling
    import pickle                                   # Importing Pickle Library to save results
    from scipy.interpolate import interp1d          # Importing Scipy Library for Interpolation
    
    t = time.time()                                 # Start Time
    
    # Input Variables for Three_Step_EnKF 
    
    sH              = len(H_observ)                 # number of pressure sensors    
    M_H_all         = np.eye(nn)                    # there are 940 nodes in this WDN, but 5 are T/R
    M_H             = M_H_all[H_observ-1,:]         # Measurement matrix of HEADS measurement
    vzH             = vzH_v*np.ones([sH,1])         # mean of the HEAD measurement errors
    RzH             = RzH_v*np.eye(sH)              # variance of HEAD measurement errors
    
    sQ              = len(Q_observ)                 # number of flow sensors    
    M_Q_all         = np.eye(npi)                   # there are 317 pipes in this WDN
    M_Q             = M_Q_all[Q_observ-1,:]         # Measurement matrix of FLOWS
    vzQ             = vzQ_v*np.ones([sQ,1])         # mean of the FLOWS measurement errors
    RzQ             = RzQ_v*np.eye(sQ)              # variance of FLOWS measurement errors

    sq              = len(q_observ)                 # number of demand sensors    
    M_q_all         = np.eye(nn);                   # there are 272 nodes in this WDN, but 4 are T/R
    M_q             = M_q_all[q_observ-1,:]         # Measurement matrix of Demand measurement
    vzq             = vzq_v*np.ones([sq,1])         # mean of the Demand measurement errors
    Rzq             = Rzq_v*np.eye(sq)              # variance of Demand measurement errors
    
    basedemandsrd   = pd.read_csv( bdemfile,   header = None)
    basedemands     = basedemandsrd.T
    assets          = pd.read_csv( assetfile,  header = None)
    Aij             = pd.read_csv( topofile,   header = None)
    curves          = []
    #link_ids        = pd.read_csv(lnkidfile,   header = None)
    #node_ids        = pd.read_csv(nodidfile,   header = None)

    
    # build the matrices A12, A10, A21 (Pipe-node Incidency matrix)
    A12 = np.zeros([npi, (nn+no)])
    
    for ilink in range (assets.shape[0]):
        A12[ilink, Aij.iloc[ilink,0]-1] = -1
        A12[ilink, Aij.iloc[ilink,1]-1] = +1
        
    A10 = A12[:, -1-(no-1):]                        # Isolating the last 4 columns of the array A12
    A12 = (A12[:,  0 : -no])                        # Removing the last 4 columns of the array A12
    A21 = (A12.T)                                   # Transposing the matrix A12 into A21
    
    # find the tmpINV matrix
    tmpINV = -(np.linalg.solve((np.matmul(A21,A12)), A21))
    
    # find the K roughness of each pipe
    K11 = K_HazenWilliams.K_HazenWilliams(assets.iloc[:,1],assets.iloc[:,2],assets.iloc[:,0])
      
    # N1 = Aij.iloc[:,0]
    # N2 = Aij.iloc[:,1]
    
    # List to store final results
    
    # Heads
    Hobs_t          = []                            # Array of Head Observations 
    H_qj_t          = []                            # Initialized Head Values      
    H_qj_zH_t       = []                            # Array of Heads with 1 step  of Assimilation (Head)  
    H_qj_zH_zQ_t    = []                            # Array of Heads with 2 steps of Assimilation (Head, Flow) 
    H_qj_zH_zQ_zq_t = []                            # Array of Heads with 3 steps of Assimilation (Head, Flow, Demand) 
    
    # Demands
    qobs_t          = []                            # Array of Demand Observations 
    q_qj_t          = []                            # Initialized Demand Values  
    q_qj_zH_t       = []                            # Array of Demand with 1 step  of Assimilation (Head) 
    q_qj_zH_zQ_t    = []                            # Array of Demand with 2 steps of Assimilation (Head, Flow) 
    q_qj_zH_zQ_zq_t = []                            # Array of Demand with 3 steps of Assimilation (Head, Flow, Demand) 
    
    # Flows 
    Qobs_t          = []                            # Array of Flow Observations 
    Q_qj_t          = []                            # Initialized Flow Values 
    Q_qj_zH_t       = []                            # Array of Flow with 1 step  of Assimilation (Head)
    Q_qj_zH_zQ_t    = []                            # Array of Flow with 2 steps of Assimilation (Head, Flow)  
    Q_qj_zH_zQ_zq_t = []                            # Array of Flow with 3 steps of Assimilation (Head, Flow, Demand) 
    
    
    #%%
    # Running the EnKF Algorithm for each time step.
    for tval_new in range(0, sim_time*3600, 3600):
        
        # Create the matrices of Data Assimilation
        # Intialized Values
        H_qj          = (np.zeros([nn, ne]))      
        Q_qj          = (np.zeros([npi,ne]))       
        
        # 1 Step of Assimilation (Heads)
        H_qj_zH       = (np.zeros([nn, ne]))
        Q_qj_zH       = (np.zeros([npi,ne]))
        
        # 2 Steps of Assimilation (Heads, Flows)
        H_qj_zH_zQ    = (np.zeros([nn, ne]))
        Q_qj_zH_zQ    = (np.zeros([npi,ne]))
        
        # 3 Steps of Assimilation (Heads, Flows, Demands)
        H_qj_zH_zQ_zq = (np.zeros([nn, ne]))
        Q_qj_zH_zQ_zq = (np.zeros([npi,ne]))
    

        
        # Initializing the Demands for the Algorithm with a Random Normal Distribution
        # The standard deviation of the observation is only 0.02
        qbase = (Dmult*basedemands*(1 + 0.02 * np.random.randn(1,nn))).T 
        # qbase is the base demands which will initialize the hydraulic model
        #----------------------------------------------------------------------------------------------------------------------------#
        
        # Running the Hydraulic Model
        curr_states   = Py_DA_list_Par.HyMod(qbase, tval_new, wdsfile)  # Running the Hydraulic Model
        qmeas = curr_states[2]                                      # Retrieve the actual demands sometimes comes affected by the demands patterns 
    
        Hobs  = curr_states[1]                                      # Head Observations
        Qobs  = curr_states[4] * (assets.iloc[0, 4])                # Flow Observations
        Qobs = Qobs.T
     
        Ho      = curr_states[6]                                    # Head of tanks/reservoirs
        Dem_Res = curr_states[7]                                    # Demands on reservoirs
    
        z_H = Hobs[H_observ-1]                                      # Observations of HEADS
        z_Q = Qobs[Q_observ-1]                                      # Observations of FLOWS
        z_q = qmeas[q_observ-1]                                     # Observations of DEMANDS
    
    
        ## Step 0 - Determine the initialization of the run
        #----------------------------------------------------------------------------------------------------------------------------#
    
        qj = (np.zeros([nn, ne]))                                   # Ensemble of demands
        wb = (np.zeros([1,  ne]))                                   # Water balances
        eb = (np.zeros([1,  ne]))                                   # Energy balances
        Sp = (np.zeros([npi, ne]))                                  # Pipe Status
        
        print("\r Time Step : {0:.0f} hr" .format(tval_new/3600))
        
        # Running Each ensemble for the current time step
        for iens in range(ne):
           q = (Dmult*basedemands*(1 + 0.05*np.random.randn(1,nn))).T
           q[q < 0] = 0                                             # to avoid negative demands q<0 = 0
           curr_states_i = Py_DA_list_Par.HyMod(q, tval_new, wdsfile)   # Running the Hydraulic Model for each ensemble
       
           Hi = curr_states_i[1]                                    # Head Observations for each ensemble member
           Qi = curr_states_i[4]                                    # Flow Observations for each ensemble member
           Qi = Qi.T
            
           qj[:, iens][:,None] = curr_states_i[2]                   # Estimated demands in LPS
           Sp[:, iens] = curr_states_i[5]                           # Status' of links
           H_qj[:, iens][:,None] = Hi[:,]                           # Obtained Node heads
           Q_qj[:, iens][:,None] = (Qi[:,]*Sp[0,iens])              # Pipeflows
           
           # Water Balance Calculations
           
           wb[0:iens] = sum(abs(np.matmul(A21, Q_qj[:,iens]/1000) 
                                - qj[:,iens]/1000))                 # Water Balance
           
           # Energy Balance Calculations
           eb1 = np.matmul(A10, Ho)[:,None]                            
           eb2 = np.matmul(A12, Hi)
           eb3 = np.matmul(np.diag(K11), 
                           abs(Qi/1000)**0.852*Qi/1000) 
           eb4 = eb3 + eb2 + eb1
           eb5 = sum(abs(eb4))
           eb[0:iens] = eb5                                          # Array of Energy Balance
    
           print(end ="\r Loading Ensembles  {0:.0f} %" .format(iens/(ne-1)*100))
           
           del q, Hi, Qi, iens
        #   print("Loading Ensembles  {0:.0f} %" .format(((iens+1)*100)/ne))
        # end # iens
        # [wb; eb]
    
    
        ## Step 1 - Once the ensemble has been estimated then it is time to assimilate the samples in the observations
        #----------------------------------------------------------------------------------------------------------------------------#
        P_H, xe = Kerror.Kerror(H_qj);                              # PH is the ensemble prior variance of the Heads 
        P_HbM_HT = np.matmul(P_H, M_H.T)                            # Intermediate to calculate Kalman Gains
        tmpH = np.linalg.solve((np.matmul(M_H, P_HbM_HT) 
                                + RzH),np.eye(sH))                  # Taking into account the number of head sensors
        
        # Kalman Gain Heads
        K_H = np.matmul(P_HbM_HT, tmpH)                             # Calculating the Kalman Gain for Head
        
        # Assimilation of Head measurements
        H_qj_zH = Kassimilate.Kassimilate(H_qj, K_H, z_H, M_H, vzH) # Assimilation of head Measurements
    
        # Estimate the posteriori flows
        for ilink in range(npi):
            ni = Aij.iloc[ilink, 0]
            nj = Aij.iloc[ilink, 1]
        
            if ni <= nn:
                H1 = H_qj_zH[ni-1,:] # + Cxy(ni,3);
            else:
                H1 = Ho[ni-1-nn]
        
            if nj <= nn:
                H2 = H_qj_zH[nj-1,:] # + Cxy(nj,3);
            else:
                H2 = Ho[nj-1-nn]
        
           # Estimate gradient of head 
           # dH = (pressure + elevation)_i - (pressure + elevation)_j
            dH = ( H1 - H2 )[:,None]
            if assets.iloc[ilink,3] < 2:         # 0: Check Valves, 1: Pipes
                if assets.iloc[ilink,3] == 0:    # Check valve
                    dH[dH < 0] = 0 # NO gradient means dHi = 0 or NO flow
        
           # Calculating Flow with Assimilated Head
           
                sdH   = np.sign(dH)                                 # sdH = zeros(size(sdH));
                Sij   = dH / assets.iloc[ilink,1]                   # Headloss per unit length
                Dij   = assets.iloc[ilink,0] / 1000                 # Convert to meters
                Ar_ij = (np.pi/4) * Dij * Dij                       # Flow area in m2
                Rij   = assets.iloc[ilink,2]                        # Roughness
                ONij  = Sp[ilink,:]                                 # Status, if open there is flow, otherwise NO flow
                Vij   = Rij * (Dij/4) ** 0.63 * abs(Sij) ** 0.54    # Estimated flow velocity
                Q_qj_zH[ilink,:][:,None] = 0.2784 * sdH * abs(Sij) ** 0.54 * (Dij ** 2.63) * Rij * 1000
        
            
        # Pumps
            if assets.iloc[ilink, 3] == 2:  
                curve_idx  = assets.iloc[ilink,5]
                curve_data = curves[curves[:,0] ==curve_idx,3:4]
                Qc         = curve_data[:,0]
                Hc         = curve_data[:,1]
                dH         = min(Hc[0],-(dH))
                new_Q_qj   = interp1d(Hc,Qc,abs(dH));               # Interpolate the pump discharges based on heads in nodes
                
                ONij       = Sp[ilink,:] # assets(ilink,5);         # Status, if open there is flow, otherwise NO flow
                Q_qj_zH[ilink,:] = ONij*new_Q_qj #ONij.*Q_qj(ilink,:);    
        
        #    If assets[ilink,3] > 2: # Valves
        #    This requires more work for each valve type
                ONij = Sp[ilink,:];                                 #assets(ilink,5);   # Status, if open there is flow, otherwise no flow
                Q_qj_zH[ilink,:] = ONij * Q_qj[ilink,:];
         
        # Now build the ensemble of estimated demands based on the FLOWS
        q_qj_zH = np.matmul(A21,Q_qj_zH)
        #q_qj_zH = max(q_qj_zH,0); % Make q >= 0
        
        ## Step 2 _Assimilate FLOW measurements in pipes
        #----------------------------------------------------------------------------------------------------------------------------#
        P_Q, xe_2 = Kerror.Kerror(Q_qj_zH)                          # PQ is the ensemble prior variance of the Flows 
        
        P_QbM_QT   = np.matmul(P_Q, M_Q.T);
        tmpQ       = np.linalg.solve((np.matmul(M_Q, P_QbM_QT) 
                                      + RzQ),np.eye(sQ))            # Intermediate to calculate Kalman Gains
        K_Q        = np.matmul (P_QbM_QT, tmpQ)                     # Kalman Gain for Flow measurements
        
        # Assimilation of Flow measurements
        Q_qj_zH_zQ = Kassimilate.Kassimilate(Q_qj_zH, K_Q, z_Q, M_Q, vzQ);
        
        ## Now build the ensemble of estimated demands based on the FLOWS
        q_qj_zH_zQ = np.matmul(A21, Q_qj_zH_zQ)
        
        # Ensemble of estimated HEADS based on FLOWS measurements
        for iens in range (ne):
            Qcurr = Q_qj_zH_zQ[:,iens] / 1000
            # Based on hydraulic headloss node calculate heads
            A11 = np.diag(K11 * (abs(Qcurr))**0.852)
            H_qj_zH_zQ[:,iens][:,None] = np.matmul(tmpINV, np.matmul(A11, Qcurr)[:,None] + np.matmul(A10, Ho)[:,None])
        
        ## Step 3  - Assimilate demand measurements (z_q)
        #----------------------------------------------------------------------------------------------------------------------------#
        [P_Qp, xerr] = Kerror.Kerror(Q_qj_zH_zQ)                    # P_Qp is the ensemble prior variance of the Flows
        #zq - M_q*q_qj_zH_zQ - vzq
        
        ## tmp3 = 1\(M_q*A21*P_Qp*A12*(M_q') + diag(Rzq));  # commented 23-Oct-2020 21.18
        tmp3 = np.linalg.solve((np.matmul(np.matmul(np.matmul(np.matmul(M_q,A21),P_Qp),A12),(M_q.T)) + Rzq), np.eye(sq))
        K_Qp = np.matmul(np.matmul(P_Qp,(np.matmul(A12,M_q.T))),tmp3)
        
        # Special Data Assimilation of Demand measurements for Flow
        # Update the flows for all pipes based on a number of measurements of Demands 
        for i in range(ne):
            Qx1 = z_q
            Qx2 = np.matmul(M_q, q_qj_zH_zQ[:,i])[:,None]
            Qx3 = vzq
            Qx4 = Qx1 - Qx2 - Qx3
            Qx5 = np.matmul(K_Qp, (Qx4))
            Qx6 = Q_qj_zH_zQ[:,i][:,None] + Qx5
            Q_qj_zH_zQ_zq[:,i][:,None] = Qx6
              
        # Now build the ensemble of estimated demands based on the Flows
        q_qj_zH_zQ_zq = np.matmul(A21, Q_qj_zH_zQ_zq)
        
        # Ensemble of estimated HEADS based on FLOWS measurements
        
        for iens in range(ne):
          Qcurr = Q_qj_zH_zQ_zq[:,iens]/1000;
          A11 =  np.diag(K11*(abs(Qcurr))**0.852);
          x11 = np.matmul(A10, Ho)
          x22 = np.matmul(A11, Qcurr)
          x33 = np.matmul(tmpINV, x11 + x22)[:,None]
          H_qj_zH_zQ_zq[:,iens][:,None] = x33
          
        # End of 3-Step DA
        #----------------------------------------------------------------------------------------------------------------------------#      
        # Appending Results
        Hobs_t.append(Hobs)
        H_qj_t.append(H_qj)       
        H_qj_zH_t.append(H_qj_zH)
        H_qj_zH_zQ_t.append(H_qj_zH_zQ)
        H_qj_zH_zQ_zq_t.append(H_qj_zH_zQ_zq) 
        
        qobs_t.append(qmeas)
        q_qj_t.append(qj)
        q_qj_zH_t.append(q_qj_zH) 
        q_qj_zH_zQ_t.append(q_qj_zH_zQ) 
        q_qj_zH_zQ_zq_t.append(q_qj_zH_zQ_zq) 
        
        Qobs_t.append(Qobs)
        Q_qj_t.append(Q_qj)
        Q_qj_zH_t.append(Q_qj_zH) 
        Q_qj_zH_zQ_t.append(Q_qj_zH_zQ) 
        Q_qj_zH_zQ_zq_t.append(Q_qj_zH_zQ_zq) 
          
        # Caluclating the simulation for each time step
        totaltime = time.time() - t
        print("\n Run Time:  {0:.2f} s" .format(totaltime))

    # Final Result Matrices
    # Heads
    Hobs_ti             = np.stack(Hobs_t, axis=2)    
    H_qj_ti             = np.stack(H_qj_t, axis=2)  
    H_qj_zH_ti          = np.stack(H_qj_zH_t, axis=2) 
    H_qj_zH_zQ_ti       = np.stack(H_qj_zH_zQ_t, axis=2)  
    H_qj_zH_zQ_zq_ti    = np.stack(H_qj_zH_zQ_zq_t, axis=2)
    
    # Demands
    qobs_ti             = np.stack(qobs_t, axis=2)   
    q_qj_ti             = np.stack(q_qj_t, axis=2)
    q_qj_zH_ti          = np.stack(q_qj_zH_t, axis=2) 
    q_qj_zH_zQ_ti       = np.stack(q_qj_zH_zQ_t, axis=2)
    q_qj_zH_zQ_zq_ti    = np.stack(q_qj_zH_zQ_zq_t, axis=2) 
    
    # Flows
    Qobs_ti             = np.stack(Qobs_t, axis=2)     
    Q_qj_ti             = np.stack(Q_qj_t, axis=2)
    Q_qj_zH_ti          = np.stack(Q_qj_zH_t, axis=2) 
    Q_qj_zH_zQ_ti       = np.stack(Q_qj_zH_zQ_t, axis=2)
    Q_qj_zH_zQ_zq_ti    = np.stack(Q_qj_zH_zQ_zq_t, axis=2) 
    
    
    # Calculating the Total Variance for each System State Using the TV Module
    TV_H_qj_t           = TV.Total_Variance(H_qj_ti,           Hobs_ti, nn, ne, sim_time)
    TVR_H_qj            = TV.Total_Variance(H_qj_ti,           Hobs_ti, nn, ne, sim_time)  / TV_H_qj_t  
    TVR_H_qj_zH         = TV.Total_Variance(H_qj_zH_ti,        Hobs_ti, nn, ne, sim_time)  / TV_H_qj_t   
    TVR_H_qj_zH_zQ      = TV.Total_Variance(H_qj_zH_zQ_ti,     Hobs_ti, nn, ne, sim_time)  / TV_H_qj_t
    TVR_H_qj_zH_zQ_zq   = TV.Total_Variance(H_qj_zH_zQ_zq_ti,  Hobs_ti, nn, ne, sim_time)  / TV_H_qj_t
    
    TV_q_qj_t           = TV.Total_Variance(q_qj_ti,           qobs_ti, nn, ne, sim_time)
    TVR_q_qj            = TV.Total_Variance(q_qj_ti,           qobs_ti, nn, ne, sim_time)  / TV_q_qj_t    
    TVR_q_qj_zH         = TV.Total_Variance(q_qj_zH_ti,        qobs_ti, nn, ne, sim_time)  / TV_q_qj_t 
    TVR_q_qj_zH_zQ      = TV.Total_Variance(q_qj_zH_zQ_ti,     qobs_ti, nn, ne, sim_time)  / TV_q_qj_t
    TVR_q_qj_zH_zQ_zq   = TV.Total_Variance(q_qj_zH_zQ_zq_ti,  qobs_ti, nn, ne, sim_time)  / TV_q_qj_t 
    
    TV_Q_qj_t           = TV.Total_Variance(Q_qj_ti,           Qobs_ti, npi, ne, sim_time)    
    TVR_Q_qj            = TV.Total_Variance(Q_qj_ti,           Qobs_ti, npi, ne, sim_time)  / TV_Q_qj_t     
    TVR_Q_qj_zH         = TV.Total_Variance(Q_qj_zH_ti,        Qobs_ti, npi, ne, sim_time)  / TV_Q_qj_t  
    TVR_Q_qj_zH_zQ      = TV.Total_Variance(Q_qj_zH_zQ_ti,     Qobs_ti, npi, ne, sim_time)  / TV_Q_qj_t   
    TVR_Q_qj_zH_zQ_zq   = TV.Total_Variance(Q_qj_zH_zQ_zq_ti,  Qobs_ti, npi, ne, sim_time)  / TV_Q_qj_t  
    
    # List of Total Variance Results to be saved in the Output file
    TV_List      = [TV_H_qj_t, TVR_H_qj, TVR_H_qj_zH, TVR_H_qj_zH_zQ, TVR_H_qj_zH_zQ_zq, 
                    TV_q_qj_t, TVR_q_qj, TVR_q_qj_zH, TVR_q_qj_zH_zQ, TVR_q_qj_zH_zQ_zq,
                    TV_Q_qj_t, TVR_Q_qj, TVR_Q_qj_zH, TVR_Q_qj_zH_zQ, TVR_Q_qj_zH_zQ_zq]
    
                            
    # Combined Total Variance Ratio
    TVR_T = TVR_H_qj_zH + TVR_H_qj_zH_zQ + TVR_H_qj_zH_zQ_zq + TVR_q_qj_zH + TVR_q_qj_zH_zQ + TVR_q_qj_zH_zQ_zq + TVR_Q_qj_zH + TVR_Q_qj_zH_zQ + TVR_Q_qj_zH_zQ_zq
    TVR   = np.sum(TVR_T)/(sim_time * 9)
    
    # List of Input Variables to save in the Output File
    Sim_Var = [sim_time, ne, npi, nn, no, H_observ, q_observ, Q_observ, wdsfile,
               bdemfile, topofile, assetfile, lnkidfile, nodidfile ,Dmult, 
                sH, H_observ, vzH_v, RzH_v,
                sQ, Q_observ, vzQ_v, RzQ_v,
                sq, q_observ, vzq_v, Rzq_v]                                         
    
    # List of the Result Matrices
    Sim_Res = [Hobs_ti, H_qj_ti, H_qj_zH_ti, H_qj_zH_zQ_ti, H_qj_zH_zQ_zq_ti,      
               qobs_ti, q_qj_ti, q_qj_zH_ti, q_qj_zH_zQ_ti, q_qj_zH_zQ_zq_ti,
               Qobs_ti, Q_qj_ti, Q_qj_zH_ti, Q_qj_zH_zQ_ti, Q_qj_zH_zQ_zq_ti]
    
    # Final Run Time for the Simulation
    totaltime = time.time() - t
    print("\n Run Time:  {0:.2f} s" .format(totaltime))

    # List for the Final Output to be saved as a pickle file
    Output = [Sim_Var, Sim_Res, TV_List, TVR, totaltime]                                         
    
    # Saving the Results with Pickle to the folder specified by Resfname
    with open(Res_fname, 'wb') as filehandle:
        # Store the data as a binary data stream
        pickle.dump(Output, filehandle)
    return Output
