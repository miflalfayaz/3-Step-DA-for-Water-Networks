# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 15:56:42 2022

@author: ibrah
"""

def K_HazenWilliams(L,C,D):
#K_HazenWilliams Estimates the K of the headloss for the Hazen-Williams equation
# calculation in SI. 
#
# L = Lenght of pipe(s) (m)
# C = Hazen Williams coefficient of pipe(s)
# D = Diamter of pipe (usually from Epanet 2.x it is given in mm)
#
# K estimated coeffient of HW equation
# 
# Subsequently the pipe(s) headloss (hf) can be estiamted as 
#     hf = K*abs(Q)*Q^0.85 
# Where Q is the flow in the pipe(s) in l/s
#
  D = D/1000 # from mm to m
  #K = (10.583.*L)./(C.^1.852.*D.^4.87);
  K = (10.67*L)/(C**1.852*D**4.87) # from Wikipedia
  
  return K
