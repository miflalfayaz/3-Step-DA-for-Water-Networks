# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 22:21:10 2022

@author: ibrah
"""
def plot_ensembles_values(y1,y2,y3,y4,ynames,yrange,yobs):
    import numpy as np
    from bokeh.plotting import figure, show
    from bokeh.layouts import column
    nx = np.shape(y1)[0]
    xx = np.arange(1,nx+1)
    
    yl1 = np.min(y1, axis = 1)
    yh1 = np.max(y1, axis = 1)
    
    yl2 = np.min(y2, axis = 1)
    yh2 = np.max(y2, axis = 1)
    
    yl3 = np.min(y2, axis = 1)
    yh3 = np.max(y2, axis = 1)
    
    yl4 = np.min(y2, axis = 1)
    yh4 = np.max(y2, axis = 1)
    
     
    f1 = figure(title= ynames[0], x_axis_label=' ', y_axis_label=ynames[0], height=300, width = 2440)
    f1.varea(x=xx, y1=yh1, y2=yl1 ,fill_color="red", fill_alpha=0.5)
    f1.line(xx, yl1, legend_label="Maximum Error.", color="crimson", line_width=0.2)
    f1.line(xx, yh1, legend_label="Minimum Error.", color="darkred", line_width=0.2)
    f1.circle(xx, yobs, color='blue', size = 5)
    
    f2 = figure(title= ynames[1], x_axis_label=' ', y_axis_label=ynames[1], height=300, width = 2440)
    f2.varea(x=xx, y1=yh2, y2=yl2 ,fill_color="red", fill_alpha=0.5)
    f2.line(xx, yl2, legend_label="Maximum Error.", color="crimson", line_width=0.2)
    f2.line(xx, yh2, legend_label="Minimum Error.", color="darkred", line_width=0.2)
    f2.circle(xx, yobs, color='blue', size = 5)
    
    f3 = figure(title= ynames[2], x_axis_label=' ', y_axis_label=ynames[2], height=300, width = 2440)
    f3.varea(x=xx, y1=yh3, y2=yl3 ,fill_color="red", fill_alpha=0.5)
    f3.line(xx, yl3, legend_label="Maximum Error.", color="crimson", line_width=0.2)
    f3.line(xx, yh3, legend_label="Minimum Error.", color="darkred", line_width=0.2)
    f3.circle(xx, yobs, color='blue', size = 5)
    
    f4 = figure(title= ynames[3], x_axis_label='Node', y_axis_label=ynames[3], height=300, width = 2440)
    f4.varea(x=xx, y1=yh4, y2=yl4 ,fill_color="red", fill_alpha=0.5)
    f4.line(xx, yl4, legend_label="Maximum Error.", color="crimson", line_width=0.2)
    f4.line(xx, yh4, legend_label="Minimum Error.", color="darkred", line_width=0.2)
    f4.circle(xx, yobs, color='blue', size = 5)
    
    return show(column(f1, f2, f3, f4))