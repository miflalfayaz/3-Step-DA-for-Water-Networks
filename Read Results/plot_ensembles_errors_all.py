# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 22:21:10 2022

@author: ibrah
"""
def plot_ensembles_errors_all(y1,y2,y3,y4,ynames,yrange,yobs):
    import numpy as np
    from bokeh.plotting import figure, show
    from bokeh.layouts import column
    nx = np.shape(y1)[0]
    xx = np.arange(1,nx+1)
    
    yl1 = np.min(y1, axis = 1) - yobs
    yh1 = np.max(y1, axis = 1) - yobs
    
    yl2 = np.min(y2, axis = 1) - yobs
    yh2 = np.max(y2, axis = 1) - yobs
    
    yl3 = np.min(y2, axis = 1) - yobs
    yh3 = np.max(y2, axis = 1) - yobs
    
    yl4 = np.min(y2, axis = 1) - yobs
    yh4 = np.max(y2, axis = 1) - yobs
    
    yobse = np.mean(y1,1)- yobs
     
    f1 = figure(title= "Combined Errors", x_axis_label=' Nodes ', y_axis_label=' ' , height=1200, width = 2440)
    f1.varea(x=xx, y1=yh1, y2=yl1 , fill_alpha=0.5, legend_label = ynames[0], color="crimson")
    # f1.line(xx, yl1, legend_label="Maximum Error.", color="crimson", line_width=0.2)
    # f1.line(xx, yh1, legend_label="Minimum Error.", color="darkred", line_width=0.2)
    f1.circle(xx, np.mean(y1,1)- yobs, size = 5)
    

    f1.varea(x=xx, y1=yh2, y2=yl2 , fill_alpha=0.5, legend_label = ynames[1], color="darkred")
    # f1.line(xx, yl2, legend_label="Maximum Error.", color="crimson", line_width=0.2)
    # f1.line(xx, yh2, legend_label="Minimum Error.", color="darkred", line_width=0.2)
    f1.circle(xx, np.mean(y2,1)- yobs, size = 5)
    

    f1.varea(x=xx, y1=yh3, y2=yl3 , fill_alpha=0.5, legend_label = ynames[2], color="red")
    # f1.line(xx, yl3, legend_label="Maximum Error.", color="crimson", line_width=0.2)
    # f1.line(xx, yh3, legend_label="Minimum Error.", color="darkred", line_width=0.2)
    f1.circle(xx, np.mean(y3,1)- yobs, size = 5)
    

    f1.varea(x=xx, y1=yh4, y2=yl4 , fill_alpha=0.5, legend_label = ynames[3],  color="orange")
    # f1.line(xx, yl4, legend_label="Maximum Error.", color="crimson", line_width=0.2)
    # f1.line(xx, yh4, legend_label="Minimum Error.", color="darkred", line_width=0.2)
    f1.circle(xx, np.mean(y4,1)- yobs, size = 5)
    
    return show(f1)