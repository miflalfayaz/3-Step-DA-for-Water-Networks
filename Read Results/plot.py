# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 22:21:10 2022

@author: ibrah
"""
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show
from bokeh.layouts import column
    
def ensembles_values(y1,y2,y3,y4,ynames,yrange,yobs):

    nx = np.shape(y1)[0]
    xx = np.arange(1,nx+1)
    
    yl1 = np.min(y1, axis = 1)
    yh1 = np.max(y1, axis = 1)
    
    yl2 = np.min(y2, axis = 1)
    yh2 = np.max(y2, axis = 1)
    
    yl3 = np.min(y3, axis = 1)
    yh3 = np.max(y3, axis = 1)
    
    yl4 = np.min(y4, axis = 1)
    yh4 = np.max(y4, axis = 1)
    
     
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


def ensembles_errors(y1,y2,y3,y4,ynames,yrange,yobs, ti, Khaze):

    nx = np.shape(y1)[0]
    xx = np.arange(1,nx+1)
    
    yl1 = np.min(y1, axis = 1) - yobs
    yh1 = np.max(y1, axis = 1) - yobs 
    
    yl2 = np.min(y2, axis = 1) - yobs 
    yh2 = np.max(y2, axis = 1) - yobs 
    
    yl3 = np.min(y3, axis = 1) - yobs 
    yh3 = np.max(y3, axis = 1) - yobs 
    
    yl4 = np.min(y4, axis = 1) - yobs 
    yh4 = np.max(y4, axis = 1) - yobs 
    
    if yrange == [0, 0]:
        ymin = np.min([yl1, yl2, yl3, yl4])-1
        ymax = np.max([yh1, yh2, yh3, yh4])+0.2
        yrange = (ymin  , ymax )
    #yobse = np.mean(y1,1)- yobs
     
    fig, ax = plt.subplots(4, figsize=(10, 9),constrained_layout=True, dpi = 300)
    fig.suptitle('Ensemble Error for {},  {},  {},  {} at Timestep: {} for Modena with ne: 100'.format(ynames[0], ynames[1], ynames[2], ynames[3], ti), fontsize=16)
    
    ax[0].fill_between(xx, yl1, yh1, color='darkred', alpha=0.5)
    ax[0].plot(xx, yh1, '-', color='crimson', linewidth=2, label='Maximum Ensemble Error')
    ax[0].plot(xx, yl1, '-', color='darkred', linewidth=2, label='Minimum Ensemble Error')
    ax[0].plot(xx, np.mean(y1,1)- yobs, 'o', color='blue', markersize=5, label='Measured Value')
    ax[0].set_xlabel('Node')
    ax[0].set_ylabel(ynames[0])
    ax[0].set_title(ynames[0])
    ax[0].set_ylim(yrange)
    ax[0].legend(loc='lower center', borderaxespad= 0.2, ncol=4)
    
    ax[1].fill_between(xx, yl2, yh2, color='crimson', alpha=0.5)
    ax[1].plot(xx, yh2, '-', color='crimson', linewidth=2, label='Maximum Ensemble Error')
    ax[1].plot(xx, yl2, '-', color='darkred', linewidth=2, label='Minimum Ensemble Error')
    ax[1].plot(xx, np.mean(y2,1)- yobs, 'o', color='blue', markersize=5, label='Measured Value')
    ax[1].set_xlabel('Node')
    ax[1].set_ylabel(ynames[1])
    ax[1].set_title(ynames[1])
    ax[1].set_ylim(yrange)
    ax[1].legend(loc='lower center', borderaxespad= 0.2, ncol=4)
    
    ax[2].fill_between(xx, yl3, yh3, color='orange', alpha=0.5)
    ax[2].plot(xx, yh3, '-', color='crimson', linewidth=2, label='Maximum Ensemble Error')
    ax[2].plot(xx, yl3, '-', color='darkred', linewidth=2, label='Minimum Ensemble Error')
    ax[2].plot(xx, np.mean(y3,1)- yobs, 'o', color='blue', markersize=5, label='Measured Value')
    ax[2].set_xlabel('Node')
    ax[2].set_ylabel(ynames[2])
    ax[2].set_title(ynames[2])
    ax[2].set_ylim(yrange)
    ax[2].legend(loc='lower center', borderaxespad= 0.2, ncol=4)
    
    ax[3].fill_between(xx, yl4, yh4, color='green', alpha=0.5)
    ax[3].plot(xx, yh4, '-', color='crimson', linewidth=2, label='Maximum Ensemble Error')
    ax[3].plot(xx, yl4, '-', color='darkred', linewidth=2, label='Minimum Ensemble Error')
    ax[3].plot(xx, np.mean(y4,1)- yobs, 'o', color='blue', markersize=5, label='Measured Value')
    ax[3].set_xlabel('Node')
    ax[3].set_ylabel(ynames[3])
    ax[3].set_title(ynames[3])
    ax[3].set_ylim(yrange)
    ax[3].legend(loc='lower center', borderaxespad= 0.2, ncol=4)
    
    #handles, labels = ax[0].get_legend_handles_labels()
    #fig.legend(handles, labels, loc='lower center', borderaxespad=4.5, ncol=4)

    return plt.show() 


def ensembles_errors_all(y1,y2,y3,y4,ynames,yrange,yobs):
    
    nx = np.shape(y1)[0]
    xx = np.arange(1,nx+1)
    
    yl1 = np.min(y1, axis = 1) - yobs
    yh1 = np.max(y1, axis = 1) - yobs
    
    yl2 = np.min(y2, axis = 1) - yobs
    yh2 = np.max(y2, axis = 1) - yobs
    
    yl3 = np.min(y4, axis = 1) - yobs
    yh3 = np.max(y3, axis = 1) - yobs
    
    yl4 = np.min(y4, axis = 1) - yobs
    yh4 = np.max(y4, axis = 1) - yobs
    
    #yobse = np.mean(y1,1)- yobs
     
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