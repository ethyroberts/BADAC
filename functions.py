#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 14:23:01 2019

@author: ethan
"""

#Hello! If you came here looking for beautiful code, look elsewhere.

import numpy as np

import matplotlib.pyplot as plt
import functools as funct
import operator as op

from matplotlib import gridspec
from matplotlib.patches import Patch
from matplotlib.patches import Polygon
from scipy import interpolate

def gauss(x,mn,sd):
    return 1/np.sqrt(2*np.pi*sd**2) * np.exp(-((x-mn)/sd)**2)


def PI(x):
    """
    x: input parameter (array)
    returns the product of elements in the array"""
    
    return funct.reduce(op.mul,x)


def spline(x,y,n):
    xs = np.linspace(x[0],x[-1],n)
    if len(x) < 4:
        f = interpolate.interp1d(x,y,kind='linear')
    else: 
        f = interpolate.interp1d(x,y,kind='cubic')
    return f(xs)


def add_sausage(x,y,yerr,n,c,**kwargs):
    top_curve = y+yerr
    bottom_curve = y-yerr
    
    xspline = np.linspace(x[0],x[-1],n)
    yspline_top = spline(x,top_curve,n)
    yspline_bottom = spline(x,bottom_curve,n)
    
    plt.fill_between(xspline,yspline_top,yspline_bottom,color=c,alpha=0.2,**kwargs)
    plt.plot(xspline,yspline_top,color=c)
    plt.plot(xspline,yspline_bottom,color=c)

    
def Pi_lin(di,sigdi,yi,sigyi):
    """
    Returns the probability of one point being similar to another point
    Input parameters:
    di: datum being measured
    sigdi: datum errorbar
    yi: training datum
    sigyi: training datum errorbar
    ybari: approximated mean for assumed distribution
    Ryi: approximated errorbar for assumed distribution
       
    Output:
    p1-p2+p3: unnormalised log probability"""
    
    C1 = sigdi**-2
    C2 = sigyi**-2
    
    p1 = np.log((2*np.pi*(C1+C2))**-0.5 / (sigdi*sigyi))
    p2 = 0.5 * (C1*di**2 + C2*yi**2)
    p3 = 0.5 * ((di*C1 + yi*C2)**2) / (C1 + C2)
    return np.exp(p1-p2+p3)


def schematic_data(d1,d2,derr1,derr2):
    global ymax,ymin,x,yo0,yo1,d,sig_yo0,sig_yo1,sig_d,y_sub1,y_sub2,yo0_fit_yt1,yo1_fit_yt1,yo0_fit_yt2,yo1_fit_yt2
    ymax = 7.0
    ymin = 0.5
    n_sub = 10000
    
    x       = np.array([1.0,2.0,3.0,4.0])
    yo0     = np.array([3.4,3.9,2.5,2.5])
    yo1     = np.array([2.0,2.5,4.0,4.0])
    d       = np.array([2.5,d1,d2,4.0])
    sig_yo0 = np.array([0.8,0.8,1.0,1.0])
    sig_yo1 = np.array([0.5,0.5,0.8,0.8])
    sig_d   = np.array([0.8,derr1,derr2,0.8])

    prob10=Pi_lin(d[1],sig_d[1],yo0[1],sig_yo0[1])
    prob20=Pi_lin(d[2],sig_d[2],yo0[2],sig_yo0[2])
    prob11=Pi_lin(d[1],sig_d[1],yo1[1],sig_yo1[1])
    prob21=Pi_lin(d[2],sig_d[2],yo1[2],sig_yo1[2])
    print(prob10)
    print(prob20)
    print(prob11)
    print(prob21)
    
    y_sub1 = np.linspace(ymin,ymax,n_sub)
    y_sub2 = np.linspace(ymin,ymax,n_sub)
    yo0_fit_yt1 = prob20 * gauss(y_sub1,d[1],sig_d[1]) * gauss(y_sub2,yo0[1],sig_yo0[1])
    yo1_fit_yt1 = prob21 * gauss(y_sub1,d[1],sig_d[1]) * gauss(y_sub2,yo1[1],sig_yo1[1])
    yo0_fit_yt2 = prob10 * gauss(y_sub1,d[2],sig_d[2]) * gauss(y_sub2,yo0[2],sig_yo0[2])
    yo1_fit_yt2 = prob11 * gauss(y_sub1,d[2],sig_d[2]) * gauss(y_sub2,yo1[2],sig_yo1[2])

    
def schematic_figure():
    c0='#43abc8'
    c1='#f58b4b'
    c2='#093145'
    c00=67/255
    c01=171/255
    c02=200/255
    c10=245/255
    c11=139/255
    c12=75/255

    fig = plt.figure(figsize=[9,5],dpi=100)
    gs = gridspec.GridSpec(1, 3, width_ratios=[4,1,1])

    custom_patches = [Polygon(((0,1),(0,0),(1,0)),facecolor=(c00,c01,c02,0.2),
                              edgecolor=(c00,c01,c02,1),label=r'$\tau=0$'),
                     Polygon(((0,1),(0,0),(1,0)),facecolor=(c10,c11,c12,0.2),
                              edgecolor=(c10,c11,c12,1),label=r'$\tau=1$')
                     ]

    ax1 = fig.add_subplot(gs[0,0])
    add_sausage(x,yo0,sig_yo0,100,c0,label=r'$y_o(\tau=0)$')
    add_sausage(x,yo1,sig_yo1,100,c1,label=r'$y_o(\tau=1)$')
    ax1.errorbar(x,d,yerr=sig_d,fmt='v',ms=10,capsize=8,label=r'$d$',c=c2)
    ax1.set_xlim([1.5,3.5])
    ax1.set_ylim([ymin,ymax])
    ax1.set_xticks([])
    ax1.set_ylabel('$y$',rotation=0)
    ax1.yaxis.set_label_coords(-0.07,0.5)
    handles, labels = ax1.get_legend_handles_labels()
    handles[0] = custom_patches[0]
    handles[1] = custom_patches[1]
    ax1.legend(handles,labels)

    ax2 = fig.add_subplot(gs[0,1])
    ax2.plot(yo0_fit_yt1,y_sub1,c=c0)
    ax2.plot(yo1_fit_yt1,y_sub1,c=c1)
    ax2.fill(yo0_fit_yt1,y_sub1,c=c0,alpha=0.2,label=r'$\tau=0$')
    ax2.fill(yo1_fit_yt1,y_sub1,c=c1,alpha=0.2,label=r'$\tau=1$')
    ax2.set_xlim([0,None])
    ax2.set_ylim([ymin,ymax])
    ax2.set_xticks([])
    ax2.set_ylabel('$y_{t_1}$',rotation=0)
    ax2.set_xlabel(r'$P(y_{t_1}|d,y_o,\tau)$')
    ax2.yaxis.set_label_coords(-0.32,0.5)
    ax2.legend(handles=custom_patches)

    ax3 = fig.add_subplot(gs[0,2])
    ax3.plot(yo0_fit_yt2,y_sub2,c=c0)
    ax3.plot(yo1_fit_yt2,y_sub2,c=c1)
    ax3.fill(yo0_fit_yt2,y_sub2,c=c0,alpha=0.2,label=r'$\tau=0$')
    ax3.fill(yo1_fit_yt2,y_sub2,c=c1,alpha=0.2,label=r'$\tau=1$')
    ax3.set_xlim([0,None])
    ax3.set_ylim([ymin,ymax])
    ax3.set_xticks([])
    ax3.set_ylabel('$y_{t_2}$',rotation=0)
    ax3.set_xlabel(r'$P(y_{t_2}|d,y_o,\tau)$')
    ax3.yaxis.set_label_coords(-0.32,0.5)
    ax3.legend(handles=custom_patches)

    fig.tight_layout()