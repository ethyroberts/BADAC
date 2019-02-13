#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:24:28 2017

@author: ethan
"""

import numpy as np

def bhm(d,sigd,y,sigy,labels):
    """
    Returns the normalised log probability of belonging to each of 2 datasets,
    here type0 and type1.
    
    Input Parameters:
    d: 1D array (observed data points)
    sigd: 1D array (observed errorbars)
    y: 2D array (template data points) - each row represents the i-th data curve
    sigy: 2D array (template errorbars) - each row represents the i-th data curve
    labels: 1D array with element format int (labels for input curves 'd')
    
    Output: 
    P0-sumPi: Total normalised log probability of belonging to type 0
    P1-sumPi: Total normalised log probability of belonging to type 1"""
    
    def Pi(di,sigdi,yi,sigyi):
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
        return p1-p2+p3
    
    """
    #class 0
    """
    index0 = np.where(labels==0)
    y0 = y[index0,:][0]
    sigy0 = sigy[index0,:][0]
    m0,n0 = np.shape(y0) #m number of training curves, n number of datapoints per curve
    
    P0matrix = Pi(d,sigd,y0,sigy0)
    P0array = np.sum(P0matrix,axis=1)    
    
    Ptau0 = 0.5
    P0 = (np.log(np.sum(np.exp(np.float128(P0array))))) + np.log(Ptau0) - np.log(m0)
    
    """
    #class 1
    """
    index1 = np.where(labels==1)
    y1 = y[index1,:][0]
    sigy1 = sigy[index1,:][0]
    m1,n1 = np.shape(y1) #m number of training curves, n number of datapoints per curve
    
    P1matrix = Pi(d,sigd,y1,sigy1)
    P1array = np.sum(P1matrix,axis=1)
    
    Ptau1 = 0.5
    P1 = (np.log(np.sum(np.exp(np.float128(P1array))))) + np.log(Ptau1) - np.log(m1)
    return P0,P1

def bhm_window(d,sigd,y,sigy,labels,window):
    n = len(d)
    n_windows = n - window + 1
    
    P0 = np.empty(n_windows)
    P1 = np.empty(n_windows)
    
    for i in range(n_windows):
        P0[i],P1[i]=bhm(d[i:window+i],sigd[i:window+i],y[:,i:i+window],sigy[:,i:i+window],labels)
    return P0,P1

def bhm_normalised(d,sigd,y,sigy,labels):
    p0,p1 = bhm(d,sigd,y,sigy,labels)
    Ptot = np.log(np.exp(np.float128(p0)) + np.exp(np.float128(p1)))
    P0 = p0 - Ptot
    P1 = p1 - Ptot
    return P0,P1