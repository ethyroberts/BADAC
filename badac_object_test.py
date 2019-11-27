#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 20:44:33 2019

@author: ethan
"""

import numpy as np


class badac_gauss:
    def __init__(self,X_train,Xerr_train,Y_train):
        self.X_train = X_train
        self.Xerr_train = Xerr_train
        self.Y_train = Y_train
        
        self.class_list =  np.unique(Y_train)
        self.nclasses = self.class_list.shape[0]
    
    def test(self,X_test,Xerr_test,contamination=0):
        self.X_test = X_test
        self.Xerr_test = Xerr_test
        self.contamination = contamination
        
        self.ntest = X_test.shape[0]
        self.prob_matrix = np.zeros(shape=[self.ntest,self.nclasses])
        
        for i in range(self.nclasses):
            self.index = np.where(self.Y_train == self.class_list[i])
            self.prob_matrix[:,i] = class_i(self.X_test,self.Xerr_test,self.X_train[self.index],
                            self.Xerr_train[self.index],self.ntest).evaluate()
            delattr(self,'index')
    
    def print_log_proba(self):
        return self.prob_matrix


class class_i(badac_gauss):
    def __init__(self,X_test,Xerr_test,X_train,Xerr_train,ntest):
        self.X_test = X_test
        self.Xerr_test = Xerr_test
        self.X_train = X_train
        self.Xerr_train = Xerr_train
        self.ntest = ntest
    
    def evaluate(self):
        def Pi(di,sigdi,yi,sigyi):
            C1 = sigdi**-2
            C2 = sigyi**-2
            
            p1 = np.log((2*np.pi*(C1+C2))**-0.5 / (sigdi*sigyi))
            p2 = 0.5 * (C1*di**2 + C2*yi**2)
            p3 = 0.5 * ((di*C1 + yi*C2)**2) / (C1 + C2)
            return p1-p2+p3
        
        self.prob_list = np.zeros(self.ntest)
        self.ntrain = self.X_train.shape[0]
        
        for i in range(self.ntest):
            self.P = np.sum(np.exp(np.float128(np.sum(Pi(self.X_test[i,:],self.Xerr_test[i,:],self.X_train,
                                                         self.Xerr_train),axis=1))))
            self.prob_list[i] = np.log(self.P) + np.log(0.5) - np.log(self.ntrain)
            delattr(self,'P')
            
        return self.prob_list
        

X_train = np.array([[1,2,3],[4,5,6],[7,8,9],[1,2,4]])
Xerr_train = np.array([[1,1,1],[1,1,1],[1,1,1],[1,1,1]])
Y_train = np.array([0,1,1,0])

X_test = np.array([[1,2,3],[4,5,6],[7,8,9],[1,2,4]])
Xerr_test = np.array([[1,1,1],[1,1,1],[1,1,1],[1,1,1]])

clf = badac_gauss(X_train,Xerr_train,Y_train)
clf.test(X_test,Xerr_test)
x = clf.print_log_proba()

print(x)

def bhm(d,sigd,y,sigy,labels):
    def Pi(di,sigdi,yi,sigyi):
        C1 = sigdi**-2
        C2 = sigyi**-2
        
        p1 = np.log((2*np.pi*(C1+C2))**-0.5 / (sigdi*sigyi))
        p2 = 0.5 * (C1*di**2 + C2*yi**2)
        p3 = 0.5 * ((di*C1 + yi*C2)**2) / (C1 + C2)
        return p1-p2+p3
    
    index0 = np.where(labels==0)
    y0 = y[index0,:][0]
    sigy0 = sigy[index0,:][0]
    m0,n0 = np.shape(y0) #m number of training curves, n number of datapoints per curve
    
    P0matrix = Pi(d,sigd,y0,sigy0)
    P0array = np.sum(P0matrix,axis=1)    
    
    Ptau0 = 0.5
    P0 = (np.log(np.sum(np.exp(np.float128(P0array))))) + np.log(Ptau0) - np.log(m0)
    return P0