#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:53:48 2019

@author: ethan
"""

import numpy as np


class BADAC:
    def __init__(self, contamination):
        self.contamination = contamination
        
    def fit(self, X, Xerr, Y):
        self.X = X
        self.Xerr = Xerr
        self.Y = Y
        
        self.class_list =  np.unique(self.Y)
        self.nclasses = self.class_list.shape[0]
    
    def predict(self, Xtest, Xerrtest):
        self.Xtest = Xtest
        self.Xerrtest = Xerrtest
        
        self.ntest = len(self.Xtest)
        
        self.probmatrix = np.zeros(shape=[self.ntest,self.nclasses])
        for self.i in range(self.nclasses):
            self.index = np.where(self.Y == self.class_list[self.i])
            
            self.Xtrain = self.X[self.index]
            self.Xerrtrain = self.Xerr[self.index]
            
            self.probmatrix[:,self.i] = maineq(self.Xtest, self.Xerrtest,
                                               self.Xtrain, self.Xerrtrain)
        return self.probmatrix
    
    def classify(self):
        return self.probmatrix.argmax(axis=1)


def maineq(Xtest, Xerrtest, Xtrain, Xerrtrain):
    def Pi(di,sigdi,yi,sigyi):
        C1 = sigdi**-2
        C2 = sigyi**-2
        
        p1 = np.log((2*np.pi*(C1+C2))**-0.5 / (sigdi*sigyi))
        p2 = 0.5 * (C1*di**2 + C2*yi**2)
        p3 = 0.5 * ((di*C1 + yi*C2)**2) / (C1 + C2)
        return p1-p2+p3
    
    ntest = len(Xtest)
    ntrain = len(Xtrain)
    
    problist = np.zeros(ntest)
    for i in range(ntest):
        p = np.sum(Pi(Xtest[i], Xerrtest[i], Xtrain, Xerrtrain), axis=1)
        P = np.sum(np.exp(np.float128(p)))
        problist[i] = np.log(P) + np.log(0.5) - np.log(ntrain)
    return problist