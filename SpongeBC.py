#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 10:18:57 2017

@author: edwin
"""
import numpy as np

class SpongeBC:
    
    def __init__(self, N, x, L, idealGas, bcX0, bcX1):
        self.N = N
        
        self.spongeAvgT     = 1.0
        self.spongeEpsP     = 0.005
        self.spongeP        = 1/idealGas.gamma
        self.spongeStrength = 2
        self.spongeLength   = 0.125*L
        
        self.spongeSigma   = np.zeros(self.N)
        self.spongeRhoAvg  = np.zeros(self.N)
        self.spongeRhoUAvg = np.zeros(self.N)
        self.spongeRhoEAvg = np.zeros(self.N)
        
        if bcX0 == "SPONGE":           
            for i in range(self.N):
                if x[i] < self.spongeLength:
                    spongeX = (self.spongeLength-x[i])/self.spongeLength;
                    self.spongeSigma[i] = self.spongeStrength * (0.068*(spongeX)**2 
                               + 0.845*(spongeX)**8)   
                    
        if self.bcX1 == "SPONGE":
            for i in range(self.N):
                if x[i] > L - self.spongeLength:
                    spongeX = (x[i]-(L-self.spongeLength))/self.spongeLength
                    self.spongeSigma[i] = self.spongeStrength*(0.068*(spongeX)**2 
                               + 0.845*(spongeX)**8) 
                    
                    
                    