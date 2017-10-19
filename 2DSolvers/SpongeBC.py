#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 10:18:57 2017

@author: edwin
"""
import numpy as np

class SpongeBC2D:
    
    def __init__(self, domain, idealGas, bc):
        
        self.Nx = domain.Nx
        self.x = domain.x
        self.Lx = domain.Lx
        self.dx = domain.dx
        
        self.Ny = domain.Ny
        self.y = domain.y
        self.Ly = domain.Ly
        self.dy = domain.dy
        
        self.bcX0 = bc.bcX0
        self.bcX1 = bc.bcX1
        self.bcY0 = bc.bcY0
        self.bcY1 = bc.bcY1
        
        self.spongeAvgT     = 10.0
        self.spongeEpsP     = 0.005
        self.spongeP        = 1/idealGas.gamma
        self.spongeStrength = 2
        self.spongeLengthX  = 0.125*self.Lx
        self.spongeLengthY  = 0.125*self.Ly
        
        self.spongeSigma   = np.zeros((self.Nx, self.Ny))
        self.spongeRhoAvg  = np.zeros((self.Nx, self.Ny))
        self.spongeRhoUAvg = np.zeros((self.Nx, self.Ny))
        self.spongeRhoVAvg = np.zeros((self.Nx, self.Ny))
        self.spongeRhoEAvg = np.zeros((self.Nx, self.Ny))
        
        if self.bcY0 == "SPONGE":           
            for i in range(self.Nx):
                if self.x[i] < self.spongeLengthX:
                    spongeX = (self.spongeLengthX-self.x[i])/self.spongeLengthX
                    for j in range(self.Ny):
                        self.spongeSigma[i,j] = max(self.spongeStrength * (0.068*(spongeX)**2 
                               + 0.845*(spongeX)**8), self.spongeSigma[i,j])   
                    
        if self.bcY1 == "SPONGE":
            for i in range(self.Nx):
                if self.x[i] > self.Lx - self.spongeLengthX:
                    spongeX = (self.x[i]-(self.Lx-self.spongeLengthX))/self.spongeLengthX
                    for j in range(self.Ny):
                        self.spongeSigma[i,j] = max(self.spongeStrength*(0.068*(spongeX)**2 
                               + 0.845*(spongeX)**8), self.spongeSigma[i,j]) 
                        
        if self.bcX0 == "SPONGE":           
            for j in range(self.Ny):
                if self.y[j] < self.spongeLengthY:
                    spongeY = (self.spongeLengthY-self.y[j])/self.spongeLengthY
                    for i in range(self.Nx):
                        self.spongeSigma[i,j] = max(self.spongeStrength * (0.068*(spongeY)**2 
                               + 0.845*(spongeY)**8), self.spongeSigma[i,j])   
                    
        if self.bcX1 == "SPONGE":
            for j in range(self.Ny):
                if self.y[j] > self.Ly - self.spongeLengthY:
                    spongeY = (self.y[j]-(self.Ly-self.spongeLengthY))/self.spongeLengthY
                    for i in range(self.Nx):
                        self.spongeSigma[i,j] = max(self.spongeStrength*(0.068*(spongeY)**2 
                               + 0.845*(spongeY)**8), self.spongeSigma[i,j])                     
                        
                    