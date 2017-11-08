#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 10:18:57 2017

@author: edwin
"""
import numpy as np

class SpongeBC3D:
    
    def __init__(self, domain, idealGas, bc):
        
        self.Nx = domain.Nx
        self.x = domain.x
        self.Lx = domain.Lx
        self.dx = domain.dx
        
        self.Ny = domain.Ny
        self.y = domain.y
        self.Ly = domain.Ly
        self.dy = domain.dy
        
        self.Nz = domain.Nz
        self.z = domain.z
        self.Lz = domain.Lz
        self.dz = domain.dz
        
        self.bcX0 = bc.bcX0
        self.bcX1 = bc.bcX1
        self.bcY0 = bc.bcY0
        self.bcY1 = bc.bcY1
        self.bcZ0 = bc.bcZ0
        self.bcZ1 = bc.bcZ1
        
        self.spongeAvgT     = 10.0
        self.spongeEpsP     = 0.005
        self.spongeP        = 1/idealGas.gamma
        self.spongeStrength = 12
        self.spongeLengthX  = 0.25*self.Lx
        self.spongeLengthY  = 0.25*self.Ly
        self.spongeLengthZ  = 0.25*self.Ly
        
        self.spongeSigma   = np.zeros((self.Nx, self.Ny, self.Nz))
        self.spongeRhoAvg  = np.zeros((self.Nx, self.Ny, self.Nz))
        self.spongeRhoUAvg = np.zeros((self.Nx, self.Ny, self.Nz))
        self.spongeRhoVAvg = np.zeros((self.Nx, self.Ny, self.Nz))
        self.spongeRhoWAvg = np.zeros((self.Nx, self.Ny, self.Nz))        
        self.spongeRhoEAvg = np.zeros((self.Nx, self.Ny, self.Nz))
        
        if self.bcX0 == "SPONGE":           
            for i in range(self.Nx):
                if self.x[i] < self.spongeLengthX:
                    spongeX = (self.spongeLengthX-self.x[i])/self.spongeLengthX
                    for j in range(self.Ny):
                        for k in range(self.Nz):
                            self.spongeSigma[i,j,k] = max(self.spongeStrength * (0.068*(spongeX)**2 
                                    + 0.845*(spongeX)**8), self.spongeSigma[i,j,k])   
                    
        if self.bcX1 == "SPONGE":
            for i in range(self.Nx):
                if self.x[i] > self.Lx - self.spongeLengthX:
                    spongeX = (self.x[i]-(self.Lx-self.spongeLengthX))/self.spongeLengthX
                    for j in range(self.Ny):
                        for k in range(self.Nz):
                            self.spongeSigma[i,j,k] = max(self.spongeStrength*(0.068*(spongeX)**2 
                                    + 0.845*(spongeX)**8), self.spongeSigma[i,j,k]) 
                        
        if self.bcY0 == "SPONGE":           
            for j in range(self.Ny):
                if self.y[j] < self.spongeLengthY:
                    spongeY = (self.spongeLengthY-self.y[j])/self.spongeLengthY
                    for i in range(self.Nx):
                        for k in range(self.Nz):
                            self.spongeSigma[i,j,k] = max(self.spongeStrength * (0.068*(spongeY)**2 
                                    + 0.845*(spongeY)**8), self.spongeSigma[i,j,k])   
                    
        if self.bcY1 == "SPONGE":
            for j in range(self.Ny):
                if self.y[j] > self.Ly - self.spongeLengthY:
                    spongeY = (self.y[j]-(self.Ly-self.spongeLengthY))/self.spongeLengthY
                    for i in range(self.Nx):
                        for k in range(self.Nz):
                            self.spongeSigma[i,j,k] = max(self.spongeStrength*(0.068*(spongeY)**2 
                                    + 0.845*(spongeY)**8), self.spongeSigma[i,j,k])
                            
        if self.bcZ0 == "SPONGE":           
            for k in range(self.Nz):
                if self.z[k] < self.spongeLengthZ:
                    spongeZ = (self.spongeLengthZ-self.z[k])/self.spongeLengthZ
                    for i in range(self.Nx):
                        for j in range(self.Ny):
                            self.spongeSigma[i,j,k] = max(self.spongeStrength * (0.068*(spongeZ)**2 
                                    + 0.845*(spongeZ)**8), self.spongeSigma[i,j,k])   
                    
        if self.bcZ1 == "SPONGE":
            for k in range(self.Nz):
                if self.z[k] > self.Lz - self.spongeLengthZ:
                    spongeZ = (self.z[k]-(self.Lz-self.spongeLengthZ))/self.spongeLengthZ
                    for i in range(self.Nx):
                        for j in range(self.Ny):
                            self.spongeSigma[i,j,k] = max(self.spongeStrength*(0.068*(spongeZ)**2 
                                    + 0.845*(spongeZ)**8), self.spongeSigma[i,j,k])
                        
                    