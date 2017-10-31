# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 09:11:33 2017

@author: mat.mathews
"""

import numpy as np
import matplotlib.pyplot as plt

from CollocatedSolver import Domain3D
from CollocatedSolver import BC3D
from CollocatedSolver import TimeStepping
from CollocatedSolver import CSolver3D

plt.ion()

#Domain information
Nx = 50    
Ny = 8
Nz = 8
Lx = 1.0
Ly = 0.1
Lz = 0.1
x = np.linspace(0,Lx,Nx)
y = np.linspace(0,Ly,Ny)
z = np.linspace(0,Lz,Nz)
domain = Domain3D(Nx, Ny, Nz, x, y, z, Lx, Ly, Lz)
[X, Y, Z] = np.meshgrid(x, y, z)

#Boundary Condition information
bcXType = "PERIODIC"
bcX0 = "PERIODIC"
bcX1 = "PERIODIC"
bcYType = "PERIODIC"
bcY0 = "PERIODIC"
bcY1 = "PERIODIC"
bcZType = "PERIODIC"
bcZ0 = "PERIODIC"
bcZ1 = "PERIODIC"
bc = BC3D(bcXType, bcX0, bcX1, bcYType, bcY0, bcY1, bcZType, bcZ0, bcZ1)

#Time stepping information
CFL = 0.15
maxTimeStep = 10
maxTime = 1000
plotStep = 1
filterStep = 1
timestepping = TimeStepping(CFL, maxTimeStep, maxTime, plotStep, filterStep)

#Filtering stuff
alphaF = 0.45
#Fluid stuff
mu_ref = 0.01

##Ghia results
#yy = np.array([0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.50, 0.6172, 0.7344, 0.8516,  0.9531, 0.9609, 0.9688, 0.9766, 1.0])
#uu100 = np.array([0.0, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.0]) 
#xx = np.array([0.0, 0.0625, 0.0703, 0.0781, 0.0928, 0.1563, 0.2266, 0.2344, 0.5, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0])
#vv100 = np.array([0.0, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864, -0.07391, -0.05906, 0.0])

##Create our solver object
csolver = CSolver3D(domain, bc, timestepping, alphaF, mu_ref)


#Allocate initial condition variables we need
U0   = np.ones((Nx,Ny,Nz))
V0   = np.ones((Nx,Ny,Nz))
W0   = np.ones((Nx,Ny,Nz))
rho0 = np.ones((Nx,Ny,Nz))
p0   = np.ones((Nx,Ny,Nz))

#Initial conditions
for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz):
#                U0[i,j,k] = 0.0
#                V0[i,j,k] = 0.0
#                W0[i,j,k] = 0.0
#                if x[i] > 0.5:
#                    rho0[i,j,k] = 1.0
#                    p0[i,j,k] = 1.0/csolver.idealGas.gamma 
#                else:
#                    rho0[i,j,k] = 0.825
#                    p0[i,j,k] = 0.8/csolver.idealGas.gamma
            
            if x[i] > Lx/4 and x[i] < 3*Lx/4:
                U0[i,j,k]   = 0.0
                V0[i,j,k]   = 0.0
                W0[i,j,k]   = 0.0
                rho0[i,j,k] = 1 + 0.05*np.exp(-((x[i]-(Lx/2))**2 + (y[j]-(Ly/3))**2)/0.001)
                p0[i,j,k]   = (1 + 0.005*np.exp(-((x[i]-(Lx/2))**2 + (y[j]-(Ly/3))**2)/0.001))/csolver.idealGas.gamma
            else:
                U0[i,j,k]   = 0.0
                V0[i,j,k]   = 0.0
                W0[i,j,k]   = 0.0
                rho0[i,j,k] = 1.0
                p0[i,j,k]   = 1.0/csolver.idealGas.gamma

#Set the initial conditions in the solver        
csolver.setInitialConditions(rho0, U0, V0, W0, p0)

while csolver.done == False:
  
    csolver.calcDtFromCFL()
    
    #RK Step 1
    rkStep = 1
    
    #csolver.preStepBCHandling(csolver.rho1, csolver.rhoU1, csolver.rhoV1, csolver.rhoW1, csolver.rhoE1)
    
    csolver.preStepDerivatives()
    
    csolver.solveContinuity(csolver.rho1, csolver.rhoU1, csolver.rhoV1, csolver.rhoW1, csolver.rhoE1)
    csolver.solveXMomentum_PV(csolver.rho1, csolver.rhoU1, csolver.rhoV1, csolver.rhoW1, csolver.rhoE1)
    csolver.solveYMomentum_PV(csolver.rho1, csolver.rhoU1, csolver.rhoV1, csolver.rhoW1, csolver.rhoE1)
    csolver.solveZMomentum_PV(csolver.rho1, csolver.rhoU1, csolver.rhoV1, csolver.rhoW1, csolver.rhoE1)        
    csolver.solveEnergy_PV(csolver.rho1, csolver.rhoU1, csolver.rhoV1, csolver.rhoW1, csolver.rhoE1)
#
#    #csolver.postStepBCHandling(csolver.rho1, csolver.rhoU1, csolver.rhoV1, csolver.rhoW1, csolver.rhoE1)
#    
    csolver.updateConservedData(rkStep)
    csolver.updateNonConservedData(rkStep)

    #RK Step 2
    rkStep = 2
    
    #csolver.preStepBCHandling(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoWk, csolver.rhoEk)
    
    csolver.preStepDerivatives()
    
    csolver.solveContinuity(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoWk, csolver.rhoEk)
    csolver.solveXMomentum_PV(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoWk, csolver.rhoEk)
    csolver.solveYMomentum_PV(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoWk, csolver.rhoEk)
    csolver.solveZMomentum_PV(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoWk, csolver.rhoEk)
    csolver.solveEnergy_PV(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoWk, csolver.rhoEk)
    
    #csolver.postStepBCHandling(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoWk, csolver.rhoEk)

    csolver.updateConservedData(rkStep)
    csolver.updateNonConservedData(rkStep)   
    
    #RK Step 3
    rkStep = 3
    
    #csolver.preStepBCHandling(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoWk, csolver.rhoEk)
    
    csolver.preStepDerivatives()
    
    csolver.solveContinuity(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoWk, csolver.rhoEk)
    csolver.solveXMomentum_PV(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoWk, csolver.rhoEk)
    csolver.solveYMomentum_PV(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoWk, csolver.rhoEk)
    csolver.solveZMomentum_PV(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoWk, csolver.rhoEk)
    csolver.solveEnergy_PV(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoWk, csolver.rhoEk)
    
    #csolver.postStepBCHandling(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoWk, csolver.rhoEk)

    csolver.updateConservedData(rkStep)
    csolver.updateNonConservedData(rkStep)    

    
    #RK Step 4
    rkStep = 4
    
    #csolver.preStepBCHandling(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoWk, csolver.rhoEk)
    
    csolver.preStepDerivatives()
    
    csolver.solveContinuity(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoWk, csolver.rhoEk)
    csolver.solveXMomentum_PV(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoWk, csolver.rhoEk)
    csolver.solveYMomentum_PV(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoWk, csolver.rhoEk)
    csolver.solveZMomentum_PV(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoWk, csolver.rhoEk)
    csolver.solveEnergy_PV(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoWk, csolver.rhoEk)
    
    #csolver.postStepBCHandling(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoWk, csolver.rhoEk)

    csolver.updateConservedData(rkStep)    

    #End of step stuff
    
    #Filter and/or move data to _1 containers
    csolver.filterConservedValues()
    
    #Update the sponge if applicable
    csolver.updateSponge()

    #Update the non conserved data after we filter
    csolver.updateNonConservedData(rkStep)       
    
    #plot/check solution
    csolver.checkSolution()
    
    #csolver.done = True


