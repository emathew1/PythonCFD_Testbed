# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 09:11:33 2017

@author: mat.mathews
"""

import numpy as np
import matplotlib.pyplot as plt

from CollocatedSolver import Domain2D
from CollocatedSolver import BC2D
from CollocatedSolver import TimeStepping
from CollocatedSolver import CSolver2D

plt.ion()

#Domain information
Nx = 75
Ny = 150
Lx = 1
Ly = 2
x = np.linspace(0,Lx,Nx)
y = np.linspace(0,Ly,Ny)
domain = Domain2D(Nx, Ny, x, y, Lx, Ly)
[X, Y] = np.meshgrid(x,y)

#Boundary Condition information
bcYType = "PERIODIC"
bcY0 = "PERIODIC"
bcY1 = "PERIODIC"
bcXType = "DIRICHLET"
bcX0 = "SPONGE"
bcX1 = "SPONGE"
bc = BC2D(bcXType, bcX0, bcX1, bcYType, bcY0, bcY1)

#Time stepping information
CFL = 0.25
maxTimeStep = 10000
maxTime = 1000
plotStep = 25
filterStep = 1
timestepping = TimeStepping(CFL, maxTimeStep, maxTime, plotStep, filterStep)

#Filtering stuff
alphaF = 0.49
#Fluid stuff
mu_ref = 0.0001

##Ghia results
#yy = np.array([0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.50, 0.6172, 0.7344, 0.8516,  0.9531, 0.9609, 0.9688, 0.9766, 1.0])
#uu100 = np.array([0.0, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.0]) 
#xx = np.array([0.0, 0.0625, 0.0703, 0.0781, 0.0928, 0.1563, 0.2266, 0.2344, 0.5, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0])
#vv100 = np.array([0.0, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864, -0.07391, -0.05906, 0.0])

##Create our solver object
csolver = CSolver2D(domain, bc, timestepping, alphaF, mu_ref)


#Allocate initial condition variables we need
U0   = np.ones((Nx,Ny))
V0   = np.ones((Nx,Ny))
rho0 = np.ones((Nx,Ny))
p0   = np.ones((Nx,Ny))

#Initial conditions
for i in range(Nx):
    for j in range(Ny):
        if x[i] > Lx/2:
            V0[i,j] = 0.7
            U0[i,j] = 0.0
            rho0[i,j] = 1.0
            p0[i,j] = 1.0/csolver.idealGas.gamma + np.random.rand(1)/20
        else:
            V0[i,j] = -0.7
            U0[i,j] = 0.0
            rho0[i,j] = 1.0
            p0[i,j] = 1.0/csolver.idealGas.gamma + np.random.rand(1)/20
        
#        if x[i] > Lx/4 and x[i] < 3*Lx/4 and y[j] > Ly/8 and y[j] < 2*Ly/3 :
#            U0[i,j]   = 0.0
#            V0[i,j]   = 0.0
#            rho0[i,j] = 1 + 0.05*np.exp(-((x[i]-(Lx/2))**2 + (y[j]-(Ly/3))**2)/0.001)
#            p0[i,j]   = (1 + 0.05*np.exp(-((x[i]-(Lx/2))**2 + (y[j]-(Ly/3))**2)/0.001))/csolver.idealGas.gamma
#        else:
#            U0[i,j]   = 0.0
#            V0[i,j]   = 0.0
#            rho0[i,j] = 1.0
#            p0[i,j]   = 1.0/csolver.idealGas.gamma

#Set the initial conditions in the solver        
csolver.setInitialConditions(rho0, U0, V0, p0)

while csolver.done == False:
  
    csolver.calcDtFromCFL()
    
    #RK Step 1
    rkStep = 1
    
    csolver.preStepBCHandling(csolver.rho1, csolver.rhoU1, csolver.rhoV1, csolver.rhoE1)
    
    csolver.preStepDerivatives()
    
    csolver.solveContinuity(csolver.rho1, csolver.rhoU1, csolver.rhoV1, csolver.rhoE1)
    csolver.solveXMomentum_PV(csolver.rho1, csolver.rhoU1, csolver.rhoV1, csolver.rhoE1)
    csolver.solveYMomentum_PV(csolver.rho1, csolver.rhoU1, csolver.rhoV1, csolver.rhoE1)
    csolver.solveEnergy_PV(csolver.rho1, csolver.rhoU1, csolver.rhoV1, csolver.rhoE1)

    csolver.postStepBCHandling(csolver.rho1, csolver.rhoU1, csolver.rhoV1, csolver.rhoE1)
    
    csolver.updateConservedData(rkStep)
    csolver.updateNonConservedData(rkStep)

    #RK Step 2
    rkStep = 2
    
    csolver.preStepBCHandling(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)
    
    csolver.preStepDerivatives()
    
    csolver.solveContinuity(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)
    csolver.solveXMomentum_PV(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)
    csolver.solveYMomentum_PV(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)
    csolver.solveEnergy_PV(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)
    
    csolver.postStepBCHandling(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)

    csolver.updateConservedData(rkStep)
    csolver.updateNonConservedData(rkStep)   
    
    #RK Step 3
    rkStep = 3
    
    csolver.preStepBCHandling(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)
    
    csolver.preStepDerivatives()
    
    csolver.solveContinuity(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)
    csolver.solveXMomentum_PV(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)
    csolver.solveYMomentum_PV(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)
    csolver.solveEnergy_PV(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)
    
    csolver.postStepBCHandling(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)

    csolver.updateConservedData(rkStep)
    csolver.updateNonConservedData(rkStep)       

    
    #RK Step 4
    rkStep = 4
    
    csolver.preStepBCHandling(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)
    
    csolver.preStepDerivatives()
    
    csolver.solveContinuity(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)
    csolver.solveXMomentum_PV(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)
    csolver.solveYMomentum_PV(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)
    csolver.solveEnergy_PV(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)
    
    csolver.postStepBCHandling(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)

    csolver.updateConservedData(rkStep)
    

    #End of step stuff
    
    #Filter and/or move data to _1 containers
    csolver.filterPrimativeValues()
    
    #Update the sponge if applicable
    csolver.updateSponge()

    #Update the non conserved data after we filter
    csolver.updateNonConservedData(rkStep)       
    
    #plot/check solution
    csolver.checkSolution()
    



