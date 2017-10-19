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
Nx = 100
Ny = 200
Lx = 1
Ly = 2
x = np.linspace(0,Lx,Nx)
y = np.linspace(0,Ly,Ny)
domain = Domain2D(Nx, Ny, x, y, Lx, Ly)
[X, Y] = np.meshgrid(x,y)

#Boundary Condition information
bcXType = "DIRICHLET"
bcX0 = "ADIABATIC_WALL"
bcX1 = "ADIABATIC_WALL"
bcYType = "DIRICHLET"
bcY0 = "ADIABATIC_WALL"
bcY1 = "ADIABATIC_WALL"
bc = BC2D(bcXType, bcX0, bcX1, bcYType, bcY0, bcY1)

#Time stepping information
CFL = 0.25
maxTimeStep = 10000
maxTime = 100
plotStep = 10
filterStep = 2
timestepping = TimeStepping(CFL, maxTimeStep, maxTime, plotStep, filterStep)

#Filtering stuff
alphaF = 0.4
#Fluid stuff
mu_ref = 0.0001



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
        if x[i] > Lx/4 and x[i] < 3*Lx/4 and y[j] > Ly/8 and y[j] < 2*Ly/3 :
            U0[i,j]   = 0.0
            V0[i,j]   = 0.0
            rho0[i,j] = 1 + 0.05*np.exp(-((x[i]-(Lx/2))**2 + (y[j]-(Ly/3))**2)/0.001)
            p0[i,j]   = (1 + 0.05*np.exp(-((x[i]-(Lx/2))**2 + (y[j]-(Ly/3))**2)/0.001))/csolver.idealGas.gamma
        else:
            U0[i,j]   = 0.0
            V0[i,j]   = 0.0
            rho0[i,j] = 1.0
            p0[i,j]   = 1.0/csolver.idealGas.gamma

#Set the initial conditions in the solver        
csolver.setInitialConditions(rho0, U0, V0, p0)

while csolver.done == False:
  
    csolver.calcDtFromCFL()
    
    #RK Step 1
    rkStep = 1
    
    csolver.preStepBCHandling(csolver.rho1, csolver.rhoU1, csolver.rhoV1, csolver.rhoE1)
    
    csolver.preStepDerivatives()
    
    csolver.solveContinuity(csolver.rho1, csolver.rhoU1, csolver.rhoV1, csolver.rhoE1)
    csolver.solveXMomentum(csolver.rho1, csolver.rhoU1, csolver.rhoV1, csolver.rhoE1)
    csolver.solveYMomentum(csolver.rho1, csolver.rhoU1, csolver.rhoV1, csolver.rhoE1)
    csolver.solveEnergy(csolver.rho1, csolver.rhoU1, csolver.rhoV1, csolver.rhoE1)

    csolver.postStepBCHandling(csolver.rho1, csolver.rhoU1, csolver.rhoV1, csolver.rhoE1)
    
    csolver.updateConservedData(rkStep)
    csolver.updateNonConservedData(rkStep)

    #RK Step 2
    rkStep = 2
    
    csolver.preStepBCHandling(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)
    
    csolver.preStepDerivatives()
    
    csolver.solveContinuity(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)
    csolver.solveXMomentum(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)
    csolver.solveYMomentum(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)
    csolver.solveEnergy(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)
    
    csolver.postStepBCHandling(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)

    csolver.updateConservedData(rkStep)
    csolver.updateNonConservedData(rkStep)   
    
    #RK Step 3
    rkStep = 3
    
    csolver.preStepBCHandling(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)
    
    csolver.preStepDerivatives()
    
    csolver.solveContinuity(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)
    csolver.solveXMomentum(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)
    csolver.solveYMomentum(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)
    csolver.solveEnergy(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)
    
    csolver.postStepBCHandling(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)

    csolver.updateConservedData(rkStep)
    csolver.updateNonConservedData(rkStep)       

    
    #RK Step 4
    rkStep = 4
    
    csolver.preStepBCHandling(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)
    
    csolver.preStepDerivatives()
    
    csolver.solveContinuity(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)
    csolver.solveXMomentum(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)
    csolver.solveYMomentum(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)
    csolver.solveEnergy(csolver.rhok, csolver.rhoUk, csolver.rhoVk, csolver.rhoEk)
    
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
    



