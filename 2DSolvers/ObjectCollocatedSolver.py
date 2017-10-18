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
Nx = 50
Ny = 50
Lx = 2*np.pi
Ly = np.pi
x = np.linspace(0,Lx,Nx)
y = np.linspace(0,Ly,Ny)
domain = Domain2D(Nx, Ny, x, y, Lx, Ly)
[X, Y] = np.meshgrid(x,y)

#Boundary Condition information
bcXType = "DIRICHLET"
bcX0 = "SPONGE"
bcX1 = "SPONGE"
bcYType = "DIRICHLET"
bcY0 = "SPONGE"
bcY1 = "SPONGE"
bc = BC2D(bcXType, bcX0, bcX1, bcYType, bcY0, bcY1)

#Time stepping information
CFL = 0.2
maxTimeStep = 10000
maxTime = 100
plotStep = 25
filterStep = 1.0
timestepping = TimeStepping(CFL, maxTimeStep, maxTime, plotStep, filterStep)

#Filtering stuff
alphaF = 0.49
#Fluid stuff
mu_ref = 0.00000001



##Create our solver object
csolver = CSolver2D(domain, bc, timestepping, alphaF, mu_ref)
#
#
##Allocate initial condition variables we need
#U0   = np.ones(N)
#rho0 = np.ones(N)
#p0   = np.ones(N)
#
##Initial conditions
#for i in range(N):
#    if x[i] > L/4 and x[i] < 3*L/4:
#        U0[i]   = 0.0
#        rho0[i] = 1 + 0.05*np.exp(-(x[i]-(L/2))**2/0.001)
#        p0[i]   = (1 + 0.05*np.exp(-(x[i]-(L/2))**2/0.001))/csolver.idealGas.gamma
#    else:
#        U0[i]   = 0.0
#        rho0[i] = 1.0
#        p0[i]   = 1.0/csolver.idealGas.gamma
#
##Set the initial conditions in the solver        
#csolver.setInitialConditions(rho0, U0, p0)
#
#while csolver.done == False:
#    
#    csolver.calcDtFromCFL()
#    
#    #RK Step 1
#    rkStep = 1
#    
#    csolver.preStepBCHandling(csolver.rho1, csolver.rhoU1, csolver.rhoE1)
#    
#    csolver.solveContinuity(csolver.rho1, csolver.rhoU1, csolver.rhoE1)
#    csolver.solveXMomentum(csolver.rho1, csolver.rhoU1, csolver.rhoE1)
#    csolver.solveEnergy(csolver.rho1, csolver.rhoU1, csolver.rhoE1)
#
#    csolver.postStepBCHandling(csolver.rhok2, csolver.rhoUk2, csolver.rhoEk2)
#    
#    csolver.updateConservedData(rkStep)
#    csolver.updateNonConservedData(rkStep)
#
#    #RK Step 2
#    rkStep = 2
#    
#    csolver.preStepBCHandling(csolver.rhok, csolver.rhoUk, csolver.rhoEk)
#    
#    csolver.solveContinuity(csolver.rhok, csolver.rhoUk, csolver.rhoEk)
#    csolver.solveXMomentum(csolver.rhok, csolver.rhoUk, csolver.rhoEk)
#    csolver.solveEnergy(csolver.rhok, csolver.rhoUk, csolver.rhoEk)
#    
#    csolver.postStepBCHandling(csolver.rhok2, csolver.rhoUk2, csolver.rhoEk2)
#
#    csolver.updateConservedData(rkStep)
#    csolver.updateNonConservedData(rkStep)   
#    
#    #RK Step 3
#    rkStep = 3
#    
#    csolver.preStepBCHandling(csolver.rhok, csolver.rhoUk, csolver.rhoEk)
#    
#    csolver.solveContinuity(csolver.rhok, csolver.rhoUk, csolver.rhoEk)
#    csolver.solveXMomentum(csolver.rhok, csolver.rhoUk, csolver.rhoEk)
#    csolver.solveEnergy(csolver.rhok, csolver.rhoUk, csolver.rhoEk)
#    
#    csolver.postStepBCHandling(csolver.rhok2, csolver.rhoUk2, csolver.rhoEk2)
#
#    csolver.updateConservedData(rkStep)
#    csolver.updateNonConservedData(rkStep)       
#
#    
#    #RK Step 4
#    rkStep = 4
#    
#    csolver.preStepBCHandling(csolver.rhok, csolver.rhoUk, csolver.rhoEk)
#    
#    csolver.solveContinuity(csolver.rhok, csolver.rhoUk, csolver.rhoEk)
#    csolver.solveXMomentum(csolver.rhok, csolver.rhoUk, csolver.rhoEk)
#    csolver.solveEnergy(csolver.rhok, csolver.rhoUk, csolver.rhoEk)
#    
#    csolver.postStepBCHandling(csolver.rhok2, csolver.rhoUk2, csolver.rhoEk2)
#
#    csolver.updateConservedData(rkStep)
#
#    #End of step stuff
#    
#    #Filter and/or move data to _1 containers
#    csolver.filterPrimativeValues()
#    
#    #Update the sponge if applicable
#    csolver.updateSponge()
#
#    #Update the non conserved data after we filter
#    csolver.updateNonConservedData(rkStep)       
#    
#    #plot/check solution
#    csolver.checkSolution()
#    



