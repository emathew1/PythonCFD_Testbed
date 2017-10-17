#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 14:18:56 2017

@author: edwin
"""

#Load Numerical Libraries
import numpy as np
import time as tm

#Bring in functions we need
from CompactSchemes import StaggeredInterp
from CompactSchemes import StaggeredDeriv
from CompactSchemes import CompactFilter

#Plotting Libraries
import matplotlib.pyplot as plt
from drawnow import drawnow
plt.ion()
plt.figure()    

#Size of the 1-D Function
N = 500

#Time stepping data
timeStep = 0
time = 0.0
maxTimeStep = 500
maxTime = 3
plotStep = 10
CFL = 0.25

#Filter info
filterStep = 1
alphaF = 0.45

#Solving Domain
x = np.linspace(0,1.0,N)
dx = x[1]-x[0]

#Generate our Compact Schemes
interp = StaggeredInterp(N)
deriv  = StaggeredDeriv(N,dx)
filt   = CompactFilter(N,alphaF)

#Fluid properties
gamma = 1.4;
Pr = 0.7;
p_ref   = 1/gamma;
rho_ref = 1.0;
T_ref   = 1.0;
mu      = 0.00005;

R_gas = p_ref/rho_ref/T_ref;
cp = R_gas*gamma/(gamma-1.0);
k  = cp*mu/Pr;

U0   = np.ones(N)
rho0 = np.ones(N)
p0   = np.ones(N)

#Initial conditions
#for i in range(N):
#        U0[i] = 0.75
#        rho0[i] = 0.6*np.exp(-(x[i]-0.125)**2/0.0005)+1.0
#        p0[i] = 1/gamma

for i in range(N):
    if x[i] < 0.5:
        U0[i]   = 0.0
        rho0[i] = 1.0
        p0[i]   = 1.0/gamma
    else:
        U0[i]   = 0.0
        rho0[i] = 0.125
        p0[i]   = 0.1/gamma


done = False
while done == False:
    
    #increment timestep
    timeStep += 1
    
    #Initialize data at time step 1
    if timeStep == 1:
        U1    = interp.compactInterpHalfRight(U0)
        rho1  = rho0
        p1    = p0
        rhoU1 = interp.compactInterpHalfRight(rho0*U0)
        rhoE1 = p0/(gamma-1) + (1/2)*rho0*U0*U0
        T1    = p0/(rho0*R_gas)
        sos   = np.sqrt(gamma*p0/rho0)
    
    #Calculate the time step from constant CFL
    UChar = np.fabs(U1) + sos
    dt   = np.min(CFL*dx/UChar)
    
    
    ###########
    #RK Step 1#
    ###########
    
    #Pre-calculations
    rhoUHalfLeft = interp.compactInterpHalfLeft(rhoU1)
    UHalfLeft    = rhoUHalfLeft/rho1
    dUdxHalfLeft = deriv.compactDerivHalfLeft(U1)
    
    rhoEHalfRight = interp.compactInterpHalfRight(rhoE1)
    pHalfRight    = interp.compactInterpHalfRight(p1)
    dUdxHalfRight = deriv.compactDerivHalfRight(UHalfLeft)
    dTdxHalfRight = deriv.compactDerivHalfRight(T1)
    
    #Continuity
    rhok1  = -dt*deriv.compactDerivHalfLeft(rhoU1)
    
    #Momentum
    rhoUk1 = -dt*deriv.compactDerivHalfRight(rhoUHalfLeft*UHalfLeft + p1 -
                                      (4/3)*mu*dUdxHalfLeft) 
    
    #Energy
    rhoEk1 = -dt*deriv.compactDerivHalfLeft(rhoEHalfRight*U1 + U1*pHalfRight -
                                      (mu/Pr/(gamma-1))*dTdxHalfRight +
                                      (4/3)*mu*U1*dUdxHalfRight)
    
    #Update conserved data for next substep
    rhok  = rho1  + rhok1/2
    rhoUk = rhoU1 + rhoUk1/2
    rhoEk = rhoE1 + rhoEk1/2
    
    #Update primative data for next substep
    rhoHalfRight = interp.compactInterpHalfRight(rhok)
    rhoUHalfLeft = interp.compactInterpHalfLeft(rhoUk)
    U1 = rhoUk/rhoHalfRight
    p1 = (gamma-1)*(rhoEk - 0.5*rhoUHalfLeft*rhoUHalfLeft/rhoHalfRight)
    T1 = (p1/(rhok*R_gas))
    
    ###########
    #RK Step 2#
    ###########
    
    #Pre-calculations  
    UHalfLeft = rhoUHalfLeft/rhok
    dUdxHalfLeft = deriv.compactDerivHalfLeft(U1)
    
    rhoEHalfRight = interp.compactInterpHalfRight(rhoEk)
    pHalfRight    = interp.compactInterpHalfRight(p1)
    dUdxHalfRight = deriv.compactDerivHalfRight(UHalfLeft)
    dTdxHalfRight = deriv.compactDerivHalfRight(T1)
    
    #Continuity
    rhok2  = -dt*deriv.compactDerivHalfLeft(rhoUk)
    
    #Momentum
    rhoUk2 = -dt*deriv.compactDerivHalfRight(rhoUHalfLeft*UHalfLeft + p1 -
                                      (4/3)*mu*dUdxHalfLeft) 
    
    #Energy
    rhoEk2 =  -dt*deriv.compactDerivHalfLeft(rhoEHalfRight*U1 + U1*pHalfRight -
                                      (mu/Pr/(gamma-1))*dTdxHalfRight +
                                      (4/3)*mu*U1*dUdxHalfRight)
    
    #Update conserved data for next substep
    rhok  = rho1  + rhok2/2
    rhoUk = rhoU1 + rhoUk2/2
    rhoEk = rhoE1 + rhoEk2/2
    
    #Update primative data for next substep
    rhoHalfRight = interp.compactInterpHalfRight(rhok)
    rhoUHalfLeft = interp.compactInterpHalfLeft(rhoUk)
    U1 = rhoUk/rhoHalfRight
    p1 = (gamma-1)*(rhoEk - 0.5*rhoUHalfLeft*rhoUHalfLeft/rhoHalfRight)
    T1 = (p1/(rhok*R_gas))    
    
    ###########
    #RK Step 3#
    ###########
    
    #Pre-calculations  
    UHalfLeft = rhoUHalfLeft/rhok
    dUdxHalfLeft = deriv.compactDerivHalfLeft(U1)
    
    rhoEHalfRight = interp.compactInterpHalfRight(rhoEk)
    pHalfRight    = interp.compactInterpHalfRight(p1)
    dUdxHalfRight = deriv.compactDerivHalfRight(UHalfLeft)
    dTdxHalfRight = deriv.compactDerivHalfRight(T1)   
    
    #Continuity
    rhok3  = -dt*deriv.compactDerivHalfLeft(rhoUk)
    
    #Momentum
    rhoUk3 = -dt*deriv.compactDerivHalfRight(rhoUHalfLeft*UHalfLeft + p1 -
                                      (4/3)*mu*dUdxHalfLeft) 
    
    #Energy
    rhoEk3 =  -dt*deriv.compactDerivHalfLeft(rhoEHalfRight*U1 + U1*pHalfRight -
                                      (mu/Pr/(gamma-1))*dTdxHalfRight +
                                      (4/3)*mu*U1*dUdxHalfRight)
    
    #Update conserved data for next substep
    rhok  = rho1  + rhok3
    rhoUk = rhoU1 + rhoUk3
    rhoEk = rhoE1 + rhoEk3
    
    #Update primative data for next substep
    rhoHalfRight = interp.compactInterpHalfRight(rhok)
    rhoUHalfLeft = interp.compactInterpHalfLeft(rhoUk)
    U1 = rhoUk/rhoHalfRight
    p1 = (gamma-1)*(rhoEk - 0.5*rhoUHalfLeft*rhoUHalfLeft/rhoHalfRight)
    T1 = (p1/(rhok*R_gas)) 
    
    ###########
    #RK Step 4#
    ###########
    
    #Pre-calculations  
    UHalfLeft = rhoUHalfLeft/rhok
    dUdxHalfLeft = deriv.compactDerivHalfLeft(U1)
    
    rhoEHalfRight = interp.compactInterpHalfRight(rhoEk)
    pHalfRight    = interp.compactInterpHalfRight(p1)
    dUdxHalfRight = deriv.compactDerivHalfRight(UHalfLeft)
    dTdxHalfRight = deriv.compactDerivHalfRight(T1)   
    
    #Continuity
    rhok4  = -dt*deriv.compactDerivHalfLeft(rhoUk)
    
    #Momentum
    rhoUk4 = -dt*deriv.compactDerivHalfRight(rhoUHalfLeft*UHalfLeft + p1 -
                                      (4/3)*mu*dUdxHalfLeft) 
    
    #Energy
    rhoEk4 =  -dt*deriv.compactDerivHalfLeft(rhoEHalfRight*U1 + U1*pHalfRight -
                                      (mu/Pr/(gamma-1))*dTdxHalfRight +
                                      (4/3)*mu*U1*dUdxHalfRight)    
    
    #Update Final Solution
    rho2  = rho1  + rhok1/6  + rhok2/3  + rhok3/3  + rhok4/6
    rhoU2 = rhoU1 + rhoUk1/6 + rhoUk2/3 + rhoUk3/3 + rhoUk4/6
    rhoE2 = rhoE1 + rhoEk1/6 + rhoEk2/3 + rhoEk3/3 + rhoEk4/6   
   
    if(timeStep%filterStep == 0):
        rho1  = filt.compactFilter(rho2)
        rhoU1 = filt.compactFilter(rhoU2)
        rhoE1 = filt.compactFilter(rhoE2)
        print("Filtering...")
    else:
        rho1  = rho2
        rhoU1 = rhoU2
        rhoE1 = rhoE2
    
    
    #Update primative data and SOS for dt calc
    rhoHalfRight = interp.compactInterpHalfRight(rho1)
    rhoUHalfLeft = interp.compactInterpHalfLeft(rhoU1)
    U1 = rhoU1/rhoHalfRight
    p1 = (gamma-1)*(rhoEk - 0.5*rhoUHalfLeft*rhoUHalfLeft/rhoHalfRight)
    T1 = (p1/(rho1*R_gas))
    sos   = np.sqrt(gamma*p1/rho1)        
    
    #Check if we've hit the end of the timestep condition
    if timeStep > maxTimeStep:
        done = True
    
    #Check if we've hit the end of the max time condition
    if time > maxTime:
        done = True
        
    if(timeStep%plotStep == 0):
        drawnow(plotFigure)
        print(timeStep)
        
    if((np.isnan(rhoE1)).any() == True or (np.isnan(rho1)).any() == True or (np.isnan(rhoU1)).any() == True):
        done = True
        print(-1)
    
plt.plot(x,rho1)
plt.show()
        