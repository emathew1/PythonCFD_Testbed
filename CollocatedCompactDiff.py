#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 20:27:29 2017

@author: edwin
"""

#Load Numerical Libraries
import numpy as np
import time as tm

#Bring in functions we need
from CompactSchemes import CollocatedDeriv
from CompactSchemes import CompactFilter

#Plotting Libraries
import matplotlib.pyplot as plt
from drawnow import drawnow
plt.ion()
#plt.figure()    

#Size of the 1-D Function
N = 250

#Time stepping data
timeStep = 0
time = 0.0
maxTimeStep = 10000
maxTime = 30
plotStep = 25
CFL = 0.25

#filter info
filterStep = 1
alphaF = 0.49

#Solving Domain
L = 1
x = np.linspace(0,L,N)
dx = x[1]-x[0]

#Set the boundary conditions
bcType = "PERIODIC"
bcX1 = "ADIABATIC_WALL"
bcX0 = "ADIABATIC_WALL"

if bcX0 == "SPONGE" or bcX1 == "SPONGE":
    spongeFlag = 1
else:
    spongeFlag = 0

#Generate our Compact Schemes
deriv = CollocatedDeriv(N,dx,bcType)
filt  = CompactFilter(N,alphaF,bcType)

#Fluid properties
gamma = 1.4
Pr = 0.7
p_ref   = 1/gamma
rho_ref = 1.0
T_ref   = 1.0
mu_ref   = 0.00001

R_gas = p_ref/rho_ref/T_ref
cp = R_gas*gamma/(gamma-1.0)

#Sponge Properties
spongeAvgT     = 10.0
spongeEpsP     = 0.005
spongeP        = 1/gamma
spongeStrength = 2
spongeLength   = 0.125*L

spongeSigma   = np.zeros(N)
spongeRhoAvg  = np.zeros(N)
spongeRhoUAvg = np.zeros(N)
spongeRhoEAvg = np.zeros(N)


if bcX0 == "SPONGE":           
    for i in range(N):
        if x[i] < spongeLength:
            spongeX = (spongeLength-x[i])/spongeLength;
            spongeSigma[i] = spongeStrength * (0.068*(spongeX)**2 
                   + 0.845*(spongeX)**8)   

if bcX1 == "SPONGE":
    for i in range(N):
        if x[i] > L - spongeLength:
            spongeX = (x[i]-(L-spongeLength))/spongeLength
            spongeSigma[i] = spongeStrength*(0.068*(spongeX)**2 
                       + 0.845*(spongeX)**8) 
            


def calcSpongeSource(f,f_avg, spongeFlag):
    if spongeFlag == 1:
        source = spongeSigma*(f_avg-f)
    else:
        source = 0
    return source

#Allocate Variables we need
U0   = np.ones(N)
rho0 = np.ones(N)
p0   = np.ones(N)

#Initial conditions
for i in range(N):
    if x[i] > L/4 and x[i] < 3*L/4:
        U0[i]   = 0.0
        rho0[i] = 1 + 0.005*np.exp(-(x[i]-(L/2))**2/0.001)
        p0[i]   = (1 + 0.005*np.exp(-(x[i]-(L/2))**2/0.001))/gamma
    else:
        U0[i]   = 0.0
        rho0[i] = 1.0
        p0[i]   = 1.0/gamma       

        
#Start timestepping
done = False
while done == False:
    
    #increment timestep
    timeStep += 1
    
    #Initialize data at time step 1
    if timeStep == 1:
        U1    = U0
        rho1  = rho0
        p1    = p0
        rhoU1 = rho0*U0
        rhoE1 = p0/(gamma-1) + (1/2)*rho0*U0*U0
        T1    = p0/(rho0*R_gas)
        mu    = mu_ref*(T1/T_ref)**0.76
        k     = cp*mu/Pr
        sos   = np.sqrt(gamma*p0/rho0)
        
        if bcX0 == "ADIABATIC_WALL":
            T1[0]  = deriv.calcNeumann0(T1)
            U1[0]  = 0.0
            rhoU1[0]  = 0.0
            
        if bcX1 == "ADIABATIC_WALL":
            T1[-1] = deriv.calcNeumannEnd(T1)
            U1[-1] = 0.0
            rhoU1[-1] = 0.0

        if bcX0 == "ADIABATIC_WALL" or bcX1 == "ADIABATIC_WALL":
            p1 = T1*rho1*R_gas
            sos   = np.sqrt(gamma*p1/rho1)
            rhoE1 = p1/(gamma-1) + (1/2)*rho1*U1*U1             
            
        if bcX0 == "SPONGE" or bcX1 == "SPONGE": 
            spongeRhoAvg  = rho1
            spongeRhoUAvg = rhoU1
            spongeRhoEAvg = rhoE1
    
    #Calculate the time step from constant CFL
    UChar = np.fabs(U1) + sos
    dt   = np.min(CFL*dx/UChar)
    
    ###########
    #RK Step 1#
    ###########
    
    if bcX0 == "ADIABATIC_WALL":
        U1[0]  = 0.0
        rhoU1[0]  = 0.0
        T1[0]  = deriv.calcNeumann0(T1)
        p1[0]  = T1[0]*rho1[0]*R_gas
        
    if bcX1 == "ADIABATIC_WALL":
        U1[-1] = 0.0
        rhoU1[-1] = 0.0
        T1[-1] = deriv.calcNeumannEnd(T1)
        p1[-1] = T1[-1]*rho1[-1]*R_gas
      
        
    #Continuity
    rhok1  = (-dt*deriv.compact1stDeriv(rhoU1) +
                                     calcSpongeSource(rho1,spongeRhoAvg,spongeFlag))
    
    #Momentum
    rhoUk1 = (-dt*(deriv.compact1stDeriv(rhoU1*U1 + p1) -
                                      (4/3)*mu*deriv.compact2ndDeriv(U1)) +
                                      calcSpongeSource(rhoU1,spongeRhoUAvg,spongeFlag))
    #Energy
    rhoEk1 = (-dt*(deriv.compact1stDeriv(rhoE1*U1 + U1*p1) -
                                      (mu/Pr/(gamma-1))*deriv.compact2ndDeriv(T1) +
                                      (4/3)*mu*U1*deriv.compact2ndDeriv(U1)) +
                                      calcSpongeSource(rhoE1,spongeRhoEAvg,spongeFlag))
    
    if bcType == "DIRICHLET":
        if bcX0 == "ADIABATIC_WALL":
            rhok1[0]   = rhok1[0]
            rhoUk1[0]  = 0
            rhoEk1[0]  = rhoEk1[0]
        else:
            rhok1[0]   = 0
            rhoUk1[0]  = 0
            rhoEk1[0]  = 0
            
        if bcX1 == "ADIABATIC_WALL":
            rhok1[-1]  = rhok1[-1]
            rhoUk1[-1] = 0
            rhoEk1[-1] = rhoEk1[-1]
        else:
            rhok1[-1]  = 0
            rhoUk1[-1] = 0
            rhoEk1[-1] = 0           
          
    
    #Update conserved data for next substep
    rhok  = rho1  + rhok1/2
    rhoUk = rhoU1 + rhoUk1/2
    rhoEk = rhoE1 + rhoEk1/2
    
    #Update primative data for next substep
    U1 = rhoUk/rhok
    p1 = (gamma-1)*(rhoEk - 0.5*rhoUk*rhoUk/rhok)
    T1 = (p1/(rhok*R_gas))
    
    ###########
    #RK Step 2#
    ###########
    
    if bcX0 == "ADIABATIC_WALL":
        U1[0]  = 0.0
        rhoUk[0]  = 0.0
        T1[0]  = deriv.calcNeumann0(T1)
        p1[0]  = T1[0]*rhok[0]*R_gas
        
    if bcX1 == "ADIABATIC_WALL":
        U1[-1] = 0.0
        rhoUk[-1] = 0.0
        T1[-1] = deriv.calcNeumannEnd(T1)
        p1[-1] = T1[-1]*rhok[-1]*R_gas
       
    
    #Continuity
    rhok2  = (-dt*deriv.compact1stDeriv(rhoUk) +
                                         calcSpongeSource(rhok,spongeRhoAvg,spongeFlag))

    
    #Momentum
    rhoUk2 = (-dt*(deriv.compact1stDeriv(rhoUk*U1 + p1) -
                                      (4/3)*mu*deriv.compact2ndDeriv(U1)) +
                                         calcSpongeSource(rhoUk,spongeRhoUAvg,spongeFlag))
    
    #Energy
    rhoEk2 = (-dt*(deriv.compact1stDeriv(rhoEk*U1 + U1*p1) -
                                      (mu/Pr/(gamma-1))*deriv.compact2ndDeriv(T1) +
                                      (4/3)*mu*U1*deriv.compact2ndDeriv(U1)) +
                                      calcSpongeSource(rhoEk,spongeRhoEAvg,spongeFlag))
    
    #Update conserved data for next substep
    rhok  = rho1  + rhok2/2
    rhoUk = rhoU1 + rhoUk2/2
    rhoEk = rhoE1 + rhoEk2/2
    
    #Update primative data for next substep
    U1 = rhoUk/rhok
    p1 = (gamma-1)*(rhoEk - 0.5*rhoUk*rhoUk/rhok)
    T1 = (p1/(rhok*R_gas)) 
    mu = mu_ref*(T1/T_ref)**0.76
    k  = cp*mu/Pr
    
    ###########
    #RK Step 3#
    ###########
    
    if bcX0 == "ADIABATIC_WALL":
        U1[0]  = 0.0
        rhoUk[0]  = 0.0
        T1[0]  = deriv.calcNeumann0(T1)
        p1[0]  = T1[0]*rhok[0]*R_gas
        
    if bcX1 == "ADIABATIC_WALL":
        U1[-1] = 0.0
        rhoUk[-1] = 0.0
        T1[-1] = deriv.calcNeumannEnd(T1)
        p1[-1] = T1[-1]*rhok[-1]*R_gas
       
    
    #Continuity
    rhok3  = (-dt*deriv.compact1stDeriv(rhoUk) +
                                         calcSpongeSource(rhok,spongeRhoAvg,spongeFlag))
    
    #Momentum
    rhoUk3 = (-dt*(deriv.compact1stDeriv(rhoUk*U1 + p1) -
                                      (4/3)*mu*deriv.compact2ndDeriv(U1)) +
                                      calcSpongeSource(rhoUk,spongeRhoUAvg,spongeFlag))

    
    #Energy
    rhoEk3 = (-dt*(deriv.compact1stDeriv(rhoEk*U1 + U1*p1) -
                                      (mu/Pr/(gamma-1))*deriv.compact2ndDeriv(T1) +
                                      (4/3)*mu*U1*deriv.compact2ndDeriv(U1)) +
                                      calcSpongeSource(rhoEk,spongeRhoEAvg,spongeFlag))


    if bcType == "DIRICHLET":
        if bcX0 == "ADIABATIC_WALL":
            rhok3[0]   = rhok3[0]
            rhoUk3[0]  = 0
            rhoEk3[0]  = rhoEk3[0]
        else:
            rhok3[0]   = 0
            rhoUk3[0]  = 0
            rhoEk3[0]  = 0
            
        if bcX1 == "ADIABATIC_WALL":
            rhok3[-1]  = rhok3[-1]
            rhoUk3[-1] = 0
            rhoEk3[-1] = rhoEk3[-1]
        else:
            rhok3[-1]  = 0
            rhoUk3[-1] = 0
            rhoEk3[-1] = 0    
    
    #Update conserved data for next substep
    rhok  = rho1  + rhok3
    rhoUk = rhoU1 + rhoUk3
    rhoEk = rhoE1 + rhoEk3
    
    #Update primative data for next substep
    U1 = rhoUk/rhok
    p1 = (gamma-1)*(rhoEk - 0.5*rhoUk*rhoUk/rhok)
    T1 = (p1/(rhok*R_gas))     
    mu = mu_ref*(T1/T_ref)**0.76
    k  = cp*mu/Pr
    
    ###########
    #RK Step 4#
    ###########
    
    if bcX0 == "ADIABATIC_WALL":
        U1[0]  = 0.0
        rhoUk[0]  = 0.0
        T1[0]  = deriv.calcNeumann0(T1)
        p1[0]  = T1[0]*rhok[0]*R_gas
        
    if bcX1 == "ADIABATIC_WALL":
        U1[-1] = 0.0
        rhoUk[-1] = 0.0
        T1[-1] = deriv.calcNeumannEnd(T1)
        p1[-1] = T1[-1]*rhok[-1]*R_gas
       

    #Continuity
    rhok4  = (-dt*deriv.compact1stDeriv(rhoUk) +
                                        calcSpongeSource(rhok,spongeRhoAvg,spongeFlag))
    
    #Momentum
    rhoUk4 = (-dt*(deriv.compact1stDeriv(rhoUk*U1 + p1) -
                                      (4/3)*mu*deriv.compact2ndDeriv(U1)) +
                                      calcSpongeSource(rhoUk,spongeRhoUAvg,spongeFlag))

    
    #Energy
    rhoEk4 = (-dt*(deriv.compact1stDeriv(rhoEk*U1 + U1*p1) -
                                      (mu/Pr/(gamma-1))*deriv.compact2ndDeriv(T1) +
                                      (4/3)*mu*U1*deriv.compact2ndDeriv(U1)) +
                                      calcSpongeSource(rhoEk,spongeRhoEAvg,spongeFlag))


    if bcType == "DIRICHLET":
        if bcX0 == "ADIABATIC_WALL":
            rhok4[0]   = rhok4[0]
            rhoUk4[0]  = 0
            rhoEk4[0]  = rhoEk4[0]
        else:
            rhok4[0]   = 0
            rhoUk4[0]  = 0
            rhoEk4[0]  = 0
            
        if bcX1 == "ADIABATIC_WALL":
            rhok4[-1]  = rhok4[-1]
            rhoUk4[-1] = 0
            rhoEk4[-1] = rhoEk4[-1]
        else:
            rhok4[-1]  = 0
            rhoUk4[-1] = 0
            rhoEk4[-1] = 0    
    
    #Update Final Solution
    rho2  = rho1  + rhok1/6  + rhok2/3  + rhok3/3  + rhok4/6
    rhoU2 = rhoU1 + rhoUk1/6 + rhoUk2/3 + rhoUk3/3 + rhoUk4/6
    rhoE2 = rhoE1 + rhoEk1/6 + rhoEk2/3 + rhoEk3/3 + rhoEk4/6   

    if(timeStep%filterStep == 0):
        rho1  = filt.compactFilter(rho2)
        rhoU1 = filt.compactFilter(rhoU2)
        rhoE1 = filt.compactFilter(rhoE2)
        print("Filtering...")
        
        if bcType == "DIRICHLET":
            rho1[0]   = rho2[0]
            rho1[-1]  = rho2[-1]
            rhoU1[0]  = rhoU2[0]
            rhoU1[-1] = rhoU2[-1]
            rhoE1[0]  = rhoE1[0]
            rhoE1[-1] = rhoE1[-1]
        
    else:
        rho1  = rho2
        rhoU1 = rhoU2
        rhoE1 = rhoE2

    if spongeFlag == 1:
        #Update the Sponge average
        eps = 1.0/(spongeAvgT/dt+1.0)
        spongeRhoAvg  += eps*(rho1  - spongeRhoAvg)
        spongeRhoUAvg += eps*(rhoU1 - spongeRhoUAvg)
        spongeRhoEAvg += eps*(rhoE1 - spongeRhoEAvg)
        spongeRhoEAvg = (spongeEpsP*spongeRhoEAvg + 
            (1.0 - spongeEpsP)*(spongeP/(gamma-1) + 
             0.5*(spongeRhoUAvg**2)/spongeRhoAvg))
    
        #Use the running average value as the dirichlet condition
        if bcX0 == "SPONGE":
            rho1[0]   = spongeRhoAvg[0]
            rhoU1[0]  = spongeRhoUAvg[0]
            rhoE1[0]  = spongeRhoEAvg[0]

        if bcX1 == "SPONGE": 
            rho1[-1]  = spongeRhoAvg[-1]
            rhoU1[-1] = spongeRhoUAvg[-1]
            rhoE1[-1] = spongeRhoEAvg[-1]

    #Update primative data for next substep
    U1 = rhoU1/rho1
    p1 = (gamma-1)*(rhoE1 - 0.5*rhoU1*rhoU1/rho1)
    T1 = (p1/(rho1*R_gas))   
    mu = mu_ref*(T1/T_ref)**0.76
    k  = cp*mu/Pr
    sos   = np.sqrt(gamma*p1/rho1)
    
    print(rho1[-2])
    
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