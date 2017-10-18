

import numpy as np
from drawnow import drawnow
import matplotlib.pyplot as plt

#Bring in functions we need
from CompactSchemes import CollocatedDeriv
from CompactSchemes import CompactFilter
from IdealGas import IdealGas
from SpongeBC import SpongeBC

#############################
#############################
# Collocated Compact Solver #
#############################
#############################

class Domain:
    
    def __init__(self, N, x, L):
        self.N = N
        self.x = x
        self.L = L
        self.dx = x[1]-x[0]
        
class BC:
    
    def __init__(self, bcType, bcX0, bcX1):
        self.bcType = bcType
        self.bcX0 = bcX0
        self.bcX1 = bcX1
        
class TimeStepping:
    
    def __init__(self, CFL, maxTimeStep, maxTime, plotStep, filterStep):
        self.CFL = CFL
        self.maxTimeStep = maxTimeStep
        self.maxTime = maxTime
        self.plotStep = plotStep
        self.filterStep = filterStep

class CSolver:
    
    def __init__(self, domain, bc, timeStepping, alphaF, mu_ref):
        
        self.done = False
        
        #grid info
        self.N = domain.N
        self.x = domain.x
        self.L = domain.L
        self.dx = domain.dx
        
        #gas properties
        self.idealGas = IdealGas(mu_ref)
        
        #BC info
        self.bcType = bc.bcType
        self.bcX0 = bc.bcX0
        self.bcX1 = bc.bcX1
        
        #initial conditions
        self.U0   = np.zeros(self.N)
        self.rho0 = np.zeros(self.N)
        self.p0   = np.zeros(self.N)
        
        #non-primative data
        self.U   = np.zeros(self.N)
        self.T   = np.zeros(self.N)
        self.p   = np.zeros(self.N)
        self.mu  = np.zeros(self.N)
        self.k   = np.zeros(self.N)
        self.sos = np.zeros(self.N)
        
        #primative data
        self.rho1  = np.zeros(self.N)
        self.rhoU1 = np.zeros(self.N)
        self.rhoE1 = np.zeros(self.N)
        
        self.rhok  = np.zeros(self.N)
        self.rhoUk = np.zeros(self.N)
        self.rhoEk = np.zeros(self.N)

        self.rhok2  = np.zeros(self.N)
        self.rhoUk2 = np.zeros(self.N)
        self.rhoEk2 = np.zeros(self.N)        
        
        self.rho2  = np.zeros(self.N)
        self.rhoU2 = np.zeros(self.N)
        self.rhoE2 = np.zeros(self.N)
        
        #time data
        self.timeStep = 0
        self.time     = 0.0
        self.maxTimeStep = timeStepping.maxTimeStep
        self.maxTime     = timeStepping.maxTime
        self.plotStep    = timeStepping.plotStep
        self.CFL = timeStepping.CFL
        
        #filter data
        self.filterStep = timeStepping.filterStep
        self.alphaF = alphaF
        
        if self.bcX0 == "SPONGE" or self.bcX1 == "SPONGE":
            self.spongeFlag = 1
            self.spongeBC = SpongeBC(self.N, self.x, self.L, self.idealGas, self.bcX0, self.bcX1)
        else:
            self.spongeFlag = 0
        
        #Generate our derivatives and our filter    
        self.deriv = CollocatedDeriv(self.N,self.dx,self.bcType)
        self.filt  = CompactFilter(self.N,self.alphaF,self.bcType)
                      
    def setInitialConditions(self, rho0, U0, p0):
        self.rho0 = rho0
        self.U0 = U0
        self.p0 = p0
        
        self.U     = U0
        self.rho1  = rho0
        self.p     = p0
        self.rhoU1 = rho0*U0
        self.rhoE1 = self.idealGas.solveRhoE(rho0,U0,p0)
        self.T     = self.idealGas.solveT(rho0, p0)
        self.mu    = self.idealGas.solveMu(self.T)
        self.k     = self.idealGas.solveK(self.mu)
        self.sos   = self.idealGas.solveSOS(self.rho1, self.p)
        
        if self.bcX0 == "ADIABATIC_WALL":
            self.T[0]  = self.deriv.calcNeumann0(self.T)
            self.U[0]  = 0.0
            self.rhoU1[0]  = 0.0
            
        if self.bcX1 == "ADIABATIC_WALL":
            self.T[-1] = self.deriv.calcNeumannEnd(self.T)
            self.U[-1] = 0.0
            self.rhoU1[-1] = 0.0

        if self.bcX0 == "ADIABATIC_WALL" or self.bcX1 == "ADIABATIC_WALL":
            self.p     = self.idealGas.solvePIdealGas(self.rho1, self.T)
            self.sos   = self.idealGas.solveSOS(self.rho1, self.p)
            self.rhoE1 = self.idealGas.solveRhoE(self.rho1, self.U, self.p)            
            
        if self.bcX0 == "SPONGE" or self.bcX1 == "SPONGE": 
            self.spongeBC.spongeRhoAvg  = self.rho1
            self.spongeBC.spongeRhoUAvg = self.rhoU1
            self.spongeBC.spongeRhoEAvg = self.rhoE1
        
    def calcDtFromCFL(self):
        UChar = np.fabs(self.U) + self.sos
        self.dt   = np.min(self.CFL*self.dx/UChar)
        
        #Increment timestep
        self.timeStep += 1
        self.time += self.dt


        
    def calcSpongeSource(self,f,eqn):
        if self.spongeFlag == 1:
            if eqn == "CONT":
                source = self.spongeBC.spongeSigma*(self.spongeBC.spongeRhoAvg-f)
            elif eqn == "XMOM":
                source = self.spongeBC.spongeSigma*(self.spongeBC.spongeRhoUAvg-f)                
            elif eqn == "ENGY":
                source = self.spongeBC.spongeSigma*(self.spongeBC.spongeRhoEAvg-f)
            else:
                source = 0
        else:
            source = 0
        return source
    
    def preStepBCHandling(self, rho, rhoU, rhoE):
        if self.bcX0 == "ADIABATIC_WALL":
            self.U[0]  = 0.0
            rhoU[0]    = 0.0
            self.T[0]  = self.deriv.calcNeumann0(self.T)
            self.p[0]  = self.idealGas.solvePIdealGas(rho[0],self.T[0])
        
        if self.bcX1 == "ADIABATIC_WALL":
            self.U[-1] = 0.0
            rhoU[-1] = 0.0
            self.T[-1] = self.deriv.calcNeumannEnd(self.T)
            self.p[-1] = self.idealGas.solvePIdealGas(rho[-1],self.T[-1])
    
    def solveContinuity(self, rho, rhoU, rhoE):
        self.rhok2  = (-self.dt*self.deriv.compact1stDeriv(rhoU) +
                                     self.calcSpongeSource(rho,"CONT"))
        
    def solveXMomentum(self, rho, rhoU, rhoE):
        self.rhoUk2 = (-self.dt*(self.deriv.compact1stDeriv(rhoU*self.U + self.p) -
                                      (4/3)*self.mu*self.deriv.compact2ndDeriv(self.U)) +
                                      self.calcSpongeSource(rhoU,"XMOM"))
    
    def solveEnergy(self, rho, rhoU, rhoE):
        self.rhoEk2 = (-self.dt*(self.deriv.compact1stDeriv(rhoE*self.U + self.U*self.p) -
                                      (self.mu/self.idealGas.Pr/(self.idealGas.gamma-1))*self.deriv.compact2ndDeriv(self.T) +
                                      (4/3)*self.mu*self.U*self.deriv.compact2ndDeriv(self.U)) +
                                      self.calcSpongeSource(rhoE,"ENGY"))
    
    
    def postStepBCHandling(self, rho, rhoU, rhoE):
        if self.bcType == "DIRICHLET":
            if self.bcX0 == "ADIABATIC_WALL":
                rho[0]   = rho[0]
                rhoU[0]  = 0
                rhoE[0]  = rhoE[0]
            else:
                rho[0]   = 0
                rhoU[0]  = 0
                rhoE[0]  = 0
                
            if self.bcX1 == "ADIABATIC_WALL":
                rho[-1]  = rho[-1]
                rhoU[-1] = 0
                rhoE[-1] = rhoE[-1]
            else:
                rho[-1]  = 0
                rhoU[-1] = 0
                rhoE[-1] = 0  
    
    def updateConservedData(self,rkStep):
        if rkStep == 1:
            self.rho2  = self.rho1 + self.rhok2/6
            self.rhoU2 = self.rhoU1 + self.rhoUk2/6
            self.rhoE2 = self.rhoE1 + self.rhoEk2/6
        elif rkStep == 2:
            self.rho2  += self.rhok2/3
            self.rhoU2 += self.rhoUk2/3
            self.rhoE2 += self.rhoEk2/3          
        elif rkStep == 3:
            self.rho2  += self.rhok2/3
            self.rhoU2 += self.rhoUk2/3
            self.rhoE2 += self.rhoEk2/3               
        elif rkStep == 4:
            self.rho2  += self.rhok2/6
            self.rhoU2 += self.rhoUk2/6
            self.rhoE2 += self.rhoEk2/6

        if rkStep == 1:
            self.rhok  = self.rho1  + self.rhok2/2
            self.rhoUk = self.rhoU1 + self.rhoUk2/2
            self.rhoEk = self.rhoE1 + self.rhoEk2/2
        elif rkStep == 2:
            self.rhok  = self.rho1  + self.rhok2/2
            self.rhoUk = self.rhoU1 + self.rhoUk2/2
            self.rhoEk = self.rhoE1 + self.rhoEk2/2
        elif rkStep == 3:
            self.rhok  = self.rho1  + self.rhok2
            self.rhoUk = self.rhoU1 + self.rhoUk2
            self.rhoEk = self.rhoE1 + self.rhoEk2
            
    def updateNonConservedData(self,rkStep):
        if rkStep == 1 or rkStep == 2 or rkStep == 3:
            self.U = self.rhoUk/self.rhok
            self.p = (self.idealGas.gamma-1)*(self.rhoEk - 
                         0.5*self.rhoUk*self.rhoUk/self.rhok)
            self.T = (self.p/(self.rhok*self.idealGas.R_gas))   
            self.mu = self.idealGas.mu_ref*(self.T/self.idealGas.T_ref)**0.76
            self.k  = self.idealGas.cp*self.mu/self.idealGas.Pr
            self.sos   = np.sqrt(self.idealGas.gamma*self.p/self.rhok)
        elif rkStep == 4:
            self.U = self.rhoU1/self.rho1
            self.p = (self.idealGas.gamma-1)*(self.rhoE1 - 
                         0.5*self.rhoU1*self.rhoU1/self.rho1)
            self.T = (self.p/(self.rho1*self.idealGas.R_gas))   
            self.mu = self.idealGas.mu_ref*(self.T/self.idealGas.T_ref)**0.76
            self.k  = self.idealGas.cp*self.mu/self.idealGas.Pr
            self.sos   = np.sqrt(self.idealGas.gamma*self.p/self.rho1)
            
    def filterPrimativeValues(self):
        if(self.timeStep%self.filterStep == 0):
            self.rho1  = self.filt.compactFilter(self.rho2)
            self.rhoU1 = self.filt.compactFilter(self.rhoU2)
            self.rhoE1 = self.filt.compactFilter(self.rhoE2)
            print("Filtering...")
        
        if self.bcType == "DIRICHLET":
            self.rho1[0]   = self.rho2[0]
            self.rho1[-1]  = self.rho2[-1]
            self.rhoU1[0]  = self.rhoU2[0]
            self.rhoU1[-1] = self.rhoU2[-1]
            self.rhoE1[0]  = self.rhoE2[0]
            self.rhoE1[-1] = self.rhoE2[-1]
        else:
            self.rho1  = self.rho2
            self.rhoU1 = self.rhoU2
            self.rhoE1 = self.rhoE2
        
            
    def updateSponge(self):
        if self.spongeFlag == 1:
            eps = 1.0/(self.spongeBC.spongeAvgT/self.dt+1.0)
            self.spongeBC.spongeRhoAvg  += eps*(self.rho1  - self.spongeBC.spongeRhoAvg)
            self.spongeBC.spongeRhoUAvg += eps*(self.rhoU1 - self.spongeBC.spongeRhoUAvg)
            self.spongeBC.spongeRhoEAvg += eps*(self.rhoE1 - self.spongeBC.spongeRhoEAvg)
            self.spongeBC.spongeRhoEAvg = (self.spongeBC.spongeEpsP*self.spongeBC.spongeRhoEAvg + 
                (1.0 - self.spongeBC.spongeEpsP)*(self.spongeBC.spongeP/(self.idealGas.gamma-1) + 
                 0.5*(self.spongeBC.spongeRhoUAvg**2)/self.spongeBC.spongeRhoAvg))
            
            if self.bcX0 == "SPONGE":
                self.rho1[0]   = self.spongeBC.spongeRhoAvg[0]
                self.rhoU1[0]  = self.spongeBC.spongeRhoUAvg[0]
                self.rhoE1[0]  = self.spongeBC.spongeRhoEAvg[0]
                
            if self.bcX1 == "SPONGE": 
                self.rho1[-1]  = self.spongeBC.spongeRhoAvg[-1]
                self.rhoU1[-1] = self.spongeBC.spongeRhoUAvg[-1]
                self.rhoE1[-1] = self.spongeBC.spongeRhoEAvg[-1]
    
    def plotFigure(self):
        plt.plot(self.x,self.rho1)
        plt.axis([0, self.L, 0.95, 1.05])

    def checkSolution(self):
        
        print(self.timeStep)
        
        #Check if we've hit the end of the timestep condition
        if self.timeStep >= self.maxTimeStep:
            self.done = True
    
        #Check if we've hit the end of the max time condition
        if self.time >= self.maxTime:
            self.done = True
            
        if(self.timeStep%self.plotStep == 0):
            drawnow(self.plotFigure)
            print(self.timeStep)
            
        if((np.isnan(self.rhoE1)).any() == True or (np.isnan(self.rho1)).any() == True or (np.isnan(self.rhoU1)).any() == True):
            self.done = True
            print(-1)

    
        