

import numpy as np

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
        
        #grid info
        self.N = domain.N
        self.x = domain.x
        self.L = domain.L
        
        #gas properties
        self.idealGas = IdealGas(mu_ref)
        
        #BC info
        self.bcType = bc.bcType
        self.bcX0 = bc.bcX0
        self.bcX1 = bc.bcX1
        
        #initial conditions
        self.U0   = np.zeros(N)
        self.rho0 = np.zeros(N)
        self.p0   = np.zeros(N)
        
        #non-primative data
        self.U   = np.zeros(N)
        self.T   = np.zeros(N)
        self.p   = np.zeros(N)
        self.mu  = np.zeros(N)
        self.k   = np.zeros(N)
        self.sos = np.zeros(N)
        
        #primative data
        self.rho1  = np.zeros(N)
        self.rhoU1 = np.zeros(N)
        self.rhoE1 = np.zeros(N)
        
        self.rhok  = np.zeros(N)
        self.rhoUk = np.zeros(N)
        self.rhoEk = np.zeros(N)
        
        self.rho2  = np.zeros(N)
        self.rhoU2 = np.zeros(N)
        self.rhoE2 = np.zeros(N)
        
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
            self.spongeBC = SpongeBC(N, x, L, self.idealGas, bcX0, bcX1)
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
            self.T[0]  = self.deriv.calcNeumann0(T1)
            self.U[0]  = 0.0
            self.rhoU1[0]  = 0.0
            
        if self.bcX1 == "ADIABATIC_WALL":
            self.T[-1] = self.deriv.calcNeumannEnd(T1)
            self.U[-1] = 0.0
            self.rhoU1[-1] = 0.0

        if self.bcX0 == "ADIABATIC_WALL" or self.bcX1 == "ADIABATIC_WALL":
            self.p     = self.idealGas.solvePIdealGas(self.rho1, self.T)
            self.sos   = self.idealGas.solveSOS(self.rho1, self.p)
            self.rhoE1 = solveRhoE(self.rho1, self.U, self.p)            
            
        if self.bcX0 == "SPONGE" or self.bcX1 == "SPONGE": 
            self.spongeBC.spongeRhoAvg  = self.rho1
            self.spongeBC.spongeRhoUAvg = self.rhoU1
            self.spongeBC.spongeRhoEAvg = self.rhoE1
        
    def calcDtFromCFL(self):
        UChar = np.fabs(self.U) + self.sos
        self.dt   = np.min(self.CFL*self.dx/UChar)

        
    def calcSpongeSource(self,f,f_avg):
        if self.spongeFlag == 1:
            source = self.spongeBC.spongeSigma*(f_avg-f)
        else:
            source = 0
        return source
    
    def preStepBCHandling(self, rho, rhoU, rhoE):
        if self.bcX0 == "ADIABATIC_WALL":
            self.U[0]  = 0.0
            rhoU[0]    = 0.0
            self.T[0]  = self.deriv.calcNeumann0(self.T)
            self.p[0]  = self.idealGas.calcPIdealGas(rho[0],self.T[0])
        
        if self.bcX1 == "ADIABATIC_WALL":
            self.U[-1] = 0.0
            rhoU[-1] = 0.0
            self.T[-1] = self.deriv.calcNeumannEnd(self.T)
            self.p[-1] = self.idealGas.calcPIdealGas(rho[-1],self.T[-1])
    
    def updateSponge(self):
        eps = 1.0/(self.spongeBC.spongeAvgT/self.dt+1.0)
        self.spongeBC.spongeRhoAvg  += eps*(self.rho1  - self.spongeBC.spongeRhoAvg)
        self.spongeBC.spongeRhoUAvg += eps*(self.rhoU1 - self.spongeBC.spongeRhoUAvg)
        self.spongeBC.spongeRhoEAvg += eps*(self.rhoE1 - self.spongeBC.spongeRhoEAvg)
        self.spongeBC.spongeRhoEAvg = (self.spongeBC.spongeEpsP*self.spongeBC.spongeRhoEAvg + 
            (1.0 - self.spongeBC.spongeEpsP)*(self.spongeBC.spongeP/(self.spongeBC.gamma-1) + 
             0.5*(self.spongeBC.spongeRhoUAvg**2)/self.spongeBC.spongeRhoAvg))
        
        if self.bcX0 == "SPONGE":
            self.rho1[0]   = self.spongeBC.spongeRhoAvg[0]
            self.rhoU1[0]  = self.spongeBC.spongeRhoUAvg[0]
            self.rhoE1[0]  = self.spongeBC.spongeRhoEAvg[0]
            
        if self.bcX1 == "SPONGE": 
            self.rho1[-1]  = self.spongeBC.spongeRhoAvg[-1]
            self.rhoU1[-1] = self.spongeBC.spongeRhoUAvg[-1]
            self.rhoE1[-1] = self.spongeBC.spongeRhoEAvg[-1]



    
        