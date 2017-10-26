

import numpy as np
from drawnow import drawnow
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#Bring in functions we need
from CompactSchemes import CollocatedDeriv
from CompactSchemes import CompactFilter
from IdealGas import IdealGas
from SpongeBC import SpongeBC3D

#############################
#############################
# Collocated Compact Solver #
#############################
#############################

class Domain3D:
    
    def __init__(self, Nx, Ny, Nz, x, y, z, Lx, Ly, Lz):
        self.Nx = Nx
        self.x = x
        self.Lx = Lx
        self.dx = x[1]-x[0]
        
        self.Ny = Ny
        self.y = y
        self.Ly = Ly
        self.dy = y[1]-y[0]  
        
        self.Nz = Nz
        self.z = z
        self.Lz = Lz
        self.dz = z[1]-z[0] 
        
        [X, Y, Z] = np.meshgrid(x, y, z)
        
        self.X = X
        self.Y = Y
        self.Z = Z
        
class BC3D:
    
    def __init__(self, bcXType, bcX0, bcX1, bcYType, bcY0, bcY1, bcZType, bcZ0, bcZ1):
        self.bcXType = bcXType
        self.bcX0 = bcX0
        self.bcX1 = bcX1
        
        self.bcYType = bcYType
        self.bcY0 = bcY0
        self.bcY1 = bcY1
        
        self.bcZType = bcZType
        self.bcZ0 = bcZ0
        self.bcZ1 = bcZ1        
        
class TimeStepping:
    
    def __init__(self, CFL, maxTimeStep, maxTime, plotStep, filterStep):
        self.CFL = CFL
        self.maxTimeStep = maxTimeStep
        self.maxTime = maxTime
        self.plotStep = plotStep
        self.filterStep = filterStep

class CSolver3D:
    
    def __init__(self, domain, bc, timeStepping, alphaF, mu_ref):
        
        self.done = False
        
        self.Uwall = 0.1
        
        #grid info
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
        
        self.X = domain.X
        self.Y = domain.Y
        self.Z = domain.Z
        
        #gas properties
        self.idealGas = IdealGas(mu_ref)
        
        #BC info
        self.bcXType = bc.bcXType
        self.bcX0 = bc.bcX0
        self.bcX1 = bc.bcX1
        
        self.bcYType = bc.bcYType
        self.bcY0 = bc.bcY0
        self.bcY1 = bc.bcY1
        
        self.bcZType = bc.bcZType
        self.bcZ0 = bc.bcZ0
        self.bcZ1 = bc.bcZ1
        
        #initial conditions
        self.U0   = np.zeros((self.Nx, self.Ny, self.Nz))
        self.V0   = np.zeros((self.Nx, self.Ny, self.Nz))
        self.W0   = np.zeros((self.Nx, self.Ny, self.Nz))
        self.rho0 = np.zeros((self.Nx, self.Ny, self.Nz))
        self.p0   = np.zeros((self.Nx, self.Ny, self.Nz))
        
        #non-conserved data
        self.U   = np.zeros((self.Nx, self.Ny, self.Nz))
        self.V   = np.zeros((self.Nx, self.Ny, self.Nz))
        self.W   = np.zeros((self.Nx, self.Ny, self.Nz))
        self.T   = np.zeros((self.Nx, self.Ny, self.Nz))
        self.p   = np.zeros((self.Nx, self.Ny, self.Nz))
        self.mu  = np.zeros((self.Nx, self.Ny, self.Nz))
        self.k   = np.zeros((self.Nx, self.Ny, self.Nz))
        self.sos = np.zeros((self.Nx, self.Ny, self.Nz))
        
        #Derivatives of Data
        self.Ux  = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Uy  = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Uz  = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Uxx = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Uyy = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Uzz = np.zeros((self.Nx, self.Ny, self.Nz))        
        self.Uxy = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Uxz = np.zeros((self.Nx, self.Ny, self.Nz))        
        self.Uyz = np.zeros((self.Nx, self.Ny, self.Nz))
        
        self.Vx   = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Vy   = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Vz   = np.zeros((self.Nx, self.Ny, self.Nz))        
        self.Vxx  = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Vyy  = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Vzz  = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Vxy  = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Vxz  = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Vyz  = np.zeros((self.Nx, self.Ny, self.Nz))
        
        self.Tx  = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Ty  = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Tz  = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Txx = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Tyy = np.zeros((self.Nx, self.Ny, self.Nz))
        self.Tzz = np.zeros((self.Nx, self.Ny, self.Nz))

        #conserved data
        self.rho1  = np.zeros((self.Nx, self.Ny, self.Nz))
        self.rhoU1 = np.zeros((self.Nx, self.Ny, self.Nz))
        self.rhoV1 = np.zeros((self.Nx, self.Ny, self.Nz))
        self.rhoW1 = np.zeros((self.Nx, self.Ny, self.Nz))
        self.rhoE1 = np.zeros((self.Nx, self.Ny, self.Nz))
        
        self.rhok  = np.zeros((self.Nx, self.Ny, self.Nz))
        self.rhoUk = np.zeros((self.Nx, self.Ny, self.Nz))
        self.rhoVk = np.zeros((self.Nx, self.Ny, self.Nz))
        self.rhoWk = np.zeros((self.Nx, self.Ny, self.Nz))
        self.rhoEk = np.zeros((self.Nx, self.Ny, self.Nz))

        self.rhok2  = np.zeros((self.Nx, self.Ny, self.Nz))
        self.rhoUk2 = np.zeros((self.Nx, self.Ny, self.Nz))
        self.rhoVk2 = np.zeros((self.Nx, self.Ny, self.Nz))
        self.rhoWk2 = np.zeros((self.Nx, self.Ny, self.Nz))       
        self.rhoEk2 = np.zeros((self.Nx, self.Ny, self.Nz))        
        
        self.rho2  = np.zeros((self.Nx, self.Ny, self.Nz))
        self.rhoU2 = np.zeros((self.Nx, self.Ny, self.Nz))
        self.rhoV2 = np.zeros((self.Nx, self.Ny, self.Nz))
        self.rhoW2 = np.zeros((self.Nx, self.Ny, self.Nz))
        self.rhoE2 = np.zeros((self.Nx, self.Ny, self.Nz))
        
        #time data
        self.timeStep = 0
        self.time     = 0.0
        self.maxTimeStep = timeStepping.maxTimeStep
        self.maxTime     = timeStepping.maxTime
        self.plotStep    = timeStepping.plotStep
        self.CFL = timeStepping.CFL
        
        #filter data
        self.filterStep = timeStepping.filterStep
        self.numberOfFilterStep = 0
        self.alphaF = alphaF
        
        if self.bcX0 == "SPONGE" or self.bcX1 == "SPONGE" or self.bcY0 == "SPONGE" or self.bcY1 == "SPONGE" or self.bcZ0 == "SPONGE" or self.bcZ1 == "SPONGE"  :
            self.spongeFlag = True
            self.spongeBC = SpongeBC3D(domain, self.idealGas, bc)
        else:
            self.spongeFlag = False
                
        #Generate our derivatives and our filters    
        self.derivX = CollocatedDeriv(self.Nx, self.dx, domain, self.bcXType, "X")
        self.derivY = CollocatedDeriv(self.Ny, self.dy, domain, self.bcYType, "Y")
        self.derivZ = CollocatedDeriv(self.Nz, self.dz, domain, self.bcZType, "Z")
        
        self.filtX  = CompactFilter(self.Nx, self.alphaF, domain, self.bcXType, "X")
        self.filtY  = CompactFilter(self.Ny, self.alphaF, domain, self.bcYType, "Y")
        self.filtZ  = CompactFilter(self.Nz, self.alphaF, domain, self.bcZType, "Z")
              
        
    def setInitialConditions(self, rho0, U0, V0, W0, p0):
        self.rho0 = rho0
        self.U0 = U0
        self.V0 = V0
        self.W0 = W0
        self.p0 = p0
        
        self.U     = U0
        self.V     = V0
        self.W     = W0
        self.rho1  = rho0
        self.p     = p0
        self.rhoU1 = rho0*U0
        self.rhoV1 = rho0*V0
        self.rhoW1 = rho0*W0
        self.rhoE1 = self.idealGas.solveRhoE(rho0,U0,V0,W0,p0)
        self.T     = self.idealGas.solveT(rho0, p0)
        self.mu    = self.idealGas.solveMu(self.T)
        self.Amu   = self.idealGas.solveAMu(self.T)
        self.k     = self.idealGas.solveK(self.mu)
        self.sos   = self.idealGas.solveSOS(self.rho1, self.p)
        
#        if self.bcX0 == "ADIABATIC_WALL":
#            self.T[0,:]  = self.derivX.calcNeumann0(self.T)
#            self.U[0,:]  = 0.0
#            self.rhoU1[0,:]  = 0.0
#            self.V[0,:]  = 0.0
#            self.rhoV1[0,:]  = 0.0
#            
#        if self.bcX1 == "ADIABATIC_WALL":
#            self.T[-1,:] = self.derivX.calcNeumannEnd(self.T)
#            self.U[-1,:] = 0.0
#            self.rhoU1[-1,:] = 0.0
#            self.V[-1,:] = 0.0
#            self.rhoV1[-1,:] = 0.0
#            
#        if self.bcY0 == "ADIABATIC_WALL":
#            self.T[:,0]  = self.derivY.calcNeumann0(self.T)
#            self.U[:,0]  = 0.0
#            self.rhoU1[:,0]  = 0.0
#            self.V[:,0]  = 0.0
#            self.rhoV1[:,0]  = 0.0
#            
#        if self.bcY1 == "ADIABATIC_WALL":
#            self.T[:,-1]  = self.derivY.calcNeumannEnd(self.T)
#            self.U[:,-1]  = 0.0
#            self.rhoU1[:,-1]  = 0.0
#            self.V[:,-1]  = 0.0
#            self.rhoV1[:,-1]  = 0.0   
#            
#        if self.bcY1 == "ADIABATIC_MOVINGWALL":
#            self.T[:,-1]  = self.derivY.calcNeumannEnd(self.T)
#            self.U[:,-1]  = self.Uwall
#            self.rhoU1[:,-1]  = self.rho1[:,-1]*self.Uwall
#            self.V[:,-1]  = 0.0
#            self.rhoV1[:,-1]  = 0.0 
#
#        if self.bcX0 == "ADIABATIC_WALL" or self.bcX1 == "ADIABATIC_WALL" or self.bcY0 == "ADIABATIC_WALL" or self.bcY1 == "ADIABATIC_WALL" or self.bcY1 == "ADIABATIC_MOVINGWALL":
#            self.p     = self.idealGas.solvePIdealGas(self.rho1, self.T)
#            self.sos   = self.idealGas.solveSOS(self.rho1, self.p)
#            self.rhoE1 = self.idealGas.solveRhoE(self.rho1, self.U, self.V, self.p)            
#            
        if self.spongeFlag == True: 
            self.spongeBC.spongeRhoAvg  = self.rho1
            self.spongeBC.spongeRhoUAvg = self.rhoU1
            self.spongeBC.spongeRhoVAvg = self.rhoV1   
            self.spongeBC.spongeRhoWAvg = self.rhoW1            
            self.spongeBC.spongeRhoEAvg = self.rhoE1
            
        
    def calcDtFromCFL(self):
        UChar_dx = ((np.fabs(self.U) + self.sos)/self.dx + 
                    (np.fabs(self.V) + self.sos)/self.dy +
                    (np.fabs(self.W) + self.sos)/self.dz)
        
        self.dt   = np.min(self.CFL/UChar_dx)
        
        #Increment timestep
        self.timeStep += 1
        self.time += self.dt

        
    def calcSpongeSource(self,f,eqn):
        if self.spongeFlag == True:
            if eqn == "CONT":
                source = self.spongeBC.spongeSigma*(self.spongeBC.spongeRhoAvg-f)
            elif eqn == "XMOM":
                source = self.spongeBC.spongeSigma*(self.spongeBC.spongeRhoUAvg-f)     
            elif eqn == "YMOM":
                source = self.spongeBC.spongeSigma*(self.spongeBC.spongeRhoVAvg-f)   
            elif eqn == "ZMOM":
                source = self.spongeBC.spongeSigma*(self.spongeBC.spongeRhoWAvg-f)                   
            elif eqn == "ENGY":
                source = self.spongeBC.spongeSigma*(self.spongeBC.spongeRhoEAvg-f)
            else:
                source = 0
        else:
            source = 0
        return source
    
#    def preStepBCHandling(self, rho, rhoU, rhoV, rhoW, rhoE):
        
#        if self.bcX0 == "ADIABATIC_WALL":
#            self.U[0,:]  = 0.0
#            rhoU[0,:]    = 0.0
#            self.V[0,:]  = 0.0
#            rhoV[0,:]    = 0.0
#            self.T[0,:]  = self.derivX.calcNeumann0(self.T)
#            self.p[0,:]  = self.idealGas.solvePIdealGas(rho[0,:],self.T[0,:])
#        
#        if self.bcX1 == "ADIABATIC_WALL":
#            self.U[-1,:] = 0.0
#            rhoU[-1,:]   = 0.0
#            self.V[-1,:] = 0.0
#            rhoV[-1,:]   = 0.0
#            self.T[-1,:] = self.derivX.calcNeumannEnd(self.T)
#            self.p[-1,:] = self.idealGas.solvePIdealGas(rho[-1,:],self.T[-1,:])
#
#        if self.bcY0 == "ADIABATIC_WALL":
#            self.U[:,0]  = 0.0
#            rhoU[:,0]    = 0.0
#            self.V[:,0]  = 0.0
#            rhoV[:,0]    = 0.0
#            self.T[:,0]  = self.derivY.calcNeumann0(self.T)
#            self.p[:,0]  = self.idealGas.solvePIdealGas(rho[:,0],self.T[:,0])
#    
#        if self.bcY1 == "ADIABATIC_WALL":
#            self.U[:,-1]  = 0.0
#            rhoU[:,-1]    = 0.0
#            self.V[:,-1]  = 0.0
#            rhoV[:,-1]    = 0.0
#            self.T[:,-1]  = self.derivY.calcNeumannEnd(self.T)
#            self.p[:,-1]  = self.idealGas.solvePIdealGas(rho[:,-1],self.T[:,-1])
# 
#        if self.bcY1 == "ADIABATIC_MOVINGWALL":
#            self.U[:,-1]  = self.Uwall
#            rhoU[:,-1]    = rho[:,-1]*self.Uwall
#            self.V[:,-1]  = 0.0
#            rhoV[:,-1]    = 0.0
#            self.T[:,-1]  = self.derivY.calcNeumannEnd(self.T[:,:])
#            self.p[:,-1]  = self.idealGas.solvePIdealGas(rho[:,-1],self.T[:,-1])
#    
    
    def preStepDerivatives(self):
        
        #First Derivatives
        self.Ux = self.derivX.df_3D(self.U)
        self.Vx = self.derivX.df_3D(self.V)
        self.Wx = self.derivX.df_3D(self.W)
        
        self.Uy = self.derivY.df_3D(self.U)
        self.Vy = self.derivY.df_3D(self.V)
        self.Wy = self.derivY.df_3D(self.W)
        
        self.Uz = self.derivZ.df_3D(self.U)
        self.Vz = self.derivZ.df_3D(self.V)
        self.Wz = self.derivZ.df_3D(self.W)       
        
        self.Tx = self.derivX.df_3D(self.T)
        self.Ty = self.derivY.df_3D(self.T)
        self.Tz = self.derivZ.df_3D(self.T)
        
        #Second Derivatives
        self.Uxx = self.derivX.d2f_3D(self.U)
        self.Vxx = self.derivX.d2f_3D(self.V)        
        self.Wxx = self.derivX.d2f_3D(self.W)                
        
        self.Uyy = self.derivY.d2f_3D(self.U)
        self.Vyy = self.derivY.d2f_3D(self.V)
        self.Wyy = self.derivY.d2f_3D(self.W)
        
        self.Uzz = self.derivZ.d2f_3D(self.U)
        self.Vzz = self.derivZ.d2f_3D(self.V)
        self.Wzz = self.derivZ.d2f_3D(self.W)
        
        self.Txx = self.derivX.d2f_3D(self.T)
        self.Tyy = self.derivY.d2f_3D(self.T)
        self.Tzz = self.derivZ.d2f_3D(self.T)

        
        #Cross Derivatives
        if self.timeStep%2 == 0:
            self.Uxy = self.derivX.df_3D(self.derivY.df_3D(self.U))
            self.Vxy = self.derivX.df_3D(self.derivY.df_3D(self.V))
            self.Wxy = self.derivX.df_3D(self.derivY.df_3D(self.W))
        else:
            self.Uxy = self.derivY.df_3D(self.derivX.df_3D(self.U))
            self.Vxy = self.derivY.df_3D(self.derivX.df_3D(self.V))  
            self.Wxy = self.derivY.df_3D(self.derivX.df_3D(self.W))
            
        if self.timeStep%2 == 0:
            self.Uyz = self.derivY.df_3D(self.derivZ.df_3D(self.U))
            self.Vyz = self.derivY.df_3D(self.derivZ.df_3D(self.V))
            self.Wyz = self.derivY.df_3D(self.derivZ.df_3D(self.W))
        else:
            self.Uyz = self.derivZ.df_3D(self.derivY.df_3D(self.U))
            self.Vyz = self.derivZ.df_3D(self.derivY.df_3D(self.V))  
            self.Wyz = self.derivZ.df_3D(self.derivY.df_3D(self.W))
            
        if self.timeStep%2 == 0:
            self.Uxz = self.derivX.df_3D(self.derivZ.df_3D(self.U))
            self.Vxz = self.derivX.df_3D(self.derivZ.df_3D(self.V))
            self.Wxz = self.derivX.df_3D(self.derivZ.df_3D(self.W))
        else:
            self.Uxz = self.derivZ.df_3D(self.derivX.df_3D(self.U))
            self.Vxz = self.derivZ.df_3D(self.derivX.df_3D(self.V))  
            self.Wxz = self.derivZ.df_3D(self.derivX.df_3D(self.W))           
            
        
#Actually solving the equations...        
        
    def solveXMomentum_Euler(self, rhoU):
        return -self.derivX.df_3D(rhoU*self.U + self.p) - self.derivY.df_3D(rhoU*self.V)
                    -self.derivZ.df_3D(rhoU*self.W)
    
    def solveYMomentum_Euler(self, rhoV):
        return -self.derivX.df_3D(rhoV*self.U) -self.derivY.df_3D(rhoV*self.V + self.p)
                    -self.derivZ.df_3D(rhoV*self.W)
                    
    def solveZMomentum_Euler(self, rhoW):
        return -self.derivX.df_3D(rhoW*self.U) -self.derivY.df_3D(rhoW*self.V)
                    -self.derivZ.df_3D(rhoW*self.W + self.p)                    

    def solveEnergy_Euler(self, rhoE):
        return -self.derivX.df_3D(rhoE*self.U + self.U*self.p) 
                    - self.derivY.df_3D(rhoE*self.V + self.V*self.p)
                        - self.derivZ.df_3D(rhoE*self.W + self.W*self.p)
    
    ##Need to write these out!
    
    def solveXMomentum_Viscous(self):
        return ((4/3)*self.Amu*self.Tx*self.Ux + (4/3)*self.mu*self.Uxx - (2/3)*self.Amu*self.Tx*self.Vy +
                  (1/3)*self.mu*self.Vxy + self.Amu*self.Ty*self.Vx + self.Amu*self.Ty*self.Uy + self.mu*self.Uyy);       

    def solveYMomentum_Viscous(self):
        return ((4/3)*self.Amu*self.Ty*self.Vy + (4/3)*self.mu*self.Vyy - (2/3)*self.Amu*self.Ty*self.Ux +
                  (1/3)*self.mu*self.Uxy + self.Amu*self.Tx*self.Uy + self.Amu*self.Tx*self.Vx + self.mu*self.Vxx);

    def solveEnergy_Viscous(self):
        return  ((self.idealGas.cp/self.idealGas.Pr)*self.Amu*(self.Tx**2 + self.Ty**2) +
                    (self.idealGas.cp/self.idealGas.Pr)*self.mu*(self.Txx + self.Tyy)  +
                    (4/3)*self.mu*(self.Ux**2 + self.Vy**2) + 
                    (4/3)*self.Amu*(self.U*self.Tx*self.Ux + self.V*self.Ty*self.Vy) +
                    (4/3)*self.mu*(self.U*self.Uxx + self.V*self.Vyy) - (4/3)*self.mu*self.Ux*self.Vy -
                    (2/3)*self.Amu*(self.U*self.Tx*self.Vy + self.V*self.Ty*self.Ux) +
                    (1/3)*self.mu*(self.U*self.Vxy + self.V*self.Uxy) +
                    self.mu*(self.Uy**2 + self.Vx**2) + 2.*self.mu*(self.Uy*self.Vx) +
                    self.mu*(self.U*self.Uyy + self.V*self.Vxx) +
                    self.Amu*(self.V*self.Tx*self.Uy + self.V*self.Tx*self.Vx + 
                    self.U*self.Ty*self.Uy + self.U*self.Ty*self.Vx))

#Using methods that don't maximize the resolution capabilities of the compact differences
    def solveContinuity(self, rho, rhoU, rhoV, rhoE):
        
        drho = (-self.derivX.df_2D(rhoU) - self.derivY.df_2D(rhoV))
        
        self.rhok2  = self.dt*(drho + self.calcSpongeSource(rho,"CONT"))

    def solveXMomentum(self, rho, rhoU, rhoV, rhoE):
        drhoU = (-self.derivX.df_2D(rhoU*self.U + self.p +
                                    -2*self.mu*self.Ux + (2/3)*self.mu*(self.Ux + self.Vy)) +
                                         -self.derivY.df_2D(rhoU*self.V -
                                        self.mu*(self.Vx + self.Uy)))
    
        self.rhoUk2 = self.dt*(drhoU + self.calcSpongeSource(rhoU,"XMOM"))            
            
    def solveYMomentum(self, rho, rhoU, rhoV, rhoE):
        
        drhoV = (-self.derivY.df_2D(rhoV*self.V + self.p +
                                    -2*self.mu*self.Vy + (2/3)*self.mu*(self.Ux + self.Vy))
                                         -self.derivX.df_2D(rhoV*self.U -
                                        self.mu*(self.Vx + self.Uy)) )  
        
        self.rhoVk2 = self.dt*(drhoV + self.calcSpongeSource(rhoV,"YMOM"))
    
    def solveEnergy(self, rho, rhoU, rhoV, rhoE):
        drhoE = (-self.derivX.df_2D(rhoE*self.U + self.U*self.p
                    - (self.mu/self.idealGas.Pr/(self.idealGas.gamma-1))*self.derivX.df_2D(self.T) + 
                     - self.U*(2*self.mu*self.Ux - (2/3)*self.mu*(self.Ux + self.Vy)) 
                     - self.V*(self.mu*(self.Vx + self.Uy))) 
                - self.derivY.df_2D(rhoE*self.V + self.V*self.p 
                    - (self.mu/self.idealGas.Pr/(self.idealGas.gamma-1))*self.derivY.df_2D(self.T) +
                     - self.V*(2*self.mu*self.Vy - (2/3)*self.mu*(self.Ux + self.Vy))
                     - self.U*(self.mu*(self.Vx + self.Uy)))); 
        
        
        self.rhoEk2 = self.dt*(drhoE + self.calcSpongeSource(rhoE,"ENGY"))

#Using methods that do take advantage of the the spectral benefits of the compact diff's    

    def solveXMomentum_PV(self, rho, rhoU, rhoV, rhoE):
        self.rhoUk2 = self.dt*(self.solveXMomentum_Euler(rhoU) +
                               self.solveXMomentum_Viscous() +
                               self.calcSpongeSource(rhoU,"XMOM"))            

    def solveYMomentum_PV(self, rho, rhoU, rhoV, rhoE):
        self.rhoVk2 = self.dt*(self.solveYMomentum_Euler(rhoV) + 
                               self.solveYMomentum_Viscous() +                               
                               self.calcSpongeSource(rhoV,"YMOM"))

    def solveEnergy_PV(self, rho, rhoU, rhoV, rhoE):
        self.rhoEk2 = self.dt*(self.solveEnergy_Euler(rhoE) + 
                               self.solveEnergy_Viscous() +                                                              
                               self.calcSpongeSource(rhoE,"ENGY"))
        
    
    def postStepBCHandling(self, rho, rhoU, rhoV, rhoE):
        if self.bcXType == "DIRICHLET":
            
            if self.bcX0 == "ADIABATIC_WALL":
                self.rhok2[0,:] = -self.dt*self.derivX.df_2D(rhoU)[0,:]
                self.rhoUk2[0,:]  = 0
                self.rhoVk2[0,:]  = 0
                self.rhoEk2[0,:]  = (-self.dt*(self.derivX.df_2D(rhoE*self.U + self.U*self.p) -
                                      (self.mu/self.idealGas.Pr/(self.idealGas.gamma-1))*self.derivX.d2f_2D(self.T) +
                                      (4/3)*self.mu*self.U*self.derivX.d2f_2D(self.U)))[0,:]
            else:
                self.rhok2[0,:]   = 0
                self.rhoUk2[0,:]  = 0
                self.rhoVk2[0,:]  = 0
                self.rhoEk2[0,:]  = 0
                
            if self.bcX1 == "ADIABATIC_WALL":
                self.rhok2[-1,:]  = -self.dt*self.derivX.df_2D(rhoU)[-1,:]
                self.rhoUk2[-1,:] = 0
                self.rhoVk2[-1,:] = 0
                self.rhoEk2[-1,:] = (-self.dt*(self.derivX.df_2D(rhoE*self.U + self.U*self.p) -
                                      (self.mu/self.idealGas.Pr/(self.idealGas.gamma-1))*self.derivX.d2f_2D(self.T) +
                                      (4/3)*self.mu*self.U*self.derivX.d2f_2D(self.U)))[-1,:]
            else:
                self.rhok2[-1,:]  = 0
                self.rhoUk2[-1,:] = 0
                self.rhoVk2[-1,:] = 0
                self.rhoEk2[-1,:] = 0  
                
        if self.bcYType == "DIRICHLET":
                
            if self.bcY0 == "ADIABATIC_WALL":
                self.rhok2[:,0] = -self.dt*self.derivY.df_2D(rhoV)[:,0]
                self.rhoUk2[:,0]  = 0
                self.rhoVk2[:,0]  = 0
                self.rhoEk2[:,0]  = (-self.dt*(self.derivY.df_2D(rhoE*self.V + self.V*self.p) -
                                      (self.mu/self.idealGas.Pr/(self.idealGas.gamma-1))*self.derivY.d2f_2D(self.T) +
                                      (4/3)*self.mu*self.V*self.derivY.d2f_2D(self.V)))[:,0]
            else:
                self.rhok2[:,0]   = 0
                self.rhoUk2[:,0]  = 0
                self.rhoVk2[:,0]  = 0
                self.rhoEk2[:,0]  = 0
            
                
            if self.bcY1 == "ADIABATIC_WALL" or self.bcY1 == "ADIABATIC_MOVINGWALL":
                self.rhok2[:,-1]  = -self.dt*self.derivY.df_2D(rhoV)[:,-1]
                self.rhoUk2[:,-1] = 0
                self.rhoVk2[:,-1] = 0
                self.rhoEk2[:,-1]  = (-self.dt*(self.derivY.df_2D(rhoE*self.V + self.V*self.p) -
                                      (self.mu/self.idealGas.Pr/(self.idealGas.gamma-1))*self.derivY.d2f_2D(self.T) +
                                      (4/3)*self.mu*self.V*self.derivY.d2f_2D(self.V)))[:,-1]
            else:
                self.rhok2[:,-1]   = 0
                self.rhoUk2[:,-1]  = 0
                self.rhoVk2[:,-1]  = 0
                self.rhoEk2[:,-1]  = 0
    
    def updateConservedData(self,rkStep):
        if rkStep == 1:
            self.rho2  = self.rho1 + self.rhok2/6
            self.rhoU2 = self.rhoU1 + self.rhoUk2/6
            self.rhoV2 = self.rhoV1 + self.rhoVk2/6
            self.rhoE2 = self.rhoE1 + self.rhoEk2/6
        elif rkStep == 2:
            self.rho2  += self.rhok2/3
            self.rhoU2 += self.rhoUk2/3
            self.rhoV2 += self.rhoVk2/3            
            self.rhoE2 += self.rhoEk2/3          
        elif rkStep == 3:
            self.rho2  += self.rhok2/3
            self.rhoU2 += self.rhoUk2/3
            self.rhoV2 += self.rhoVk2/3
            self.rhoE2 += self.rhoEk2/3               
        elif rkStep == 4:
            self.rho2  += self.rhok2/6
            self.rhoU2 += self.rhoUk2/6
            self.rhoV2 += self.rhoVk2/6
            self.rhoE2 += self.rhoEk2/6

        if rkStep == 1:
            self.rhok  = self.rho1  + self.rhok2/2
            self.rhoUk = self.rhoU1 + self.rhoUk2/2
            self.rhoVk = self.rhoV1 + self.rhoVk2/2
            self.rhoEk = self.rhoE1 + self.rhoEk2/2
        elif rkStep == 2:
            self.rhok  = self.rho1  + self.rhok2/2
            self.rhoUk = self.rhoU1 + self.rhoUk2/2
            self.rhoVk = self.rhoV1 + self.rhoVk2/2
            self.rhoEk = self.rhoE1 + self.rhoEk2/2
        elif rkStep == 3:
            self.rhok  = self.rho1  + self.rhok2
            self.rhoUk = self.rhoU1 + self.rhoUk2
            self.rhoVk = self.rhoV1 + self.rhoVk2
            self.rhoEk = self.rhoE1 + self.rhoEk2
            
    def updateNonConservedData(self,rkStep):
        if rkStep == 1 or rkStep == 2 or rkStep == 3:
            self.U = self.rhoUk/self.rhok
            self.V = self.rhoVk/self.rhok
            self.p = (self.idealGas.gamma-1)*(self.rhoEk - 
                         0.5*(self.rhoUk*self.rhoUk + self.rhoVk*self.rhoVk)/self.rhok)
            self.T = (self.p/(self.rhok*self.idealGas.R_gas))   
            self.mu = self.idealGas.mu_ref*(self.T/self.idealGas.T_ref)**0.76
            self.k  = self.idealGas.cp*self.mu/self.idealGas.Pr
            self.sos   = np.sqrt(self.idealGas.gamma*self.p/self.rhok)
            self.Amu   = self.idealGas.solveAMu(self.T)
            
        elif rkStep == 4:
            self.U = self.rhoU1/self.rho1
            self.V = self.rhoV1/self.rho1
            self.p = (self.idealGas.gamma-1)*(self.rhoE1 - 
                         0.5*(self.rhoU1*self.rhoU1 + self.rhoV1*self.rhoV1)/self.rho1)
            self.T = (self.p/(self.rho1*self.idealGas.R_gas))   
            self.mu = self.idealGas.mu_ref*(self.T/self.idealGas.T_ref)**0.76
            self.k  = self.idealGas.cp*self.mu/self.idealGas.Pr
            self.sos   = np.sqrt(self.idealGas.gamma*self.p/self.rho1)
            self.Amu   = self.idealGas.solveAMu(self.T)
            
    def filterPrimativeValues(self):
        if(self.timeStep%self.filterStep == 0):
            self.numberOfFilterStep += 1
            
            #Need to flip the order of the filtering every other time
            if self.numberOfFilterStep%2 == 0:
                self.rho1  = self.filtX.filt_2D(self.rho2)
                self.rhoU1 = self.filtX.filt_2D(self.rhoU2)
                self.rhoV1 = self.filtX.filt_2D(self.rhoV2)
                self.rhoE1 = self.filtX.filt_2D(self.rhoE2)
                
                self.rho1  = self.filtY.filt_2D(self.rho1)
                self.rhoU1 = self.filtY.filt_2D(self.rhoU1)
                self.rhoV1 = self.filtY.filt_2D(self.rhoV1)
                self.rhoE1 = self.filtY.filt_2D(self.rhoE1)
            else:
                self.rho1  = self.filtY.filt_2D(self.rho2)
                self.rhoU1 = self.filtY.filt_2D(self.rhoU2)
                self.rhoV1 = self.filtY.filt_2D(self.rhoV2)
                self.rhoE1 = self.filtY.filt_2D(self.rhoE2)
                
                self.rho1  = self.filtX.filt_2D(self.rho1)
                self.rhoU1 = self.filtX.filt_2D(self.rhoU1)
                self.rhoV1 = self.filtX.filt_2D(self.rhoV1)
                self.rhoE1 = self.filtX.filt_2D(self.rhoE1)
                
            print("Filtering...")
        
            #Is there something here that needs to be done about corners for
            #DIRICHLET/DIRICHLET, PERIODIC/DIRICHLET?
            if self.bcXType == "DIRICHLET":
                self.rho1[0,:]   = self.rho2[0,:]
                self.rho1[-1,:]  = self.rho2[-1,:]
                self.rhoU1[0,:]  = self.rhoU2[0,:]
                self.rhoU1[-1,:] = self.rhoU2[-1,:]
                self.rhoV1[0,:]  = self.rhoV2[0,:]
                self.rhoV1[-1,:] = self.rhoV2[-1,:]            
                self.rhoE1[0,:]  = self.rhoE2[0,:]
                self.rhoE1[-1,:] = self.rhoE2[-1,:]
            
            if self.bcYType == "DIRICHLET":
                self.rho1[:,0]   = self.rho2[:,0]
                self.rho1[:,-1]  = self.rho2[:,-1]
                self.rhoU1[:,0]  = self.rhoU2[:,0]
                self.rhoU1[:,-1] = self.rhoU2[:,-1]
                self.rhoV1[:,0]  = self.rhoV2[:,0]
                self.rhoV1[:,-1] = self.rhoV2[:,-1]            
                self.rhoE1[:,0]  = self.rhoE2[:,0]
                self.rhoE1[:,-1] = self.rhoE2[:,-1]           
            
        else:
            self.rho1  = self.rho2
            self.rhoU1 = self.rhoU2
            self.rhoV1 = self.rhoV2
            self.rhoE1 = self.rhoE2
        
            
    def updateSponge(self):
        if self.spongeFlag == True:
            eps = 1.0/(self.spongeBC.spongeAvgT/self.dt+1.0)
            self.spongeBC.spongeRhoAvg  += eps*(self.rho1  - self.spongeBC.spongeRhoAvg)
            self.spongeBC.spongeRhoUAvg += eps*(self.rhoU1 - self.spongeBC.spongeRhoUAvg)
            self.spongeBC.spongeRhoVAvg += eps*(self.rhoV1 - self.spongeBC.spongeRhoVAvg)
            self.spongeBC.spongeRhoEAvg += eps*(self.rhoE1 - self.spongeBC.spongeRhoEAvg)
            self.spongeBC.spongeRhoEAvg = (self.spongeBC.spongeEpsP*self.spongeBC.spongeRhoEAvg + 
                (1.0 - self.spongeBC.spongeEpsP)*(self.spongeBC.spongeP/(self.idealGas.gamma-1) + 
                 0.5*(self.spongeBC.spongeRhoUAvg**2 + self.spongeBC.spongeRhoVAvg**2)/self.spongeBC.spongeRhoAvg))
            
            if self.bcX0 == "SPONGE":
                self.rho1[0,:]   = self.spongeBC.spongeRhoAvg[0,:]
                self.rhoU1[0,:]  = self.spongeBC.spongeRhoUAvg[0,:]
                self.rhoV1[0,:]  = self.spongeBC.spongeRhoVAvg[0,:]
                self.rhoE1[0,:]  = self.spongeBC.spongeRhoEAvg[0,:]
                
            if self.bcX1 == "SPONGE": 
                self.rho1[-1,:]  = self.spongeBC.spongeRhoAvg[-1,:]
                self.rhoU1[-1,:] = self.spongeBC.spongeRhoUAvg[-1,:]
                self.rhoV1[-1,:] = self.spongeBC.spongeRhoVAvg[-1,:]
                self.rhoE1[-1,:] = self.spongeBC.spongeRhoEAvg[-1,:]
                
            if self.bcY0 == "SPONGE":
                self.rho1[:,0]   = self.spongeBC.spongeRhoAvg[:,0]
                self.rhoU1[:,0]  = self.spongeBC.spongeRhoUAvg[:,0]
                self.rhoV1[:,0]  = self.spongeBC.spongeRhoVAvg[:,0]
                self.rhoE1[:,0]  = self.spongeBC.spongeRhoEAvg[:,0]
                
            if self.bcY1 == "SPONGE": 
                self.rho1[:,-1]  = self.spongeBC.spongeRhoAvg[:,-1]
                self.rhoU1[:,-1] = self.spongeBC.spongeRhoUAvg[:,-1]
                self.rhoV1[:,-1] = self.spongeBC.spongeRhoVAvg[:,-1]
                self.rhoE1[:,-1] = self.spongeBC.spongeRhoEAvg[:,-1]  
              
    
    def plotFigure(self):
        #plt.plot(self.x,self.rho1[:,0])
#        plt.imshow(np.rot90(self.Vx[1:-2,1:-2] - self.Uy[1:-2,1:-2]), cmap="RdBu",interpolation='bicubic')
#        plt.imshow(np.rot90(self.Ux[1:-2,1:-2] + self.Ux[1:-2,1:-2]), cmap="RdBu",interpolation='bicubic')
        plt.imshow(np.rot90(self.U), cmap="RdBu",interpolation='bicubic')
        plt.colorbar()
        plt.axis("equal")

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

    
        