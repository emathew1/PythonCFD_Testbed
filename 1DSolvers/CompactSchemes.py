#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 21:17:56 2017

@author: edwin
"""

import numpy as np
import scipy.sparse as spspar
import scipy.sparse.linalg as spsparlin


#####################
#####################
# STAGGERED SCHEMES #
#####################
#####################

###########################
# Staggered Interpolation #
###########################

class StaggeredInterp:
    
    def __init__(self, N, typeBC):
        self.N = N
        self.typeBC = typeBC
        self.createStaggeredInterpMatrices()
        

    def createStaggeredInterpMatrices(self):
        N = self.N
        typeBC = self.typeBC
        
        alphaI = 3/10
        aI     = 1/8 * (9 + 10*alphaI)
        bI     = 1/8 * (6*alphaI - 1)
    
        #Interpolation Universal LHS Matrix
        LHI = spspar.lil_matrix((N,N))
        LHI.setdiag(np.ones(N),0)
        LHI.setdiag(alphaI*np.ones(N-1),-1)
        LHI.setdiag(alphaI*np.ones(N-1),1)
        if typeBC == "PERIODIC":
            LHI[0,-1] = alphaI
            LHI[-1,0] = alphaI
        self.LHI = LHI.tocsr()
        
        #Interp to 1/2 Left
        RHI1 = spspar.lil_matrix((N,N))
        RHI1.setdiag((bI/2)*np.ones(N-2),-2)
        RHI1.setdiag((aI/2)*np.ones(N-1),-1)
        RHI1.setdiag((aI/2)*np.ones(N),0)
        RHI1.setdiag((bI/2)*np.ones(N-1),1)
        if typeBC == "PERIODIC":
            RHI1[0,-1] = aI/2
            RHI1[0,-2] = bI/2
            RHI1[1,-1] = bI/2
            RHI1[-1,0] = bI/2
        else:
            raise ValueError('Unknown boundary condition ' + typeBC)
                        
        self.RHI1 = RHI1.tocsr()
        
        #Interp to 1/2 Right
        RHI2 = spspar.lil_matrix((N,N))
        RHI2.setdiag((bI/2)*np.ones(N-1),-1)
        RHI2.setdiag((aI/2)*np.ones(N),0)
        RHI2.setdiag((aI/2)*np.ones(N-1),1)
        RHI2.setdiag((bI/2)*np.ones(N-2),2)
        if typeBC == "PERIODIC":
            RHI2[0,-1] = (bI/2)
            RHI2[-1,0] = (aI/2)
            RHI2[-1,1] = (bI/2)
            RHI2[-2,0] = (bI/2)
        else:
            raise ValueError('Unknown boundary condition ' + typeBC)
            
        self.RHI2 = RHI2.tocsr()
            
    def compactInterpHalfLeft (self,f): return spsparlin.spsolve(self.LHI,spspar.csr_matrix.dot(self.RHI1,f))
    
    def compactInterpHalfRight(self,f): return spsparlin.spsolve(self.LHI,spspar.csr_matrix.dot(self.RHI2,f))

#########################
# Staggered Derivatives #
#########################

class StaggeredDeriv:
    
    def __init__(self, N, dx, typeBC):
        self.N = N
        self.dx = dx
        self.typeBC = typeBC
        self.createStaggeredDerivMatrices()
        
    def createStaggeredDerivMatrices(self):
        N  = self.N
        dx = self.dx 
        typeBC = self.typeBC
        
        alphaD = 9/62
        aD     = 3/8 * (3 - 2*alphaD)
        bD     = 1/8 * (22*alphaD -1)
        
        #Derivatives Universal LHS Matrix
        LHD = spspar.lil_matrix((N,N))
        LHD.setdiag(np.ones(N),0)
        LHD.setdiag(alphaD*np.ones(N-1),-1)
        LHD.setdiag(alphaD*np.ones(N-1),1)
        if typeBC == "PERIODIC":
            LHD[0,-1] = alphaD
            LHD[-1,0] = alphaD
        self.LHD = LHD.tocsr()
        
        #Derivative to 1/2 Left
        RHD1 = spspar.lil_matrix((N,N))
        RHD1.setdiag(-(bD/3)*np.ones(N-2),-2)
        RHD1.setdiag(-(aD)*np.ones(N-1),-1)
        RHD1.setdiag( (aD)*np.ones(N),0)
        RHD1.setdiag( (bD/3)*np.ones(N-1),1)
        if typeBC == "PERIODIC":
            RHD1[0,-1] = -aD
            RHD1[0,-2] = -(bD/3)
            RHD1[1,-1] = -(bD/3)
            RHD1[-1,0] =  (bD/3)
        else:
            raise ValueError('Unknown boundary condition ' + typeBC)
            
        RHD1 = RHD1/dx
        self.RHD1 = RHD1.tocsr()
        
        #Derivative to 1/2 Right
        RHD2 = spspar.lil_matrix((N,N))
        RHD2.setdiag(-(bD/3)*np.ones(N-1),-1)
        RHD2.setdiag(-(aD)*np.ones(N),0)
        RHD2.setdiag( (aD)*np.ones(N-1),1)
        RHD2.setdiag( (bD/3)*np.ones(N-2),2)
        if typeBC == "PERIODIC":
            RHD2[0,-1] = -(bD/3)
            RHD2[-1,0] =  aD
            RHD2[-1,1] =  (bD/3)
            RHD2[-2,0] =  (bD/3)
        else:
            raise ValueError('Unknown boundary condition ' + typeBC)
            
        RHD2 = RHD2/dx
        self.RHD2 = RHD2.tocsr()

    def compactDerivHalfLeft (self,f): return spsparlin.spsolve(self.LHD,spspar.csr_matrix.dot(self.RHD1,f))
    def compactDerivHalfRight(self,f): return spsparlin.spsolve(self.LHD,spspar.csr_matrix.dot(self.RHD2,f))


######################
######################
# COLLOCATED SCHEMES #
######################
######################


class CollocatedDeriv:

    def __init__(self, N, dx, typeBC):
        self.N = N
        self.dx = dx
        self.typeBC = typeBC
        self.createCollocatedDerivMatrices()
    
    def createCollocatedDerivMatrices(self):
        
        N  = self.N
        dx = self.dx
        typeBC = self.typeBC
        
        #####################
        # First Derivatives #
        #####################
        
        #Periodic Coefficients
        alpha1D = 1/3
        a1D     = (2/3)*(2 + alpha1D)
        b1D     = (1/3)*(-1 + 4*alpha1D)
        
        #Dirichlet Coefficients
        alpha11 = 5
        a1 = -197/60
        b1 = -5/12
        c1 =  5
        d1 = -5/3
        e1 =  5/12
        f1 = -1/20
        
        alpha21 = 1/8
        alpha22 = 3/4
        a2 = -43/96
        b2 = -5/6
        c2 =  9/8
        d2 =  1/6
        e2 = -1/96
        
        #1st Derivatives LHS Matrix
        LH1D = spspar.lil_matrix((N,N))
        LH1D.setdiag(np.ones(N),0)
        LH1D.setdiag(alpha1D*np.ones(N-1),-1)
        LH1D.setdiag(alpha1D*np.ones(N-1),1)
        if typeBC == "PERIODIC":
            LH1D[0,-1] = alpha1D
            LH1D[-1,0] = alpha1D
        elif typeBC == "DIRICHLET":
            LH1D[0,1]   = alpha11
            LH1D[1,0]   = alpha21
            LH1D[1,2]   = alpha22
            LH1D[-1,-2] = alpha11
            LH1D[-2,-3] = alpha22
            LH1D[-2,-1] = alpha21
        else:
            raise ValueError('Unknown boundary condition ' + typeBC)
                        
        self.LH1D = LH1D.tocsr()
        
        #1st Derivative
        RH1D = spspar.lil_matrix((N,N))
        RH1D.setdiag(-(b1D/4)*np.ones(N-2),-2)
        RH1D.setdiag(-(a1D/2)*np.ones(N-1),-1)
        RH1D.setdiag( (a1D/2)*np.ones(N-1),1)
        RH1D.setdiag( (b1D/4)*np.ones(N-2),2)
        if typeBC == "PERIODIC":
            RH1D[0,-1] = -a1D/2
            RH1D[0,-2] = -b1D/4
            RH1D[1,-1] = -b1D/4
            RH1D[-2,0] =  b1D/4
            RH1D[-1,0] =  a1D/2
            RH1D[-1,1] =  b1D/4
        elif typeBC == "DIRICHLET":
            RH1D[0,0]  = a1
            RH1D[0,1]  = b1
            RH1D[0,2]  = c1
            RH1D[0,3]  = d1
            RH1D[0,4]  = e1
            RH1D[0,5]  = f1
            RH1D[1,0]  = a2
            RH1D[1,1]  = b2
            RH1D[1,2]  = c2
            RH1D[1,3]  = d2
            RH1D[1,4]  = e2
            RH1D[-1,-1] = -a1
            RH1D[-1,-2] = -b1
            RH1D[-1,-3] = -c1
            RH1D[-1,-4] = -d1
            RH1D[-1,-5] = -e1
            RH1D[-1,-6] = -f1
            RH1D[-2,-1] = -a2
            RH1D[-2,-2] = -b2
            RH1D[-2,-3] = -c2
            RH1D[-2,-4] = -d2
            RH1D[-2,-5] = -e2
        else:
            raise ValueError('Unknown boundary condition ' + typeBC)
        
        RH1D = RH1D/dx
        self.RH1D = RH1D.tocsr()
    
        ######################
        # Second Derivatives #
        ######################
        
        #Periodic Coefficients
        alpha2D = 2/11
        a2D     = 12/11
        b2D     = 3/11
        
        #Dirichlet Coefficients
        alpha11 = 10
        
        alpha4th = 1/10;
        alpha21 = alpha4th
        alpha22 = alpha4th
        
        a4th = 6/5;
        
        a1 = 145/12
        b1 = -76/3
        c1 = 29/2
        d1 = -4/3
        e1 = 1/12
        
        a2 = a4th
        b2 = -2*a2
        c2 = a4th
        
        #2nd Derivatives LHS Matrix
        LH2D = spspar.lil_matrix((N,N))
        LH2D.setdiag(np.ones(N),0)
        LH2D.setdiag(alpha2D*np.ones(N-1),-1)
        LH2D.setdiag(alpha2D*np.ones(N-1),1)
        if typeBC == "PERIODIC":
            LH2D[0,-1] = alpha2D
            LH2D[-1,0] = alpha2D
        elif typeBC == "DIRICHLET":
            LH2D[0,1]   = alpha11
            LH2D[1,0]   = alpha21
            LH2D[1,2]   = alpha22
            LH2D[-1,-2] = alpha11
            LH2D[-2,-3] = alpha22
            LH2D[-2,-1] = alpha21
        else:
            raise ValueError('Unknown boundary condition ' + typeBC)
            
        self.LH2D = LH2D.tocsr()
        
        #2nd Derivative
        RH2D = spspar.lil_matrix((N,N))
        RH2D.setdiag((b2D/4)*np.ones(N-2),-2)
        RH2D.setdiag((a2D)*np.ones(N-1),-1)
        RH2D.setdiag(-(b2D/2 + 2*a2D)*np.ones(N),0)
        RH2D.setdiag((a2D)*np.ones(N-1),1)
        RH2D.setdiag((b2D/4)*np.ones(N-2),2)
        if typeBC == "PERIODIC":
            RH2D[0,-1] =  a2D
            RH2D[0,-2] =  b2D/4
            RH2D[1,-1] =  b2D/4
            RH2D[-2,0] =  b2D/4
            RH2D[-1,0] =  a2D
            RH2D[-1,1] =  b2D/4
        elif typeBC == "DIRICHLET":
            RH2D[0,0]  = a1
            RH2D[0,1]  = b1
            RH2D[0,2]  = c1
            RH2D[0,3]  = d1
            RH2D[0,4]  = e1
            RH2D[1,0]  = a2
            RH2D[1,1]  = b2
            RH2D[1,2]  = c2
            RH2D[1,3]  = 0.0
            RH2D[-1,-1] = a1
            RH2D[-1,-2] = b1
            RH2D[-1,-3] = c1
            RH2D[-1,-4] = d1
            RH2D[-1,-5] = e1
            RH2D[-2,-1] = a2
            RH2D[-2,-2] = b2
            RH2D[-2,-3] = c2
            RH2D[-2,-4] = 0.0
        else:
            raise ValueError('Unknown boundary condition ' + typeBC)
            
        RH2D = RH2D/dx/dx
        self.RH2D = RH2D.tocsr()

    def compact1stDeriv(self,f): return spsparlin.spsolve(self.LH1D,spspar.csr_matrix.dot(self.RH1D,f))
    def compact2ndDeriv(self,f): return spsparlin.spsolve(self.LH2D,spspar.csr_matrix.dot(self.RH2D,f))

    def calcNeumann0(self,f):
        #6th order...
        alpha = 147
        a0 = 360
        a1 = -450
        a2 = 400
        a3 = -225
        a4 = 72
        a5 = -10
        
        f0 = (a0*f[1] + a1*f[2] + a2*f[3] + a3*f[4] + a4*f[5] + a5*f[6])/alpha
        
        return f0

    def calcNeumannEnd(self,f):
        #6th order...
        alpha = 147
        a0 = 360
        a1 = -450
        a2 = 400
        a3 = -225
        a4 = 72
        a5 = -10
        
        fend = (a0*f[-2] + a1*f[-3] + a2*f[-4] + a3*f[-5] + a4*f[-6] + a5*f[-7])/alpha
        
        return fend


###################
###################
# COMPACT FILTERS #
###################
###################

class CompactFilter:

    def __init__(self, N, alphaF, typeBC):
        self.N = N
        self.alphaF = alphaF
        self.typeBC = typeBC        
        self.createFilterMatrices()

    
    def createFilterMatrices(self):
      
        N = self.N
        alphaF = self.alphaF;
        typeBC = self.typeBC
        
        ############################
        # 8th Order Compact Filter #
        ############################
        a0 = (93+70*alphaF)/128;
        a1 = (7 +18*alphaF)/16;
        a2 = (-7+14*alphaF)/32;
        a3 = ( 1- 2*alphaF)/16;
        a4 = (-1+ 2*alphaF)/128;
        
        ############################
        # 6th Order Compact Filter #
        ############################
        a0_6 = (11 + 10*alphaF)/16
        a1_6 = (15 + 34*alphaF)/32
        a2_6 = (-3 +  6*alphaF)/16
        a3_6 = (1  -  2*alphaF)/32
        
        #LHS Filter Matrix 
        LHF = spspar.lil_matrix((N,N))
        LHF.setdiag(alphaF*np.ones(N-1),-1)
        LHF.setdiag(np.ones(N),0)
        LHF.setdiag(alphaF*np.ones(N-1), 1)
        if typeBC == "PERIODIC":
            LHF[0,-1] = alphaF
            LHF[-1,0] = alphaF
        elif typeBC == "DIRICHLET":
            #Explicitly filter the boundary points (Lele)
            LHF[0,1] = 0
            LHF[1,0] = 0
            LHF[1,2] = 0
            LHF[2,1] = 0
            LHF[2,3] = 0
            
            LHF[-1,-2] = 0
            LHF[-2,-1] = 0
            LHF[-2,-3] = 0
            LHF[-3,-2] = 0
            LHF[-3,-4] = 0
        else:
            raise ValueError('Unknown boundary condition ' + typeBC)
            
        
        self.LHF = LHF.tocsr()
        
        #RHS Filter Matrix
        RHF = spspar.lil_matrix((N,N))
        RHF.setdiag((a4/2)*np.ones(N-4),-4)
        RHF.setdiag((a3/2)*np.ones(N-3),-3)
        RHF.setdiag((a2/2)*np.ones(N-2),-2)
        RHF.setdiag((a1/2)*np.ones(N-1),-1)
        RHF.setdiag((a0)*np.ones(N),0)
        RHF.setdiag((a1/2)*np.ones(N-1),1)
        RHF.setdiag((a2/2)*np.ones(N-2),2)
        RHF.setdiag((a3/2)*np.ones(N-3),3)
        RHF.setdiag((a4/2)*np.ones(N-4),4)
        
        if typeBC == "PERIODIC":
            RHF[0,-1] = a1/2
            RHF[0,-2] = a2/2
            RHF[0,-3] = a3/2
            RHF[0,-4] = a4/2
            RHF[1,-1] = a2/2
            RHF[1,-2] = a3/2
            RHF[1,-3] = a4/2
            RHF[2,-1] = a3/2
            RHF[2,-2] = a4/2
            RHF[3,-1] = a4/2
            RHF[-1,0] = a1/2
            RHF[-1,1] = a2/2
            RHF[-1,2] = a3/2
            RHF[-1,3] = a4/2
            RHF[-2,0] = a2/2
            RHF[-2,1] = a3/2
            RHF[-2,2] = a4/2
            RHF[-3,0] = a3/2
            RHF[-3,1] = a4/2
            RHF[-4,0] = a4/2
        elif typeBC == "DIRICHLET":
            RHF[0,0] = 15/16
            RHF[0,1] = (1/16)*4
            RHF[0,2] = -(1/16)*6
            RHF[0,3] = (1/16)*4
            RHF[0,4] = -(1/16)
            
            RHF[1,0] = (1/16)
            RHF[1,1] = 3/4
            RHF[1,2] = (1/16)*6
            RHF[1,3] = -(1/16)*4
            RHF[1,4] = (1/16)
            RHF[1,5] = 0
            
            RHF[2,0] = -(1/16)
            RHF[2,1] = (1/16)*4
            RHF[2,2] = 5/8
            RHF[2,3] = (1/16)*4
            RHF[2,4] = -(1/16)
            RHF[2,5] = 0
            RHF[2,6] = 0
            
            RHF[3,0] = a3_6/2
            RHF[3,1] = a2_6/2
            RHF[3,2] = a1_6/2
            RHF[3,3] = a0_6
            RHF[3,4] = a1_6/2
            RHF[3,5] = a2_6/2
            RHF[3,6] = a3_6/2
            RHF[3,7] = 0
            
            RHF[-1,-1] = 15/16
            RHF[-1,-2] = (1/16)*4
            RHF[-1,-3] = -(1/16)*6
            RHF[-1,-4] = (1/16)*4
            RHF[-1,-5] = -(1/16)
            
            RHF[-2,-1] = (1/16)
            RHF[-2,-2] = 3/4
            RHF[-2,-3] = (1/16)*6
            RHF[-2,-4] = -(1/16)*4
            RHF[-2,-5] = (1/16)
            RHF[-2,-6] = 0
            
            RHF[-3,-1] = -(1/16)
            RHF[-3,-2] = (1/16)*4
            RHF[-3,-3] = 5/8
            RHF[-3,-4] = (1/16)*4
            RHF[-3,-5] = -(1/16)
            RHF[-3,-6] = 0
            RHF[-3,-7] = 0           
        
            RHF[-4,-1] = a3_6/2
            RHF[-4,-2] = a2_6/2
            RHF[-4,-3] = a1_6/2
            RHF[-4,-4] = a0_6
            RHF[-4,-5] = a1_6/2
            RHF[-4,-6] = a2_6/2
            RHF[-4,-7] = a3_6/2
            RHF[-4,-8] = 0
        
        else:
            raise ValueError('Unknown boundary condition ' + typeBC)
            
        self.RHF = RHF.tocsr()
    
    def compactFilter(self,f): return spsparlin.spsolve(self.LHF,spspar.csr_matrix.dot(self.RHF,f))
    