#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 13:54:43 2017

@author: edwin
"""
import numpy as np

class IdealGas:
    
    def __init__(self, mu_ref):
        self.gamma = 1.4
        self.Pr = 0.7
        self.p_ref = 1/self.gamma
        self.rho_ref = 1.0
        self.T_ref = 1.0
        self.mu_ref = mu_ref
        self.R_gas = self.p_ref/self.rho_ref/self.T_ref
        self.cp = self.R_gas*self.gamma/(self.gamma-1.0)

    def solveRhoE(self,rho,U,V,p):
        return p/(self.gamma-1) + (1/2)*rho*(U*U + V*V)
    
    def solveT(self,rho, p):
        return p/(rho*self.R_gas)
    
    def solvePIdealGas(self, rho, T):
        return T*rho*self.R_gas
    
    def solvePPrimative(self, rho, rhoU, rhoV, rhoE):
        return (self.gamma-1)*(rhoE - 0.5*(rhoU*rhoU + rhoV*rhoV)/rho)
    
    def solveMu(self,T):
        return self.mu_ref*(T/self.T_ref)**0.76
    
    def solveAMu(self, T):
        return 0.76*(self.mu_ref/(self.T_ref**0.76))*T**(0.76-1.0)
    
    def solveK(self, mu):
        return self.cp*mu/self.Pr
    
    def solveSOS(self, rho, p):
        return np.sqrt(self.gamma*p/rho)

        