# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 08:35:28 2017

@author: mat.mathews
"""
import scipy.sparse as spspar
import scipy.sparse.linalg as spsparlin
import scipy.linalg.lapack as sll
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time

def cyclic(a, b, c, alpha, beta, r, n, bb, u, x, z):
    start = time.time()
    gamma = -b[0]
    bb[:] = b[:]
    end = time.time()
    print("1")
    print(end-start)
    
    start = time.time()
    bb[0] = b[0]-gamma
    bb[-1]  = b[-1]-alpha*beta/gamma
    end = time.time()
    print("2")
    print(end-start)
    
    start = time.time()
    x = sll.dgtsv(a,bb,c,r)[3]
    end = time.time()
    print("3")
    print(end-start)

    
    start = time.time()
    u[0] = gamma
    u[-1] = alpha
    end = time.time()
    print("4")
    print(end-start)
    
    start = time.time()
    z = sll.dgtsv(a,bb,c,u)[3]

    end = time.time()
    print("5")
    print(end-start)

 
    start = time.time()
    fact = (x[0]+beta*x[-1]/gamma)/(1.0+z[0]+beta*z[-1]/gamma)
    x = x - fact*z
    end = time.time()
    print("6")
    print(end-start)

    return x

                
    
    
    

N = 100
a = np.ones(N-1,dtype=np.double)
b = -2*np.ones(N,dtype=np.double)
c = np.ones(N-1,dtype=np.double)

d = np.ones(N,dtype=np.double)

start = time.time()
for i in range(10000):
    x = sll.dgtsv(a,b,c,d)
end = time.time()
print(end-start)


lmat = spspar.lil_matrix((N,N))
lmat.setdiag(np.ones(N-1),-1)
lmat.setdiag(-2*np.ones(N),0)
lmat.setdiag(np.ones(N-1),1)
lmat = lmat.tocsr()

start = time.time()
for i in range(10000):
    xsparse = spsparlin.spsolve(lmat,d)
end = time.time()
print(end-start)


lmatc = spspar.lil_matrix((N,N))
lmatc.setdiag(np.ones(N-1),-1)
lmatc.setdiag(-2*np.ones(N),0)
lmatc.setdiag(np.ones(N-1),1)
lmatc[0,-1] = 1.0
lmatc[-1,0] = 1.0
lmatc = lmatc.tocsr()

start = time.time()
for i in range(10000):
    xsparsec = spsparlin.spsolve(lmatc,d)
end = time.time()
print(end-start)

work1 = np.zeros(N,dtype=np.double)
work2 = np.zeros(N,dtype=np.double)
work3 = np.zeros(N,dtype=np.double)
work4 = np.zeros(N,dtype=np.double)
start = time.time()
for i in range(10000):
    x = cyclic(a,b,c,1.0,1.0,d,N,work1,work2,work3,work4)
end = time.time()
print(end-start)
