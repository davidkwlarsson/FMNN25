# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 09:25:34 2018

@author: David_000
"""

import numpy.linalg as nl
import scipy.linalg as sl
import numpy as np




A = np.random.rand(5,5)
A = 1/2*(A+A.T)
eig = sl.eigvals(A)
Q,R = sl.qr(A)
Anew = R@Q

def practQR(Ak, rtol = 1e-8):
    n = Ak.shape[0]
    itercount = 0
    
    super_diagonal = np.tri(n, k=1, dtype=bool) * np.tri(n, dtype=bool)
    isclose = np.isclose(Ak[super_diagonal], 0, rtol = rtol/2)
    
    while not isclose.any():
        itercount += 1
        
        mu = Ak[-1,-1]
        Q,R = sl.qr(Ak-mu*np.eye(n))
        Ak = R@Q + mu*np.eye(n)
        isclose = np.isclose(A[super_diagonal], 0, rtol = rtol/2)
            
        
    if not isclose.all():
        for j in range(n-1):
            if isclose[j]:
                break
            
        return_list = []
        for B in [Ak[:j+1,:j+1], A[j+1:,j+1:]]:
            l,i = practQR(B)
            return_list += l
            itercount += 1
            
        return return_list,itercount
    else:
        diagonal = np.diag([True]*n)
        return list(A[diagonal]), itercount
        
    
    
Adiag, inbr = practQR(Anew)
    
print(Adiag)
print(eig)
print(np.allclose(eig,np.diag(Anew),rtol=1e-8))
