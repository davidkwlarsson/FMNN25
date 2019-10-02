# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 12:20:09 2018

@author: David_000
"""
import numpy as np
import scipy as sc
import scipy.linalg as sl
import matplotlib.pyplot as plt

A = np.array([[5,0,0,-1],
     [1,0,-3,1], 
     [-1.5,1,-2,1], 
     [-1,5,3,-3]])

p = np.linspace(0,1,50)
for i in range(len(p)): 
    Ap = p[i]*A
    for i in range(len(A)):
        Ap[i,i] = A[i,i]
        
    eig = sl.eigvals(Ap)
    
    X = [x.real for x in eig]
    Y = [x.imag for x in eig]
    plt.plot(X,Y,'ro')

Adiag = np.diag(A)
Xdiag = [x.real for x in Adiag]
Ydiag = [x.imag for x in Adiag]
plt.plot(Xdiag, Ydiag,'*')
plt.show