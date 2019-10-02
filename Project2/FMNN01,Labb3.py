# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 18:24:19 2018

@author: David_000
"""
import numpy.linalg as nl
import scipy.linalg as sl
import numpy as np

A = sl.hilbert(5)
Ainv = sl.invhilbert(5)
u,s,vh = nl.svd(A)
Anorm = nl.norm(A,2)
Ainvnorm = nl.norm(Ainv,2)
smat = np.diag(s)


#Choose b in the direction corresponding to the maximum eigenval for A
#Choose db in the direction corresponding to the minimum eigenval for A
#np.zeros(50)
#b[0] = 1
#np.zeros(50)

b = u[:,0]
db = u[:,-1]

print()



condnbr = Anorm*Ainvnorm
condnbrS = s[0]/s[-1]

print("Condition number = ", condnbr)

x = Ainv@b
dx = Ainv@db
RHS = condnbr*sl.norm(db,2)/sl.norm(b,2)
LHS = sl.norm(dx,2)/sl.norm(x,2)

print(LHS, " = ", RHS)