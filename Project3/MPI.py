# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 12:23:49 2018
@author: annai
"""

import scipy as sc
import pylab as pl
from mpi4py import MPI
import numpy as np 
import scipy.linalg as sl


#    comm.recv(source, tag)
#    comm.send(name, dest, tag)

    #returns Delta u(i,j)
def laplaceMatrix(u):
        (m,n) = sc.shape(u)
        size = m*n
        
        diag = [-4]*size
        sub1 = [1]*(size-1) + [0]
        sup1 = [0] + [1]*(size-1)
        subm = [1]*(size-m) + [0]*m
        supm = [0]*m + [1]*(size-m)
        
        N = sc.array([supm,sup1,diag,sub1,subm])
   
    # room is an int
    # i rader, j columner
def setBoundaries(u, room):
    [i,j] = sc.shape(u);
    if room == 1:
        u[:,0] = 40
        #u[:,-1] = neumann   
        u[0,:] = 15
        u[-1,:] = 15
        
        u[0,0] = (40+15)/2
        u[-1,0] = (40+15)/2
        u[-1,-1] = (5+15)/2
    
            
    if room == 2:
        u[0,:] = 40
        u[-1,:] = 5
        for k in range(i/2):
            u[k,0] = 15
         #   u[k,-1] = dirichlet    
         #   u[k+i/2,0] = dirichlet
            u[k+i/2,-1]= 15
            
        u[0,0] = (40+15)/2
        u[-1,0] = (5+15)/2
        u[-1,-1] = (5+15)/2 
        u[0,-1] = (40+15)/2 
            
    if room == 3:
    #       u[:,0] = neumann
        u[:,-1] = 40
        u[0,:] = 15
        u[-1,:] = 15
        
        u[0,0] = (40+15)/2 
        u[0,-1] =(40+15)/2 
        u[-1,-1] = (40+15)/2 
              
def main():
    comm = MPI.COMM_WORLD	
    rank = comm.Get_rank()
    size = comm.Get_size()
    intitialtemp = 10
    deltax = 1/20
    n = int(1/deltax)
    w = 0.8
    initial_guess = True
    if rank == 0:
        u1 = np.ones((int(1/deltax), int(1/deltax)))*intitialtemp
        u2 = np.ones((int(2/deltax), int(1/deltax)))*intitialtemp
        u1L = laplaceMatrix(u1)
        u2L = laplaceMatrix(u2)
        setBoundaries(u1,1)
        setBoundaries(u2,2)
        u2 = np.hstack(u2.T)
        u1 = np.hstack(u1.T)
    if rank == 1:
        u3 = np.ones((int(1/deltax), int(1/deltax)))*intitialtemp
        u3L = laplaceMatrix(u3)
        setBoundaries(u3,3)
        u3 = np.hstack(u3.T)
        
    for i in range(10):     #iteration solver
        if rank == 0:
            b2 = np.zeros(2*n,n)
            setBoundaries(b2,2)
            
            if initial_guess:
                u2k3 = u2[-2*n+1:-n-1]
                u2k1 = u2[n-1:2*n-2-1]
                initial_guess = False
            else:
                u2k3 = comm.recv(source=1, tag=12)
            
            b2[n:,0] = u2k1
            b2[0:n,-1] = u2k3
            b2 = np.hstack(b2.T)
            u2new = sl.solvebanded(u2L,-b2) #dx is in both RHS and LHS
            u2 = w*u2new + (1-w)*u2
            
            u2k3 = u2[-2*n+1:-n-1] #boundary to room 3
            u2k1 = u2[n-1:2*n-2-1]  #boundary to room 2
            
            comm.send(u2k3, dest=1, tag=11)
            b1 = np.zeros(n,n)
            setBoundaries(b1,2)
            b1[:,-1] = u2k1
            b1 = np.hstack(b1.T)
            u1new = sl.solvebanded(u1L,-b1)
            u1 = w*u1new + (1-w)*u1
            u2k1 = u1[-n:]
            
        if rank == 1:
            u2k = comm.recv(source=0, tag=11)
            b3 = np.zeros(n,n)
            setBoundaries(b3,3)
            b3[:n] = u2k
            u3new = sl.solvebanded(u3L,-b3)
            u3 = w*u3new + (1-w)*u3
            
            u3k2 = u3[0:n]
            comm.send(u3k2, dest=0, tag=12)
    
    
    

main()