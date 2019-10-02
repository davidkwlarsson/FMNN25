# -*- coding: utf-8 -*-
"""
Created on Wed Oct  10 10:23:49 2018
@author: David
"""

from mpi4py import MPI
import scipy as sc
import numpy as np 
import scipy.linalg as sl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


#    comm.recv(source, tag)
#    comm.send(name, dest, tag)

    #returns Delta u(i,j)

def laplaceMatrix(u,roomnbr):
    (m,n) = sc.shape(u)
    size = m*n
    
    A = np.eye(size)*(-4)
    for i in range(size-1):
        A[i,i+1] = 1
        A[i+1,i] = 1
    for i in range(size-m):
        A[i+m-2,i] = 1
        A[i,i+m-2] = 1   
    if roomnbr == 1:
        for i in range(n):
            A[-n+i,-n+i] = -3
        #diag[-n:] = [-3]*n
        
    elif roomnbr == 3:
        for i in range(n):
            A[i,i] = -3
        #diag[:n] = [-3]*n

        
    #N = sc.array([supm,sup1,diag,sub1,subm])
    return A


def give_b_vector(roomnbr,n):
    if roomnbr == 2:
        b = np.zeros((2*n-1,n-1))
        for i in range(2*n-1):
            b[i,0] += -15
            b[i, n-2] += -15
        for i in range(n-1):
            b[0,i] += -40
            b[2*n-2, i] += -5
        return b
            
    if roomnbr == 1:
        b = np.zeros((n-1,n)) #need the points for neuman condition
        for i in range(n-1):    #the heated wall
            b[i-1, 0] += -40
        for i in range(n):      #top and bottom
            b[0,i] += -15
            b[n-2, i] += -15
        return b
            

    # room is an int
    # i rader, j columner
def setBoundaries(u, room):
    [i,j] = sc.shape(u);
    if room == 1:
        u[:,0] = 40
        u[0,:] = 15
        u[-1,:] = 15
        #hörn
        u[0,0] = (40+15)/2
        u[-1,0] = (40+15)/2
        #u[-1,-1] = (5+15)/2
               
    if room == 2:
        u[0,:] = 40
        u[-1,:] = 5
        u[0:int(i/2) + 1,0] = 15
        u[int(i/2):,-1] = 15
        #hörn
        u[0,0] = (40+15)/2
        u[-1,0] = (5+15)/2
        u[-1,-1] = (5+15)/2 
        u[0,-1] = (40+15)/2 
            
    if room == 3:
        u[:,-1] = 40
        u[0,:] = 15
        u[-1,:] = 15
        #hörn
        #u[0,0] = (40+15)/2 
        u[0,-1] =(40+15)/2 
        u[-1,-1] = (40+15)/2 
        
        
def my_plot(u, deltax, room) :
    if room == 2:
        a = u.reshape(int(1/deltax)+1, int(2/deltax)+1).T
        plt.imshow(a, cmap='hot', interpolation='nearest')
        plt.show()
    else:
        a = u.reshape(int(1/deltax)+1, int(1/deltax)+1).T
        plt.imshow(a, cmap='hot', interpolation='nearest')
        plt.show()   
        
    if room == 3:    
        plt.colorbar()

              
def main():
   # comm = MPI.COMM_WORLD	
    #rank = comm.Get_rank()
    #size = comm.Get_size()
    intitialtemp = 15
    deltax = 1/3
    n = int(1/deltax)
    w = 0.8
    u1 = np.ones((n+1, n-1))*intitialtemp
    u2 = np.ones((2*n + 1, n+1))*intitialtemp
    u3 = np.ones((n+1, n+1))*intitialtemp
    u1L = laplaceMatrix(u1,1)
    u2L = laplaceMatrix(u2,2)
    u3L = laplaceMatrix(u3,3)
    setBoundaries(u1,1)
    setBoundaries(u2,2)
    setBoundaries(u3,3)
    u2 = np.hstack(u2.T)
    u1 = np.hstack(u1.T)
    u3 = np.hstack(u3.T)  
    print(u1L)
        
    for i in range(1):     #iteration solver
        #boundarys for 2
        b2 = np.zeros((2*n+1,n+1))
        setBoundaries(b2,2)
        if i==0:
            u2k3 = u2[-2*n-1:-n].copy() #oklart om detta hjällper
            u2k1 = u2[n:2*n+1].copy()
        else:
            u2k3 = u3[:n+1].copy() #dirichlet between 3 an 2
            u2k1 = u1[-n-1:].copy() #dirichlet between 2 to 1
        
        b2[n:,0] = u2k1 
        b2[:n+1,-1] = u2k3
        b2 = np.hstack(b2.T)
        u2new = sl.solve(u2L,-b2) #dx is in both RHS and LHS
        
        #reshapar för att lättare kunna göra operationer
        u2 = u2.reshape(int(1/deltax)+1, int(2/deltax)+1).T
        u2new = u2new.reshape(int(1/deltax)+1, int(2/deltax)+1).T

        u2 = w*u2new + (1-w)*u2
        #u2[:n+1,-1] = u2k3 
        #u2[n:,0] = u2k1
        setBoundaries(u2,2)
        u2 = np.hstack(u2.T)
        #u2 updaterad

        u2k3 = (u2[-2*n-1:-n] - u2[-4*n-2:-3*n-1])/deltax #neuman between 2 and 3
        u2k1 = (u2[3*n+1:4*n+2] - u2[n:2*n+1])/deltax  #neuman between 2 and 1
        #boundarys for 1
        b1 = np.zeros((n+1,n-1))
        setBoundaries(b1,1) 
        b1[:,-1] = u2k1*deltax #neuman between 2 and 1?
        b1 = np.hstack(b1.T)
        u1new = sl.solve(u1L,-b1) #dx is in both RHS and LHS
        #reshapar för att lättare kunna göra operationer
        u1 = u1.reshape(int(1/deltax)-1, int(1/deltax)+1).T
        u1new = u1new.reshape(int(1/deltax)-1, int(1/deltax)+1).T
        u1 = w*u1new + (1-w)*u1
       
        
        u1 = np.hstack(u1.T)
        
        #setBoundaries(u1,1)
        
        #u1 updaterad
        
        
        b3 = np.zeros((n+1,n+1))
        setBoundaries(b3,3)
        b3[:,0] = - u2k3*deltax#neuman between 2 and 3?
        b3 = np.hstack(b3.T)
        u3new = sl.solve(u3L, -b3)
        #reshapar för att lättare kunna göra operationer
        u3 = u3.reshape(int(1/deltax)+1, int(1/deltax)+1).T
        u3new = u3new.reshape(int(1/deltax)+1, int(1/deltax)+1).T
        u3 = w*u3new + (1-w)*u3
        #u3[:,0] = u2k3
        setBoundaries(u3,3)
        u3 = np.hstack(u3.T)
        #u3 updaterad
        
    
    u1 = u1.reshape(int(1/deltax)-1, int(1/deltax)+1).T
    uprint = np.zeros((n+1,n+1))
    uprint[:,1:-1] = u1
    setBoundaries(uprint,1)
    heatplot = plt.imshow(uprint)
    heatplot.set_cmap('hot')
    plt.colorbar()
    plt.show()
    
    
    

main()
