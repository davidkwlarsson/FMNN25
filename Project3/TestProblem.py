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
    print('m',m,'n',n)
    size = m*n
    
    A = np.eye(size)*(-4)
    for i in range(size-1):
        A[i,i+1] = 1
        A[i+1,i] = 1
    for i in range(size-m):
        A[i+m-1,i] = 1
        A[i,i+m-1] = 1    
        
    if roomnbr == 1:
        for i in range(n):
            A[-n+i,-n+i] = -3
    return A


    # room is an int
    # i rader, j columner
def setBoundaries(u, room):
    [i,j] = sc.shape(u);               
    if room == 2:
        u[0,:] = 40
        u[-1,:] = 5
        #u[0:int(i/2) + 1,0] = 15
        #u[int(i/2):,-1] = 15
        u[:,0] = 15     #endast rum 2
        u[:,-1] = 15
        #hörn
        u[0,0] = (40+15)/2
        u[-1,0] = (5+15)/2
        u[-1,-1] = (5+15)/2 
        u[0,-1] = (40+15)/2 
    if room == 1:
        u[:,0] = 40
        u[0,:] = 15
        u[-1,:] = 15
        #hörn
        u[0,0] = (40+15)/2
        u[-1,0] = (40+15)/2
        #u[-1,-1] = (5+15)/2
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
            

def my_plot(u, deltax, room) :
    if room == 2:
        a = u.reshape(int(1/deltax)+1, int(2/deltax)+1).T
        plt.imshow(a, cmap='hot', interpolation='nearest')
        plt.show()
        plt.colorbar() 
        
              
def main1():
   # comm = MPI.COMM_WORLD	
    #rank = comm.Get_rank()
    #size = comm.Get_size()
    intitialtemp = 0
    deltax = 1/4
    n = int(1/deltax)
    w = 0.8
    u2 = np.ones((2*n + 1, n+1))*intitialtemp
    u2L = laplaceMatrix(u2,2)
    setBoundaries(u2,2)
    u2 = np.hstack(u2.T)
        
    for i in range(100):     #iteration solver
        #boundarys for 2
        b2 = np.zeros((2*n+1,n+1))
        setBoundaries(b2,2) 
        b2 = np.hstack(b2.T)
        u2new = sl.solve(u2L,-b2) #dx is in both RHS and LHS
        #reshapar för att lättare kunna göra operationer
        u2 = u2.reshape(int(1/deltax)+1, int(2/deltax)+1).T
        u2new = u2new.reshape(int(1/deltax)+1, int(2/deltax)+1).T
        u2[1:-1,1:-1] = w*u2new[1:-1,1:-1] + (1-w)*u2[1:-1,1:-1]
        setBoundaries(u2,2)
        u2 = np.hstack(u2.T)
        #u2 updaterad
 
    u2 = u2.reshape(int(1/deltax)+1, int(2/deltax)+1).T
    heatplot = plt.imshow(u2)
    heatplot.set_cmap('hot')
    plt.colorbar()
    plt.show()
   
    
def main2():
    intitialtemp = 0
    deltax = 1/3
    n = int(1/deltax)
    w = 0.8
    u1 = np.ones((n-1, n))*intitialtemp
    u1L = laplaceMatrix(u1,1)
    #setBoundaries(u1,1)
    u1 = np.hstack(u1.T)
    print(u1)
        
    for i in range(100):     #iteration solver
        #boundarys for 2
        b1 = give_b_vector(1,n)
        b1 = b1.flatten()
        print(b1)
        u1new = sl.solve(u1L,b1) #dx is in both RHS and LHS
        #reshapar för att lättare kunna göra operationer
        u1 = u1.reshape(int(1/deltax)-1, int(1/deltax)).T
        u1new = u1new.reshape(int(1/deltax)-1, int(1/deltax)).T
        u1 = w*u1new + (1-w)*u1
       
        u1 = np.hstack(u1.T)
        #u2 updaterad
 
    u1 = u1.reshape(int(1/deltax)-1, int(1/deltax)).T
    uprint = np.zeros((n+1,n+1))
    
    setBoundaries(uprint,1)
    heatplot = plt.imshow(u1)
    heatplot.set_cmap('hot')
    plt.colorbar()
    plt.show()

main2()
