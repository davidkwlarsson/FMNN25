
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 22:25:42 2018

@author: David_000
"""
from mpi4py import MPI
import numpy as np
import scipy as sc
import numpy.linalg as nl
import matplotlib.pyplot as plt

def laplaceMidMatrix(dimension):
    (m,n) = dimension
    size = (m-2)*(n-2) #dosent care about the dirichlet conditions (inner points)
    
    A = np.eye(size)*(-4)
    for i in range(1,size):
        if i%(n-2):
            A[i-1][i] = 1
            A[i][i-1] = 1
    for i in range(size-n+2):
        A[i][i+n-2] = 1
        A[i+n-2][i] = 1        
    #N = sc.array([supm,sup1,diag,sub1,subm])
    return A

def laplaceSideMatrix(size,roomnbr):
    diaglength = (size-1)*(size-2)
    A = np.eye(diaglength)*(-4)
    for i in range(1,diaglength):
        if i%(size-1):   #dont go into this for the boundaries
            A[i-1][i] = 1
            A[i][i-1] = 1
    for i in range(size-1,diaglength):
        A[i-size+1][i] = 1
        A[i][i-size+1] = 1  

    for i in range(size-2): #for every row
        if roomnbr == 3:
            bi = (size - 1)*i #the left side of room 3
            A[bi][bi] = -3
        elif roomnbr == 1:
            bi = (size - 2) + (size-1)*i #the right side of room 1
            A[bi][bi] = -3
    return A
            

def gradfunc(u_left, u_right, dx):
    return (u_right - u_left)/dx
            
def give_b_vector(left,top,right,bot):
    m = len(left)
    n = len(top)
    size = m*n #size for the flattend b vector
    b = np.zeros(size) #creates the boundary vector with top left as 0
    b[0] = top[0]+left[0] #first element
    b[n-1] = top[-1] + right[0]
    b[size-n] = left[-1] + bot[0]
    b[-1] = bot[-1] + right[-1] # last element
    b[1:n-1] = top[1:-1]
    b[size-n+1:-1] = bot[1:-1]
    for i in range(1,m-1):
        b[i*n] = left[i] #n times any scalar will be the left boundaries
        b[i*n + n -1] = right[i] # + n - 1 to get to the right side
        
        
    return b


def main():
    inv_dx = 20
    dx = 1/inv_dx
    w = 0.8
    
    #The calculation needs only to be done to the inner points since we have
    #dirichlet conditions on the boundaries for the middle room. -> (2n-1)x(n-1)
    
    #For the small rooms we need to include the points with the the neumann
    #conditions aswell. -> (n-1)x(n-1+1)
    if rank == 0: #rank 0 for the middle and 3 room
    #bounary vectors for the middle room:
        left2 = np.ones(inv_dx*2 - 1)*15
        top2 = np.ones(inv_dx - 1)*40
        bot2 = np.ones(inv_dx - 1)*5
        right2 = np.ones(inv_dx*2 - 1)*15
        top13 = np.ones(inv_dx)*15
        bot13 = np.ones(inv_dx)*15
        right3 = np.ones(inv_dx - 1)*40
        u2L = laplaceMidMatrix((inv_dx*2 + 1, inv_dx +1))/(dx**2) #the size is + 1 for the points
        u3L = laplaceSideMatrix(inv_dx+1,3)/(dx**2)
        
    
    #boundary for the small rooms:
    if rank == 1: #rank 1 for the 1 room
        left1 = np.ones(inv_dx - 1)*40
        top13 = np.ones(inv_dx)*15
        bot13 = np.ones(inv_dx)*15
        u1L = laplaceSideMatrix(inv_dx+1,1)/(dx**2)
    
    #the laplaceMatricies for the rooms:
    
    for i in range(100):
        #starting with solving the right room, dirichlet beeing in the boundary vectors
        #initially set to 15
        if rank == 0:
            b2 = give_b_vector(left2,top2,right2,bot2)/(dx**2)
            b2 = np.hstack(b2)
            u2new = nl.solve(u2L,-b2)
            
            if i == 0:
                u2 = u2new   
            else:
                u2 = w*u2 + (1-w)*u2new
                
            #the neuman conditions for the left room:
            tempneu1 = u2.reshape((2*inv_dx - 1, inv_dx - 1))[-inv_dx + 1:,0]
            neu1 = gradfunc(left2[-inv_dx + 1:],tempneu1,dx)
            
            comm.send(neu1, dest=1, tag=11)
            #neumann for right room:
            tempneu3 = u2.reshape((2*inv_dx - 1, inv_dx - 1))[:inv_dx-1,-1]
            neu3 = gradfunc(tempneu3,right2[:inv_dx -1],dx)
        
            #solve right
            b3 = give_b_vector(-dx*neu3,top13,right3,bot13)/(dx**2)
            b3 = np.hstack(b3)
            u3new = nl.solve(u3L,-b3)
            if i == 0:
                u3 = u3new
            else:
                u3 = w*u3 + (1-w)*u3new
            u1send = comm.recv(source=1,tag=12)
            #putting the solution from room 1 and 3 to the boundary vectors
            left2[-inv_dx + 1:] = u1send
            right2[:inv_dx -1] = u3.reshape((inv_dx-1,inv_dx))[:,0]
            
        if rank == 1:
            #hopefully this worked
            #solve left
            neu1 = comm.recv(source=0, tag=11)
            b1 = give_b_vector(left1,top13,neu1*dx,bot13)/(dx**2)
            b1 = np.hstack(b1)
            
            u1new = nl.solve(u1L,-b1)
            if i == 0:
                u1 = u1new
            else:
                u1 = w*u1 + (1-w)*u1new
            u1send = u1.reshape((inv_dx-1,inv_dx))[:,-1]
            comm.send(u1send,dest=0,tag=12)
        
    if rank == 1:
        u1 = u1.reshape((inv_dx-1,inv_dx))
        comm.send(u1,dest = 0, tag = 20)
    if rank == 0:
        u3 = u3.reshape((inv_dx-1,inv_dx))
        u2 = u2.reshape((2*inv_dx - 1, inv_dx - 1))
        u1 = comm.recv(source = 1, tag = 20)

        mid_h,mid_w = u2.shape
        left_h,left_w = u1.shape
        right_h,right_w = u3.shape
        
        plot_matrix_width = mid_w+left_w+right_w+2
        plot_matrix_height = mid_h+2
        plot_matrix = np.zeros((plot_matrix_height,plot_matrix_width))
        
        
        plot_matrix[-inv_dx:-1,1:inv_dx+1] = u1
        plot_matrix[1:-1,inv_dx+1:2*inv_dx] = u2
        plot_matrix[1:inv_dx,2*inv_dx:-1] = u3
        plot_matrix[-1,1:inv_dx+1] = np.ones(inv_dx)*15
        plot_matrix[-1,inv_dx+1:2*inv_dx] = np.ones(inv_dx-1)*5
        plot_matrix[-inv_dx-1:,0] = np.ones(inv_dx+1)*40
        plot_matrix[-inv_dx-1,1:inv_dx+1] = np.ones(inv_dx)*15
        plot_matrix[:inv_dx,inv_dx] = np.ones(inv_dx)*15
        plot_matrix[0,inv_dx+1:2*inv_dx] = np.ones(mid_w)*40
        plot_matrix[:inv_dx+1,-1] = np.ones(inv_dx+1)*40
        plot_matrix[inv_dx+1:,2*inv_dx] = np.ones(inv_dx)*15
        plot_matrix[inv_dx,2*inv_dx:-1] = np.ones(inv_dx)*15
        plot_matrix[0,2*inv_dx:-1] = np.ones(right_w)*15
        
        heatplot = plt.imshow(plot_matrix)
        heatplot.set_cmap('hot')
        plt.colorbar()
        plt.show()
    

comm = MPI.COMM_WORLD	
rank = comm.Get_rank()
size = comm.Get_size()
main()
    
    
    
    
    