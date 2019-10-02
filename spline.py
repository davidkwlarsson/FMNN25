#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 10:13:10 2018

@author: samuel
"""

import scipy as sc
import scipy.linalg as sl
import numpy as np
import matplotlib.pyplot as mpl

class Spline:
    #constructor for our spline, gives the attributes controlpoints
    #interpolate boolean and subgrid as a modified linspace
    def __init__(self, controlpoints, subgrid, interpolate):
        self.controlpoints = controlpoints
        self.interpolate = interpolate
        if(((subgrid[0]==subgrid[1])*(subgrid[1]==subgrid[2]))*
           ((subgrid[-1]==subgrid[-2])*(subgrid[-2]==subgrid[-3]))):
            self.grid = subgrid
        else:
            grid = sc.r_[subgrid[0], subgrid[0], subgrid, subgrid[-1], subgrid[-1]]
            self.grid = grid
            
    #Finds the hot interval and calculetes s(u)
    def __call__(self, u): 
        self.grid.sort()
        idx=self.grid.searchsorted([u])
        i = idx[0]
        i = i-1
        if self.interpolate:    #runs the interpolation
            return np.transpose(self.interpol())
        else:                   #runs the blosssom recursion to obtain s(u)
            return self.blossom(i,i+1,u)      
    
    #Method to plot results from spline (and interpolation)
    def my_plot(self):
        if(self.interpolate):
            interspline = Spline(self.__call__(0),self.grid,False)
            interspline.my_plot()
            
        else:
            mpl.plot(self.controlpoints[:,0], self.controlpoints[:,1], '-.*')  #Plottar controllpunkerna + polygon
            u = np.linspace(self.grid[0], self.grid[-1], num = 100*len(self.grid))
            su = len(u)*[0]
            for k in range(len(su)):
                su[k] = self.__call__(u[k])
            
            #Because s(u) is in a inconvinient format
            sux = len(su)*[0]            
            suy = len(su)*[0]
            for c in range(len(sux)):
                sux[c] = su[c][0]
                suy[c] = su[c][1]
            mpl.plot(sux[:], suy[:], '-')
        
    
    #Recursive method to calculate d(u,u,u)
    def blossom(self, right,left,u):
        if right - left == 2:
            return self.controlpoints[left,:]
        if self.grid[right+1]-self.grid[left-1] == 0:
            alpha = 0
            return (alpha*self.blossom(right,left-1,u)+
                    (1-alpha)*self.blossom(right+1,left,u))
        else:
            alpha = ((self.grid[right+1]-u)/
                     (self.grid[right+1]-self.grid[left-1]))
            return (alpha*self.blossom(right,left-1,u)+
                    (1-alpha)*self.blossom(right+1,left,u))
                   
    #function for the base functions N, math from slides
    #k= order, i = index in grid, u = xi, determined i interpol
    # 0/0 = 0 
    def baseFunc(self, k, i, u):
        if(k==0):  
            if ((self.grid[i-1]==self.grid[i])):
                return 0
            elif (self.grid[i-1]<=u)*(u<self.grid[i]):
                return 1
            else:
                return 0
        else:
            n1=0
            n2=0
            if i > 0:
                if(self.grid[i+k-1] != self.grid[i-1]):
                    n1 = ((u-self.grid[i-1])/(self.grid[i+k-1]-self.grid[i-1]))*self.baseFunc(k-1,i,u)
            if(i+k < len(self.grid)):
                if(self.grid[i+k]!=self.grid[i]):
                    n2 = ((self.grid[i+k]-u)/(self.grid[i+k]-self.grid[i]))*self.baseFunc(k-1,i+1,u)                    
            N = n1 + n2 
        return N
    
    #calculates the non-zero values of the vandermonde matrix.
    #the new controlpoints, d, is solved for the intepolation problem Nd=x
    def interpol(self): 
        xi = ([(self.grid[i]+self.grid[i+1]+self.grid[i+2])/3 for i in range(len(self.grid)-2)])
        xi[-1] = xi[-1] - 10**-16       #to avoid problems with the open intervall
        
        sup1 = [0] + ([self.baseFunc(3,i+1,xi[i]) for i in range(len(xi)-1)])
        diagonal = ([self.baseFunc(3,i,xi[i]) for i in range(len(xi))])
        sub1 = ([self.baseFunc(3,i,xi[i+1]) for i in range(len(xi)-1)]) + [0]
        sub2 = ([self.baseFunc(3,i,xi[i+2]) for i in range(len(xi)-2)]) + [0,0]
       
        Nxi = sc.array([sup1,diagonal,sub1, sub2])
        dx = sl.solve_banded((2,1), Nxi, self.controlpoints[:,0])
        dy = sl.solve_banded((2,1), Nxi, self.controlpoints[:,1])
        return sc.array([dx,dy])
        
    #Returns the basefunction
    def N(self):
        f = lambda index,u: self.baseFunc(3,index,u)
        return f
        
#plots some basfunctions N
def testN():
    test = sc.array([[4.5,4], [5,2], [3,0], [1,7], [4,8], [4.5,5], [3.2,4]])
    grid = np.linspace(0, 1, num=len(test)-2) #vi lägger till 4 punkter och L=K-2
    cspline = Spline(test,grid,True)
    func = cspline.N()  
    u = np.linspace(grid[0], grid[-1], num = 100*len(grid))
    Nj = (len(u))*[0]
    for j in range(5):
        for i in range(len(u)):
            Nj[i] = func(j+1,u[i])
        
        mpl.plot(u,Nj)
    mpl.show()


def main(c, bol):
    if c==1:
        #test = sc.array([[4.5,4], [5,2], [3,0], [1,7], [4,8], [4.5,5], [3.2,4]])
        test = sc.array([[1,3], [5,4], [2,7], [4,10], [8,8], [7,2]])
        #test = sc.array([[9,3], [5,4], [3,1], [5,2], [6,0], [7,2]])
        #test = sc.array([[5,9], [2,2], [2.5,6], [4,0], [6,8], [3,7]])
        #test = sc.array(np.mat('1.,0.;1.,2.;2.5,2.;0.5,4.5;1.5,7.5;0.5,2.75;-.5,7.50;0.5,4.5;-1.5,2.0;0.,2.;0.,.0'))
        if bol:
            mpl.plot(test[:,0], test[:,1], '*')
        
        grid = np.linspace(0, 1, num=len(test)-2) 
        cspline = Spline(test,grid,bol)
        cspline.my_plot()
    else:
        testN()

main(1, False)     # 1 for spline/interpol, orther for basisfunc
    
