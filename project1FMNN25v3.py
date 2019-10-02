# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 20:37:28 2018

@author: David_000
"""

# -*- coding: utf-8 -*-
"""
    Spyder Editor
    
    This is a temporary script file.
    """

import scipy as sc
import scipy.linalg as sl
import numpy as np
import matplotlib.pyplot as mpl

class Spline:
    def __init__(self, controlpoints, subgrid, interpolate):
        #Fixa conditional init
        self.controlpoints = controlpoints
        self.interpolate = interpolate
        if(((subgrid[0]==subgrid[1])*(subgrid[1]==subgrid[2]))*
           ((subgrid[-1]==subgrid[-2])*(subgrid[-2]==subgrid[-3]))):
            self.grid = subgrid
           
        else:
            grid = sc.r_[subgrid[0], subgrid[0], subgrid, subgrid[-1], subgrid[-1]]
            self.grid = grid
            
        print(self.grid.shape)    
        print(self.controlpoints.shape) 
        if(len(self.grid)-2 != len(controlpoints)):
            print('fel på grid')
    
    def __call__(self, u): #detta ska vara s(u)
        #find the hot interval for u
        self.grid.sort()
        idx=self.grid.searchsorted([u])
        i = idx[0]
        i = i-1
        #select the corresponding four control points d_i-2,...,d_i+1
        if(self.interpolate):
            return np.transpose(self.interpol())
        else:
            return self.blossom(i,i+1,u)      #run the blosssom recursion to obtain s(u)
    
    #Method to plot the control polygon and control points
    
    
    def my_plot(self):
        if(self.interpolate):
            interspline = Spline(self.__call__(1),self.grid,False)
            interspline.my_plot()
            
        else:
            mpl.plot(self.controlpoints[:,0], self.controlpoints[:,1], '-.*')  #Plottar controllpunkerna + polygon
            u = np.linspace(self.grid[0], self.grid[-1], num = 30*len(self.grid))
            su = (len(u)-4)*[0]
            for k in range(len(u)-4):
                su[k] = self.__call__(u[k+2])
        
            sux = (len(u)-4)*[0]            #Detta då su har massa arrays som är jobbiga
            suy = (len(u)-4)*[0]
            
            for c in range(len(sux)):
                sux[c] = su[c][0]
                suy[c] = su[c][1]
            mpl.plot(sux[:], suy[:], '-x')
        
    
    #function for the base functions N
   

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
            if(self.grid[i+k-1] != self.grid[i-1]):
                n1 = ((u-self.grid[i-1])/(self.grid[i+k-1]-self.grid[i-1]))*self.baseFunc(k-1,i,u)
            if(i+k < len(self.grid)):
                if(self.grid[i+k]!=self.grid[i]):
                    n2 = ((self.grid[i+k]-u)/(self.grid[i+k]-self.grid[i]))*self.baseFunc(k-1,i+1,u)
                    
            
            N = n1 + n2
            
        return N
    
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
    
    def interpol(self):
        xi = ([(self.grid[i]+self.grid[i+1]+self.grid[i+2])/3 for i in range(len(self.grid)-2)])
        
        #NEED TO FIND HOW TO USE THE INDEX IN THE FOR-LOOP, NOT THE ELEMENT
        print(self.baseFunc(3,2,xi[0]))
        
        sup2 = ([self.baseFunc(3,i+2,xi[i]) for i in range(len(xi)-2)])
        sup2 = [0,0] + sup2
        sup1 = ([self.baseFunc(3,i+1,xi[i]) for i in range(len(xi)-1)])
        sup1 = [0] + sup1
        diagonal = ([self.baseFunc(3,i,xi[i]) for i in range(len(xi))])
        sub1 = ([self.baseFunc(3,i,xi[i+1]) for i in range(len(xi)-1)])
        sub1 = sub1 + [0]
        
        Nxi = sc.array([sup2,sup1,diagonal,sub1])
        print(Nxi)
    
        print(self.controlpoints.shape)
        dx = sl.solve_banded((1,2),Nxi,self.controlpoints[:,0])
        dy = sl.solve_banded((1,2),Nxi,self.controlpoints[:,1])

        return sc.array([dx,dy])
    
    def test_blossomvsbase(self):
        #utest = array([[4.5,4], [5,2], [3,0], [1,7], [4,8], [4.5,5], [3.2,4]])
        #grid = np.linspace(0, 1, num=len(utest))
        #cspline = self.__init__(utest,grid)
        u = np.linspace(self.grid[0], self.grid[-1], num = 30*len(self.grid))
        blo = self.__call__(u)
        #find hot interval1
        self.grid.sort()
        idx=self.grid.searchsorted([u])
        I = idx[0]
        su = zeros(len(u))
        for x in range(0,len(u)-1):
            i = I[x]
            for y in range(-2,1):
                su[x] = su[x] + self.controlpoints[i+y]*self.baseFunc(3,i+y,u[x])
                
        assertAlmostEqual(blo,su,6)
        
    def N(self):
        
        f = lambda index,u: self.baseFunc(3,index,u)
        return f
        
def testN():
    test = sc.array([[4.5,4], [5,2], [3,0], [1,7], [4,8], [4.5,5], [3.2,4]])
    grid = np.linspace(0, 1, num=len(test)-2) #vi lägger till 4 punkter och L=K-2
    cspline = Spline(test,grid,True)
    func = cspline.N()  
    u = np.linspace(grid[0], grid[-1], num = 100*len(grid))
    Nj = (len(u))*[0]
    for j in range(6):
        for i in range(len(u)):
            Nj[i] = func(j,u[i])
        
        mpl.plot(u,Nj)
    mpl.show()    


def main():
    test = sc.array([[4.5,4], [5,2], [3,0], [1,7], [4,8], [4.5,5], [3.2,4]])
    #       test = array([[1,3], [5,4], [2,7], [4,10], [8,8], [7,2]])
    #       test = array([[9,3], [5,4], [3,1], [5,2], [6,0], [7,2]])
    #        test = sc.array([[5,9], [2,2], [2.5,6], [4,0], [6,8], [3,7]])
    mpl.plot(test[:,0], test[:,1], '-.b')
    grid = np.linspace(0, 1, num=len(test)-2) #vi lägger till 4 punkter och L=K-2
    cspline = Spline(test,grid,True)
    cspline.my_plot()
    

testN()
    
