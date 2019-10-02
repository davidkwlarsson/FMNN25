#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Fri Sep 28 09:51:02 2018
    
    @author: samuel
    """
import scipy as sc
import pylab as pl
import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as mpl

class General:
    
    def __init__(self, function, initial, linesearch, gradient):
        self.f=function
        self.pos=sc.array(initial)
        self.inexact = linesearch
        self.exactg = gradient is not None
        if(self.exactg):
            self.gfunc = gradient
        self.gp = self.getGradient(self.pos)
        self.Hp = self.getHessian(self.pos)
    
    def __call__(self):
        for i in range(5000):
            if(self.gp@self.gp < 1e-20):
                if self.checkPD():
                    return self.pos, self.f(self.pos)
            self.step()
        return 'oh noo, we failed'
    
    def step(self):
        #calculate direction
        dirr = -sl.solve(self.Hp,self.gp)
        
        #calculate alpha
        alpha = self.lineSearch(dirr)
            #alpha = 0.3
        self.pos= self.pos + alpha*dirr
        self.gp = self.getGradient(self.pos)
        self.Hp = self.getHessian(self.pos)
    #print('här är jag', self.pos)
#print('hessian:', self.Hp)
#print('gradient:', self.gp)


    def getGradient(self, newpos):
        if(self.exactg):
            g = sc.array([i(self.pos)for i in self.gfunc])
            return g
        h=1e-04
        g = sc.array([.0] * len(newpos))
        temp = [.0]*len(newpos)
        for i in range(len(newpos)):
            temp[i] = h
            g[i] = self.f(newpos+temp)/h - self.f(newpos)/h
            temp[i]=.0
        
        return g
    
    
    def getHessian(self, newpos):
        return [0]


    def checkPD(self):
        Ghat = 1/2*(self.Hp + self.Hp.transpose())
        L = sl.cholesky(Ghat, lower=True)
        if (L*L.transpose()-Ghat+ 1e-16).all:
            return True
        else:
            raise Exception('The Hessian is not positive definite.')


    def lineSearch(self,dirr):
        maxIter = 10000;
        it = 0;
            
        if not self.inexact:
            tol = 1e-15
            al = 1
            grad = self.gp
            gamma = 0.1
            fun = lambda alpha: self.f(self.pos + alpha * dirr)
            
            grad =  getGradForAnyFunc(fun,self.pos)
            # alpha = argmin f(x+alpha*d)
            while(grad < tol):
                if it == maxIter:
                    raise Exception('Linesearch failed.')
                it = it + 1
                grad = self.getGradient(x(al))
        
        if self.inexact:
            rho = 0.1 #should be in [0,0.5)
            alphaL = 0 #initial lower bound
            alphaU = 1e99 #initial upper bound
            sigma = 0.7
            tau = 0.1
            xsi = 9
            xL = self.pos + alphaL*dirr
            fL = self.f(xL)
            fpL = self.getGradient(xL)@dirr
            alpha0 = np.linalg.norm(self.gp)**2/(self.gp@self.Hp@self.gp)
            x0 = self.pos + alpha0*dirr
            f0 = self.f(x0)
            
            # Goldstein conditions
            
            LC = f0 >= fL + (1-rho)*(alpha0-alphaL)*fpL
            RC = f0 <= fL + rho*(alpha0-alphaL)*fpL
            
            
            while not (LC*RC):
                if it == maxIter:
                    raise Exception('Linesearch failed.')
                it = it + 1
                if not LC:
                    fp0 = self.getGradient(x0)@dirr
                    deltaAplha1 = ((alpha0-alphaL)*fp0)/(fpL-fp0)
                    deltaAlpha2 = max(deltaAplha1,tau*(alpha0-alphaL))
                    deltaAlpha = min(deltaAlpha2,xsi*(alpha0-alphaL))
                    
                    alphaL = alpha0
                    alpha0 = alpha0 + deltaAlpha
                if LC:
                    if not RC:
                        alphaU = min(alpha0,alphaU)
                        alpha0 = alphaL + ((alpha0-alphaL)**2 * fpL)/(2*(fL-f0+(alpha0-alphaL)*fpL))
                        alpha0 = max(alpha0,alphaL+tau*(alphaU-alphaL))
                        alpha0 = min(alpha0,alphaU-tau*(alphaU-alphaL))
        
                xL = self.pos + alphaL*dirr
                fL = self.f(xL)
                fpL = self.getGradient(xL)@dirr
                x0 = self.pos + alpha0*dirr
                f0 = self.f(x0)
                
                LC = f0 >= fL + (1-rho)*(alpha0-alphaL)*fpL
                RC = f0 <= fL + rho*(alpha0-alphaL)*fpL

        return alpha0

    def my_plot(self):
        my_res = self.__call__()
        x_star = my_res[0]
        f_star = my_res[1]
        print('x* is', x_star, 'and f(x*) is', f_star)
    
        if len(x_star) == 1:        #multidim-plot hur?
            mpl.plot(x_star, f_star, '*')
            lx = sc.linspace(x_star-5,x_star+5,1000)
        f_disc = [0]*len(lx)
        for ii, ll in enumerate(lx):
            f_disc[ii]=self.f([ll])
        mpl.plot(lx, f_disc, '-')
        mpl.show()


    class Newton(General):
    
        def __init__(self, function, initial, linesearch, *gradient):
            General.__init__(self, function, initial, linesearch, *gradient)
    
        def getHessian(self, newpos):
            H0 = [.0]*len(newpos)
            H = sc.array([H0]*len(newpos))
            h = 1e-03
            ti = [.0]*len(newpos)
            tj = [.0]*len(newpos)
            for i in range(len(newpos)):
                ti[i] = h
                H[i,i] = (self.f(newpos+ti)+self.f(newpos-ti)-2*self.f(newpos))/(h**2)
                for j in range(i):
                    tj[j] = h
                    H[i,j] = (self.f(newpos+ti+tj)+self.f(newpos-ti-tj)-self.f(newpos+ti-tj)-self.f(newpos-ti+tj))/(2*h**2)
                    H[j,i] = H[i,j]
                    tj[j] = .0
                ti[i] = .0
            return H




            
    class QNMethod(General):
    
        def __init__(self, function, initial, linesearch, *gradient):
            General.__init__(self, function, initial, linesearch, *gradient)
            
            
            
        def __call__(self):
        
            self.Hinv = np.eye(len(self.pos))
        
            for i in range(500):
                if(self.gp@self.gp < 1e-10):
                    if checkPD(self):
                        return self.pos, f(self.pos)
                self.step()
            return 'oh noo, we failed'
            
        def step(self):
            #calculate direction
            x = self.pos
            gp = self.gp
            dirr = -self.Hinv@gp
            #calculate alpha
            alpha = self.lineSearch(dirr)
            x_new = x + alpha@(dirr)
            gp_new = self.getGradient(self, x)
            delta = x_new - x
            gamma = gp_new - gp
            self.Hinv = getHessianInv(self, Hinv, delta, gamma)
            self.gp = gp_new
            self.pos = x_new
            
        
        class Broyden(QNMethod):
            
            def __init__(self, function, initial, linesearch, *gradient):
                QNMethod.__init__(self, function, initial, linesearch, *gradient)
                
            def getHessianInv(infunc, Hinv,delta,gamma):  
                t = np.dot(delta-np.dot(Hinv,gamma),np.dot(delta.T,Hinv))
                n = np.dot(delta.T,np.dot(Hinv,gamma))
                return Hinv + t/n
                 
        class BadBroyden(QNMethod):

            def __init__(self, function, initial, linesearch, *gradient):
                QNMethod.__init__(self, function, initial, linesearch, *gradient)
                
            def getHessianInv(infunc, Hinv, delta, gamma):
                t = np.dot(delta-np.dot(Hinv,gamma),gamma.T)
                n = np.dot(gamma.T,gamma)
                return Hinv + t/n
                
        class DFP(QNMethod):
        
            def __init__(self, function, initial, linesearch, *gradient):
                QNMethod.__init__(self, function, initial, linesearch, *gradient)
                
            def getHessianInv(infunc, Hinv, delta, gamma): 
                temp1 = np.dot(delta,delta.T)/np.dot(delta.T,gamma)
                temp2 = np.dot(np.dot(Hinv,np.dot(gamma,gamma.T)),Hinv)/np.dot(np.dot(gamma.T,Hinv),gamma)
                
                return Hinv + temp1 + temp2
                
        class BFGS(QNMethod):
            
            def __init__(self, function, initial, linesearch, *gradient):
                    QNMethod.__init__(self, function, initial, linesearch, *gradient)
                    
            def getHessianInv(infunc, Hinv, delta, gamma):
                temp1 = np.eye(len(self.pos)) - np.dot(delta,gamma.T)/np.dot(gamma.T,delta)
                temp2 = np.eye(len(self.pos)) - np.dot(gamma,delta.T)/np.dot(gamma.T,delta)
                temp3 = np.dot(delta,delta.T)/np.dot(gamma.T,delta)
                return np.dot(np.dot(temp1,Hinv),temp2) + temp3
                
    
        
        
        


def main():
    infunc = lambda x: x[0]**2 + x[1]**2
    g1 = lambda x: 2*x[0]
    g2 = lambda x: 2*x[1]
    grad= [g1, g2]
    # if  lineSearch == True -> inexact method
    qn = Newton(infunc, [1,1], True, grad)
    #punkt = qn.__call__()
    #qn = Broyden(infunc, 1, lineSearch == False)
    #qn = BadBroyden(infunc, 1, lineSearch == False)
    #qn = DFP(infunc, 1, lineSearch == False)
    #qn = BFGS(infunc, 1, lineSearch == False)
    qn.my_plot()

main()



