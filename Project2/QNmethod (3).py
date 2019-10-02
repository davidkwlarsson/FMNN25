#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 09:51:02 2018

@author: David
"""
import scipy as sc
import pylab as pl
import numpy as np
import scipy.linalg as sl
import scipy.optimize as scop
import matplotlib.pyplot as mpl
import math
#from Chebyquad import gradchebyquad, chebyquad_fcn

class General:
    
    def __init__(self, function, initial, linesearch, gradient = None):
        self.f=function
        self.pos=sc.array(initial)
        self.exact = linesearch
        self.exactg = gradient is not None
        if(self.exactg):
            self.gfunc = gradient
        self.gp = self.getGradient(self.pos)
        self.Hp = self.getHessian(self.pos)
        self.allPos = [self.pos]
            
    def __call__(self):
        s=100#allokerar
        for i in range(1000):
            if(sl.norm(s) < 1e-05):
                if self.checkPD():
                    return self.pos, self.f(self.pos), 'succes'
                else:
                    raise Exception('The Hessian is not positive definite.')
                    break
            s = self.step()
            if type(s) is str:
                
                return self.pos, self.f(self.pos), 'curve to step, gradient or hessian is nan'
        return self.pos, self.f(self.pos), 'oh noo, we failed'
    
    def step(self):
            #calculate direction
            try:
                dirr = -sl.solve(self.Hp,self.gp)
            except sl.LinAlgError:
                fake_Hp = self.Hp + 1e-16*sc.eye(len(self.Hp))
                dirr = -sl.solve(fake_Hp, self.gp)
            #calculate alpha
            alpha = self.lineSearch(dirr)
            #alpha = 0.3
           
            self.pos= self.pos + alpha*dirr
            self.allPos = self.allPos + [self.pos]
            self.gp = self.getGradient(self.pos)
            self.Hp = self.getHessian(self.pos)
            #fångar lite problem
            #if math.isnan(self.Hp) or math.isnan(self.gp):
            #     return 'to_steep'
            
            if math.isnan(sum(sum(self.Hp))) or math.isnan(sum(self.gp)):
                return 'to_steep'
            if sl.norm(self.Hp) > 1e+10 or sl.norm(self.gp) > 1e+10:
                return 'to_steep'
           
            return dirr
    
    
    def getGradient(self, newpos):
        if(self.exactg):
            g = sc.array(self.gfunc(newpos))
            return g
        h=1e-6
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
            return False
        
        
    def lineSearch(self,dirr):
        maxIter = 1000;
        it = 0;
        
        if self.exact:
            def alpha_f(alpha):
                return self.f(self.pos+alpha*dirr)
            
            return scop.minimize_scalar(alpha_f).x
       
        if not self.exact:
            rho = 0.1 #should be in [0,0.5)
            alphaL = .0 #initial lower bound
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
                        print(f0)
                        alpha01 = alphaL + ((alpha0-alphaL)**2 * fpL)/(2*(fL-f0+(alpha0-alphaL)*fpL))
                        alpha02 = max(alpha01,alphaL+tau*(alphaU-alphaL))
                        
                        alpha0 = min(alpha02,alphaU-tau*(alphaU-alphaL))
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
        mess = my_res[2]
        print(mess, 'x* is', x_star, 'and f(x*) is', f_star)
        
        
        
        if len(x_star) == 1:        #multidim-plot hur?
            for x in self.allPos:
                y = self.f(x)
                mpl.plot(x, y, '.k')
            mpl.plot(x, y, '*r',markersize = 10)
            
            l = self.allPos[0]
            u = x_star
            r = abs(l-u)
            if l>r:
                lt = r
                r = l
                l = lt
            
            lx = sc.linspace(l+r*0.1,u-r*0.1,1000)
            f_disc = [0]*len(lx)
            for ii, ll in enumerate(lx):
                f_disc[ii]=self.f([ll])
            mpl.plot(lx, f_disc, '-')
            mpl.show()
            
        if len(x_star) == 2:
            xl  = self.allPos[0][0]
            xu = x_star[0]
            yl = self.allPos[0][1]
            yu = x_star[1]
            
            xr = abs(xl-xu)
            yr = abs(yl-yu)
            
            if xl>xu:
                lt = xl
                xr = xl
                xl = lt
                
            if yl>yu:
                lt = yl
                yr = yl
                yl = lt
            
            lx = sc.linspace(xl+xr*0.1, xu-xr*0.1, 1000)
            ly = sc.linspace(yl+yr*0.1, yu-yr*0.1, 1000)
            
    
            F = np.empty(shape=(len(lx),len(ly)))
            
            #detta tar tid men funkar borde kolla meshgrid också...
            for i in range(len(lx)):
                for j in range(len(ly)):
                    x = np.array([lx[i],ly[j]])
                    F[i,j] = self.f(x)
            
            
            
            fig, ax = mpl.subplots()
            CS = ax.contourf(lx, ly, F,30,cmap=mpl.cm.binary)
            fig.colorbar(CS)
            
            for i in range(len(self.allPos)):
                x = self.allPos[i]
                xx = x[0]
                xy = x[1]
                mpl.plot(xx, xy, 'k.',markersize=3)
                #if i > 1:
                #    x = np.array(x)
                #    xp = np.array(self.allPos[i-1])
                #    mpl.plot(xp,x,'k--',linewidth=5)
            
            mpl.plot(xx,xy,'r*',markersize=10)
            
            
            mpl.show
        
    
class Newton(General):    
    
    def __init__(self, function, initial, linesearch, *gradient):
        General.__init__(self, function, initial, linesearch, *gradient)
        
    def getHessian(self, newpos):
        H0 = [.0]*len(newpos)
        H = sc.array([H0]*len(newpos))       
        h = 1e-04
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
        
            
        
    
class QNMethod(Newton):

    def __init__(self, function, initial, linesearch, *gradient):
        General.__init__(self, function, initial, linesearch, *gradient)
        
        
        self.Hinv = np.eye(len(self.pos))
        
        

        
    def step(self):
        #calculate direction
        x = self.pos
        gp = self.gp
        dirr = -self.Hinv@gp
        #calculate alpha
        alpha = self.lineSearch(dirr)
        x_new = x + alpha*dirr
        gp_new = self.getGradient(x_new)
        self.pos = x_new
        delta = x_new - x
        gamma = gp_new - gp
        self.Hinv = self.getHessianInv(self.Hinv, delta, gamma)
        self.gp = gp_new
        self.allPos = self.allPos + [self.pos]
        
        if math.isnan(sum(sum(self.Hinv))) or math.isnan(sum(self.gp)):
            return 'to_steep'
        if sl.norm(self.Hinv) > 1e+10 or sl.norm(self.gp) > 1e+10:
            return 'to_steep'
        
        return -self.Hinv@self.gp
        
    
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
        temp1 = np.eye(len(Hinv)) - np.dot(delta,gamma.T)/np.dot(gamma.T,delta)
        temp2 = np.eye(len(Hinv)) - np.dot(gamma,delta.T)/np.dot(gamma.T,delta)
        temp3 = np.dot(delta,delta.T)/np.dot(gamma.T,delta)
        return np.dot(np.dot(temp1,Hinv),temp2) + temp3

      
def main():
    rosen = lambda x: 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
    rgrad = lambda x: [400*(x[0]**3)-400*x[0]*x[1]-2+2*x[0], 200*x[1]-200*(x[0]**2)]
    
    # if  lineSearch == True -> exact method
    #qn = Newton(infunc, g1,[0.8,1.1], True, None)
    
    infunc = lambda x: x[0]**4 + (x[0]-1)**3+x[1]**4 + (x[1]-1)**3
    grad = lambda x: [4*x[0]**3+ 3*(x[0]-5)**2, 4*x[1]**3 +3*(x[1]-5)**2]
    qn = Newton(infunc, [0,0], False)
    qn.my_plot()
    
    #chebyquad_fcn tar in och returnerar en vektor av längd n, vad fan ska vi göra
    #cheb = lambda x: chebyquad_fcn(x)
    #chebgrad = lambda x: gradchebyquad(x)
    #ch = Newton(cheb, 2*[1], True, chebgrad)
    #ch.my_plot()

main()        
        
        

