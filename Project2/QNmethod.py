from  scipy import *
from  pylab import *
import numpy as np
import scipy.linalg as sl

class General:
    
    def __init__(self, function, initial, linesearch, *gradient):
        self.f=function
        self.pos=initial
        self.dir=array(len(pos) * [1])
        self.inexact = linesearch
        self.exactg = gradient is not none
        if(exactg):
            self.gfunc = gradient
        self.gp = getgradient(self,self.pos)
        self.Hp = getHessian(self, self.pos)
            
    def __call__(self):
        for i in range(5000):
            if(self.gp@self.gp < 1e-10):
                if checkPD(self):
                    return self.pos, f(self.pos)
            self.step()
        return 'oh noo, we failed'
    
    
    def getGradient(self,newpos):
        if(exactg):
            return self.gfunc(newpos)
        h=1e-14
        g = array([0] * len(newpos))
        for i in range(len(newpos)):
            temppos = newpos.copy()
            temppos(i) = temppos(i)+h
            g(i) = (self.f(temppos)-self.f(newpos))/h
        return g        
                
    
    def getHessian(self, newpos):
        return 0
        
    
    def checkPD(self):
        Ghat = 1/2*(self.Hp + self.Hp.transpose())
        L = sl.cholesky(Ghat, lower=True) 
        if (L*L.transpose() == Ghat):
            return true
        else: 
            #return false
            raise Exception('The Hessian is not positive definite.')
            
    def step(self):
        alpha = self.lineSearch()
        self.pos=self.pos + alpha@(-self.Hp@self.gp) #Check
        self.gp = getGradient(self, self.pos)
        self.Hp = getHessian(self, self.pos)
        
    def lineSearch(self):
        
        if inexact:
            rho = 0.1 #should be in [0,0.5)
            alphaL = 0 #initial lower bound
            alphaU = 1e99 #initial upper bound 
            sigma = 0.7
            tau = 0.1
            xsi = 9
            xL = self.pos + alphaL*self.d
            fL = self.f(xL)
            fpL = np.transpose(self.getGradient(xL))*self.d
            alpha0 = np.linalg.norm(self.gp)**2/(np.transpose(self.gp)*self.Hp*self.gp)
            x0 = self.pos + alpha0*self.d
            f0 = self.f(x0)
            
            # Goldstein conditions
            
            LC = f0 >= fL + (1-rho)*(alpha0-alphaL)*fpL
            RC = f0 <= fL + rho*(alpha0-alphaL)*fpL
            
            
            while not LC*RC:
                if not LC:
                    fp0 = np.transpose(self.getGradient(x0))*self.d
                    deltaAplha = ((alpha0-alphaL)*fp0)/(fpL-fp0)
                    deltaAlpha = max(deltaAlpha,tau*(alpha0-alphaL))
                    deltaAlpha = min(deltaAlpha,xsi*(alpha0-alphaL))
                    
                    alphaL = alpha0
                    alpha0 = alpha0 + deltaAlpha
                if LC:
                    if not RC:
                        alphaU = min(alpha0,alphaU)
                        alpha0 = alphaL + ((alpha0-alphaL)**2 * fpL)/(2*(fL-f0+(alpha0-alphaL)*fpL))
                        alpha0 = max(alpha0,alphaL+tau*(alphaU-alphaL))
                        alpha0 = min(alpha0,alphaU-tau*(alphaU-alphaL))
                
                xL = self.pos + alphaL*self.d
                fL = self.f(xL)    
                fpL = np.transpose(self.getGradient(xL))*self.d
                x0 = self.pos + alpha0*self.d
                f0 = self.f(x0)
            
                LC = f0 >= fL + (1-rho)*(alpha0-alphaL)*fpL
                RC = f0 <= fL + rho*(alpha0-alphaL)*fpL
            
        return alpha0     
        
    def plot(self):
        plot
    
    class Newton(General):    
        
        def __init__(self, function, initial, linesearch, *gradient):
            General.__init__(self, function, initial, linesearch, *gradient)
            
        def getHessian(self, newpos):
            H = [0]*len(newpos)
            h=1e-14
            for i in range(len(newpos)):
                H = [0] *len(newpos)
            H = array(H)
            for i in range(len(newpos)):
                temp1 = newpos.copy()
                temp1(i) = temp1(i)+h
                temp2 = newpos.copy()
                temp2(i) = temp2(i)-h
                temp3 = newpos.copy()
                temp3(i) = temp1(i)
                temp4 = newpos.copy()
                temp4(i) = temp2(i)
                H(i,i) = -(f(temp1)+f(temp2)-2*f(newpos))/h**2
                for k in range(i):
                    temp1(k) = temp1(k)+h
                    temp2(k) = temp1(k)-h
                    temp3(k) = temp2(k)
                    temp4(k) = temp1(k)
                    H(i,k) = (f(temp1)+f(temp2)-f(temp3)-f(temp4))/(2*h**2)
                    H(k,i) = H(i,k)
                    temp1(k) = newpos(k)
                    temp2(k) = temp1(k)
                    temp3(k) = temp1(k)
                    temp4(k) = temp1(k)
            self.Hp=H    
                    
                
            
            
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
                
    
        
        
        
