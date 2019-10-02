import scipy as sc
import pylab as pl
import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as mpl

class General:
    
    def __init__(self, function, initial, linesearch, gradient):
        self.f=function
        self.pos=initial
        self.dir=sc.array(len(self.pos) * [1])
        self.inexact = linesearch
        self.exactg = gradient is not None
        if(self.exactg):
            self.gfunc = gradient
        self.gp = self.getGradient(self.pos)
        self.Hp = self.getHessian(self.pos)
            
    def __call__(self):
        for i in range(5000):
            if(self.gp@self.gp < 1e-10):
                if self.checkPD():
                    return self.pos, self.f(self.pos)
            self.step()
        return 'oh noo, we failed'
    
    
    def getGradient(self, newpos):
        if(self.exactg):
            return self.gfunc(newpos)
        h=1e-14
        g = sc.array([0] * len(newpos))
        for i in range(len(newpos)):
            temppos = newpos.copy()
            temppos[i] = temppos[i]+h
            g[i] = (self.f(temppos)-self.f(newpos))/h
        return g        
                
    
    def getHessian(self, newpos):
        return 0
        
    
    def checkPD(self):
        Ghat = 1/2*(self.Hp + self.Hp.transpose())
        L = sl.cholesky(Ghat, lower=True) 
        if (L*L.transpose() == Ghat):
            return True
        else: 
            #return false
            raise Exception('The Hessian is not positive definite.')
            
    def step(self):
        alpha = self.lineSearch()
        self.pos = self.pos + alpha@(-self.Hp@self.gp)
        self.gp = self.getGradient(self, self.pos)
        self.Hp = self.getHessian(self, self.pos)
        
    def lineSearch(self):
        
        if self.inexact:
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
        
    def my_plot(self):
        mpl.plot(self.__call__(), '*')  #Plottar minpunkten
        mpl.plot(self.f, '-')
    
    
class Newton(General):    
    
    def __init__(self, function, initial, linesearch, *gradient):
        General.__init__(self, function, initial, linesearch, *gradient)
        
    def getHessian(self, newpos):
        H = [0]*len(newpos)
        h=1e-14
        for i in range(len(newpos)):
            H = [0] *len(newpos)
        H = sc.array(H)
        for i in range(len(newpos)):
            temp1 = newpos.copy()
            temp1[i] = temp1[i]+h
            temp2 = newpos.copy()
            temp2[i] = temp2[i]-h
            temp3 = newpos.copy()
            temp3[i] = temp1[i]
            temp4 = newpos.copy()
            temp4[i] = temp2[i]
            H[i,i] = -(self.f(temp1)+self.f(temp2)-2*self.f(newpos))/h**2
            for k in range(i):
                temp1[k] = temp1[k]+h
                temp2[k] = temp1[k]-h
                temp3[k]= temp2[k]
                temp4[k] = temp1[k]
                H[i,k] = (self.f(temp1)+self.f(temp2)-self.f(temp3)-self.f(temp4))/(2*h**2)
                H[k,i] = H[i,k]
                temp1[k] = newpos[k]
                temp2[k] = temp1[k]
                temp3[k] = temp1[k]
                temp4[k]= temp1[k]
        self.Hp=H    
                
            
        
        
class QNMethod(General):

    def __init__(self, function, initial, linesearch, *gradient):
        General.__init__(self, function, initial, linesearch, *gradient)
    
class Broyden(QNMethod):
    
    def __init__(self, function, initial, linesearch, *gradient):
        QNMethod.__init__(self, function, initial, linesearch, *gradient)
        
    def getHessian(infunc, x): 
        return 0
         
class BadBroyden(QNMethod):

    def __init__(self, function, initial, linesearch, *gradient):
        QNMethod.__init__(self, function, initial, linesearch, *gradient)
        
    def getHessian(infunc, x): 
        return 0
        
class DFP(QNMethod):

    def __init__(self, function, initial, linesearch, *gradient):
        QNMethod.__init__(self, function, initial, linesearch, *gradient)
        
    def getHessian(infunc, x): 
        return 0
        
class BFGS(QNMethod):
    
    def __init__(self, function, initial, linesearch, *gradient):
            QNMethod.__init__(self, function, initial, linesearch, *gradient)
            
    def getHessian(infunc, x):
        return 0
        
            
def main():
    #infunc = lambda x: x**2
    #grad = lambda x: 2*x
    def infunc(x): 
        return x**2
    def grad(x):
        return 2*x
    
    # if  lineSearch == True -> inexact method
    qn = Newton(infunc, [1,1], True, grad)
    #qn = Broyden(infunc, 1, lineSearch == False)
    #qn = BadBroyden(infunc, 1, lineSearch == False)
    #qn = DFP(infunc, 1, lineSearch == False)
    #qn = BFGS(infunc, 1, lineSearch == False)
    qn.my_plot()

main()        
        
        
