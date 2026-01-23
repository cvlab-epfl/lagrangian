#!/usr/bin/env python3

#%%%--------------------------------------------------------------------------
#                                IMPORTS
#-----------------------------------------------------------------------------
#%%
import torch
import torch_optimizer as topt
import scipy
import numpy as np

try:
    from   ademamix import AdEMAMix
except:
    def AdEMAMix(**kwargs):
        return None

from auxfuncs import printTensor,makeTensor,fromTensor,raiseError,dumpToFile

def pytorchLagrangeOpt(objF,cstF,x0,nIt=10,lambd=1e-6,rho=100.0,kktP=True,verbP=False):
    
    if(kktP):
        lag = pytorchKktOptim(objF,cstF)
        x1  = lag.optim(x0,nIt=nIt,lambd=lambd,rho=rho,verbP=verbP)
    else:
        lag = pytorchAlmOptim(objF,cstF)
        x1  = lag.optim(x0,nIt=nIt,rho=rho,verbP=verbP)

    return x1

#%%---------------------------------------------------------------------------
#                     Constrained LSQ Optimization
#=============================================================================

class pytorchCnstOptim():
    
    def testF(self,x):
        x = makeTensor(x,gradP=False)
        testBatchJac(self.objF,x)
        
    def testC(self,x):
        x = makeTensor(x,gradP=False)
        testBatchJac(self.cstF,x)
        
#%%---------------------------------------------------------------------------
#                      KKT Newton step for each batch element
#
# Minimize F(X) + lam C(X) + 0.5 * rho C(X)^T C(X) sbj to C(X) = 0
# 
# Recursively solve the block system
#
#   [ J^tJ  + pho A^T A  A^T ] [dX] = -[ J^t F + pho A^T C]
#   [ A                  0   ] [dL] =  [ C                ]
#
# followed by x -= dx with: 
#    X : b x n   (X    in R^n)
#    F : b x r   (F(X) in R^r)
#    C : b x m   (C(X) in R^m)
#    L : b x m   (Lagrange multipliers)
#    J ; b x r x n
#    A : b x m x n
# ----------------------------------------------------------------------------
#%%
class pytorchKktOptim(pytorchCnstOptim):
    
    def __init__(self,objF,cstF):
        
        self.objF = objF
        self.cstF = cstF
      
    def optim(self,x0,nIt=10,lambd=0.0,rho=0.0,verbP=False,dbgP=False):
        
        x      = makeTensor(x0,gradP=False)
        augmL  = torch.inf*torch.ones(x.size(0),device = x.device)
        
        self.print(x,it=0,verbP=verbP)
        
        for itN in range(nIt):
            # Compute objective function, constraints, and Jacobian
            F,J = self.objF(x) 
            C,A = self.cstF(x) 
            # Solve the KKt equestions, return dx and lagrange multiplier
            dx,lm = solveKkt(F,J,C,A,lambd=lambd,rho=np.abs(rho),dbgP=dbgP)
            if(rho>0.0):
                assert(lm is not None)
                # Pick a step size that decreases the augmented lagrangian
                s   = 1.0
                for _ in range(5):
                    xs   = x-s*dx
                    Fs,_ = self.objF(xs) 
                    Cs,_ = self.cstF(xs)
                    # Augmented Lagrangian
                    currL= 0.5*Fs.pow(2).sum(dim=1) - (lm * Cs).sum(dim=1)+ (0.5*rho)*Cs.pow(2).sum(dim=1)
                    # Augmented Lagrangian is lower
                    ids = (currL <= augmL)
                    augmL[ids] = currL[ids]
                    x[ids]     = xs[ids]
                    s /= 2.0
            else:
                # Use a fixed step size
                x = x-dx
                
            self.print(x,it=itN+1,verbP=verbP)
                
            # Increase rho
            rho = rho*1.5
        
        return x
    
    def print(self,x,it=None,verbP=True):
       
        if(verbP):
            F,_ = self.objF(x)
            C,_ = self.cstF(x)
            if(it is not None):
                print('Iteration {:d}'.format(it))
            print('F:',end=' ')
            printTensor(F.pow(2).sum(dim=1))
            print('C:',end=' ')
            printTensor(C.pow(2).sum(dim=1))
            
def solveKkt(F,J,C,A,lambd=0.0,rho=0.0,dbgP=False,sparseP=False):
    
    if(dbgP):
        print('Save matrices onto file matrix.dat')
        dumpToFile('matrix.dat',[fromTensor(F),fromTensor(J),fromTensor(C),fromTensor(A)])
        assert(0)
        
    if(rho==0):
        if(J is None):
            # J is assumed to be the identity matrix
            return solveReducedKkt(F,C,A,lambd=0.0,sparseP=sparseP)
        elif(len(J.size())==2):
            # J is assumed to be a diagonal matrix
            return solveDiagonalKkt(F,J,C,A,lambd=0.0,sparseP=sparseP)
    
    return solveFullKkt(F,J,C,A,lambd=lambd,rho=rho,sparseP=sparseP)
    

# Return state increment and Lagrangian multiplier    
def solveFullKkt(F,J,C,A,lambd=0.0,rho=0.0,sparseP=False):
    
    _,m,n = A.size()
    
    K,Y = makeKktMatrices(F,J,C,A,rho=rho)
    if(lambd>0.0):
        b,m,l=K.size()  
        Id = lambd*torch.eye(l,device=K.device).view((1,l,l)).repeat((b,1,1))
        K += Id
    
    dX = batchSolve(K,Y,sparseP=sparseP).squeeze()
    if(dX is None):
        print('solveFullKtt: batchSolve failed for sparseP {:}'.format(sparseP))
        return None,None
    else:
        return dX[:,0:n],dX[:,n:]

# J is assumed to be a diagonal matrix represented only by its diaganal
# Solve  A (JtJ^{-1}) A^t L  = C - A F 
# Return dx      = (JtJ^{-1}) (F + A^t L)
def solveDiagonalKkt(F,J,C,A,lambd=0.0,sparseP=False):
    
    assert(2==len(J.size())) # Only diagonal is specified as 1D vector for each matrix
    
    J0  = torch.diag_embed(J)
    J1  = torch.diag_embed(1.0  /  J)
    J2  = torch.diag_embed(1.0  /  J.pow(2))
        
    At   = A.transpose(1,2)
    AtA  = torch.bmm(A,torch.bmm (J2,At))
    
    if(lambd>0.0):
        b,n,_= AtA.size()  
        Id   = lambd*torch.eye(n,device=A.device).view((1,n,n)).repeat((b,1,1))
        AtA += Id  
    lmbW = batchSolve(AtA,torch.bmm(A,torch.bmm(J1,F.unsqueeze(2))).squeeze(2)-C,sparseP=sparseP)
    if(lmbW is None):
        print('solveDiagonalKkt(: batchSolve failed for sparseP {:}'.format(sparseP)) 
        return None,None
         
    dx   = torch.bmm(J0,F.unsqueeze(2)) - torch.bmm(At,lmbW.unsqueeze(2))
    dx   = torch.bmm (J2,dx)
    return dx.squeeze(2),lmbW

# J is assumed to be the identity matrix
# Solve  AA^t L  = C - A F 
# Return dx      = F + A^t L 
def solveReducedKkt(F,C,A,lambd=0.0,sparseP=False):
    
    At   = A.transpose(1,2)
    AtA  = torch.bmm(A,At)
    if(lambd>0.0):
        b,n,_= AtA.size()  
        Id   = lambd*torch.eye(n,device=A.device).view((1,n,n)).repeat((b,1,1))
        AtA += Id  
    b    = F.unsqueeze(2)
    lmbW = batchSolve(AtA,C-torch.bmm(A,b).squeeze(2),sparseP=sparseP)
    if(lmbW is None):
        print('solveReducedKkt: batchSolve failed for sparseP {:}'.format(sparseP)) 
        return None,None
         
    dx   = b + torch.bmm(At,lmbW.unsqueeze(2))
    return dx.squeeze(2),None
    
def makeKktMatrices(F,J,C,A,rho=0.0):
    
    b,m,n,r =  getMatrixDimensions(F,J,C,A)
    device  = F.device
    
    At   = A.transpose(1,2)
    if(J is None):
        # No J specified. Make the upper left corner of the matrix identity
        assert (n==r)
        JtJ  = torch.eye(n,device=C.device).view((1,n,n)).repeat((b,1,1))
        JtF  = F
    else:
        JtJ  = torch.bmm(J.transpose(1,2),J)
        JtF  = torch.bmm(J.transpose(1,2),F.view((b,r,1))).squeeze()
    
    if(rho>0.0):
        JtJ += rho*torch.bmm(At,A)
        JtF += rho*torch.bmm(At,C.view((b,m,1))).squeeze()
           
    K=torch.zeros([b,n+m,n+m],device=device)
    Y=torch.zeros([b,n+m,1],device=device)
    
    K[:,0:n,0:n]   = JtJ 
    K[:,0:n,n:n+m] = At
    K[:,n:n+m,0:n] = A 

    Y[:,0:n,0]     = JtF
    Y[:,n:n+m,0]   = C

    return K,Y

# b: batch size, n: dimension of input, r: dimension of output, m: number of constraints
# F output of objF, b x r
# C output of cstF, b x m
# J jacobian of objF, b x r x n
# A jacobian of cstF, b x m x n
def getMatrixDimensions(F,J,C,A):
    
    b,m,n  = A.size()
    _,r    = F.size()
   
    # Check matrix dimensions
    if(not ((F.size(0)==b) and (C.size(0)==b) and (C.size(1)==m))):
        print('F',F.size())
        print('C',C.size())
        print('A',A.size())
        raiseError('makeKktMatrices: F,C, and A have incomptiable dimensions')
    if(J is not None):
        if(not((J.size(0)==b) and (J.size(1)==r) and (J.size(2)==n))):
            raiseError('makeKktMatrices: F,C, and J have incomptiable dimensions')
            
    return b,m,n,r
    
def batchSolve(A,y,sparseP=False):
    
    if(sparseP):
        return batchSparseSolve(A.to_sparse(),y.squeeze())
    else:
        try:
            return torch.linalg.solve(A,y)
        except:
            return None    

def batchSparseSolve(A,y):
    
    raiseError('batchSparseSolve is not imlemented')
#%%---------------------------------------------------------------------------
#                          Augmented Lagrangian
#
# Minimize F(X) + lam C(X) + L C * 0.5 * rho C(X)^T C(X) 
#
# where L are the Lagrange multipliers
#%%
class pytorchAlmOptim(pytorchCnstOptim): 
    
    def __init__(self,objF,cstF,dspF=None):
    
        self.objF = objF
        self.cstF = cstF
        self.dspF = dspF
        
    def optim(self,x0,nIt=10,nRep=10,lr=0.1,rho=0.0,verbP=False):
               
        x     = makeTensor(x0,gradP=True)
        x1    = torch.zeros_like(x).requires_grad_(False)
        b,n   = x.size()
        
        optimizer = torch.optim.Adam([x], lr=lr)
        lam       = None    # Lagrange multipliers, will be initialized when the number of contraitns is known
        
        self.print(x,it=0,verbP=verbP)
        if(self.dspF is not None):
            self.dspF(x)
       
        for itN in range(nIt):
        
            bestL     = torch.inf*torch.ones(b,device=x.device)
            # Inner loop
            for _ in range(nRep): 
                
                optimizer.zero_grad()
                # constraint
                obj,_  = self.objF(x)
                cst,_  = self.cstF(x)
                # augmented Lagrangian
                if(2==len(obj.size())):
                    currL = obj.pow(2).sum(dim=1) + (rho/2) * cst.pow(2).sum(dim=1)
                elif(1==len(obj.size())):
                    currL = obj + (rho/2) * cst.pow(2).sum(dim=1)
                else:
                    raiseError('Dimension of obj is not handled')
                if (lam is not None):
                    currL += (lam*cst).sum(dim=1)
                
                ids        = currL<bestL
                x1[ids]    = x[ids].data
                bestL[ids] = currL[ids]
               
                computeAllGrads(currL,sumP=True)
                optimizer.step()
         
            x.data[:]=x1.data

            # Update lagrange multipliers
            cst,_  = self.cstF(x)
            if(lam is None):
                lam = rho * cst.detach().clone().requires_grad_(False)
            else:
                lam = lam + rho * cst.data
                
            self.print(x,it=itN,verbP=verbP)
            if(self.dspF is not None):
                self.dspF(x)
            
        return x
                
    def print(self,x,it=None,verbP=True):
        
        if(verbP):
            if(it is not None):
                print('It',it+1)
            obj,_  = self.objF(x)
            cst,_  = self.cstF(x)
            if(2==len(obj.size())):
                obj    = obj.pow(2).sum(dim=1) 
            cst    = cst.pow(2).sum(dim=1)
            
            printTensor(obj.t(),prefix='F:')
            printTensor(cst.t(),prefix='C:')
#%%---------------------------------------------------------------------------
#                    Unconstrained Adam Optimization
#=============================================================================
#%%
class pytorchAdamOptim(): 
    
    def __init__(self,lossF,dspF=None):
    
        self.objF = lossF
        self.dspF = dspF
        
    def optim(self,x0,nIt=10,lr=0.1,rho=0.0,schedP=False,verbP=False):
       
        x     = makeTensor(x0,gradP=True)
        x1    = makeTensor(x0,gradP=False)
        b,n   = x.size()
        
        optim = torch.optim.Adam([x], lr=lr)
        if(schedP):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim,nIt)
        
        self.print(x,it=0,verbP=verbP)        
        
        bestL     = torch.inf*torch.ones(b,device=x.device)
        # Inner loop
        for itN in range(nIt):
            
            optim.zero_grad()
           
            currL  = self.objF(x)
            
            ids        = currL<bestL
            x1[ids]    = x[ids].data
            bestL[ids] = currL[ids]
            
            computeAllGrads(currL,sumP=True)
            optim.step()
            
            if(schedP):
                scheduler.step()
         
            self.print(x,it=itN,verbP=verbP)
            if(self.dspF is not None):
                self.dspF(x)
            
        return x1
                
    def print(self,x,it=None,verbP=True):
        
        if(verbP):
            if(it is not None):
                print('It',it+1)
            obj  = self.objF(x)
            printTensor(obj.t(),prefix='F:')
#%%---------------------------------------------------------------------------
#                      Use Scipy instead of pytorch
#-----------------------------------------------------------------------------
#%% lossF is expected to take at least two arguments, the sample index i and the sample itself x          
class pytorchScipyOptim(): 
    
     
    def __init__(self,lossF,cstF=None,cstA=None):
    
        self.lossF = lossF
        self.cnstF = cstF
    
    def optim(self,x0,verbP=False,**kwargs):
            
        def cstF(x):
            assert(self.cnstF is not None)
            x = makeTensor(x,gradP=False,dblP=True)
            C,_ = self.cnstF(x)
            return fromTensor(C)
        
        def cstA(x): 
            assert(self.cnstF is not None)
            x = makeTensor(x,gradP=False,dblP=True)
            _,A = self.cnstF(x)
            return fromTensor(A)
        
        xs = fromTensor(x0)
        ys = np.zeros_like(xs)
        for i,x in enumerate(xs):
            auxF = lambda x : self.lossF(i,x,**kwargs)
            if(self.cnstF is None):
                x,fun=scipyMinimizeLoss(auxF,x,verbP=False)
            else:
                x,fun=scipyMinimizeLoss(auxF,x,verbP=False,cstF=cstF,cstA=cstA)
            ys[i,:]=x
        return ys
#%%----------------------------------------------------------------------------
#                             Optimizers
#------------------------------------------------------------------------------

def getTorchOptimizer(z,lr=0.01,typ=None):
    
    if(typ=='MadGrad'):
        return topt.MADGRAD([z],lr=lr,momentum=0.9,weight_decay=0,eps=1e-6)
    elif(typ=='LFBGS'):
        return torch.optim.LBFGS([z],lr=lr,line_search_fn='strong_wolfe')
    elif(typ=='ademamix'):
        return AdEMAMix(params=[z], lr=lr, betas=(0.9, 0.999, 0.9999), alpha=2.0, beta3_warmup=None, alpha_warmup=None, weight_decay=0.0)
    return torch.optim.Adam([z],lr=lr)

def computeAllGrads(losses,sumP=True):
    if(sumP):
        # The sum is never actually used but this forces the computation of all derivatives
        losses.sum().backward()
    else:
        # Explicity loop thru the loss vector
        for loss in losses:
            loss.backward(retain_graph=True)
    
def scipyMinimizeLoss(lossF,x,verbP=False,cstF=None,cstA=None,**kwargs):
    
    def objF(x): 
        x = makeTensor(x,gradP=False,dblP=True)
        y = lossF(x,**kwargs)
        return y.item()

    def objG(x):
        x = makeTensor(x,gradP=True,dblP=True)
        if(1):
            auxF = lambda x : lossF(x,**kwargs)
            return fromTensor(torch.autograd.functional.jacobian(auxF,x))
        else:
            y = lossF(x,**kwargs)
            y.backward()
            return fromTensor(x.grad)
    
    cons = None
    if(cstF is not None):
        cons = ({'type': 'eq', 'fun': cstF, 'jac': cstA})
    opt = scipy.optimize.minimize(objF,x,jac=objG,method='SLSQP',constraints=cons)
   
    if(verbP):
        print('{}: {}'.format(opt.message,opt.fun))
        
    return opt.x,opt.fun
        
#%%---------------------------------------------------------------------------
#                        Test Functions
#-----------------------------------------------------------------------------

#%% Check objF assumed to return a value F and its gradient G
def testBatchGrad(objF,x0,eps=1e-4):
    b,n = x0.size()
    x0   = makeTensor(x0)
    x    = makeTensor(x0)
    F0,G = objF(x)
    for i in range(n):
        x[:,i]=x0[:,i]+eps
        F1,_ = objF(x)
        dF = (F1-F0)/eps
        print(i)
        printTensor(G[:,i])
        print('----')
        printTensor(dF)
        x[:,i]=x0[:,i]
#%% Check objF assumed to return a vector F and its Jacobian J
def testBatchJac(objF,x0,eps=1e-4):
    b,n = x0.size()
    x    = makeTensor(x0)
    F0,J = objF(x)
    if(J is None):
        print('testBatchJac: No jabobian computed')
        return
    m    = F0.size(1)
    for i in range(n):
        x[:,i]=x0[:,i]+eps
        F1,_ = objF(x)
        for j in range(m):
            dF = (F1[:,j]-F0[:,j])/eps
            print('coord {:2d}, cnst {:2d}'.format(i,j))
            printTensor(J[:,j,i])
            printTensor(dF)
        x[:,i]=x0[:,i]
