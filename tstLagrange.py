#!/usr/bin/env python3

#%%%--------------------------------------------------------------------------
#                                IMPORTS
#-----------------------------------------------------------------------------
#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
from   torch.autograd.functional import jacobian

from auxfuncs  import plotCircle,getFig,pltInline,printTensor,fromTensor,makeTensor,currentDevice
from torchopt  import pytorchKktOptim,pytorchAlmOptim,pytorchScipyOptim
#%%---------------------------------------------------------------------------
#                           Projection on a Circle
#-----------------------------------------------------------------------------
#%%
global xg,xt,insideP,analyticP
n = 2
m = 1
b = 5

# Batches
def objF(x):
    
    device=currentDevice()
    
    b,n = x.size()
    r   = n
    
    F = x-makeTensor(xt)
    J = torch.zeros((b,r,n),device=device)
    
    for i in range(n):
        J[:,i,i]  = 1.0
    
    return F,J
    
def cstF (x):
    
    device=currentDevice()
    C = torch.zeros((b,m),device=device)
    A = torch.zeros((b,m,n),device=device)
    
    if(analyticP):
        # Analytic derivatives
        C[:,0]   = 0.5*(torch.sum(x*x,dim=1)-1.0) 
        A[:,0,:] = x
        if(insideP):
            outP         = (C[:,0] < 0.0).flatten()
            C [outP]  = 0.0
            A [outP]  = 0.0
    else:
        # Autograd  derivatives
        if(insideP):
            distF = lambda x : torch.relu(torch.sqrt(torch.sum(x*x,dim=1))-1.0)
        else:
            distF = lambda x : torch.sqrt(torch.sum(x*x,dim=1))-1.0
  
        dst = distF(x)
        jac = jacobian(distF,x)
        
        C[:,0]   = dst
        for i in range(A.size(0)):
           A[i,0,:] = jac[i,i,:]
        
    return C,A

# Single sample
def lossF1(i,x):
    assert(1==len(x.size()))
    xi = makeTensor(xt[i])
    return 0.5 * (x-xi).pow(2).sum()

def cstF1(x):
    assert(1==len(x.size()))
    device=currentDevice()
    C = torch.zeros((m),device=device)
    A = torch.zeros((m,n),device=device)
    # Analytic derivatives
    C[0]   = 0.5*(torch.sum(x*x)-1.0)
    A[0,:] = x
        
    return C,A
#%%
xt = np.random.randn(b,n)                         # Attractors
xg = xt / np.linalg.norm(xt,axis=1,keepdims=True) # Desired result
x0 = xt + 0.1 * np.random.randn(b,n)
#%% KKt optimization
insideP  =  1
analyticP = 1
lag = pytorchKktOptim(objF,cstF) 
x0 =  makeTensor(x0)
x1 = lag.optim(x0,nIt=10,lambd=1e-6,rho=10.0,verbP=True)
x0 = fromTensor(x0)
x1 = fromTensor(x1)
#%% Augmented Lagrange optimization
lag = pytorchAlmOptim(objF,cstF)
x1  = lag.optim(x0,nIt=10,rho=100.0,verbP=True)
x0  = fromTensor(x0)
x1  = fromTensor(x1)
#%% Use scipy instead of pytorch
lag = pytorchScipyOptim(lossF1,cstF1)
x0  = fromTensor(x0,dblP=True)
x1  = lag.optim(x0,verbP=True)
#%%
plotCircle(0.0,0.0,1.0,'g')
plt.plot(xg[:,0],xg[:,1],'xg')    # What the result should be
plt.plot(xt[:,0],xt[:,1],'.r')    # Target point 
plt.plot(x1[:,0],x1[:,1],'.b')    # Optimized result
plt.plot([xt[:,0],x1[:,0]],[xt[:,1],x1[:,1]],'--g')
ax=plt.gca()
ax.set_aspect('equal')
#%%---------------------------------------------------------------------------
#               Projection on the interesection of 2 circles
#-----------------------------------------------------------------------------
#%%
n = 2
m = 2
b = 20
xc = 0.1
yc = 0.2

def objF(x):
    
    device=currentDevice()
    
    b,n = x.size()
    r   = n
    
    F = x-makeTensor(xt)
    J = torch.zeros((b,r,n),device=device)
    
    for i in range(n):
        J[:,i,i]  = 1.0
    
    return F,J
    
def cstF (x):
    
    device=currentDevice()
    C = torch.zeros((b,m),device=device)
    A = torch.zeros((b,m,n),device=device)
    
    y = x.clone()
    y[:,0] -= xc
    y[:,1] -= yc
    
    C[:,0]   = 0.5*(torch.sum(x*x,dim=1)-1.0)
    C[:,1]   = 0.5*(torch.sum(y*y,dim=1)-1.0)
    A[:,0,:] = x
    A[:,1,:] = y
    
    if(insideP):
        outP        = (C[:,0] < 0.0).flatten()
        C [outP,0]  = 0.0
        A [outP,0]  = 0.0
        outP        = (C[:,1] < 0.0).flatten()
        C [outP,1]  = 0.0
        A [outP,1]  = 0.0
    
    return C,A
#%%
xt =  np.random.randn(b,n)                     # Attractors
x0 =  xt + 0.1 * np.random.randn(b,n)
#%%
insideP  = 0 
lag = pytorchKktOptim(objF,cstF) 
x1 =  lag.optim(x0,nIt=10,lambd=1.e-6,rho=100.0,verbP=True)
x0 = fromTensor(x0)
x1 = fromTensor(x1)
#%% Augmented Lagrange optimization
lag = pytorchAlmOptim(objF,cstF)
x1 =  lag.optim(x0,nIt=10,nRep=50,rho=100.0,verbP=True)
x0 = fromTensor(x0)
x1 = fromTensor(x1)
#%%
plotCircle(0.0,0.0,1.0,'g')
plotCircle(xc,yc,1.0,'g')
plt.plot(xt[:,0],xt[:,1],'.r')
plt.plot(x1[:,0],x1[:,1],'.b')
plt.plot([xt[:,0],x1[:,0]],[xt[:,1],x1[:,1]],'--g')
ax=plt.gca()
ax.set_aspect('equal')
#%%---------------------------------------------------------------------------
#               Projection on the interesection of 2 spheres
#-----------------------------------------------------------------------------
#%%
n = 3
m = 2
b = 100
xc = 0.1
yc = 0.2
zc = 0.3

def objF(x):
    
    device = currentDevice()
    
    b,n = x.size()
    r   = n
    
    F = x-makeTensor(xt)
    J = torch.zeros((b,r,n),device=device)
    
    for i in range(n):
        J[:,i,i]  = 1.0
    
    return F,J

# Two intersecting spheres
def cstFS (x):
    
    device = currentDevice()
    C = torch.zeros((b,m),device=device)
    A = torch.zeros((b,m,n),device=device)
    
    y = x.clone()
    y[:,0] -= xc
    y[:,1] -= yc
    y[:,2] -= zc
    
    C[:,0]   = 0.5*(torch.sum(x*x,dim=1)-1.0)
    C[:,1]   = 0.5*(torch.sum(y*y,dim=1)-1.0)
    A[:,0,:] = x
    A[:,1,:] = y
    
    if(insideP):
        outP        = (C[:,0] < 0.0).flatten()
        C [outP,0]  = 0.0
        A [outP,0]  = 0.0
        outP        = (C[:,1] < 0.0).flatten()
        C [outP,1]  = 0.0
        A [outP,1]  = 0.0
    
    return C,A 

# Intersection of a sphere and a plane
def cstFP (x):
    
    device = currentDevice()
    C = torch.zeros((b,m),device=device)
    A = torch.zeros((b,m,n),device=device)
    
    y = x.clone()
    y[:,0] -= xc
    y[:,1] -= yc
    y[:,2] -= zc
    
    C[:,0]   = 0.5*(torch.sum(x*x,dim=1)-1.0)
    C[:,1]   = x.sum(dim=1)
    A[:,0,:] = x
    A[:,1,:] = 1.0
    
    if(insideP):
        outP        = (C[:,0] < 0.0).flatten()
        C [outP,0]  = 0.0
        A [outP,0]  = 0.0
    
    return C,A
#%%  
xt      = np.random.randn(b,n)   
x0      =  xt + 0.1 * np.random.randn(b,n)
insideP = True
cstF = cstFP
#%%
lag = pytorchKktOptim(objF,cstF) 
x0 = makeTensor(x0)
x1 =  lag.optim(x0,nIt=10,lambd=1.e-6,rho=10.0,verbP=True)
x0 = fromTensor(x0)
x1 = fromTensor(x1)
#%% Augmented Lagrange optimization
lag = pytorchAlmOptim(objF,cstF)
x1 =  lag.optim(x0,nIt=10,nRep=50,rho=1000.0,verbP=True)
x0 = fromTensor(x0)
x1 = fromTensor(x1)
#%%
try:
    pltInline(False)
except:
    pass
fig = getFig()
ax  = fig.add_subplot(111, projection='3d')
ax.scatter(x0[:,0],x0[:,1],x0[:,2],'.r')
ax.scatter(x1[:,0],x1[:,1],x1[:,2],'.b')
#%%---------------------------------------------------------------------------
#                 Test ObjF and CstF 
#-----------------------------------------------------------------------------
#%% Check objF
eps = 1e-4
x0   = makeTensor(x0)
x    = makeTensor(x0)
F0,J = objF(x)
for i in range(n):
    x[:,i]=x0[:,i]+eps
    F1,_ = objF(x)
    dF = (F1-F0)/eps
    print(i)
    printTensor(J[:,:,i])
    print('----')
    printTensor(dF)
    x[:,i]=x0[:,i]
#%% Check cstF
eps = 1e-4
x    = makeTensor(x0)
C0,A = cstF(x)
for i in range(n):
    x[:,i]=x0[:,i]+eps
    C1,_ = cstF(x)
    dC = (C1[:,0]-C0[:,0])/eps
    print(i)
    printTensor(A[:,0,i])
    printTensor(dC)
    x[:,i]=x0[:,i]
