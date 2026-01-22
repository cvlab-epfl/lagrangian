#!/usr/bin/env python3

#%%%--------------------------------------------------------------------------
#                                 IMPORTS
#-----------------------------------------------------------------------------
#%%

import torch
import sys,pickle
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.getipython import get_ipython

#%%----------------------------------------------------------------------------
#                                  Arrays
#------------------------------------------------------------------------------

def currentDevice():
    if(torch.cuda.is_available()):
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
    
def makeTensor(x,floatP=True,dblP=False,gradP=False):
    
    if(torch.is_tensor(x)):
        x = x.clone().detach().requires_grad_(gradP)
        x = x.to(currentDevice())
    else:
        if(floatP):
            x = np.asarray(x,dtype=np.float32)
        else:
            x = np.asarray(x,dtype=np.int32)
        x = torch.tensor(x,device=currentDevice(),requires_grad=gradP)
        
    return x

def fromTensor(x,dblP=False):
    if(torch.is_tensor(x)):
        data = x.data.cpu().numpy() 
        if(dblP):
            return np.asarray(data,dtype=np.float64)
        return data
    elif(dblP):
        return np.asarray(x,dtype=np.float64)
    return x
    
def printTensor(x,form='{:2.4f}',prefix=None):
    x = fromTensor(x)
    printArray(x,form=form,prefix=prefix)
       
def printArray(xs,form='{:2.4f}',trP=False,prefix=None):
    
    if(prefix is not None):
        print(prefix,end=' ')
        
    if(1==len(xs.shape)):
        for x in xs:
            print(form.format(x),end=' ')
        print(' ')
    elif(2==len(xs.shape)):
        if(trP):
            x = x.t
        for row in xs:
            for x in row:
                print(form.format(x),end=' ')
            print(' ')
    else:
        print('printFlats cannot print arrays of size {:}'.format(len(xs.shape)))

#%%----------------------------------------------------------------------------
#                                  Files
#------------------------------------------------------------------------------

def dumpToFile(fileName,obj,verbP=True):
    if(fileName is not None):
        try:
            with open(fileName,'wb') as f:
                pickle.dump(obj,f)
        except IOError:
            if(verbP):
                print('dumpToFile: cannot open', fileName)     
                
#%%----------------------------------------------------------------------------
#                                  Errors
#------------------------------------------------------------------------------

def raiseError(message="Runtime error"):
    if(None==sys.exc_info()[0]):
        print(message, )
    else: 
        print(message,sys.exc_info()[0])
    raise RuntimeError
    
#%%----------------------------------------------------------------------------
#                                  Plot
#------------------------------------------------------------------------------
#%%
def plotCircle(x,y,r,col='r',scale=1.0):
    
    plotArc(x/scale,y/scale,r/scale,r/scale,0,360,col)

def plotArc(x,y,rx,ry,th1=0,th2=360,col='r'):
    
    #fig=getFig(fig)
    
    phi=(np.pi/180.0)*np.arange(th1,th2,1)
    xs=x+rx*np.cos(phi)
    ys=y+ry*np.sin(phi)
    plt.plot(xs,ys, c=col,ls='-')
    
def pltInline(inLineP=True):
    if(inLineP):
        get_ipython().run_line_magic('matplotlib', 'inline')
    else:
        get_ipython().run_line_magic('matplotlib', 'qt')
        
def getFig(fig=None,clearP=True):
    
    if(fig is None):
        fig = plt.gcf()
    if(clearP):
        fig.clf()
        
    return(fig)