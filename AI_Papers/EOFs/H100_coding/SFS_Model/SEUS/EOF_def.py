###Function statement for EOFs. 
from eofs.standard import Eof
import numpy as np
import math
import pandas as pd
import xarray as xr 
import pickle 

def EOF_def(solver, modes):
    ##select the desired number of eofs
    nmode = modes
    #print(nmode)
    EOF = solver.eofs(neofs=nmode,eofscaling=0) 
    EOF_nw = EOF 
    #print(type(EOF),np.shape(EOF))
    ##make the EOF 2-dimensional
    EOF2d = EOF.reshape(EOF.shape[0],EOF.shape[-2]*EOF.shape[-1])
    #print(np.shape(EOF2d))
    pv = np.dot(EOF2d,np.transpose(EOF2d))
    EOF_nw2d = EOF2d
    #print(pv.shape)
    eigenv = solver.eigenvalues(neigs=nmode)
    VarEx = solver.varianceFraction(neigs=nmode)*100
    PC = solver.pcs(npcs=nmode,pcscaling=1)
    print(type(PC),np.shape(PC))
    del pv
    pv = np.dot(np.transpose(PC),PC)
    return EOF_nw, EOF_nw2d,eigenv, VarEx, PC;
