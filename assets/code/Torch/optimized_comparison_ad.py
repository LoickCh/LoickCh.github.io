#Author: Chambon Loick
'''
Desc: Comparison between forward AD and reverse AD using 
torch.autograd.functionalities
'''
#--------------------------------------------------------------------
# Imports:
import torch
from torch import nn
import torch.autograd.forward_ad as fwAD

import time
#--------------------------------------------------------------------
# Main:
if __name__ == '__main__':
    N,M=16,1024
    
    x=torch.randn(N)
    model=nn.Linear(N,M)
    
    # Forward 
    start=time.perf_counter()
    jacob_fwd=torch.autograd.functional.jacobian(model, x, vectorize=True, strategy='forward-mode')
    end=time.perf_counter()
    print('Time: {:10.3f}'.format(end-start))
    
    # Reverse
    start=time.perf_counter()
    jacob_rev=torch.autograd.functional.jacobian(model, x)
    end=time.perf_counter()
    print('Time: {:10.3f}'.format(end-start))
    
    torch.allclose(jacob_rev, jacob_fwd)
    
    # jvp
    v=torch.randn(N)
    jvp=torch.autograd.functional.jvp(model, x, v)
    torch.allclose(jvp[1], jacob_rev @ v)
    pass