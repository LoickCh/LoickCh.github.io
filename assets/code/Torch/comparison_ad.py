#Author: Chambon Loick
'''
Desc: Comparison between forward AD and reverse AD.
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
    N,M=1024,16
    
    x=torch.randn(N)
    model=nn.Linear(N,M)
    
    # Forward AD
    primal=x.clone()
    tangents=torch.eye(N)
    
    start=time.perf_counter()
    jacob_fwd=[]
    for tangent in tangents:
        with fwAD.dual_level():
            dual_input = fwAD.make_dual(x, tangent)
            y = model(dual_input)
            jvp = fwAD.unpack_dual(y).tangent
        jacob_fwd.append(jvp)
    jacob_fwd=torch.stack(jacob_fwd)
    end=time.perf_counter()
    
    print(f'Forward AD: {end-start:>21.5f}s shape {jvp.shape}')
    
    # Reverse AD
    inp=x.clone().requires_grad_(True)
    gradients=torch.eye(M)
    
    start=time.perf_counter()
    y=model(inp)
    jacob_rev=[]
    for grad in gradients:
        torch.autograd.backward(y, grad_tensors=grad, retain_graph=True)
        jacob_rev.append(inp.grad)
    jacob_rev=torch.stack(jacob_rev)
    end=time.perf_counter()
    print(f'Backward AD: {end-start:>20.5f}s shape {inp.grad.shape}')
    
    torch.allclose(jacob_rev, jacob_fwd.T)
    pass