#Author: Chambon Loick
'''
Desc: Forward AD
'''
#--------------------------------------------------------------------
# Imports:
import torch
import torch.autograd.forward_ad as fwAD
from torch import nn
#--------------------------------------------------------------------
# Main:
if __name__ == '__main__':    
    # Params
    in_f, inter_f, out_f=3,128,2

    # Setup
    x=torch.arange(in_f, dtype=torch.float32)
    tangent = torch.ones(in_f)
    lin_in, lin_inter, lin_out=(nn.Linear(in_f,inter_f), 
                                nn.Linear(inter_f, inter_f), 
                                nn.Linear(inter_f, out_f))
    model=nn.Sequential(lin_in, lin_inter, lin_out)

    # Forward
    with fwAD.dual_level():
        dual_input = fwAD.make_dual(x, tangent)
        y = model(dual_input)
        jvp = fwAD.unpack_dual(y).tangent
    
    # Sanity check
    prod=lin_out.weight @ lin_inter.weight @ lin_in.weight # W_3.W
    torch.allclose(jvp, torch.sum(prod, dim=1)) # Equality !
    print(jvp.shape)
    pass