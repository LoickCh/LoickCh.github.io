#Author: Chambon Loick
'''
Desc: Reverse AD
'''
#--------------------------------------------------------------------
# Imports:
import torch
from torch import nn
#--------------------------------------------------------------------
# Main:
if __name__ == '__main__':
    in_f, inter_f, out_f=3,128,2

    # Setup
    x=torch.arange(in_f, dtype=torch.float32, requires_grad=True)
    lin_in, lin_inter, lin_out=(nn.Linear(in_f,inter_f), 
                                nn.Linear(inter_f, inter_f), 
                                nn.Linear(inter_f, out_f))
    model=nn.Sequential(lin_in, lin_inter, lin_out)

    # Forward
    y= model(x)
    grad_output=torch.ones(out_f)

    # Backward: sum of gradients
    torch.autograd.backward(y, grad_tensors=grad_output)
    jacob_sum=x.grad

    # Sanity check
    prod=lin_out.weight @ lin_inter.weight @ lin_in.weight # W_3.W
    torch.allclose(jacob_sum, torch.sum(prod, dim=0)) # sum of gradients
    pass