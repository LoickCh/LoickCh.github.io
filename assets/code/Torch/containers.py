#Author: Chambon Loick
'''
Desc: Exploring torch.nn module
-> Containers: 
'''
#--------------------------------------------------------------------
# Imports:
from collections import OrderedDict

import torch
from torch import nn
#--------------------------------------------------------------------
# Classes:
class Model(nn.Module):
    def __init__(self,in_features, inter_features, out_features, mode='sequential'):
        super().__init__()
        lin_in=nn.Linear(in_features, inter_features)
        lin_out=nn.Linear(inter_features, out_features)
        self.mode=mode
        
        if mode=='sequential':
            self.layers=nn.Sequential(lin_in, lin_out)
        elif mode=='sequential_order':
            self.layers=nn.Sequential(OrderedDict([
                ('lin_in', lin_in),
                ('lin_out', lin_out)]))
        elif mode=='module_list':
            self.layers=nn.ModuleList([lin_in, lin_out])
        elif mode=='list':
            self.layers=[lin_in, lin_out]
        return

    def forward(self, x):
        match self.mode:
            case ['sequential', 'sequential_order']:
                return self.layers(x)
            case ['module_list', 'list']:
                for l in self.layers:
                    x=l(x)
                return x
#--------------------------------------------------------------------
# Functions:

#--------------------------------------------------------------------
# Main:
if __name__ == '__main__':
    in_f, inter_f, out_f=3,10,1
    x=torch.randn(1,in_f)
    
    for mode in ['sequential','sequential_order','module_list','list']:
        print('\nMode', mode)
        model=Model(in_f, inter_f, out_f, mode)
        print(model.layers)
        
        for p in model.parameters():
            print(p.shape)
        model(x)
    pass