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
        
        match mode:
            case 'sequential':
                self.layers=nn.Sequential(lin_in, lin_out)
            case 'sequential_order':
                self.layers=nn.Sequential(OrderedDict([
                    ('lin_in', lin_in),
                    ('lin_out', lin_out)]))
            case 'module_list':
                self.layers=nn.ModuleList([lin_in, lin_out])
            case 'list':
                self.layers=[lin_in, lin_out]
            case 'module_dict':
                self.layers=nn.ModuleDict({'lin_in':lin_in, 'lin_out':lin_out})
        return

    def forward(self, x):
        match self.mode:
            case 'sequential' | 'sequential_order':
                return self.layers(x)
            case 'module_list' | 'list':
                for l in self.layers:
                    x=l(x)
                return x
            case 'module_dict':
                for k,v in self.layers.items():
                    x=v(x)
                return x
#--------------------------------------------------------------------
# Functions:
def inspect_layer(l):
    print(l)

#--------------------------------------------------------------------
# Main:
if __name__ == '__main__':
    in_f, inter_f, out_f=3,10,1
    x=torch.randn(1,in_f)
    
    for mode in ['sequential','sequential_order','module_list','list','module_dict']:
        print(f'Mode:{mode}'.center(40,'-'))
        model=Model(in_f, inter_f, out_f, mode)
        print(f'Layer representation:\n {model.layers}')
        
        print('\nWhat are parameters shape registered in model.parameters() ?')
        for p in model.parameters():
            print(p.shape)
        model(x)
        
        print('\nInspect function:')
        model.apply(inspect_layer)
        print('\n')
    pass