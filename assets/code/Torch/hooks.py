#Author: Chambon Loick
'''
Desc: Exploring torch.nn module
-> Hooks
'''
#--------------------------------------------------------------------
# Imports:
import torch
from torch import nn

from collections import OrderedDict
#--------------------------------------------------------------------
# Classes:
class SimpleMLP(nn.Module):
    def __init__(self, in_f, inter_f, out_f):
        super().__init__()
        self.layers=nn.Sequential(
            OrderedDict([
            ('lin_in', nn.Linear(in_f, inter_f)),
            ('lin_out', nn.Linear(inter_f, out_f))])
            )
        
    def forward(self, x):
        return self.layers(x)
    
class VerboseModel(nn.Module):
    def __init__(self, model:nn.Module, mode='named_modules') -> None:
        super().__init__()
        self.model=model 
        
        match mode:
            case 'named_modules':
                for name, module in model.named_modules():
                    module.__name__=name
                    module.register_forward_pre_hook(inspect_shape_in)
            case 'named_children':
                for name, module in model.named_children():
                    module.__name__=name
                    module.register_forward_pre_hook(inspect_shape_in)

    def forward(self, x):
        return self.model(x)
        
#--------------------------------------------------------------------
# Functions:
def inspect_shape_in(module, input):
    if hasattr(module, '__name__'):
        name=module.__name__
    else:
        name=''
    print("Model {:15} Input shape {:10}".format(name, 
                                                 str(tuple(input[0].shape)) ))

# def inspect_shape_in_out(module, input, output):
#     print("Module {:15} in {:10} out {:10}".format(module.__name__, 
#                                        str(tuple(input[0].shape)),
#                                        str(tuple(output[0].shape))))
#--------------------------------------------------------------------
# Main:
if __name__ == '__main__':
    in_f, inter_f, out_f=3,10,1
    x=torch.zeros((2,in_f))
    
    for mode in ['apply', 'named_modules', 'named_children']:
        print('Mode: {:<15}'.format(mode))
        model=SimpleMLP(in_f, inter_f, out_f)
        
        match mode:
            case 'apply':
                model.apply(lambda m: m.register_forward_pre_hook(inspect_shape_in) )
                model(x)
                
            case 'named_modules'|'named_children':
                verbose_model=VerboseModel(model, mode)
                verbose_model(x)
        print('\n')
    