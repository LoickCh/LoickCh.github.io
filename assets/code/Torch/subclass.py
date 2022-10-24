#Author: Chambon Loick
'''
Desc: Exploring torch.nn module
-> nn.Parameter()
'''
#--------------------------------------------------------------------
# Imports:
import torch
from torch import nn
#--------------------------------------------------------------------
# Classes:
# >> torch.nn.Parameter
class CustomLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight=nn.Parameter(torch.zeros(out_features, in_features))
        self.bias=nn.Parameter(torch.ones(out_features))
        return
    
    def forward(self,x):
        return x @ self.weight.T + self.bias

class CustomLayer_dict(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Simple initialization
        self.params=nn.ParameterDict({'weight':nn.Parameter(torch.zeros(out_features, in_features)),
                                     'bias':nn.Parameter(torch.ones(out_features))
                                     })
        return
    
    def forward(self,x):
        return x @ self.params['weight'].T + self.params['bias']
#--------------------------------------------------------------------
# Main:
if __name__ == '__main__':
    # Parameters
    in_dim,in_features=1,3
    out_features=2
    
    # Code
    x=torch.zeros((in_dim,in_features))
    layer=CustomLayer(in_features, out_features)
    for p in layer.parameters():
        print(p)
        
    # Define an optimizer using layer.parameters() !
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)

    