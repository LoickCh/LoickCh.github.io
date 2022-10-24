#Author: Chambon Loick
'''
Desc: Gradient context manager 
'''
#--------------------------------------------------------------------
# Imports:
import torch
from torch import nn
from torch import autograd
#--------------------------------------------------------------------
# Main:
if __name__ == '__main__':
    # Params
    in_f, out_f=128,1

    # Setup
    x=torch.arange(in_f, dtype=torch.float32, requires_grad=True)
    model=nn.Sequential(nn.Linear(in_f,out_f))
    
    with torch.autograd.profiler.profile(enabled=True, profile_memory=True) as prof:
        for _ in range(10):
            y=model(x)     
            a=y**2  
    
    if prof is not None:
        # Save profiling
        prof.export_chrome_trace('result_profiling')
        print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    pass