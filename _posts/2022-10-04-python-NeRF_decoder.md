---
layout: post
title:  NeRF in PyTorch (3/4)
date:   2022-10-04
description: How to build a NeRF neural network from scratch using PyTorch.
tags: implementation
categories: sample-posts
---

State: in progress

<p> <br> </p>

# Table of content

* Parameterised cameras.
* Sample points.
* Decoder architecture.
* Ray marcher.

<p> <br> </p>

## Decoder architecture.

So far, we have only used computer graphics techniques to model cameras, rays and to sample
points. Now, we need to find for each point, an estimated color and an estimated depth.
To do it, we will use two different modules: one embedder and one decoder.

1. Embedder.

Embedding consists on mapping inputs to a higher dimensional space 
using high frequency functions before passing them to the network. The idea behind is to
enable better fitting of data that contains high frequency variation. In theory,
we can chose any function:

$$\gamma : \mathbb{R} \rightarrow \mathbb{R}^{M} \text{, with } M>>1 $$ 

In practice, they chose to use sinus-cosinus encoding parameterised by L, 
controlling the output space dimension:

$$\gamma : \mathbb{R} \rightarrow \mathbb{R}^{2.L} \text{ , } \gamma(p)=(sin(2^0\pi p), cos(2^0\pi p), ... sin(2^{L-1}\pi p), cos(2^{L-1}\pi p))$$

Using the previous formula, the code is quite direct. For their experiments, they 
have set the embedding dimension of position to 20 (L=10) and of viewing to 8 (L=4).
The space of variation associated with changes in position seems to be larger than the one
associated with viewpoint changes.

```python 
class Embedder:
    def __init__(self, L={'pts':10,'views':4}):    
        self.embed={}
        for mode in L.keys():
            self.embed[mode]=self.get_embed(L[mode])
    
    def get_embed(self, scale):
        bands=2.**torch.linspace(0., scale-1, steps=scale)
        mapping=[]
        for f in bands:
            mapping.append(lambda x: torch.sin(x*f) )
            mapping.append(lambda x: torch.cos(x*f) )
        return lambda x : torch.cat([fn(x) for fn in mapping], -1)
        
    def __call__(self,x,dims={'pts':slice(0,3),'views':slice(3,6)}):
        out=[]
        
        for mode in self.embed.keys():
            out.append(self.embed[mode](x[...,dims[mode]]))            
        return out
```

<p> <br> </p>

2. Decoder.

Decoder takes as inputs positional and viewing direction embedding, and outputs estimated
color and density. The architecture is composed of fully connected layers, with 
two skip connections.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/Blog/2022-10-04/NeRF_architecture.png" title="Decoder architecture" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Decoder architecture.
</div>

The associated code is rather simple since it uses simple layers. We only have to take 
care that we use most of the time ReLU activation functions except for the last layer before 
outputing density (no activation) and for the final layer (sigmoid).

```python 
class NeRFModel(nn.Module):
    def __init__(self,
                 features_in= [60, 256,256,256,256,  256+60,256,256,  256,256+24,128], 
                 features_out=[256,256,256,256,256,  256,   256,256,  256+1,128,3],
                 skip_x=4,skip_views=8,
                embedder_kwargs={'L':{'pts':10,'views':4} }):
        super().__init__()
        len_f=len(features_in)
        
        core=[]
        for n in range(len_f):
            core.append(nn.Linear(features_in[n],features_out[n]))
        self.core=nn.Sequential(*core)
        self.skip={'pts':skip_x,'views':skip_views}
        self.Embedder=Embedder()
        
    def forward(self, x):
        input_pts, input_views=self.Embedder(x)
        z=input_pts
        for l in range(len(self.core)):
            z=self.core[l](z)
            if l==self.skip['pts']:
                z=torch.cat([input_pts,z], dim=-1)
                z=F.relu(z)
            elif l==self.skip['views']:
                sigma=z[...,-1:]
                z=torch.cat([z[...,:-1], input_views], dim=-1)
            else:
                z=F.relu(z)
        rgb=z
        return rgb,sigma
```

This model has 593.924 parameters.