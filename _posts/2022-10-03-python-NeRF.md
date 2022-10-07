---
layout: post
title:  NeRF in PyTorch (2/4)
date:   2022-09-29
description: How to build a NeRF neural network from scratch using PyTorch.
tags: python
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

## Sample points

Once we have modelled the cameras, we need to sample the 3D space. 
To do it, we sample points along rays *r* defined with the camera origin *o*, 
the camera directions *d* and depths *t*: $r=o+t.d$ 
In the NeRF paper, there are two different sampling techniques: coarse sampling and fine sampling.

<p> <br> </p>

1. Coarse sampling

The coarse pass samples the points uniformly along rays. 
First, we uniformly subdivide the rays, and then select a depth in each bin created.

```python 
def sample_stratified(self, ray_oray_o, ray_start, ray_end, N_samples):
        B, N_rays, *_ = ray_o.shape

        # 1. Subdivide rays
        depth = torch.linspace(ray_start, ray_end, N_samples, device=ray_o.device).reshape(1, 1, N_samples, 1).repeat(B, N_rays, 1, 1)
        
        # Select one random point in each bin.
        delta = (ray_end - ray_start)/(N_samples - 1)
        depth += torch.rand_like(depth) * delta
        return depth
```

The approach is simple and does not take into account neither the shape of the 
object nor previous results. During the training we progressively have some knowledge over 
spatial point density. Then, a good idea is to use it to sample points near high 
density areas which might correspond to surface regions.

<p> <br> </p>

2. Fine sampling

To perform fine sampling, we need to have depths *z_vals* from the coarse sample 
and point weights *weights* obtained after ray marchering on coarse points. 
We will explain ray marching in another post, for the moment it is enought to admit 
it assigns for each ray points, a weight indicating if it contributes to the final
pixel color and density or not. If we admit it, the idea is to extract from the 
weights, a probability function guiding the sampling.


First we extract a cumulative distribution function from the weights. It is defined 
for any real-valued random variable *X* by:

$$F_X(x)= \mathbb{P}(X\leq x)$$

```python 
def sample_pdf(z_vals, weights, N_importance, det=False, eps=1e-5):
    N_rays, N_samples_ = weights.shape
    
    # 1. Construct the probability distribution associated with weights.
    weights = weights + eps
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)
```

Next, we need to sample points according to the cumulated ditribution function. 
We use the following property:

$$ \text{If } Y \sim \mathcal{U}[0,1], \text{ then } F^{-1}(Y) \text{ is distributed as } F $$

```python
    [...]
    # 2. Sample points
    u = torch.rand(N_rays, N_importance, device=z_vals.device) # uniform sample between 0 and 1.
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    z_vals_g = torch.gather(z_vals, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 
    
    return z_vals_g[...,0] + (u-cdf_g[...,0])/denom * (z_vals_g[...,1]-z_vals_g[...,0])
```

