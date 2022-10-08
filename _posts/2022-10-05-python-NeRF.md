---
layout: post
title:  NeRF in PyTorch (4/4)
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

## Ray marcher

After decoding points, we obtain color and density of points. Since one pixel corresponds
to one ray, we need to find a way to assign a ray color and a ray density using the 
points on it. That is the role of the ray marcher.


To estimate color of a ray $\hat{C}(\textbf{r})$ with ray previously defined by its origin, destination and depth: 

$$r := o + t.d$$ 

discretised in N points using different depths $t := (t_i)_{(i \in [1,N])}$, 
we need the distance between consecutive samples $\delta = ( t_{i+1} - t_i)_{(i \in [1,N])}$, 
point density $\sigma := (\sigma_i)_{i \in [1,N]}$ and point color $\textbf{c}= (\textbf{c}_i)_{i \in [1,N]}$. Then we use a volume rendering formula:

$$ T_i = \exp{\Big( - \sum_{j=1}^{i-1} \sigma_j \delta_j \Big)} $$
$$ \hat{C}(\textbf{r})= \sum_{i=1}^N T_i ( 1 - \exp{(-\sigma_i \delta_i)})  \textbf{c}_i$$

To implement the following formula, we begin to define distances between samples
 while adding an infinite distance at the end. Since $e^{-\infty}=0$, it has no 
 effect on the result. Furthermore, since ray_d represents an image plane, 
 the distance of the points from ray_o varies. In order to obtain the true 
 distance to the origin, we multiply by the norm of ray_d.

```python 
def ray_marcher(rgb,sigma,z_vals,ray_d):
    B,N_rays,N_sample,_=rgb.shape
    deltas = z_vals[:,:,1:] - z_vals[:,:,:-1]
    deltas = torch.cat([deltas, torch.Tensor([1e10]).expand(deltas[:,:,:1,:].shape).to(device=deltas.device)], -2)
    deltas = deltas * torch.norm(ray_d[...,None,None,:], dim=-1)
```

Then we use the fact that an exponential of a sum is the product of exponential to
calculate weights. In the decoder, we use only relu activation function except after
the density layer and the final rgb, so they are "unconstrained". Densities must be
between 0 and 1, so we use a relu activation.
```python 
    [...]
    rgb=torch.sigmoid(rgb)
    alpha = 1.-torch.exp(-F.relu(sigma)*deltas)

    # 1e-10 to avoid multiply by zero.
    # 1-alpha=e^(-sigma . delta)
    T = torch.cumprod(torch.cat([torch.ones((B,N_rays,1,1),device=deltas.device),1.-alpha+1e-10],-2),-2)[...,:-1,:]
    weights = alpha * T
    
    rgb_map = torch.sum(weights * rgb, -2)
    depth_map = torch.sum(weights * z_vals, -1)
    return rgb_map, weights
```