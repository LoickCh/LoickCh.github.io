---
layout: post
title:  NeRF in PyTorch
date:   2022-09-30
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

## Parameterised cameras

A NeRF dataset can be seen as a collection of images with their 25 camera parameters. The 25 camera parameters are composed of 16 extrinsic parameters and 9 intrinsic parameters. 

The extrinsic matrix (or camera to world matrix *c2w*) can be written using a rotation matrix $(R_{i,j})_{(i,j) \in \{1,2,3\}}$ and a translation matrix $(T_{i})_{(i) \in \{1,2,3\}}$. The rotation matrix can also being represented using three eulerian angles, aircraft principal angles (yaw, pitch, roll) or another representation.

The intrinsic matrix (K) can be written using the focal length ($f_x, f_y$), the principal point coordinates ($c_x,c_y$) and the skew $s_k$.

$$
c2w=
\left(\begin{array}{cc} 
r11 & r12 & r13 & t1 \\
r21 & r22 & r23 & t2 \\
r31 & r32 & r33 & t3 \\
0   & 0   & 0   & 1
\end{array}\right)
,
K=
\left(\begin{array}{cc} 
fx & sk & cx \\
0  & fy & cy \\
0  & 0  & 1 
\end{array}\right)
$$ 


The first step of building a NeRF neural network is to get ray directions (image plane) and ray origins (origin of the camera). To do so, we need to create a camera grid, to multiply it by the invert of the intrisic matrix and before to multiply it by the extrinsic matrix. Let us explain how can we code it.

1. Parameters representation

A common way to represent the 25 parameters is to flatten the tensor. The first 16 parameters represent the extrinsic parameters and the last the intrinsic parameters. In addition, there often are assumptions on the intrinsic parameters: skew is often set to 0, focal length on x and on y are identical, and the principal point is at the middle of the resolution. That is why we often have the following representation:
$$
K=
\left(\begin{array}{cc} 
f & 0 & W/2 \\
0 & f & H/2 \\
0 & 0  & 1 
\end{array}\right)
$$ 


```python 
def get_rays(labels):
    """Return ray origins and ray directions.
    
    Args:
        - labels: contains 25 parameters representing the extrinsic and the 
        intrinsic matrices. (B,25)
    """
    B,_=labels.shape
    
    # Unpack intrinsic and extrinsic matrix
    intrinsic=labels[:,16:].reshape(B,3,3) # B,3,3
    extrinsic=labels[:,:16].reshape(B,4,4) # B,4,4
    
    # Get focal length and principal point coordinates
    cy=intrinsic[0,1,2].int().item()
    cx=intrinsic[0,0,2].int().item()
    focal=intrinsic[:,0,0].unsqueeze(-1)

    # The principal point is often localted at a middle location.
    N_rays=H*W
```

2. Create a camera grid-pixel correspondance

Now we have to create a camera grid where one point correspond to one pixel on the image. To do it, we generate a grid using *torch.meshgrid*. Then, we need to go from 2D coordinates to 3D, so we use computer graphic conventions localising the image plane at $z=1$.

```python 
    [...]

    # 1. Set camera grid
    # -> Sensor discretization
    uv=torch.stack(torch.meshgrid(torch.linspace(0,W-1,W),
                   torch.linspace(0,H-1,H),indexing='xy')).to(labels.device)
    
    uv = uv.reshape(2, -1) 
    uv = uv.unsqueeze(0).repeat(B, 1, 1)
    x = uv[:,0,:].view(B, N_rays)
    y = uv[:,1,:].view(B, N_rays)
    z = torch.ones((B,N_rays), device=uv.device)
```

3. Invert the intrinsic

Now, we have unlocated 3D points. Then, we need to multiply by the inverse intrinsic matrix. Since it is a triangular upper matrix, it is equivalent to apply the following transformation:

$$
P:=
\left(\begin{array}{cc} 
x \\
y \\
z 
\end{array}\right)

\cdot 

K^{-1}
= 
\left(\begin{array}{cc} 
(x- 0.5*W)/f \\
(y- 0.5*H)/f\\
z 
\end{array}\right)
$$ 

```python 
    [...]
    
    # 2. Multiply by the invert of the intrisic matrix.
    x = (x - 0.5*W) / focal
    y = (y - 0.5*H) / focal
    cam_rel_points=torch.stack((x,-y,-z), dim=1)
```

4. Apply the extrinsic

Finally we need to apply camera to world matrix multiplication in order to change the point of view of the image. It is a simple batch matrix multiplication.


```python 
    [...]
    
    # 3. Apply camera to world transformation (+ remove homogeneous coordinates)
    world_rel_points = torch.bmm(extrinsic[:,:3,:3], cam_rel_points)
```

5. Get ray origins and directions

Ray origins *o* and ray directions *d* are defined such that for any ray *r*: $r = o + f*d$ where f is the depth (often bounded between a far and a near plane).

```python 
    [...]
    # Prepare outputs
    ray_d = world_rel_points.permute((0,2,1))
    ray_o = extrinsic[:, :3, 3].unsqueeze(1).repeat(1, ray_d.shape[1], 1)
    return ray_o, ray_d
```

---------------------------

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
        depth = torch.linspace(ray_start, ray_end, N_samples, device=ray_o.device)
        depth=depth.reshape(1, 1, N_samples, 1).repeat(B, N_rays, 1, 1)
        
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

---------------------------

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

---------------------------

<p> <br> </p>

## Ray marcher

After decoding points, we obtain color and density of points. Since one pixel corresponds
to one ray, we need to find a way to assign a ray color and a ray density using the 
points on it. That is the role of the ray marcher.


To estimate color of a ray $$\hat{C}(\textbf{r})$$ with ray previously defined by its origin, destination and depth: 

$$r := o + t.d$$ 

discretised in N points using different depths $$t := (t_i)_{(i \in [1,N])}$$, 
we need the distance between consecutive samples $$\delta = ( t_{i+1} - t_i)_{(i \in [1,N])}$$, 
point density $$\sigma := (\sigma_i)_{i \in [1,N]}$$ and point color $$\textbf{c}= (\textbf{c}_i)_{i \in [1,N]}$$. Then we use a volume rendering formula:

\begin{aligned}
&T_i  = \exp{\Big( - \sum_{j=1}^{i-1} \sigma_j \delta_j \Big)}  \\
&\hat{C}(\textbf{r}) = \sum_{i=1}^N T_i ( 1 - \exp{(-\sigma_i \delta_i)})  \textbf{c}_i
\end{aligned}


To implement the following formula, we begin to define distances between samples
 while adding an infinite distance at the end. Since $$e^{-\infty}=0$$, it has no 
 effect on the result. Furthermore, since ray_d represents an image plane, 
 the distance of the points from ray_o varies. In order to obtain the true 
 distance to the origin, we multiply by the norm of ray_d.

```python 
def ray_marcher(rgb,sigma,z_vals,ray_d):
    B,N_rays,N_sample,_=rgb.shape
    device=z_vals.device
    deltas = z_vals[:,:,1:] - z_vals[:,:,:-1]
    deltas = torch.cat([deltas, 
        torch.Tensor([1e10]).expand(deltas[:,:,:1,:].shape).to(device)], -2)
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
    T=torch.cumprod(torch.cat([
        torch.ones((B,N_rays,1,1),device),1.-alpha+1e-10],-2),-2)[...,:-1,:]
    weights = alpha * T
    
    rgb_map = torch.sum(weights * rgb, -2)
    depth_map = torch.sum(weights * z_vals, -1)
    return rgb_map, weights
```