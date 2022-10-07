---
layout: post
title:  NeRF in PyTorch (1/4)
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
      labels: contains 25 parameters representing the extrinsic and the intrinsic matrices. (B,25)
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