---
layout: post
title:  Torch module illustrated (torch.nn)
date:   2022-10-23
description: Exploring and explaining torch functionalities.
tags: python
categories: library
---

Today we will dive in torch.nn module, one of them most important Pytorch's module
that you must manipulate if you want to create neural networks or customized existing 
code.

<p> <br> </p>

# Table of contents:

- nn.Parameter


<p> <br> </p>

# Specific subclass

### nn.Parameter

**Sum up:** nn.Parameter is a subclass of torch.Tensor used when we need to optimize tensors during the gradient
descent. It automatically add tensors to the parameters() iterator allowing us to simply
define an optimizer.


**Example:** To illustrate the notion, let us implement a linear layer. To recall, a linear layer
modifies the number of channels of an input tensor *x* applying a single hidden layer.
It is defined with a weight matrix *A* and bias matrix *b* by the following equation:
$$y=x.A^T + b$$
Where y has shape $(*,H_{out})$ and x has shape $(*,H_{in})$.

Thus, to implement the layer, we need to define *A* and *b*, i.e to initialize a weight 
and a bias matrix and to specify that they need to be updated during optimization.
A common practice to define an optimizer is to call *model.parameters()* returning
an iterator with all model registered parameters:

```python
# SGD optimizer
# >> use model.parameters()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

This behaviour constraints *A* and *b* to be in the *model.parameters()* iterator.
To do it, we must define them using *torch.nn.Parameter* (!) instead of *torch.Tensor*.
Hence, a simple definition of a linear layer can be:

```python
class SimpleLinearLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Simple initialization
        self.weight=nn.Parameter(torch.zeros(out_features, in_features))
        self.bias=nn.Parameter(torch.ones(out_features))
        return
    
    def forward(self,x):
        return x @ self.weight.T + self.bias
```

Then, when we inspect the parameters() iterator, we get:

```python
layer=CustomLayer(in_features, out_features)
for p in layer.parameters():
    print(p)

>>> Parameter containing:
>>> tensor([[0., 0., 0.],
>>>         [0., 0., 0.],
>>>         [0., 0., 0.],
>>>         [0., 0., 0.],
>>>         [0., 0., 0.]], requires_grad=True)
>>> Parameter containing:
>>> tensor([1., 1., 1., 1., 1.], requires_grad=True)
```

<p> <br> </p>

# How to properly define model ?

Defining a model requires to create layers and to add an order between them. First, 
we need to subclass the *Module* class. It is the base class for all neural networks modules.
Then, we need to define layers in the *\__init__* method. The best way to define 
layers depend on the architecture. We can use:
- Sequential: If layers are sequentially executed.
- ModuleList or ModuleDict: If we want to register layers but define the call logic later.
- ParameterList or ParameterDict: If instead of layers we work directly with their parameters.


{% mermaid %}
sequenceDiagram
    participant John
    participant Alice
    Alice->>John: Hello John, how are you?
    John-->>Alice: Great!
{% endmermaid %}