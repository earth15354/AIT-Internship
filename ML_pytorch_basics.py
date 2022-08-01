import torch
import numpy as np
import torchvision

# TENSOR BASICS 
'''
Tensors are similar to arrays/lists/vectors, but are the default data
types used in pytorch. Tensors can have many different dimensions.
'''

x1 = torch.empty(2, 1, 3) # 3D empty tensor
x2 = torch.rand(2, 2) # 2D random tensor (values from 0-1)
x3 = torch.zeros(2) # 1D zeros tensor
x4 = torch.ones(4, dtype=torch.float16) # 1D ones tensor of float16 (by default, it's float32)
x5 = torch.tensor([1,2,3]) # Create a tensor from an array
y = torch.rand(2, 2)
print(x2, y)

z = x2 + y # this will add x and y... can also use torch.add(x,y)
#y.add_(x2) # this will modify y to be sum of x and y <-- any function with a "_" will modify y
print(z)

z = x2 - y # this will subtract x and y... torch.sub(x,y)
print(z)

z = x2 * y # torch.mul(x,y) <--- element-wise multiplication
print(z)

z = x2 / y # torch.div(x,y) <--- element-wise division
print(z)

x6 = x2[1,1].item() # Gets element in position 1,1... item() gets the actual object
x7 = x2.view(4) # Converts to 1D tensor... can convert size too as long as # of elements is same

a1 = x4.numpy() # Convert tensor to numpy array... be careful though! Both objects share memory location!

a2 = np.ones(5)
x8 = torch.from_numpy(a2) # Convert numpy array to tensor

x9 = torch.ones(5, requires_grad=True) # Requires the calculation of gradient in future steps

#————————————————————

# AUTOGRAD

x = torch.randn(3, requires_grad=True) # Must specifiy requires grad
print(x)

y = x+2
print(y)

z = y*y*2
print(z)

a = z.mean() # turn into a scalar
print(a)

# a.backward() # calculates gradient w respect to x (dz/dx)
# print(x.grad)

gradient_vector = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float64)

z.backward(gradient_vector) # need gradient vector cuz not scalar
print(x.grad)

# How to prevent tracking of gradient?
# 1. x.requires_grad_(False)
# 2. x.detach()
# 3. with torch.no_grad():
with torch.no_grad():
    y_nograd = x + 2
x_nograd = x.detach()
x.requires_grad_(False)

# NOTE: Gradients accumulate!

weights = torch.ones(4, requires_grad=True)

for epoch in range(2):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)

    weights.grad.zero_() # resets grad! Important!

mlp = torchvision.ops.MLP(in_channels=10, hidden_channels=[20,25])

