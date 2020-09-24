from __future__ import print_function
import torch
import numpy as np

x = torch.empty(5, 3) #5x3 matrix
print(x)

x = torch.rand(5, 3) #random 5x3 matrix
print(x)

x =  torch.zeros(5, 3, dtype=torch.long) #0 5x3 matrix type long
print(x)

x = torch.tensor([5.5, 3]) #directly construct tensor
print(x)

x = x.new_ones(5, 3, dtype=torch.double) #new_* take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float) #override dtype, random value
print(x)

print(x.size()) #get size

######################

y = torch.rand(5, 3)
print(x + y) #syntax 1
print(torch.add(x, y)) #syntax 2

result = torch.empty(5, 3)
torch.add(x, y, out=result) #output x + y result to result var
print(result)

y.add_(x) #adds x to y, *_ method will do permanent change
print(y)

print(x[:, 1]) #standard numPy indexing

x = torch.randn(4, 4)
y = x.view(16) # resize/reshape tensor
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())

x = torch.randn(1)
print(x)
print(x.item()) #get value as regular number

a = torch.ones(5)
print(a)
b = a.numpy() #b as numpy array
print(b)

a.add_(1)
print(a)
print(b)

a = np.ones(5)
b = torch.from_numpy(a) #convert numpy array to torch tensor
np.add(a, 1, out=a)
print(a)
print(b)

#use CUDA for tensors
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device) #send x to GPU
    z = x + y
    print(z)
    print(z.to("cpu", torch.double)) #send to cpu and change type