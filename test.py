from __future__ import print_function
import torch

x = torch.empty(5, 3) #5x3 matrix
print(x)

x = torch.rand(5, 3) #random 5x3 matrix
print(x)

x =  torch.zeros(5, 3, dtype=torch.long) #0 5x3 matrix type long
print(x)

x = torch.tensor([5.5, 3]) #directly construct tensor
print(x)