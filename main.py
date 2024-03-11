import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# load dataset
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter = ',')

# split data to inputs (X) and output (y)
inputs = dataset[:,0:8]
output = dataset[:,8]

# convert numpy matrices to 32bit float tensors
# to avoid implicit conversion issues
inputs = torch.tensor(inputs, dtype = torch.float32)
output = torch.tensor(output, dtype = torch.float32).reshape(-1, 1)

print(output.shape)