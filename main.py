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

# create the classifier model
model_3layer = nn.Sequential(
  # fully connected hidden layer | 8 => 12
  nn.Linear(8, 12),
  # rectified linear unit activation layer (12 neuron)
  nn.ReLU(),
  # fully connected hidden layer | 12 => 8
  nn.Linear(12, 8),
  # rectified linear unit activation layer (8 neuron)
  nn.ReLU(),
  # fully connect hidden layer | 8 => 1
  nn.Linear(8, 1)
  # sigmoid activation layer (1 neuron)
  nn.Sigmoid()
)