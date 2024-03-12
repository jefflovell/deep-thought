import datetime
import timeit
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

#logging
start_time = datetime.datetime.now()

#load dataset
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter = ',')

#split data to features (inputs) and label (outputs)
#    y = fn(X)
features = dataset[:,0:8]
label = dataset[:,8]

#convert numpy matrices to 32bit float tensors
#   to avoid implicit conversion issues
features = torch.tensor(features, dtype = torch.float32)
label = torch.tensor(label, dtype = torch.float32).reshape(-1, 1)

#create the classifier model
model_3layer = nn.Sequential(
    #fully connected hidden layer | 8 => 12
    nn.Linear(8, 12),
    #rectified linear unit activation layer (12 neuron)
    nn.ReLU(),
    #fully connected hidden layer | 12 => 8
    nn.Linear(12, 12),
    #rectified linear unit activation layer (8 neuron)
    nn.ReLU(),
    #fully connected hidden layer | 12 => 8
    nn.Linear(12, 8),
    #rectified linear unit activation layer (8 neuron)
    nn.ReLU(),
    #fully connect hidden layer | 8 => 1
    nn.Linear(8, 1),
    #sigmoid activation layer (1 neuron)
    nn.Sigmoid()
)

#select loss function & optimizer algo
#   binary cross entropy
loss_function = nn.BCELoss()
#   Adam (Adaptive Moment Estimation) optimizer
#   with the default learning rate of 0.001
#   This is gradient descent with momentum
optimizer_algorithm = optim.Adam(model_3layer.parameters(), lr=0.001)

#train the model
number_of_epochs = 1000
batch_size = 10

for epoch in range(number_of_epochs):
    for i in range(0, len(features), batch_size):
        batch_start = datetime.datetime.now()
        features_batch = features[i:i + batch_size]
        label_prediction = model_3layer(features_batch)
        label_batch = label[i:i + batch_size]
        loss = loss_function(label_prediction, label_batch)
        optimizer_algorithm.zero_grad()
        loss.backward()
        optimizer_algorithm.step()
        batch_finish = datetime.datetime.now()
        with open("model_3layer_training.log", "a") as log:
            log.write(f"Starting epoch {epoch}, batch {i} at {datetime.datetime.now()}\n\tFinished batch {i} of {batch_size * number_of_epochs} at time: {batch_finish}")
    with open("model_3layer_training.log", "a") as log:
        log.write(f"Starting epoch {epoch} of {number_of_epochs} at {datetime.datetime.now()}\n\tFinished epoch {epoch}, latest loss {loss}")
    print(f"Current Time: {datetime.datetime.now()} | Finished epoch {epoch}, latest loss {loss}")

print(f"Finished training {number_of_epochs} epochs at {datetime.datetime.now()} | latest loss {loss}\nCheck logs for more info.\nGoodbye.")


