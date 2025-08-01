import torch
import torch.nn as nn

class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FullyConnectedNetwork, self).__init__()
        
        # Create a list to hold the layers
        layers = []
        
        # Add the input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        
        # Add the hidden layers
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(nn.ReLU())
        
        # Add the output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        layers.append(nn.Softmax(dim=1))  # Use 'nn.Sigmoid()' if binary classification
        
        # Combine all layers into a sequential module
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

