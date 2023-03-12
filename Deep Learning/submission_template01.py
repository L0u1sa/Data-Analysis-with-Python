import numpy as np
import torch
from torch import nn

def create_model():
    NN = nn.Sequential(nn.Linear(784, 256),
                   nn.ReLU(),
                   nn.Linear(256, 16),
                   nn.ReLU(),
                   nn.Linear(16, 10))
    return NN

def count_parameters(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    return pytorch_total_params
    

