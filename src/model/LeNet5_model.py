from __future__ import absolute_import, division, print_function
# from __future__ import nested_scopes, generators, with_statement
# from __future__ import unicode_literals, generator_stop, annotations

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class LeNet5Model(nn.Module) :
    def __init__(self) -> None:
        super(LeNet5Model, self).__init__()
        self.model = nn.Sequential(
                        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5), stride=1, padding=0, dilation=1, bias=True),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0, dilation=1),
                        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5), stride=1, padding=0, dilation=1, bias=True),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0, dilation=1),
                        nn.Flatten(start_dim=1, end_dim=-1),
                        nn.Linear(in_features=16*5*5, out_features=120, bias=True),
                        nn.ReLU(),
                        nn.Linear(in_features=120, out_features=84, bias=True),
                        nn.ReLU(),
                        nn.Linear(in_features=84, out_features=10, bias=True)
                        )
    
    def forward(self, input):
        modelRet = self.model(input)
        return modelRet
    
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-- Using {device} device")
    model = LeNet5Model().to(device)
    print(model)
    print("Model's state_dict:") # Print model's state_dict
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())