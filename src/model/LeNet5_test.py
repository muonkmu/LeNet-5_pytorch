from __future__ import absolute_import, division, print_function
from __future__ import nested_scopes, generators, with_statement
from __future__ import unicode_literals, generator_stop, annotations

import os
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
import torchvision.transforms as transforms
from LeNet5_model import LeNet5Model

def GetTrainData(batch_size) :
    mnistTrans=transforms.Compose([transforms.Resize((32,32)),
                                   transforms.Grayscale(num_output_channels=1),
                                   transforms.ToTensor()])
    testData = mnist.MNIST(root='dataset.test', train=False, download=True, transform=mnistTrans)
    testnDataLoader = DataLoader(testData, batch_size=batch_size)
    return testnDataLoader

def BuildTestModel(device):
    model=LeNet5Model().to(device)
    if not (os.path.isfile('./model/LeNet5_weight.pth')) :
      print('Model weight is not exist!!')
      os.exit(0)
    model.load_state_dict(torch.load('./model/LeNet5_weight.pth'))
    model.eval()
    return  model

def DoTest(dataloader, model):
  correct = 0
  total = 0
  with torch.no_grad() :
    for inputImgae, labels in dataloader :
      outputs = model(inputImgae)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  return correct, total


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-- Using {device} device")
    batchSize = 4
    dataloader = GetTrainData(batchSize)
    model = BuildTestModel(device)
    correct, total = DoTest(dataloader, model)
    print(f"Test result : toal={total} correct={correct} accuracy={100 * correct // total}%")
