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
    trainingData = mnist.MNIST(root='dataset.train', train=True, download=True, transform=mnistTrans)
    trainDataLoader = DataLoader(trainingData, batch_size=batch_size)
    return trainDataLoader

def BuildTrainModel(device):
    model=LeNet5Model().to(device)
    model.train() # set the mode to train
    loss_Fn = nn.CrossEntropyLoss();
    optimzer= optim.SGD(model.parameters(), lr=1e-3)
    return  model, loss_Fn, optimzer

# def DoTrain(dataloader, model, loss_fn, optimizer, device):
def DoTrain(dataloader, model, loss_fn, optimizer):
    for batch, (inputImgae, label) in enumerate(dataloader):
        inputImgae, label = inputImgae.to(device), label.to(device)
        pred = model(inputImgae.float())
        loss = loss_fn(pred, label.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()

def SaveModel(model) :
    try:
        if not os.path.exists('./model'):
            os.makedirs('./model')
    except OSError:
        print("Error: Failed to create the directory.")
    torch.save(model.state_dict(), './model/LeNet5_weight.pth')


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-- Using {device} device")
    batchSize = 16
    dataloader = GetTrainData(batchSize)
    model, loss_Fn, optimizer = BuildTrainModel(device)
    loss = 1
    epochs = 0
    while (loss > 0.015) :
        epochs = epochs + 1
        print(f"Epoch {epochs} : processing........", end="", flush=True)
        start_time = datetime.now()
        loss = DoTrain(dataloader, model, loss_Fn, optimizer)
        elaps_time = datetime.now() - start_time
        print(f"\rEpoch {epochs} : loss={loss:>7f}, elapsed={elaps_time}")
    SaveModel(model)
