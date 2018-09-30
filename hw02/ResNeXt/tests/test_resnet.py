import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import random
import pytest
from torch.utils.data import Dataset, DataLoader
import numpy as np
from resnext import ResNeXt, Trainer

class PartFasionMNIST(Dataset):

    def __init__(self, length):

        transform = transforms.Compose([transforms.Resize(224),
                                        transforms.Grayscale(3), transforms.ToTensor()])
        trainset = datasets.FashionMNIST('.', train=True, download=True, transform=transform)
        self.data = []
        self.len = length
        indxs = np.arange(len(trainset))
        np.random.shuffle(indxs)
        for i in indxs[:length]:
            self.data.append(trainset[i])

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


def test_resnext():
    dataset = PartFasionMNIST(50)
    dataloader = DataLoader(dataset,batch_size=4, shuffle=True, num_workers=2 )
    
    net = ResNeXt([1, 1, 1, 1], num_classes=10)
    trainer = Trainer(dataloader, net, epochs=2)
    train_net = trainer.train()
    
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.Grayscale(3), transforms.ToTensor()])
    testset =  datasets.FashionMNIST('.', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=100, shuffle=True, num_workers=2)
    inputs, labeles = iter(testloader).next()
    outputs = train_net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    
    acc = (predicted == labeles).sum().item() / len(labeles)
    rand = torch.randint(0, 9, (100, ), dtype=torch.int64)
    rand_acc = (rand == labeles).sum().item() / len(labeles)
    assert(acc > rand_acc)
