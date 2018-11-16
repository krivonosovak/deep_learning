import torch.nn as nn
import math
import torch

class DCGenerator(nn.Module):

    def __init__(self, image_size):
        super(DCGenerator, self).__init__()
        n = int(math.log2(image_size)) - 2
        accum = 2**(n - 1)
        self.layers = torch.nn.Sequential()
        
        for i in range(n):
            if i == 0:
                deconv = nn.ConvTranspose2d(100,  128 * accum, kernel_size=4, stride=1, padding=0)
            else:
                deconv = nn.ConvTranspose2d(128 * accum * 2, 128 * accum, kernel_size=4, stride=2, padding=1)
            
            self.layers.add_module('deconv' + str(i + 1), deconv)
            self.layers.add_module('bn' + str(i + 1), nn.BatchNorm2d(128 * accum))
            self.layers.add_module('activation' + str(i + 1), nn.ReLU())
            accum //= 2
            
        # output layers
        self.layers.add_module('output', nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1))
        self.layers.add_module('activation' + str(n), nn.Tanh())
            
    def forward(self, data):
        out = self.layers(data)
        return out


class DCDiscriminator(nn.Module):

    def __init__(self, image_size):
        super(DCDiscriminator, self).__init__()
        n = int(math.log2(image_size)) - 2
        self.layers = torch.nn.Sequential()
        accum = 1
        
        # input layer 
        self.layers.add_module("input", nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1) )
        self.layers.add_module('activation0', nn.LeakyReLU(0.2)) 
        
        # hiden layers
        for i in range(n - 1):
            self.layers.add_module('conv' + str(i + 1), nn.Conv2d(128 * accum, 128 * accum * 2,
                                                                                kernel_size=4, stride=2, padding=1))
            self.layers.add_module('bn' + str(i + 1), nn.BatchNorm2d(128 * accum * 2))
            self.layers.add_module('activation' + str(i + 1), nn.LeakyReLU(0.2))
            accum *= 2
        
        # output layers
        self.layers.add_module('output', nn.Conv2d(128 * accum, 1,
                                                     kernel_size=4, stride=1, padding=0)) 
        self.layers.add_module('activation' + str(n), nn.Sigmoid())
        
    def forward(self, data):
        out = self.layers(data)
        return out.squeeze()
