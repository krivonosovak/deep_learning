import torch.optim as optim
from tensorboard import SummaryWriter
import torch.nn as nn


class Trainer:
    def __init__(self, data, NetWork, lr=0.001, epochs=2, log_dir='./logs'):
        self.net = NetWork
        self.data = data
        self.lr = lr
        self.epochs = epochs
        self.log_dir = log_dir

    def train(self):
        logger = SummaryWriter(self.log_dir)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        num_iter = len(self.data)
        for epoch in range(self.epochs):
            for i, batch in enumerate(self.data):
                inputs, labels = batch
                optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = loss_function(outputs, labels)
                logger.add_scalar('loss epoch ' + str(epoch), loss.item(), i)
                loss.backward()
                optimizer.step()
        return self.net
