import argparse
import logging
import os

import torch
import torchvision.datasets as datasets
from torch.optim import Adam
from torchvision import transforms

from vae.vae import VAE, loss_function
from vae.trainer import Trainer
from tensorboardX import SummaryWriter



def get_config():
    parser = argparse.ArgumentParser(description='Training DCGAN on CIFAR10')

    parser.add_argument('--log-root', type=str, default='../logs')
    parser.add_argument('--data-root', type=str, default='data')
    parser.add_argument('--log-name', type=str, default='train_vae.log')
    parser.add_argument('--latent_size', type=int, default=20)
    parser.add_argument('--train_batch-size', type=int, default=32,
                        help='input batch size for training')
    parser.add_argument('--test_batch-size', type=int, default=64,
                        help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train ')
    parser.add_argument('--image-size', type=int, default=28,
                        help='size of images to generate')
    parser.add_argument('--log_interval', type=int, default=30,
                                                help='size of images to generate')
    config = parser.parse_args()
#    config.cuda = not config.no_cuda and torch.cuda.is_available()

    return config


def main():
    config = get_config()
    logging.basicConfig(
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.log_root,
                                             config.log_name)),
            logging.StreamHandler()],
        level=logging.INFO)

    transform = transforms.Compose([transforms.Resize(config.image_size), transforms.ToTensor()])
    train_dataset = datasets.FashionMNIST(root=config.data_root, train=True, download=True,
                               transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                             num_workers=4, pin_memory=True)
    
    test_dataset = datasets.FashionMNIST(root=config.data_root, train=False, download=True,
                                                             transform=transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=True,
                                                                                         num_workers=4, pin_memory=True)

    
    model = VAE(image_size=config.image_size, )
    optimazer = Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    trainer = Trainer(model, train_dataloader, test_dataloader,  optimazer, loss_function, device='cpu')


    for epoch in range(config.epochs):
        trainer.train(epoch)
        trainer.test(epoch, config.test_batch_size)
        
    test_dataset = datasets.FashionMNIST(root=config.data_root, train=False, download=True,
                                                             transform=transform)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=5000, shuffle=True,
                                                                                         num_workers=4, pin_memory=True)
    
    writer = SummaryWriter('vae_data')
    data_iter = iter(test_dataloader)
    data, labels = data_iter.next()
    embed = model.embed(data)
    writer.add_embedding(embed, labels, data, config.epochs)


if __name__ == '__main__':
    main()
