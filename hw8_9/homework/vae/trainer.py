import logging
import os

import torch
import torch.nn.functional as nn
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import numpy as np

class Trainer:

    def __init__(self, model, train_loader, test_loader, optimizer,
                 loss_function, device='cuda', save_root='vae_data'):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.save_root = save_root
        self.writer = SummaryWriter(save_root)
        

    def train(self, epoch, log_interval=1):
        self.model.train()
        epoch_loss = 0

        for batch_idx, (data, _) in enumerate(self.train_loader):
            # TODO your code here
            
            self.optimizer.zero_grad()
            data = data.to(self.device)
            recon_data, mu, logvar = self.model(data) 
            train_loss = self.loss_function(recon_data, data, mu, logvar)
            epoch_loss += train_loss
            norm_train_loss = train_loss / len(data)

            norm_train_loss.backward()
            self.optimizer.step()
            
            if batch_idx % log_interval == 0:
                msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data),
                    len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    norm_train_loss)
                logging.info(msg)

                batch_size = self.train_loader.batch_size
                train_size = len(self.train_loader.dataset)
                batches_per_epoch_train = train_size // batch_size
                self.writer.add_scalar(tag='data/train_loss',
                                       scalar_value=norm_train_loss,
                                       global_step=batches_per_epoch_train * epoch + batch_idx)

        epoch_loss /= len(self.train_loader.dataset)
        logging.info(f'====> Epoch: {epoch} Average loss: {epoch_loss:.4f}')
        self.writer.add_scalar(tag='data/train_epoch_loss',
                               scalar_value=epoch_loss,
                               global_step=epoch)
        self.save(os.path.join(self.save_root, f'train_loader{epoch}.pt'))

    def test(self, epoch, batch_size, log_interval=1):
        self.model.eval()
        test_epoch_loss = 0

        for batch_idx, (data, _) in enumerate(self.test_loader):
            
            with torch.no_grad():
                data = data.to(self.device)
                recon_data, mu, logvar = self.model(data) 
                test_loss = self.loss_function(recon_data, data, mu, logvar)
                test_epoch_loss += test_loss

            if batch_idx % log_interval == 0:
                msg = 'Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data),
                    len(self.test_loader.dataset),
                    100. * batch_idx / len(self.test_loader),
                    test_loss / len(data))
                logging.info(msg)

                batches_per_epoch_test = len(self.test_loader.dataset) // batch_size
                self.writer.add_scalar(tag='data/test_loss',
                                       scalar_value=test_loss / len(data),
                                       global_step=batches_per_epoch_test * (epoch - 1) + batch_idx)

        test_epoch_loss /= len(self.test_loader.dataset)
        logging.info('====> Test set loss: {:.4f}'.format(test_epoch_loss))
        self.writer.add_scalar(tag='data/test_epoch_loss',
                               scalar_value=test_epoch_loss,
                               global_step=epoch)
        self.plot_generated(epoch)

    def plot_generated(self, epoch, batch_size=64):
        
        with torch.no_grad():
            z =  torch.randn(batch_size, self.model.latent_size, device=self.device)
            gener_x = self.model.decode(z)
            gener_x = gener_x.view(batch_size, 1, 28, 28)
        x = vutils.make_grid(gener_x, normalize=True, scale_each=True)
        self.writer.add_image('img/generate', x, epoch)
        
        iter_loder = iter(self.test_loader)
        data, labels = iter_loder.next()
        data = data.to(self.device)
        noise = torch.randn(data.size()).to(self.device) * 0.07
        data_noise = data + noise
        with torch.no_grad():
            embed = self.model.embed(data)
            recon_x = self.model.decode(embed)
            recon_x = recon_x.view(batch_size, 1, 28, 28)
            noise_embed = self.model.embed(data_noise)
            denoise_x = self.model.decode(embed)
            denoise_x = denoise_x.view(batch_size, 1, 28, 28)
        y = vutils.make_grid(recon_x[:batch_size, :, :, : ], normalize=True, scale_each=True)
        self.writer.add_image('img/reconstraction', y, epoch) 
        z = vutils.make_grid(data_noise[:batch_size, :, :, : ], normalize=True, scale_each=True)
        self.writer.add_image('img/noise', z, epoch)
        t = vutils.make_grid(denoise_x[:batch_size, :, :, : ], normalize=True, scale_each=True)
        self.writer.add_image('img/denoise', t, epoch)   

    def save(self, checkpoint_path):
        dir_name = os.path.dirname(checkpoint_path)
        os.makedirs(dir_name, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)
