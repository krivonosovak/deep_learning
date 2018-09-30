
# ResNeXt Pytorch Implementation


[ResNext](https://arxiv.org/abs/1611.05431) - Aggregated Residual Transformations for Deep Neural Networks 

* uses the pytorch implementation of [ResNet](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
* test on small random subset of FashionMNIST
	

# Usage

	dataset = ...
	dataloader = DataLoader(dataset,batch_size=4, shuffle=True, num_workers=2 )
    
   	net = ResNeXt([1, 1, 1, 1], num_classes=10)
   	trainer = Trainer(dataloader, net, epochs=2,  log_dir='./logs')
   	train_net = trainer.train()