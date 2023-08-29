import os
import warnings
warnings.filterwarnings('ignore')
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torch.utils.data import random_split

import pytorch_lightning as pl # Lightning 主套件
from pytorch_lightning.plugins import DDPPlugin # 關掉一些verbose


from torchvision.datasets import MNIST
from torchvision import transforms as trans

import model_mnist



# os.system("rm -rf lightning_logs")

if __name__ == '__main__':

    BATCH_SIZE=128
    EPOCHS=20

    ## ---定義 Model---
    net=nn.Sequential(
                nn.Linear(28 * 28, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
    ## ---指定 loss function, optimizer---
    loss_fn=F.cross_entropy
    opt=partial(torch.optim.Adam,lr=1e-4)

    ## ---放進model class---
    model=model_mnist.Model(net,loss_fn,opt)


    MNIST_train = MNIST("",train=True, download=True, transform=trans.ToTensor())
    MNIST_test = MNIST("",train=False, download=True, transform=trans.ToTensor())
    trainset, valset = random_split(MNIST_train, [55000, 5000])
    train_loader = data.DataLoader(trainset, batch_size=BATCH_SIZE,num_workers=4)
    val_loader = data.DataLoader(valset, batch_size=BATCH_SIZE,num_workers=4)
    test_loader = data.DataLoader(MNIST_test, batch_size=BATCH_SIZE,num_workers=4)

    trainer = pl.Trainer(max_epochs=EPOCHS,gpus=2,
                         accelerator="ddp",
                         plugins=DDPPlugin(find_unused_parameters=False),
                         log_every_n_steps=2)
    
    trainer.fit(model,train_loader, val_loader)
    trainer.test(model,test_loader)