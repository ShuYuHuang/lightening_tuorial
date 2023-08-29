import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

from torch.utils.tensorboard import SummaryWriter

## ---最簡單的Lightning Module---
class Model(pl.LightningModule):
    # ---初始化---
    def __init__(self,net,loss_fn=F.cross_entropy,optimizer=torch.optim.Adam):
        super().__init__()
        # ---至少要放一個網路(把結構放在這邊，Sequential或者layers都可以)---
        self.net=net
        
        self.loss_fn=loss_fn
        self.optimizer=optimizer
        
    # ---網路走法(如果有skipping, concatenate, 還有tensor操作在這邊)---    
    def forward(self,x):
        x=x.view(x.shape[0],-1)
        return self.net(x)
    
    # ---Optimizer放這邊---
    def configure_optimizers(self):
        return self.optimizer(self.parameters())
    
    # ---每個training batch要做的事---    
    def training_step(self, batch, batch_idx,flag="train"):
        # ---分開data, label---
        x, y = batch
        
        # ---data進網路---
        y_hat = self(x)
        
        # ---prediction跟y算loss,acc---
        loss = F.cross_entropy(y_hat, y) # cross_entropy裡面包含softmax計算
        acc=accuracy(y_hat, y)
        
        # ---記錄到進度條---
        self.log_dict({"loss_"+flag:loss,"acc_"+flag:acc},
                      prog_bar=True,
                      logger=False)
        
        return {"loss":loss,"acc":acc}
    
    # ---記錄到logger---
    def training_epoch_end(self,outputs,flag="train"):
        tmp={}
        for arg in outputs[0]:
            arg_mean=torch.tensor([x[arg] for x in outputs]).mean()
            tmp[f"exp/{arg}"]={f"{arg}_{flag}":arg_mean}
        self.log_dict(tmp,prog_bar=False,logger=True)
    
    # ---重複利用step function---
    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx,flag="val")
        
    def validation_epoch_end(self, outputs):
        self.training_epoch_end(outputs,flag="val")
        
    def test_step(self, batch, batch_idx):
        self.training_step(batch, batch_idx,flag="test")

    