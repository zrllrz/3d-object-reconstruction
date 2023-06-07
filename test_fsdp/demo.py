import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.distributed.fsdp.wrap import wrap


class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(32, 32)
        self.block = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 32))

    def configure_sharded_model(self):
        # modules are sharded across processes
        # as soon as they are wrapped with `wrap`.
        # During the forward/backward passes, weights get synced across processes
        # and de-allocated once computation is complete, saving memory.

        # Wraps the layer in a Fully Sharded Wrapper automatically
        linear_layer = wrap(self.linear_layer)

        for i, layer in enumerate(self.block):
            self.block[i] = wrap(layer)

        self.model = nn.Sequential(linear_layer, nn.ReLU(), self.block)

    def training_step(self, batch, batch_idx):
        x = self.model(torch.zeros(4, 32))
        return torch.mean(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters())


model = MyModel()
trainer = Trainer(accelerator="gpu", devices=7, strategy="fsdp", precision=32)
trainer.fit(model)
