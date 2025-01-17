import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.distributed.fsdp.wrap import wrap

from pytorch_lightning import LightningModule, Trainer


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.frozen_layer = torch.nn.Linear(32, 32)
        self.layer = torch.nn.Linear(32, 2)
        for param in self.frozen_layer.parameters():
            param.requires_grad = False

    def configure_sharded_model(self):
        self.frozen_layer = wrap(self.frozen_layer)
        self.layer = wrap(self.layer)

    def forward(self, x):
        # for param in self.frozen_layer.parameters():
        #     param.requires_grad = False
        x = self.frozen_layer(x)
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        print()
        loss = self(batch).sum()
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


def run():
    train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    val_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    test_data = DataLoader(RandomDataset(32, 64), batch_size=2)

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        num_sanity_val_steps=0,
        max_epochs=1,
        accelerator='gpu',
        enable_model_summary=False,
        precision=16,
        strategy='fsdp'
    )
    print(model.device)
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)
    # trainer.test(model, dataloaders=test_data)


if __name__ == "__main__":
    run()
