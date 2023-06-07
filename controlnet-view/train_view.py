from share import *
import sys
sys.path.append(r'/DATA/disk1/cihai/lrz/3d-object-reconstruction/controlnet-view')
import shutil
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from dataset_view import MyDataset
from dataset_view3 import ObjaverseDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

parser = argparse.ArgumentParser(description='train_view')
parser.add_argument('--resume_path', default='./models/control_sd21_view_ini.ckpt', type=str, help='init parameter')
parser.add_argument('--bs', default=8, type=int, help='batch size')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--checkdir', default='model_checkpoint_7', type=str, help='checkpoint save dir')
parser.add_argument('--split', default='train', type=str, help='train sample save dir')
args = parser.parse_args()

# Configs
resume_path = args.resume_path  # './models/control_sd21_view_ini.ckpt'
batch_size = args.bs  # 4
grad_accum = 8
logger_freq = 1000
learning_rate = args.lr  # 1e-4
sd_locked = False
only_mid_control = False

checkpoint_callback = ModelCheckpoint(
    dirpath=args.checkdir,  # 'model_checkpoint_7',
    save_last=True,
)

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
print('preparing model')
model = create_model('./models/cldm_v21_view.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control
print('preparing model done')


# Misc
print('preparing dataset')
# dataset = MyDataset(
#     path="../../../yxd/dataset/co3d",
#     split="train",
#     resolution=512,
#     pairs=batch_size * grad_accum * 2,
#     full_dataset=False,
#     transform="center_crop",
#     kind="car",
#     dropout=0.1
# )

dataset = ObjaverseDataset(
    path="../../../yxd/zero123/zero123/views_whole_sphere",
    pairs=batch_size * grad_accum,
    zero123=True
)

print('preparing dataset done')
dataloader = DataLoader(dataset, num_workers=10, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq, split=args.split)
trainer = pl.Trainer(
    accelerator='gpu',
    devices=1,
    precision=16,
    accumulate_grad_batches=grad_accum,
    callbacks=[logger, checkpoint_callback],
    max_epochs=10000
)


# Train!
trainer.fit(model, dataloader)