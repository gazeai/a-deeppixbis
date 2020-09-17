import os
import torch
import torch.nn as nn
import torch.optim as optim

from config.config import config as cfg
from losses.loss import compute_loss
from models.deeppix import DeepPixBiS
from trainers.deeppix_trainer import Trainer
from dataloaders.deeppixdataloader import get_dataloaders

# Specify other training parameters
batch_size = 64
num_workers = 4
epochs = 5
learning_rate = 0.001
weight_decay = 0.00001
save_interval = 1
seed = 42
use_gpu = True
output_dir = cfg.save_dir
save_path = f'{output_dir}/best.pt'
os.makedirs(output_dir, exist_ok=True)

print(f"CONFIGUATION: {cfg}")

model = DeepPixBiS()
# set trainable parameters
for name, param in model.named_parameters():
    param.requires_grad = True

# optimizer initialization
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate,
                       weight_decay=weight_decay)

trainer = Trainer(model, optimizer, compute_loss, learning_rate=learning_rate, batch_size=batch_size,
                  device='cuda:2' if torch.cuda.is_available() else 'cpu', do_crossvalidation=True,
                  save_interval=save_interval, save_path=save_path)

print("Data loading started!....")
dataloader = get_dataloaders(data_type="mobile", color_mode=['rgb'], batch_size=batch_size, hard_protocol='proto3')
print("Data Loaded!")
trainer.train(dataloader, n_epochs=epochs, output_dir=output_dir)
