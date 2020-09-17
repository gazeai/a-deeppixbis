import copy
import os
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn


class Trainer(object):

    def __init__(self, network, optimizer, compute_loss, learning_rate=0.0001, batch_size=64,
                 device='cuda:0', do_crossvalidation=False, save_interval=2, save_path=''):

        self.network = network
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.compute_loss = compute_loss
        self.device = device
        self.learning_rate = learning_rate
        self.save_interval = save_interval
        self.save_path = save_path

        self.do_crossvalidation = do_crossvalidation

        if self.do_crossvalidation:
            phases = ['train', 'val']
        else:
            phases = ['train']
        self.phases = phases

        self.network.to(self.device)

    def load_model(self, model_filename):

        cp = torch.load(model_filename)
        self.network.load_state_dict(cp['state_dict'])
        start_epoch = cp['epoch']
        start_iter = cp['iteration']
        losses = cp['loss']
        return start_epoch, start_iter, losses

    def save_model(self, output_dir, epoch=0, iteration=0, losses=None):

        saved_filename = 'model_{}_{}.pth'.format(epoch, iteration)
        saved_path = os.path.join(output_dir, saved_filename)
        cp = {'epoch': epoch,
              'iteration': iteration,
              'loss': losses,
              'state_dict': self.network.cpu().state_dict()
              }
        torch.save(cp, saved_path)

        print(f"INFO: MODEL SAVED AT {saved_path}")
        self.network.to(self.device)

    def train(self, dataloader, n_epochs=25, output_dir='out', model=None, freeze=False):

        # if model exists, load it
        if model is not None:
            start_epoch, start_iter, losses = self.load_model(model)
            print(
                f'STARTING TRAINING AT  {start_epoch}, ITERATION {start_iter} - LAST LOSS VALUE IS {losses[-1]}')
        else:
            start_epoch = 0
            start_iter = 0
            losses = []
            print('STARTING TRAINING FROM SCRACTH')

        for name, param in self.network.named_parameters():
            if param.requires_grad:
                print(f'LAYER TO BE ADAPTED FROM GRAD CHECK : {name}')

        # setup optimizer
        self.network.train(True)
        best_model_wts = copy.deepcopy(self.network.state_dict())
        best_loss = float("inf")
        if freeze:
            self.network.base_line.backbone.requires_grad = False
            self.network.base_line.enc.requires_grad = False
            print(f'DENSENET BLOCK FORZEN')

        # STARTING TRAINING LOOP
        for epoch in tqdm(range(start_epoch, n_epochs)):
            if epoch == 4 and freeze:
                self.network.base_line.backbone.requires_grad = True
                self.network.base_line.enc.requires_grad = True
                freeze = False
                print(f'DENSENET BLOCK UNFREEZED')

            train_loss_history = []
            val_loss_history = []

            for phase in self.phases:
                # Set model to training mode
                if phase == 'train':
                    self.network.train()
                # Set model to evaluate mode
                else:
                    self.network.eval()

                for i, data in enumerate(dataloader[phase], 0):
                    if i >= start_iter:
                        start = time.time()
                        img, labels = data
                        img = img['image']

                        self.optimizer.zero_grad()
                        with torch.set_grad_enabled(phase == 'train'):
                            loss = self.compute_loss(
                                self.network, img, labels, self.device)
                            if phase == 'train':
                                loss.backward()
                                self.optimizer.step()
                                train_loss_history.append(loss.item())
                            else:
                                val_loss_history.append(loss.item())
                        end = time.time()

                        print(
                            f"[{epoch}/{n_epochs}][{i}/{len(dataloader[phase])}] => LOSS: {loss.item()} (ELAPSED TIME: {(end - start)}), PHASE: {phase}")
                        losses.append(loss.item())
            epoch_train_loss = np.mean(train_loss_history)

            print(f"TRAIN LOSS: {epoch_train_loss}  EPOCH: {epoch}")

            if self.do_crossvalidation:
                epoch_val_loss = np.mean(val_loss_history)
                print(f"VAL LOSS: {epoch_val_loss}  EPOCH: {epoch}")

                if phase == 'val' and epoch_val_loss < best_loss:
                    best_loss = epoch_val_loss
                    print(
                        f"VAL LOSS IMPROVED FROM: {epoch_val_loss} TO: {best_loss}, COPYING OVER NEW SWEIGHTS")
                    best_model_wts = copy.deepcopy(self.network.state_dict())

            print(f"EPOCH {epoch + 1} DONE")

            # save the last model, and the ones in the specified interval
            if (epoch + 1) == n_epochs or epoch % self.save_interval == 0:
                self.save_model(output_dir, epoch=(epoch + 1),
                                iteration=0, losses=losses)

        self.network.load_state_dict(best_model_wts)
        torch.save(self.network, self.save_path)
        self.save_model(output_dir, epoch=0, iteration=0, losses=losses)
