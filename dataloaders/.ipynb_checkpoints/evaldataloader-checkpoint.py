import torch
import numpy as np
from random import shuffle
import cv2
from torchvision import transforms
import glob
import os
from torch.utils import data

data_type = "oulu"


class DeepPixDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform, color_mode=['rgb']):
        self.path = path
        self.transform = transform
        self.color_mode = color_mode
        print(self.path[0])

    def __getitem__(self, idx):
        im_path = self.path[idx]

        label = self.path[idx].split("/")[-2]

        if label == "spoof" or "device" in label or "print" in label or "video-replay" in label:
            lab_14 = torch.tensor(np.zeros((14, 14)))
            label = {"pixel_mask": lab_14, "binary_target": torch.tensor(0)}
        elif label == "live" or label == "real":
            lab_14 = torch.tensor(np.ones((14, 14)))
            label = {"pixel_mask": lab_14, "binary_target": torch.tensor(1)}
        img = cv2.imread(im_path)

        images = {}
        if len(self.color_mode) > 1:
            for mode in self.color_mode:
                im = cv2.cvtColor(img, eval(f"cv2.COLOR_BGR2{mode.upper()}"))
                im = cv2.resize(im, (224, 224))
                im = self.transform(im)
                images[mode] = im
        else:
            img = cv2.cvtColor(img, eval(f"cv2.COLOR_BGR2{self.color_mode[0].upper()}"))
            img = cv2.resize(img, (224, 224))
            img = self.transform(img)
            images['image'] = img

        return images, label

    def __len__(self):
        return len(self.path)

# def oulu(config, split='Dev'):
#     oulu_path = config.oulu_data_path
#     sessions = config.proto3[f'{split.lower()}'][config.combination]['sess']
#     phones = config.proto3[f'{split.lower()}'][config.combination]['phone']
#     imgs  = []
    
#     dirs = [f"{oulu_path}/{split}/phones/{phone}/sessions/{sess}" for sess in sessions for phone in phones]
#     print(dirs)
#     for d in dirs:
#         imgs.extend(glob.glob(f"{d}/**/*"))
    
#     print(len(imgs))
#     return imgs


def oulu(config, hard_protocol=None, split='Dev'):
    oulu_path = "/home/jupyter/datasets/spoof_data/cropped_faces/oulu/NO_ALIGNMENT/"

    imgs = glob.glob(f"{oulu_path}{hard_protocol}/{split}/**/**")
    print(len(imgs))
    return imgs


def mobile(config, split='Dev'):
    oulu_path = config.oulu_data_path
    sessions = config.proto3[f'{split.lower()}'][config.combination]['sess']
    phones = config.proto3[f'{split.lower()}'][config.combination]['phone']
    imgs  = []
    
    dirs = [f"{oulu_path}/{split}/phones/{phone}/sessions/{sess}" for sess in sessions for phone in phones]
    print(dirs)
    for d in dirs:
        imgs.extend(glob.glob(f"{d}/**/*"))
    
    print(len(imgs))
    return imgs


def get_dataloaders(config, data_type="oulu", batch_size=128, hard_protocol=None, split='Dev'):
    get_data = globals().get(data_type)
    im_data = get_data(config, hard_protocol=hard_protocol, split=split)    
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset =  DeepPixDataset(im_data, tfm, ['rgb'])
    dataloder = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=64)            
    
    return dataloder
