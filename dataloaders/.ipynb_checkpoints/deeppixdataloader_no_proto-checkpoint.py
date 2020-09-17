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
                im = self.transform[mode](im)
                images[mode] = im
        else:
            img = cv2.cvtColor(img, eval(f"cv2.COLOR_BGR2{self.color_mode[0].upper()}"))
            img = cv2.resize(img, (224, 224))
            img = self.transform[self.color_mode[0]](img)
            images['image'] = img

        return images, label

    def __len__(self):
        return len(self.path)
    

def get_real_spoof_split(path):
    real_files = glob.glob(f"{path}/real/*")
    spoof_dirs = [folder for folder in os.listdir(path) if "real" not in folder]
    divide = int(len(real_files)/len(spoof_dirs))
    spoof_files = []
    
    for dir in spoof_dirs:
        files = glob.glob(f"{path}/{dir}/*")
        shuffle(files)
        spoof_files.extend(files[:divide])
        
    return real_files, spoof_files


def oulu(config, hard_protocol=None):
    
    sessions = config.proto3['train'][config.combination]['sess']
    phones = config.proto3['train'][config.combination]['phone']
    oulu_path = config.oulu_data_path
    train_imgs = []
    val_imgs = []
    
    print(f'SESSION: {sessions}, PHONES: {phones}')
    train_dir = [f"{oulu_path}/Train/phones/{phone}/sessions/{sess}" for sess in sessions for phone in phones]
    val_dir = [f"{oulu_path}/Dev/phones/{phone}/sessions/{sess}" for sess in sessions for phone in phones]

    for t_dir, v_dir in zip(train_dir, val_dir):
        train_real, train_spoof = get_real_spoof_split(t_dir)
        val_real, val_spoof = get_real_spoof_split(v_dir)
        print(f"TRAIN SPLIT: real->{len(train_real)} spoof->{len(train_spoof)}")
        print(f"VAL SPLIT: real->{len(val_real)} spoof->{len(val_spoof)}")
        
        train_imgs.extend(train_real + train_spoof)
        val_imgs.extend(val_real + val_spoof)
    
    print(f"TRAIN: {len(train_imgs)}, VAL: {len(val_imgs)}")
    return {"train": train_imgs, "val": val_imgs}


def msu(hardproto=None):
    train_path = "/home/jupyter/datasets/spoof_data/cropped_faces/msu/Train"
    val_path = "/home/jupyter/datasets/spoof_data/cropped_faces/msu/Test"
    train_imgs = glob.glob(f"{train_path}/**/*")
    val_imgs = glob.glob(f"{val_path}/**/*")
    return {"train": train_imgs, "val": val_imgs}


def gaze(hardproto=None):
    train_path = "/home/jupyter/datasets/spoof_data/cropped_faces/msu/Train"
    val_path = "/home/jupyter/datasets/spoof_data/cropped_faces/msu/Test"
    train_imgs = glob.glob(f"{train_path}/**/*.png")
    val_imgs = glob.glob(f"{val_path}/**/*.png")
    
    gaze_train = glob.glob("/home/jupyter/datasets/spoof_data/cropped_faces/front_cam/train/**/**/*")
    gaze_val = glob.glob("/home/jupyter/datasets/spoof_data/cropped_faces/front_cam/val/**/**/*")
    
    train_imgs.extend(gaze_train)
    val_imgs.extend(gaze_val)
    
    shuffle(train_imgs)
    shuffle(val_imgs)
    
    print(f"TRAIN SPLIT->{len(train_imgs)} ")
    print(f"VAL SPLIT->{len(val_imgs)} ")
    
    return {"train": train_imgs, "val": val_imgs}

def get_transforms(color_mode):
    rgb_train_transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.0),
         transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])  #

    rgb_val_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    hsv_train_transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor()])  #

    hsv_val_transform = transforms.Compose(
        [transforms.ToTensor()])

    ycr_cb_train_transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor()])  #

    ycr_cb_val_transform = transforms.Compose(
        [transforms.ToTensor()])

    transformations = {}
    for phase in ['train', 'val']:
        transformations[phase] = {}
        for mode in color_mode:
            transformations[phase][mode] = eval(f"{mode}_{phase}_transform")

    return transformations


def get_dataloaders(config, data_type="oulu", color_mode=['rgb'], batch_size=32, hard_protocol=None):
    transformations = get_transforms(color_mode)
    get_data = globals().get(data_type)
    im_data = get_data(config, hard_protocol)

    phases = ["train", "val"]
    dataloaders = {}
    for phase in phases:
        dataset = DeepPixDataset(im_data[phase], transformations[phase], color_mode)
        dataloaders[phase] = data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=64)
    # test_dataset = DeepPixDataset("dataset", val_transform)

    return dataloaders
