# -*- coding: utf-8 -*-

import numpy as np

import random

import torch
from torchvision import datasets
from torchvision import transforms

def get_train_loader(data_dir,
                     batch_size,
                     random_seed,
                     shuffle=True,
                     num_workers=1,
                     pin_memory=True):

    # define transforms
    trans = transforms.Compose([
        transforms.Resize(size=(256,256)),
        transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
        transforms.Normalize([0, 0, 0], [255, 255, 255])
    ])

    # load dataset
    dataset = datasets.ImageFolder(root=data_dir,
                                transform=trans)
    
    if shuffle:
        np.random.seed(random_seed)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader

def get_test_loader(data_dir,
                    batch_size,
                    num_workers=1,
                    pin_memory=True):
    # define transforms
    trans = transforms.Compose([
        transforms.Resize(size=(256,256)),
        transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
        transforms.Normalize([0, 0, 0], [255, 255, 255]),
    ])

    # load dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=trans)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader

def get_data_loader(data_dir,
                     batch_size,
                     random_seed,
                     shuffle=True,
                     num_workers=1,
                     pin_memory=True):

    # define transforms
    trans = transforms.Compose([
        transforms.Resize(size=(256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0, 0, 0], [255, 255, 255]),
    ])

    # load dataset
    dataset = datasets.ImageFolder(root=data_dir,
                                transform=trans)

    train_size = int(len(dataset) * 0.8)
    valid_size = len(dataset) - train_size

    train_set, valid_set = torch.utils.data.random_split(dataset, [train_size, valid_size], torch.Generator().manual_seed(random.randint(0,100)))
   
    if shuffle:
        np.random.seed(random_seed)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory,
    )

    # the batch_size for testing is set as the valid_size
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, valid_loader

def get_dataset(data_dir):
    # define transforms
    trans = transforms.Compose([
        transforms.Resize(size=(256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0, 0, 0], [255, 255, 255]),
    ])

    # load dataset
    dataset = datasets.ImageFolder(root=data_dir,
                                transform=trans)

    return dataset