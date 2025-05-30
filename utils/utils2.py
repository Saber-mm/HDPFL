#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch.utils import data
from torchvision import datasets, transforms
from utils.sampling import iid, noniid, mnist_noniid_unequal, iid_even_odd
import numpy as np
import time
from torch.utils.data import DataLoader, Dataset




class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)
    
   




def get_dataset(dataset='MNIST', num_users=20, iidness=True, unequal=False, user_max_class=10):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if dataset == 'CIFAR10':
        data_dir = '../data/CIFAR10/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if iidness:
            # Sample IID user data from Mnist
            user_groups = iid(train_dataset, num_users)
        else:
            # Sample Non-IID user data from Mnist
            if unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = noniid(train_dataset, num_users, user_max_class)

    elif dataset == 'MNIST' or dataset == 'FMNIST':
        
        if dataset == 'MNIST':
            data_dir = '../data/MNIST/'
            
            apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
            
            train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)
            test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)
            
            
        else:
            data_dir = '../data/FMNIST/'
            
            apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
            
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=apply_transform)
            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=apply_transform)

       

        # sample training data amongst users
        if iidness:
            # Sample IID user data from Mnist
            user_groups = iid(train_dataset, num_users)
        else:
            # Sample Non-IID user data from Mnist
            if unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, num_users)
            else:
                # Chose euqal splits for every user
                user_groups = noniid(train_dataset, num_users, user_max_class)
    
    return train_dataset, test_dataset, user_groups



def get_dataset_CIFAR(dataset='CIFAR10', num_users=20, iidness=True, unequal=False, user_max_class=10):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if dataset == 'CIFAR10':
        data_dir = '../data/CIFAR10/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)
        if iidness:
            # Sample IID data from CIFAR10 for each user
            user_groups_train = iid(train_dataset, num_users)
            user_groups_test = iid(test_dataset, num_users)
        else:
            # Sample non-IID data from CIFAR10 for each user
            if unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = noniid(train_dataset, num_users, user_max_class)

        return train_dataset, test_dataset, user_groups_train, user_groups_test
    
    elif dataset == 'CIFAR100':
        data_dir = '../data/CIFAR100/'
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transforms.ToTensor())
        if iidness:
            # Sample IID data from CIFAR100 for each user
            user_groups_train = iid(train_dataset, num_users)
            user_groups_test = iid(test_dataset, num_users)
        else:
            # Sample non-IID data from CIFAR100 for each user
            if unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = noniid(train_dataset, num_users, user_max_class)

        return train_dataset, test_dataset, user_groups_train, user_groups_test
