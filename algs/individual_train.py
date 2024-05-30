# training
import argparse
import os
from torch import optim, nn
from torch import linalg as LA
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector
import sys
import pickle
import matplotlib
import matplotlib.cm as cm
from models.models import CNN_MNIST
from utils.io import Tee, to_csv
import torch
from utils.eval import accuracy, accuracies, losses
from tqdm import tqdm
import json
import threading
from utils.concurrency import multithreads
from utils.print import print_acc
import numpy as np
from math import sqrt
import copy
from opacus.utils.batch_memory_manager import BatchMemoryManager

def individual_train(train_loader, loss_func, optimizer, model, test_loader, device, \
                    client_id, epochs, output_dir, show=True, save=True): 
    
    # device_lock.acquire()
    output_dir = os.path.join(output_dir, f'client_{client_id}')
    if save:
        os.makedirs(output_dir, exist_ok=True)
    model.to(device)
    if save:
        csv_file = os.path.join(output_dir, f'client_{client_id}_log.csv')
        to_csv(csv_file, ['epoch', 'loss', 'test acc'], mode='w')
    
    # use tqdm to monitor progress
    for epoch in range(epochs):
        if show:
            t = tqdm(train_loader)
        else:
            t = train_loader
        for images, target in t:
            images = images.to(device)
            target = target.to(device)
            # target = torch.nn.functional.one_hot(target, num_classes=10).type(torch.cuda.FloatTensor)
            outputs = model(images).to(device)
            model.zero_grad()
            loss = loss_func(outputs, target).to(device)
            loss.backward()

            optimizer.step()
            if show:
                t.set_description(f'epoch: {epoch}, client: {client_id}, loss: {loss:.6f}')
        
        acc = accuracy(model, test_loader, device, show=show)
        if save:
            to_csv(csv_file, [epoch, loss.item(), acc], mode='a')
    if save:
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), \
                'optimizer' : optimizer.state_dict(),}, output_dir  + f'/model_{client_id}_last.pth')

    return model



def individual_train_PDP(train_loader, loss_func, optimizer, model, privacy_engine, delta, test_loader, device, \
                    client_id, epochs, output_dir, show=True, save=True, max_physical_batch_size=16): 
    
    # device_lock.acquire()
    output_dir = os.path.join(output_dir, f'client_{client_id}')
    if save:
        os.makedirs(output_dir, exist_ok=True)
    model.to(device)
    if save:
        csv_file = os.path.join(output_dir, f'client_{client_id}_log.csv')
        to_csv(csv_file, ['epoch', 'loss', 'test acc'], mode='w')
    

    for epoch in range(epochs):      
        t = train_loader
        optimizer.zero_grad()
        with BatchMemoryManager(data_loader=train_loader, max_physical_batch_size=max_physical_batch_size, \
                                optimizer=optimizer) as t:
            for i, (images, target) in enumerate(t):
                assert len(images) <= max_physical_batch_size # physical batch is no more than max_physical_batch_size
                images = images.to(device)
                target = target.to(device)
                if images.shape[0] == 0:
                    break
                outputs = model(images).to(device)
                # model.zero_grad()
                loss = loss_func(outputs, target).to(device)
                loss.backward()
                # optimizer won't actually make a step unless logical batch is over
                optimizer.step()
                # optimizer won't actually clear gradients unless logical batch is over
                optimizer.zero_grad()

                if show:
                    t.set_description(f'epoch: {epoch}, client: {client_id}, loss: {loss:.6f}')
        
    acc = accuracy(model, test_loader, device, show=show)
    if save:
        to_csv(csv_file, [epoch, loss.item(), acc], mode='a')
    if save:
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), \
                'optimizer' : optimizer.state_dict(),}, output_dir  + f'/model_{client_id}_last.pth')
    
    return model


def params_norm(state_dict):
    norm = 0
    for k in state_dict.keys():
        norm += LA.norm(state_dict[k])**2
    return sqrt(norm.cpu().numpy())
