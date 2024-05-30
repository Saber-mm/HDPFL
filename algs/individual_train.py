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




if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"] ='1,2,4,5,6,7'
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        devices = [torch.device(i) for i in range(device_count)]
        device_locks = [threading.BoundedSemaphore(value=1) \
                      for _ in range(device_count)]
    else:
        devices = [torch.device('cpu')]
        
    root = '/home/guojun-z/propfair'

    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--data_dir', type=str, default='label-12')
    parser.add_argument('--output_dir', type=str, default='label-12')
    parser.add_argument('--dataset', type=str, default="MNIST")
    parser.add_argument('--num_clients', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8, help='for data loader')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--multithread', type=int, default=0, help='0 if concurrency')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    #parser.add_argument('--save', default=False, type=lambda x: (str(x).lower() == 'true'))
    #parser.add_argument('--client_id', type=int, default=0)
    #parser.add_argument('--device_id', type=int, default=0)
    args = parser.parse_args()
    args.data_dir = os.path.join(root, 'data', args.dataset, args.data_dir)
    args.output_dir = os.path.join(root, 'results', args.dataset, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    print(args)

    with open(os.path.join(args.output_dir, 'args.json'), 'w') as fp:
        json.dump(vars(args), fp)
    
    # data
    in_file = os.path.join(args.data_dir, 'in.pickle')
    out_file = os.path.join(args.data_dir, 'out.pickle')

    with open(in_file, 'rb') as f_in:
        in_data = pickle.load(f_in)
    with open(out_file, 'rb') as f_out:
        out_data = pickle.load(f_out)

    # data loaders
    train_loaders = [DataLoader(
        dataset=in_data[i],
        batch_size=args.batch_size,
        num_workers=args.num_workers,shuffle=True)
        for i in range(len(in_data))]

    test_loaders = [DataLoader(
        dataset=out_data[i],
        batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=True)
        for i in range(len(out_data))]
    
    # models
    models = [CNN() for  _ in range(args.num_clients)]
    # loss functions, optimizer
    #loss_func = nn.CrossEntropyLoss()
    loss_func = nn.MSELoss()
    #optimizers = [optim.Adam(model.parameters(), lr = 0.001) for model in models]
    if args.optimizer == 'SGD':
        optimizers = [optim.SGD(model.parameters(), lr = args.learning_rate, \
                                momentum=0.0) for model in models]
    elif args.optimizer == 'Adam':
        optimizers = [optim.Adam(model.parameters(), lr = args.learning_rate) for model in models]
    
    if args.multithread == 0:
        threads = []
        for i in range(args.num_clients):
            threads.append(threading.Thread(target=individual_train, args=(train_loaders, loss_func, \
                              optimizers[i], models, test_loaders, \
                             devices[i % device_count], device_locks[i % device_count], i, \
                             args.num_epochs, args.output_dir, True, True,)))
        multithreads(threads)
    else:
        for i in range(args.num_clients):
            individual_train(train_loaders[i], loss_func, optimizers[i], models[i], test_loaders[i], \
                             device=devices[0], device_lock=device_locks[0],\
                             client_id=i, epochs=args.num_epochs, \
                             output_dir=args.output_dir, show=True, save=True)
    server_device = devices[0]
    accs = accuracies(models, test_loaders, server_device)
    losses = losses(models, train_loaders, loss_func, server_device)
    acc_file = os.path.join(args.output_dir, 'acc.npy')
    loss_file = os.path.join(args.output_dir, 'loss.npy')
    np.save(acc_file, accs)
    np.save(loss_file, losses)
    print(accs)
    print_acc(accs)