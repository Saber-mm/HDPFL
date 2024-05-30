#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
import random
import copy
from torchvision import datasets, transforms






def iid_even_odd(dataset, num_users, num_even_users=4, p_even=1):
    np.random.seed(0)
    num_items = int(len(dataset)/(2*15.5))
    
    num_even_evens = int(num_items * p_even)
    num_odd_evens = num_items - num_even_evens
    
    num_even_odds = num_odd_evens
    num_odd_odds = num_even_evens

    dict_users = {}
    even_idxs, odd_idxs = even_odd_idxs(dataset)
    
    
    for i in range(num_users):
        if i < num_even_users: # i.e. it is the client with mostly even numbers        
            dict_users[i] = set(np.random.choice(even_idxs, num_even_evens, replace=False))
            even_idxs = list(set(even_idxs) - dict_users[i])
            dict_users[i] = set(list(dict_users[i]) + list(np.random.choice(odd_idxs, num_odd_evens, replace=False)))
            odd_idxs = list(set(odd_idxs) - dict_users[i])
            
        else:
            dict_users[i] = set(np.random.choice(odd_idxs, num_odd_odds, replace=False))
            odd_idxs = list(set(odd_idxs) - dict_users[i])
            dict_users[i] = set(list(dict_users[i]) + list(np.random.choice(even_idxs, num_even_odds, replace=False)))
            even_idxs = list(set(even_idxs) - dict_users[i])
            
        
    return dict_users



# def iid_even_odd(dataset, num_users, num_even_users=10, p_even=1):
#     np.random.seed(0)
#     print(len(dataset))
#     num_items = int(len(dataset)/(2*16))
#     print(num_items)
#     dict_users = {}
#     even_idxs, odd_idxs = even_odd_idxs(dataset)
#     for i in range(num_users):
# #         if i%2 == 0:
#         if i < 4:    
#             dict_users[i] = set(np.random.choice(even_idxs, num_items, replace=False))
#             even_idxs = list(set(even_idxs) - dict_users[i])
#         else:
#             dict_users[i] = set(np.random.choice(odd_idxs, num_items, replace=False))
#             odd_idxs = list(set(odd_idxs) - dict_users[i])
        
#     return dict_users


def iid(dataset, num_users):
    """
    Sample I.I.D. client data from "dataset"
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    np.random.seed(0)
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    # ######## temp
    # frac = 4
    # num_items = int(len(dataset)/num_users/frac)
    # dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    # random.shuffle(all_idxs)
    # all_idxs = all_idxs[:int(len(dataset)/frac)]
    # print("fix the idd sampling part")
    # print("fix the idd sampling part")
    # print("fix the idd sampling part")
    # print("fix the idd sampling part")
    # print("fix the idd sampling part")
    # print("fix the idd sampling part")
    # print("fix the idd sampling part")
    # print("fix the idd sampling part")
    # print("fix the idd sampling part")
    # print("fix the idd sampling part")
    # print("fix the idd sampling part")
    # print("fix the idd sampling part")
    # print("fix the idd sampling part")
    # ########
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def even_odd_idxs(dataset):
    """
    Splits the idxs into even and odd ones
    :param dataset:
    returns: odd idxs, even idxs
    """
    odd_idxs = []
    even_idxs = []
    
    idxs = np.arange(len(dataset))
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    for i in range(10):
        if i%2 == 0:
            even_idxs = even_idxs + list(idxs[i*6000:(i+1)*6000])
        else:
            odd_idxs = odd_idxs + list(idxs[i*6000:(i+1)*6000])
    np.random.shuffle(even_idxs)
    np.random.shuffle(odd_idxs)
    
    return even_idxs, odd_idxs



def noniid(dataset, num_users, user_max_class):
    """
    Sample non-I.I.D client data from "dataset"
    :param dataset:
    :param num_users:
    :param user_max_class:
    :return:
    """
    np.random.seed(0)
    
    num_shards_per_user = user_max_class
    num_shards = num_users * num_shards_per_user
    num_imgs = int(np.floor(len(dataset)/num_shards))
        


    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 6 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, num_shards_per_user, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    for i in range(num_users):
        np.random.shuffle(dict_users[i])
    for i in range(num_users):
        values, counts = np.unique(labels[list(map(int, list(dict_users[i])))], return_counts=True)
#         print(len(values), values, counts)

    return dict_users


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users