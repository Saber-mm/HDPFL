# training
import sys
sys.path.append("..")
import os
print(os.getcwd())
os.chdir('/fs01/home/sabermm/pdpfl/rpca_c/algs')
os.getcwd()

import copy
import argparse
import os
import torch
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
import numpy as np
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
import pickle
import threading
from tqdm import tqdm
import json
from utils.io import Tee, to_csv
from utils.eval import accuracy, accuracies, losses, epsilons
from utils.aggregate import aggregate, aggregate_lr, zero_model, aggregate_momentum, models_to_matrix, matrix_to_models, param_to_models, PCA_proj_recons, project, find_new_model, find_delta_model, assign_model, R_pca, count_parameters, pub_priv_hierarchical_cluster, pub_priv_kmeans_cluster, pub_priv_GMM_cluster
from algs.individual_train import individual_train, individual_train_PDP
from utils.concurrency import multithreads
from models.models import resnet18, CNN_MNIST, CNN_FMNIST, CNN, CNN_CIFAR10, CNN_FEMNIST, RNN_Shakespeare, RNN_StackOverflow, resnet34
from utils.print import print_acc, round_list
from utils.save import save_acc_loss, save_acc_loss_privacy
from utils.stat import mean_std
from moments_accountant import compute_z
from opacus import PrivacyEngine
###
from utils.privacy_params import get_epsilons_batchsizes
from utils.utils2 import get_dataset, get_dataset_CIFAR, DatasetSplit




root = '..'

parser = argparse.ArgumentParser(description='training')
parser.add_argument('--device', type=str, default='0')
parser.add_argument('--data_dir', type=str, default='iid-10')
parser.add_argument('--method', type=str, default='Robust_PDP', help="used method: 'FedAvg'/'epsilon_min'/'PFA'/'Robust_PDP (ours)'/'WeiAvg'/'DPFedAvg' ")
parser.add_argument('--privacy_dist', type=str, default='Dist4', help="privacy preference sampling distribution: 'Dist1'/'Dist2'/'Dist3'/'Dist4'/'Dist5'/'Dist6'/'Dist7'/'Dist8'/'Dist9' ")
parser.add_argument('--dataset', type=str, default='MNIST', help="'MNIST'/'FMNIST'/'CIFAR10'/'CIFAR100'")
parser.add_argument('--clustering_method', type=str, default='GMM', help="'GMM'/'KMeans'/'hierarchical'")
parser.add_argument('--num_clients', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_workers', type=int, default=0, help='for data loader')
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--num_local_epochs', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=0.1)
parser.add_argument('--save_epoch', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--full_batch', type=bool, default=False)
parser.add_argument('--max_per_sample_grad_norm', type=float, default=3.0)
parser.add_argument('--delta', type=float, default=0.0001)
parser.add_argument('--n_pca', type=int, default=1)
parser.add_argument('--p_prime', type=int, default=200000)
parser.add_argument('--iid', dest='iid', action='store_true')
parser.add_argument('--noniid', dest='iid', action='store_false')
parser.add_argument('--user_max_class', type=int, default=10)
parser.add_argument('--alpha', type=float, default=1.0)

args = parser.parse_args()
print(os.getcwd())
print(args)


if args.method == 'FedAvg':
    output_dir = output_dir = os.path.join(root, 'results', args.data_dir, 'FedAvg' + f'_lr{args.learning_rate}_' + 'full_batch'+f'_{args.full_batch}', f'seed_{args.seed}')
elif args.method == 'PFA':
    output_dir = os.path.join(root, 'results', args.data_dir, args.method + f'_{args.n_pca}' + f'_lr{args.learning_rate}_' + args.privacy_dist + '_' + 'full_batch'+f'_{args.full_batch}' + '_' + f'c_{args.max_per_sample_grad_norm}', f'seed_{args.seed}')
else:
    output_dir = os.path.join(root, 'results', args.data_dir, args.method + f'_lr{args.learning_rate}_' + args.privacy_dist + '_' + 'full_batch'+f'_{args.full_batch}' + '_' + f'c_{args.max_per_sample_grad_norm}', f'seed_{args.seed}')
    
    
print('output_dir: ', output_dir)
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'args.json'), 'w') as fp:
    json.dump(vars(args), fp)

args.data_dir = os.path.join(root, 'data', args.dataset, args.data_dir)

os.environ['CUDA_VISIBLE_DEVICES'] = args.device
if torch.cuda.is_available():
    device = torch.device('cuda:0')  # use the first GPU
else:
    device = torch.device('cpu')

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if args.dataset == "MNIST" or args.dataset == "FMNIST":
    train_dataset, test_dataset, user_groups_train = get_dataset(dataset=args.dataset, num_users=args.num_clients, \
                                                       iidness=args.iid, unequal=False, user_max_class=args.user_max_class)
    idxs_train = [list(user_groups_train[i])[:int(0.8*len(user_groups_train[i]))] for i in range(len(user_groups_train))]
    idxs_test = [list(user_groups_train[i])[int(0.8*len(user_groups_train[i])):] for i in range(len(user_groups_train))]
    
elif args.dataset == "CIFAR10" or args.dataset == "CIFAR100":
    train_dataset, test_dataset, user_groups_train, user_groups_test = get_dataset_CIFAR(dataset=args.dataset,
                                                                                           num_users=args.num_clients, \
                                                                                           iidness=args.iid, unequal=False,
                                                                                           user_max_class=args.user_max_class)  
    idxs_train = [list(user_groups_train[i]) for i in range(len(user_groups_train))]
    idxs_test = [list(user_groups_test[i]) for i in range(len(user_groups_test))]
    

weights = np.array([len(idxs_train[i]) for i in range(len(idxs_train))])
weights_aggregation_n = list(weights / np.sum(weights))
weights_test = np.array([len(idxs_test[i]) for i in range(len(idxs_test))])

print('total train samples: {}'.format(np.sum(weights)))
print('total test samples: {}'.format(np.sum(weights_test)))
print('total samples: {}'.format(np.sum(weights)+np.sum(weights_test)))
print('samples: ', weights)


epsilons_input, batches = get_epsilons_batchsizes(args.num_clients, args.privacy_dist)
deltas_input = np.array([args.delta] * args.num_clients)
if args.dataset == 'CIFAR100':
    epsilons_input = epsilons_input * 10


if args.clustering_method == 'GMM':
    public_idxs, n_pub = pub_priv_GMM_cluster(epsilons_input, seed=args.seed)
elif args.clustering_method == 'KMeans':   
    public_idxs, n_pub = pub_priv_kmeans_cluster(epsilons_input, seed=args.seed)
elif args.clustering_method == 'hierarchical':
    public_idxs, n_pub = pub_priv_hierarchical_cluster(epsilons_input)

print('public cluster obtained from clustering of clients, and their number: {}, {}'.format(public_idxs, n_pub))
private_idxs = list(set(range(0, args.num_clients)) - set(public_idxs))

if args.method == 'epsilon_min':
    Z = [round(compute_z(epsilon=np.min(epsilons_input), dataset_size=weights[i], batch=batches[i], \
                     local_epochs=args.num_local_epochs, global_epochs=args.num_epochs, delta=deltas_input[i]), 2) \
     for i in range(len(epsilons_input))]
elif args.method in ['Robust_PDP', 'Robust_PDP_plus', 'PFA', 'DPFedAvg', 'WeiAvg']:
    Z = [round(compute_z(epsilon=epsilons_input[i], dataset_size=weights[i], batch=batches[i], \
                         local_epochs=args.num_local_epochs, global_epochs=args.num_epochs, delta=deltas_input[i]), 2) \
         for i in range(len(epsilons_input))]
    print('clients noise scales are: {}'.format(Z))
#####


if args.full_batch:
    train_loaders = [DataLoader(DatasetSplit(train_dataset, idxs_train[i]), batch_size=len(idxs_train[i]), shuffle=True) for i in range(len(user_groups_train))]
else:
    train_loaders = [DataLoader(DatasetSplit(train_dataset, idxs_train[i]), batch_size=batches[i], shuffle=True) for i in range(len(user_groups_train))]

if args.dataset in ['MNIST', 'FMNIST']:
    test_loaders = [DataLoader(DatasetSplit(train_dataset, idxs_test[i]), batch_size=len(idxs_test[i]), shuffle=False) for i in
                    range(len(user_groups_train))]
elif args.dataset in ['CIFAR10', 'CIFAR100']:
    test_loaders = [DataLoader(DatasetSplit(test_dataset, idxs_test[i]), batch_size=len(idxs_test[i]), shuffle=False) for i in
                    range(len(user_groups_test))]
    

if args.dataset == 'MNIST':
    models = [CNN_MNIST() for _ in range(args.num_clients)]
    print('total number of parameters:{}'.format(sum(p.numel() for p in models[0].parameters() if p.requires_grad)))
    model_trial = copy.deepcopy(models[0])
    model_trial.cuda()
    summary(model_trial, (1,28,28))
elif args.dataset == 'FMNIST':
    models = [CNN_FMNIST() for _ in range(args.num_clients)]
    print('total number of parameters:{}'.format(sum(p.numel() for p in models[0].parameters() if p.requires_grad)))
    model_trial = copy.deepcopy(models[0])
    model_trial.cuda()
    summary(model_trial, (1,28,28))
elif args.dataset == 'CIFAR10':
    models = [resnet18(num_classes = 10) for _ in range(args.num_clients)]
    print('total number of parameters:{}'.format(sum(p.numel() for p in models[0].parameters() if p.requires_grad)))
    model_trial = copy.deepcopy(models[0])
    model_trial.cuda()
    summary(model_trial, (3,32,32))
elif args.dataset == 'CIFAR100':
    model_0 = resnet34(num_classes = 10)
    models = [copy.deepcopy(model_0) for _ in range(args.num_clients)]
    print('total number of parameters:{}'.format(sum(p.numel() for p in models[0].parameters() if p.requires_grad)))
    model_trial = copy.deepcopy(models[0])
    model_trial.cuda()
    summary(model_trial, (3,32,32))
    
    
# loss functions, optimizer:
loss_func = nn.CrossEntropyLoss()
optimizers = [optim.SGD(model.parameters(), lr = args.learning_rate, momentum=0.0) for model in models]
privacy_engines = [None for model in models]


if args.method != 'FedAvg':
    privacy_engines = [PrivacyEngine() for model in models]
    for j in range(len(models)):
        models[j], optimizers[j], train_loaders[j] = privacy_engines[j].make_private(
                module = models[j],
                optimizer = optimizers[j],
                data_loader = train_loaders[j],
                noise_multiplier = Z[j],
                max_grad_norm = args.max_per_sample_grad_norm,
            )
#####
model_path = output_dir  + f'/model_last.pth'
if os.path.exists(model_path):
    ckpt = torch.load(model_path, map_location='cpu')
    start_epoch = ckpt['epoch']
    for model in models:
        # model.to(device)
        model.load_state_dict(ckpt['state_dict'])
else:
    start_epoch = 0
#####
json_file = os.path.join(output_dir, 'log.json')
with open(json_file, 'w') as f:
    f.write('')

#####
acc_file = "accuracy_vals_{}_{}_{}_{}.pkl".format(args.dataset, args.method, args.privacy_dist, args.seed)
acc_file = os.path.join(output_dir, acc_file)
if os.path.exists(acc_file):
    with open(acc_file, 'rb') as f_in:
        accuracy_vals = pickle.load(f_in)
else:
    accuracy_vals = []

#####
loss_file = "loss_vals_{}_{}_{}_{}.pkl".format(args.dataset, args.method, args.privacy_dist, args.seed)
loss_file = os.path.join(output_dir, loss_file)
if os.path.exists(loss_file):
    with open(loss_file, 'rb') as f_in:
        loss_vals = pickle.load(f_in)
else:
    loss_vals = []


if args.method == 'Robust_PDP':
    weights_agg_file = "weights_agg_{}_{}_{}_{}.pkl".format(args.dataset, args.method, args.privacy_dist, args.seed)
    weights_agg_file = os.path.join(output_dir, weights_agg_file)



for t in range(start_epoch + 1, args.num_epochs):
    old_models = [copy.deepcopy(model) for model in models]
    for i in range(args.num_clients):
        if args.method == 'FedAvg':
            individual_train(train_loaders[i], loss_func, optimizers[i], models[i], test_loaders[i], device=device, \
                             client_id=i, epochs=args.num_local_epochs, output_dir=output_dir, show=False, save=False)
        else:
            individual_train_PDP(train_loaders[i], loss_func, optimizers[i], models[i], privacy_engines[i], args.delta, \
                                 test_loaders[i], device=device, client_id=i, epochs=args.num_local_epochs,
                                 output_dir=output_dir, show=False, save=False)
    #####################################
    if args.method == 'Robust_PDP':
        if t ==1:
            delta_models = [find_delta_model(models[m], old_models[m].to(device)) for m in range(len(models))]
            delta_matrix_main = models_to_matrix(delta_models)
            if args.dataset in ['CIFAR10', 'CIFAR100']:
                delta_matrix = delta_matrix_main[0:args.p_prime,:]
                noise_var_scale = delta_matrix_main.shape[0]/args.p_prime  
            else:
                delta_matrix = delta_matrix_main   
            RPCA = R_pca(delta_matrix)
            L, S = RPCA.fit(tol=5e-8)
            if args.dataset not in ['CIFAR10', 'CIFAR100']:
                print([(np.linalg.norm(S[:,i])**2) for i in range(S.shape[1])])
                S_col_norms_squared_inversed = np.array([1/(np.linalg.norm(S[:,i])**2) for i in range(S.shape[1])])
            else:
                print([noise_var_scale * (np.linalg.norm(S[:,i])**2) for i in range(S.shape[1])])
                S_col_norms_squared_inversed = np.array([1/(noise_var_scale * (np.linalg.norm(S[:,i])**2)) for i in
                                                         range(S.shape[1])])
            # U, S, V = np.linalg.svd(delta_matrix, full_matrices=False)
            # print('singular values before RPCA: {}'.format(S))
            # U, S, V = np.linalg.svd(L, full_matrices=False)
            # print('singular values after RPCA: {}'.format(S))

            weights_aggregation_noise = list(S_col_norms_squared_inversed/np.sum(S_col_norms_squared_inversed))
            print('The weights obtained from RPCA are: {}'.format(weights_aggregation_noise))
            weights_aggregation = [weights_aggregation_noise[i] for i in range(args.num_clients)]
            with open(weights_agg_file, 'wb') as f_out:
                pickle.dump(weights_aggregation, f_out)

            del delta_models
            del delta_matrix
            del L
            del S

        else:
            if os.path.exists(weights_agg_file):
                with open(weights_agg_file, 'rb') as f_in:
                    weights_aggregation = pickle.load(f_in)
            else:
                raise ValueError('previoulsy computed Robust_PDP aggregation weights are not found')

    #########################
    elif args.method == 'PFA':
        delta_models = [find_delta_model(models[m], old_models[m].to(device)) for m in range(len(models))]
        priv_delta_models = [deepcopy(delta_models[i]) for i in range(args.num_clients) if i in private_idxs]
        public_delta_models = [deepcopy(delta_models[i]) for i in range(args.num_clients) if i in public_idxs]
        del delta_models
        M_pub = models_to_matrix(deepcopy(public_delta_models))
        M_pri = models_to_matrix(deepcopy(priv_delta_models))
        M_pri_reconstructed = PCA_proj_recons(M_pub.T, M_pri.T, n_components=args.n_pca)
        del M_pub
        del M_pri
        priv_delta_models = matrix_to_models(M_pri_reconstructed, deepcopy(priv_delta_models))
        priv_new_models = [find_new_model(priv_delta_models[i], old_models[private_idxs[i]].to(device)) for i in
                           range(len(priv_delta_models))]
        del priv_delta_models

        # loading the new private models:
        j = 0
        for i in range(args.num_clients):
            if i in private_idxs:
                assign_model(models[i], priv_new_models[j])
                j += 1
        assert j == len(private_idxs), "not all private models are loaded"
        del priv_new_models
        weights_aggregation = list(epsilons_input / np.sum(epsilons_input))

    #############################
    elif args.method == 'WeiAvg':
        weights_aggregation = list(epsilons_input / np.sum(epsilons_input))

    ###############################
    elif args.method == 'DPFedAvg' or args.method == 'epsilon_min' or args.method == 'FedAvg':
        weights_aggregation = weights_aggregation_n # i.e. aggregating the uploaded noisy updates with "weights_aggregation_n"


    aggregate(models, weights = weights_aggregation)
    ### Evaluation at the end of t-th round:
    accs = accuracies(models, test_loaders, device)
    losses_ = losses(models, train_loaders, loss_func, device)
    accuracy_vals.append(accs)
    loss_vals.append(losses_)

    print(f'global epoch: {t}')
    mean, std = mean_std(accs)
    print('mean accuracy: {}'.format(mean))
    print('mean loss: {}'.format(np.mean(losses_)))
    print(f'accs: {[round(i, 3) for i in accs]}')
    print(f'losses: {round_list(losses_)}')


    ### saving the model at the end of t-th round:
    if t % args.save_epoch == 0:
        torch.save({'epoch': t, 'state_dict': models[0].state_dict()}, output_dir  + f'/model_last.pth')
    ### saving loss values and accuracy values at the end of each round:
    if t % 1 == 0:
        with open(acc_file, 'wb') as f_out:
            pickle.dump(accuracy_vals, f_out)
        with open(loss_file, 'wb') as f_out:
            pickle.dump(loss_vals, f_out)

mean, std = mean_std(accs)
print('mean: ', mean, 'std: ', std)
print(f'accs: {[round(i, 3) for i in accs]}')
