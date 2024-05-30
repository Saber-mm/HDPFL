import torch
import numpy as np1
from copy import deepcopy
import sklearn.decomposition.PCA as PCA

''' basic '''
def model_to_params(model):
    return [param.data for param in model.parameters()]

'''one model operation'''

def zero_model(model):
    zero = deepcopy(model)
    for i, param in enumerate(zero.parameters()):
        param.data = torch.zeros_like(param.data)
    return zero

def scale_model(model, scale):
    scaled = deepcopy(model)
    for i, param in enumerate(scaled.parameters()):
        model_param = model_to_params(model)[i]
        param.data = scale * model_param.data
    return scaled

'''two model operation'''
def add_models(model1, model2, alpha=1.0):
    # obtain model1 + alpha * model2 for two models of the same size
    addition = deepcopy(model1)
    layers = len(model_to_params(model1))
    for i, param_add in enumerate(addition.parameters()):
        param1, param2 = model_to_params(model1)[i], model_to_params(model2)[i] 
        with torch.no_grad():
            param_add.data = param1.data + alpha * param2.data
    return addition

def sub_models(model1, model2):
    # obtain model1 - model2 for two models of the same size
    subtract = deepcopy(model1)
    layers = len(model_to_params(model1))
    for i, param_sub in enumerate(subtract.parameters()):
        param1, param2 = model_to_params(model1)[i], model_to_params(model2)[i] 
        with torch.no_grad():
            param_sub.data = param1.data - param2.data
    return subtract

def assign_model(model1, model2):
    for i, param1 in enumerate(model1.parameters()):
        param2 = model_to_params(model2)[i] 
        with torch.no_grad():
            param1.data = deepcopy(param2.data)
    return

'''model list operation'''

def avg_models(models, weights=None):
    '''take a list of models and average, weights: a list of numbers summing up to 1'''
    if weights == None:
        total = len(models)
        weights = [1.0/total] * total
        
    avg = zero_model(models[0])
    for index, model in enumerate(models):
        for i, param in enumerate(avg.parameters()):
            model_param = model_to_params(model)[i]
            param.data += model_param * weights[index]
    return avg

def avg_models_DP(models, sigma, weights=None):
    '''take a list of models and average, weights: a list of numbers summing up to 1'''
    if weights == None:
        total = len(models)
        weights = [1.0/total] * total
        
    avg = zero_model(models[0])
    for index, model in enumerate(models):
        for i, param in enumerate(avg.parameters()):
            model_param = model_to_params(model)[i]
            param.data += model_param * weights[index]
            
    # adding Gaussian noise from N(0, sigma^2):
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')  # use the first GPU
    else:
        device = torch.device('cpu')
        
    for param in avg.parameters():
        param.data += sigma * torch.randn(param.data.size()).to(device)        
    return avg


def assign_models(models, new_model):
    ''' assign the new_model into a list of models'''
    for model in models:
        assign_model(model, new_model)
    return

def average_update(local_updates, weights, params):
    num_users = len(local_updates)
    avg_update = deepcopy(local_updates[0])
    
    for k in params:
        avg_update[k] = avg_update[k] * weights[0]
        for user_index in range(1, num_users):
            avg_update[k] += local_updates[user_index][k] * weights[user_index]
            
    return avg_update        
            
def average_loss(local_losses, weights):
    num_users = len(local_losses)
    avg_loss = 0
    for i in range(num_users):
        avg_loss += local_losses[i] * weights[i]
        
    return avg_loss
    
def global_delta(beta, local_deltas, local_losses, avg_delta, avg_loss, weights, params):
    new_weights = []
    for i in range(len(weights)):
        new_weights.append(weights[i]*(local_losses[i]-avg_loss))    
    sum_new_weights = sum(new_weights)
    
    new_avg_delta = average_update(local_deltas, new_weights, params)
    
    global_delta = deepcopy(avg_delta)
    for k in params:
        global_delta[k] += 2*beta*(new_avg_delta[k] - (sum_new_weights*avg_delta[k]))
    
    return global_delta
    

def global_delta_semi(beta, local_deltas, local_losses, avg_delta, avg_loss, weights, params):
    new_weights = []
    for i in range(len(weights)):
        if weights[i]*(local_losses[i]-avg_loss) > 0:
            new_weights.append(weights[i]*(local_losses[i]-avg_loss))
        else:
            new_weights.append(0)
            
    sum_new_weights = sum(new_weights)
    new_avg_delta = average_update(local_deltas, new_weights, params)
    
    global_delta = deepcopy(avg_delta)
    for k in params:
        global_delta[k] += 2*beta*(new_avg_delta[k] - (sum_new_weights*avg_delta[k]))
    
    return global_delta


'''aggregation'''
def aggregate(models, weights=None):   # FedAvg
    avg = avg_models(models, weights=weights)
    assign_models(models, avg)
    return

def aggregate_DP(models, sigma, weights=None):
    avg = avg_models_DP(models, sigma, weights=weights)
    assign_models(models, avg)
    return    

def global_update(model_global, global_step, params):
    state_dic = model_global.state_dict()
    for k in params:
        state_dic[k] = state_dic[k] - global_step[k]
    model_global.load_state_dict(state_dic)    
    return

def aggregate_lr(old_model, models, weights=None, global_lr=1.0): # FedAvg
    '''return old_model + global_lr * Delta, where Delta is aggregation of local updates'''
    with torch.no_grad():
        Delta = [sub_models(model, old_model) for model in models]
        avg_Delta = avg_models(Delta, weights=weights)
        new_model = add_models(old_model, avg_Delta, alpha=global_lr)
        assign_models(new_model, models)
    return

def aggregate_momentum(old_model, server_momentum, models, weights=None, global_lr=1.0, \
                 momentum_coeff=0.9): # FedAvg
    '''return old_model + global_lr * Delta + 0.9 momentum, where Delta is aggregation of local updates
    Polyak's momentum'''
    with torch.no_grad():
        Delta = [sub_models(model, old_model) for model in models]
        avg_Delta = avg_models(Delta, weights=weights)
        avg_Delta = scale_model(avg_Delta, global_lr)
        server_momentum = add_models(avg_Delta, server_momentum, momentum_coeff)
        new_model = add_models(old_model, avg_Delta)
        assign_models(new_model, models)
    return