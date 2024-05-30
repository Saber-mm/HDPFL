import torch
import numpy as np
from copy import deepcopy
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.mixture import GaussianMixture

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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def models_to_matrix(models):
    num_rows = count_parameters(models[0])
    num_cols = len(models)
    mat = np.zeros((num_rows, num_cols))
    for i, model in enumerate(models):    
        V = torch.nn.utils.parameters_to_vector(model.parameters())
        X = V.cpu().detach().numpy()
        mat[:,i] = X
    return mat

def matrix_to_models(mat, models):
    models_generated = deepcopy(models)
    for i in range(len(models_generated)):
        parameters_dict = deepcopy(models_generated[i].state_dict())
        X = mat[:,i]
        head = 0
        for name, param in models_generated[i].named_parameters():
            tail = head + param.numel()
            param_vectorized = X[head:tail]
            param_vectorized = np.reshape(param_vectorized, tuple(param.shape))
            parameters_dict[str(name)] = deepcopy(torch.tensor(param_vectorized))
            head = tail
        models_generated[i].load_state_dict(parameters_dict)
    return models_generated



def param_to_models(input_param, models):
    models_generated = deepcopy(models)
    for i in range(len(models_generated)):
        parameters_dict = deepcopy(models_generated[i].state_dict())
        X = input_param
        head = 0
        for name, param in models_generated[i].named_parameters():
            tail = head + param.numel()
            param_vectorized = X[head:tail]
            param_vectorized = np.reshape(param_vectorized, tuple(param.shape))
            parameters_dict[str(name)] = deepcopy(torch.tensor(param_vectorized))
            head = tail
        models_generated[i].load_state_dict(parameters_dict)
    return models_generated




def PCA_proj_recons(M, N, n_components=2):
    # M and N are of shape (n,d)
    scaling = StandardScaler()
    scaling.fit(M)
    M_scaled = scaling.transform(M)
    principal = PCA(n_components = n_components)
    principal.fit(M_scaled)
    # print(principal.components_): returns the "normalized" projection directions
    
    N_scaled = scaling.transform(N)
    N_scaled_projected = principal.transform(N_scaled)
    N_scaled_reconstructed = principal.inverse_transform(N_scaled_projected)
    N_reconstructed = scaling.inverse_transform(N_scaled_reconstructed)

    # N_reconstructed is of shape (n,d). We return the transpose:
    return N_reconstructed.T


def PCA_proj(M, n_components=2):
    # M is of shape (n,d)
    scaling = StandardScaler()
    scaling.fit(M)
    M_scaled = scaling.transform(M)
    principal = PCA(n_components = n_components)
    principal.fit(M_scaled)
    M_projected = principal.transform(M)
    return M_projected



def project(M, V):
    
    V = V/np.linalg.norm(V) # making the projection direction a unit vector
    n_rows = M.shape[0]
    n_cols = M.shape[1]
    M_projected = deepcopy(M)
    components = np.matmul(V.T, M)
    M_projected = np.matmul(V, components)
        
    return M_projected




def find_delta_model(new_model, old_model):
    
    delta_model = zero_model(new_model)
    for i, param in enumerate(delta_model.parameters()):
        new_model_param = model_to_params(new_model)[i]
        old_model_param = model_to_params(old_model)[i]
        param.data = new_model_param - old_model_param
        
    return delta_model

def find_new_model(delta_model, old_model):
    
    new_model = zero_model(old_model)
    for i, param in enumerate(new_model.parameters()):
        old_model_param = model_to_params(old_model)[i]
        delta_model_param = model_to_params(delta_model)[i]
        param.data = old_model_param + delta_model_param
        
    return new_model


def frobenius_norm(M):
        return np.linalg.norm(M, ord='fro')

def l1_norm(M):
    return sum([np.linalg.norm(M[:,i], ord=1) for i in range(M.shape[1])])



def svd(M, tau):
    U, S, V = np.linalg.svd(M, full_matrices=False)
    return U, S, V





class R_pca:

    def __init__(self, D, mu=None, lmbda=None):
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)

        if mu:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / (4 * np.linalg.norm(self.D, ord=1))

        self.mu_inv = 1 / self.mu

        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

    def frobenius_norm(self, M):
        return np.linalg.norm(M, ord='fro')

    def shrink(self, M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self, tol=None, max_iter=10000, iter_print=5):
        iter = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)

        if tol:
            _tol = tol
        else:
            _tol = 0.0000001 * self.frobenius_norm(self.D)
            
        print('tolerance for fitting RPCA is {}'.format(_tol))    
        #this loop implements the principal component pursuit (PCP) algorithm
        #located in the table on page 29 of https://arxiv.org/pdf/0912.3599.pdf
        while (err > _tol) and iter < max_iter:
            Lk = self.svd_threshold(self.D - Sk + self.mu_inv * Yk, self.mu_inv)            #this line implements step 3
            Sk = self.shrink(self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)    #this line implements step 4
            Yk = Yk + self.mu * (self.D - Lk - Sk)                                          #this line implements step 5
            err = self.frobenius_norm(self.D - Lk - Sk)
            iter += 1
            if iter%500 == 0:
                print('iteration: {0}, error: {1}'.format(iter, err))
                

        print('iteration: {0}, error: {1}'.format(iter, err))
        self.L = Lk
        self.S = Sk
        return Lk, Sk

    def plot_fit(self, size=None, tol=0.1, axis_on=True):

        n, d = self.D.shape

        if size:
            nrows, ncols = size
        else:
            sq = np.ceil(np.sqrt(n))
            nrows = int(sq)
            ncols = int(sq)

        ymin = np.nanmin(self.D)
        ymax = np.nanmax(self.D)
        print('ymin: {0}, ymax: {1}'.format(ymin, ymax))

        numplots = np.min([n, nrows * ncols])
        plt.figure()

        for n in range(numplots):
            plt.subplot(nrows, ncols, n + 1)
            plt.ylim((ymin - tol, ymax + tol))
            plt.plot(self.L[n, :] + self.S[n, :], 'r')
            plt.plot(self.L[n, :], 'b')
            if not axis_on:
                plt.axis('off')
                

                
def find_closests(clusters, S):
    d_min = np.inf
    idx1 = 0
    idx2 = 0
    for i in range(len(clusters)-1):
        for j in range(i+1, len(clusters)):
            d = inter_cluster_dist(clusters[i], clusters[j], S)
            if d_min > d:
                d_min = d
                idx_1 = i
                idx_2 = j
    return idx_1, idx_2            
                
def inter_cluster_dist(cluster1, cluster2, S):
    dmin = np.inf
    for idx1 in cluster1:
        for idx2 in cluster2:
            d = np.linalg.norm(S[:,idx1] - S[:,idx2]) ######
            if dmin > d:
                dmin = d
    return dmin

def pub_priv_hierarchical_cluster(epsilons_input):
    epsilons_input = epsilons_input.reshape(1, len(epsilons_input))
    clusters = list([i] for i in np.arange(0, np.shape(epsilons_input)[1]))
    while len(clusters) > 2:
        idx1, idx2 = find_closests(clusters, epsilons_input)
        clusters[idx1] = clusters[idx1] + clusters[idx2]
        clusters.pop(idx2)
    
    clusters_avg = []
    for c in clusters:
        epsilon_values = [epsilons_input[0,i] for i in c]
        clusters_avg.append(np.mean(epsilon_values))
    
    public_cluster = clusters[np.argsort(clusters_avg)[1]]
    
    return np.sort(public_cluster), len(public_cluster)

def pub_priv_kmeans_cluster(epsilons_input, seed):

    epsilons_input = epsilons_input.reshape(len(epsilons_input), 1)
    kmeans = KMeans(n_clusters=2, random_state=seed, n_init="auto").fit(epsilons_input)
    labels = kmeans.labels_
    cluster_0 = [i for i in range(epsilons_input.shape[0]) if labels[i] == 0]
    cluster_1 = [i for i in range(epsilons_input.shape[0]) if labels[i] == 1]
    clusters = [cluster_0, cluster_1]
    
    clusters_avg = []
    for c in clusters:
        epsilon_values = [epsilons_input[i, 0] for i in c]
        clusters_avg.append(np.mean(epsilon_values))
    
    public_cluster = clusters[np.argsort(clusters_avg)[1]]
    
    
    return np.sort(public_cluster), len(public_cluster)




def pub_priv_GMM_cluster(mat, num_clusters=2, seed=0):
    
    mat = mat.reshape(1, len(mat))
    scaling = MinMaxScaler()
    scaling.fit(mat.T)
    mat = scaling.transform(mat.T).T
    gm = GaussianMixture(n_components=num_clusters, random_state=seed)
    gm.fit(mat.T)
    labels = gm.predict(mat.T)
    
    cluster_0 = [i for i in range(mat.shape[1]) if labels[i] == 0]
    cluster_1 = [i for i in range(mat.shape[1]) if labels[i] == 1]
    clusters = [cluster_0, cluster_1]
    
    
    clusters_avg = []
    for c in clusters:
        epsilon_values = [mat[0, i] for i in c]
        clusters_avg.append(np.mean(epsilon_values))
    
    public_cluster = clusters[np.argsort(clusters_avg)[1]]
    
    
    return np.sort(public_cluster), len(public_cluster)




