import numpy as np
from scipy.stats import gmean

def mean_std(accs):
    return np.mean(accs), np.std(accs)

def max_min(accs):
    return np.max(accs) - np.min(accs)

def best_p(accs, percent=0.2):
    sorted_ = np.sort(accs)
    num = len(accs)
    best = sorted_[num - int(num * percent):]
    return mean_std(best)

def worst_p(accs, percent=0.2):
    sorted_ = np.sort(accs)
    num = len(accs)
    worst = sorted_[:int(num * percent)]
    return mean_std(worst)
