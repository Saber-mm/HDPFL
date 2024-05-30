# from domainbed
import sys
import csv
import json

class Tee:  
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

def to_csv(csv_file, row, mode='w'):
    with open(csv_file, mode) as f:
        writer = csv.writer(f)
        writer.writerow(row)
        
 # present things in a nice format 

def print_acc(list_):
    for elem in list_:
        print(f'{elem * 100:.2f}%', end='\t')
    print('\n')
    
def round_list(list_, dec=4):
    return [round(elem, dec) for elem in list_]


def read_json(file):
    results = []
    with open(file) as f:
        for line in f:
            j_content = json.loads(line)
            results.append(j_content)
    return results

def print_mean_std(mean, std):
    return f'{mean}' + '$_{' + f'\pm {std}' + '}$'

def plus_minus(A_means, G_means, stds, worsts, bests):
    mean, std = round_list(list(mean_std(A_means)), dec=2)
    mean_0, std_0 = round_list(list(mean_std(G_means)), dec=2)
    mean_1, std_1 = round_list(list(mean_std(stds)), dec=2)
    mean_2, std_2 = round_list(list(mean_std(worsts)), dec=2)
    mean_3, std_3 = round_list(list(mean_std(bests)), dec=2)
    print(print_mean_std(mean, std) + ' & ' + print_mean_std(mean_0, std_0) + ' & ' +\
          print_mean_std(mean_1, std_1) + ' & ' + print_mean_std(mean_2, std_2) + ' & ' +\
          print_mean_std(mean_3, std_3))
    return

def alg_to_stats(alg='FedAvg', root='/home/guojun-z/propfair/results/CIFAR10/label-10'):
    A_means = []
    stds = []
    G_means = []
    worsts = []
    bests = []
    for seed in range(3):
        log = os.path.join(root, alg, f'seed_{seed}', 'log.json')
        accs = read_json(log)[-1]['accs']
        mean, std = mean_std(accs)
        gmean_ = gmean(accs)
        worst = np.min(accs)
        best = np.max(accs)
        A_means.append(mean)
        stds.append(std)
        G_means.append(gmean_)
        worsts.append(worst)
        bests.append(best)
    plus_minus(A_means, G_means, stds, worsts, bests)
    return
