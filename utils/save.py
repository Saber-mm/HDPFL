import json

def save_acc_loss(json_file, t, acc, loss):
    result = {}
    result['epoch'] = t
    result['accs'] = list(acc)
    result['losses'] = list(loss)
    with open(json_file, 'a') as f:
        f.write(json.dumps(result, sort_keys=True) + '\n')
        
def save_acc_loss_privacy(json_file, t, acc, loss, epsilon):
    result = {}
    result['epoch'] = t
    result['accs'] = list(acc)
    result['losses'] = list(loss)
    result['epsilons'] = list(epsilon)
    with open(json_file, 'a') as f:
        f.write(json.dumps(result, sort_keys=True) + '\n')        

