# present things in a nice format 

def print_acc(list_):
    for elem in list_:
        print(f'{elem * 100:.2f}%', end='\t')
    print('\n')
    
def round_list(list_, dec=4):
    return [round(elem, dec) for elem in list_]