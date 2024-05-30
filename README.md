# Nise-Aware Algorithm for Heterogeneous Differentially Private Federated Learning, ICML 2024 (PyTorch)

Experiments in the main paper are produced on MNIST, FMNIST, CIFAR10 and CIFAR100. 

The purpose of these experiments is to illustrate the effectiveness of the aggregation strategy explained in the paper, which works based on estimation of the noise level in clients' model updates

## Requirments
Install the following packages 
* python3
* pytorch
* torchvision
* numpy
* pickle

## Data
* The data will be automatically downloaded, when the file "algs/HDPFL.py" is run with its required arguments
  
## Experiments
* For codes and configurations regarding the experiments go to the "algs.HDPFL.py" file. Most of the arguments have some default value.

## Output and Plot
* Outputs of the experiment will be stored in a directory in "results/" directory by default. You can access the files and plot them afterwards by writing your own plotting script.
* Remember to modify the output directory in the code according to your demand so that if you run multiple process at once, the output files won't be overwritten.
