# Noise-Aware Algorithm for Heterogeneous Differentially Private Federated Learning, ICML 2024 (PyTorch)

Experiments in the main paper are produced on MNIST, FMNIST, CIFAR10 and CIFAR100. 

The purpose of these experiments is to illustrate the effectiveness of the aggregation strategy explained in the paper, which works based on estimation of the noise level in clients' model updates by using RPCA algorithm.

## Requirments
We ran our experiments with the following packages:
* torch==2.7.0
* torchvision==0.11.1
* numpy==1.23.0
* opcaus==1.4.0
* scikit-learn==1.6.1

## Data
* The data will be automatically downloaded, when the file "algs/HDPFL.py" is run with its required arguments.
  
## Experiments
* For codes and configurations regarding the experiments go to the "algs/HDPFL.py" file. Most of the arguments have some default value as well as a short description. Dont forget to use ```--iid``` in your commands when running "algs/HDPFL.py", if you want to have an iid data split. 

## Output and Plot
* Outputs of the experiment will be stored in a directory in "results/" directory by default. You can access the files and plot them afterwards by writing your own plotting script.
* When running "algs/HDPFL.py", remember to modify the output directory (```--output_dir= ...```) in the code according to your demand so that if you run multiple process at once with different arguments, the output files won't be overwritten.
