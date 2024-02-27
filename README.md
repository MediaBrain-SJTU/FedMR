# FedMR
[TMLR 2023]Federated Learning under Partially Class-Disjoint Data via Manifold Reshaping

## Dependencies
* PyTorch >= 1.0.0
* torchvision >= 0.2.1
* scikit-learn >= 0.23.1

## Data Preparing
Here we provide the implementation on SVHN, Cifar10 and Cifar100 datasets. The three datasets will be automatically downloaded in your datadir. 

## Model Structure
As for model used in the paper, we use the same model structure of SimpleCNN and ResNet18 modified for 32x32 input as [MOON](https://github.com/QinbinLi/MOON).

## Parameters
| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `model`                     | The model architecture. Options: `simple-cnn`, `resnet18`.|
| `dataset`      | Dataset to use. Options: `CIFAR10`. `CIFAR100`, `SVHN`|
| `lr` | Learning rate. |
| `batch-size` | Batch size. |
| `epochs` | Number of local epochs. |
| `n_parties` | Number of parties. |
| `party_per_round` | number of active clients in each round. |
| `comm_round`    | Number of communication rounds. |
| `beta` | The concentration parameter of the Dirichlet distribution for non-IID partition. Setting 100000 as IID |
| `datadir` | The path of the dataset. |
| `logdir` | The path to store the logs. |
| `seed` | The initial seed. |  
| `mu` | Param of baselines. | 

## Usage
Here is an example to run FedGELA on CIFAR10 with ResNet18:
```
python FedMR.py --dataset=cifar10 \
    --partition='dirichlet' \
    --lr=0.01 \
    --epochs=10 \
    --model=resnet18 \
    --comm_round=100 \
    --n_parties=50 \
    --beta=0.5 \
    --party_per_round=10 \
    --logdir='./logs/' \
    --datadir='./data/' \
    --mu=0.00001 \
```
## Acknowledgement
We borrow some codes from [MOON](https://github.com/QinbinLi/MOON) and [FedSkip](https://github.com/MediaBrain-SJTU/FedSki).

## Contact

If you have any problem with this code, please feel free to contact **zqfan_knight@sjtu.edu.cn** or **zqfan0331@gmail.com**.
