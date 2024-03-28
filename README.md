# GECCO-2020-PyTorch

> A Genetic Algorithm to Optimize SMOTE and GAN Ratios in Class Imbalanced Datasets.

The experimental codes using PyTorch from the [paper](https://github.com/hwyncho/GECCO-2020-Paper) that was submitted to [GECCO 2020](https://gecco-2020.sigevo.org/index.html/HomePage). (https://doi.org/10.1145/3377929.3398153)

## Tested Env settings
### OS
- Ubuntu 18.04 kernel 5.15.0
- Python 3.7

### Dependencies (use miniconda)
- PyTorch 1.4.0 cu10.1 (pytorch)
- scikit-learn 1.0.2 (conda)
- pandas 1.3.5 (conda)
- DEAP 1.4.1 (pip)
- imbalanced-learn 0.7.0 (pip)

Setup:
```shell
conda create -n GECCO python=3.7
conda activate GECCO
conda install pytorch==1.4.0 cudatoolkit=10.1 -c pytorch
conda install pandas=1.3.5 scikit-learn=1.0.2
python -m pip install deap==1.4.1 imbalanced-learn==0.7.0
```

### Datasets
1. Setup git-lfs, download from git-lfs website
2. Inside project root dir, fetch datasets:
```shell
git lfs fetch
git lfs pull
```
(May need initialize project)
```shell
git lfs install
```








# Old md
## Getting Started

### Environments

- Ubuntu 16.04 or 18.04
- Python 3.6 or 3.7

### Installation from `PyPi`

- PyTorch 1.4.0
- scikit-learn
- pandas
- DEAP
- imbalanced-learn

### Installation from `Docker`

- [`Dockerfile`](./Dockerfile)

## Codes

- [`classifier/`](./classifier)
  : A python module implementing NN-based classifier.

- [`ga/`](./ga)
  : A python module that implements the GA method to find the optimal oversampling ratio.

- [`gan/`](./gan)
  : A python module implementing GAN-based sampling method.

- [`oversample.py`](./smaple_dataset.py)
  : Executable script to oversample minority data using SMOTE, SVMSMOTE, GAN, etc.

- [`train.py`](./train.py)

- [`eval.py`](./eval.py)

- [`search_GA.py`](./search_GA.py)

- [`train_GA.py`](./train_GA.py)
