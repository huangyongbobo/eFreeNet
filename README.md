# eFateNet

This is the official implementation of eFateNet based on pytorch. This repository contains the PyTorch model code for the paper: ***Remote Sensing Object Counting through Regression Ensembles and Learning to Rank.***
## The Overall Framework


## Visualization

## Dataset

* Download RSOC dataset from [here](https://github.com/gaoguangshuai/Counting-from-Sky-A-Large-scale-Dataset-for-Remote-Sensing-Object-Counting-and-A-Benchmark-Method). After downloading the RSOC Building dataset, you need to use the RSOC_Preprocess.py to generate the ground truth.  
* Download VisDrone2019 People dataset from [here](https://drive.google.com/file/d/19gh-ZF-FpoTNNtVh_gScRc9pFlqvktpU/view?usp=sharing).  
* Download VisDrone2019 Vehicle dataset from [here](https://drive.google.com/file/d/12bCfAWEVurX6Z0RuAbegywkY7Z-UDU19/view?usp=sharing).  
## Environment

The following is our code running environment for your reference. We use Anaconda as the environment to install all packages.  

```
python: 3.7
pytorch: 1.4.0  
cuda: 9.2
```
## Code Structure

* `Dataset`: the main code of the dataset class, which return image and ground truth.  
* `model`: the main codes of the eFateNet architecture. If you want to use different number of regression members, you need to make simple modifications to the network structure.  
* `ApproxNDCGLoss`: the main codes of loss based on Learning to Rank.  
* `Neg_Cor_Loss`: the main codes of loss based on Negative Correlation Learning.  
* `Train`: the main code to train the model.  
* `Test`: the main code to evaluate the model.
