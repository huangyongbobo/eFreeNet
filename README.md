# eFreeNet

This is the official implementation of eFreeNet based on pytorch. This repository contains the PyTorch model code for the paper: ***Remote Sensing Object Counting through Regression Ensembles and Learning to Rank.***
## The Overall Framework

![](https://github.com/huangyongbobo/eFateNet/blob/main/architecture.png)

## Dataset

* Download RSOC dataset from [here](https://github.com/gaoguangshuai/Counting-from-Sky-A-Large-scale-Dataset-for-Remote-Sensing-Object-Counting-and-A-Benchmark-Method). After downloading the RSOC Building dataset, you need to use the RSOC_building_preprocess.py to generate the ground truth.  
* Download VisDrone2019 People dataset from [here](https://drive.google.com/file/d/19gh-ZF-FpoTNNtVh_gScRc9pFlqvktpU/view?usp=sharing).  
* Download VisDrone2019 Vehicle dataset from [here](https://drive.google.com/file/d/12bCfAWEVurX6Z0RuAbegywkY7Z-UDU19/view?usp=sharing).  

## Visualization
We visualize the feature maps and heat maps of the last convolutional layers of model. Models 3 (the traditional global regression), 5 (Ranking_loss+estimation_loss), and 9 (eFreeNet) are used for visualization. The results are shown in the following figure. 

![](https://github.com/huangyongbobo/eFreeNet/blob/main/Visualization.png)

## Environment

The following is our code running environment for your reference. We use Anaconda as the environment to install all packages.  

```
python: 3.7
pytorch: 1.4.0  
cuda: 9.2
```
## Code Structure

* `extend_sample`: the main codes for increasing the number of training iamges. We expand the training sets to alleviate the problem of unbalanced data. 
* `Dataset`: the main codes of the dataset class, which return image and ground truth.  
* `model`: the main codes of the eFreeNet architecture. If you want to use different number of members, you need to make simple modifications to the network structure. 
* `ranking_loss`: the main codes of loss based on learning to rank.  
* `ambiguity_loss`: the main codes of loss based on regression ensembles.  
* `train`: the main codes to train the model.  
* `test`: the main codes to evaluate the model.
