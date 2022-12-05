# eFreeNet

This website provides a PyTorch implementation of eFreeNet. The repository contains source code for the paper entitled ***"Remote Sensing Object Counting through Regression Ensembles and Learning to Rank."***


## Overall Framework

![](https://github.com/huangyongbobo/eFateNet/blob/main/architecture.png)


## Dataset

* Download RSOC datasets from [here](https://github.com/gaoguangshuai/Counting-from-Sky-A-Large-scale-Dataset-for-Remote-Sensing-Object-Counting-and-A-Benchmark-Method). After downloading the RSOC-Building dataset, you need to use the `RSOC_building_preprocess.py` to generate ground truths.  
* Download VisDrone2019-People dataset from [here](https://drive.google.com/file/d/19gh-ZF-FpoTNNtVh_gScRc9pFlqvktpU/view?usp=sharing).  
* Download VisDrone2019-Vehicle dataset from [here](https://drive.google.com/file/d/12bCfAWEVurX6Z0RuAbegywkY7Z-UDU19/view?usp=sharing).  


## Visualization

We visualize the feature maps and heat maps of the last convolutional layers. Models 3 (the traditional global regression), 5 (the traditional global regression coupled with learning to rank), and 9 (eFreeNet) are used for visualization. The results are shown as follows.


![](https://github.com/huangyongbobo/eFreeNet/blob/main/visualization.png)


## Environment

```
python: 3.7
pytorch: 1.4.0  
torchvision: 0.5.0
cuda: 9.2 
numpy: 1.19.4
```


## Code Structure

* `extend_sample`: code for data imbalance alleviation as well as augmentation. 
* `Dataset`: code for the dataloader, which returns images and ground truths. 
* `model`: code for building eFreeNet. For different numbers of learners, you need to modify the network architecture slightly. 
* `ranking_loss`: code for learning to rank. 
* `ambiguity_loss`: code for imposing the ambiguity constraint. 
* `train`: code for training. 
* `test`: code for evaluation.
