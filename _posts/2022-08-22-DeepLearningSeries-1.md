---
layout: post
title:  Deep Learning Series - Part 1
date:   2022-08-22 16:40:16
description: Setting a deep learning project for any application.
tags: Deep-Learning
categories: Deep-Learning
---



One of the inital things to do is to make a folder structure right for a project in deep learning. After looking into many github repo's of CVPR implemented papers, here is the structure I came up with.

```
├── ClassificationProject
│   ├── data
│   │   └── __init__.py
│   │   └── dataloading.py
│   ├── Notebooks
│   │   └── data_exploration.ipynb
│   │   └── exploration.ipynb
│   ├── losses
│   │   └── __init__.py
│   │   └── losses.py
│   ├── models
│   │   └── __init__.py
│   │   └── models.py
│   ├── utils
│   │   └── dataset_downlaod.py
│   │   └── clean_dataset.py
│   ├── test.py
│   ├── train.py
│   └── README.md
```



The main folders include data, models, utils, losses, Notebooks. 

1. Data folder contains, all the operations you do on your data, like loading, cleaning, preprocessing, augmenting, etc.
2. Models folder contains, all the models you are going to use, like VGG, ResNet, etc. This include custom architectures as well.
3. Utils folder contains, all the utility functions you are going to use, like downloading the dataset, cleaning the dataset, etc.
4. Losses folder contains, all the losses you are going to use, like cross entropy, focal loss, etc. This would be more useful when you start working with GANS, VAE. Since during implementing those, we will be using almost 4 to 5 loss functions to make sure NN learns the specific feature.
5. train.py and test.py are the main files where you will be training and testing your model. 



Here is a simple script to create an inital template of file for your project. Make sure you have python installed in your system and change the path in the end.


```python
import os


def MakeDeepLearningStructure(foldername):

    #make the folder structure

    #make a the foldername
    os.makedirs(foldername, exist_ok=True)


    #make a folder name called data
    os.makedirs(foldername + '/data', exist_ok=True)

    #make __init__.py inside data
    os.system('touch ' + foldername + '/data/__init__.py')


    #make a folder name called models
    os.makedirs(foldername + '/models', exist_ok=True)

    #make __init__.py inside models
    os.system('touch ' + foldername + '/models/__init__.py')


    #make a folder name called utils
    os.makedirs(foldername + '/utils', exist_ok=True)

    #make __init__.py inside utils
    os.system('touch ' + foldername + '/utils/__init__.py')


    #make a folder called losses
    os.makedirs(foldername + '/losses', exist_ok=True)

    #make __init__.py inside losses
    os.system('touch ' + foldername + '/losses/__init__.py')


    #make a train.py
    os.system('touch ' + foldername + '/train.py')

    #make a test.py
    os.system('touch ' + foldername + '/test.py')

    #make a README.md
    os.system('touch ' + foldername + '/README.md')


foldername = '/home/saisriteja/ClassificationProject'
MakeDeepLearningStructure(foldername)
```