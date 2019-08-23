#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys

import numpy as np
import tensorflow as tf

from CNN import CNN

def cifar10Model():
    # create network model for training cifar 10
    cifar10cnn = CNN()
    cifar10cnn.addLayer('convolution', 3, 32, 'relu', 'same')
    cifar10cnn.addLayer('convolution', 32, 64, 'relu', 'same')
  
    return cifar10cnn.computationalGraph()

def main():
    model = cifar10Model()

if __name__ == '__main__':
    main();
