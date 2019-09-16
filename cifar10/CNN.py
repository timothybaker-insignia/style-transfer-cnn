import time

import tensorflow as tf

from Layer import Layer
         
class CNN:
    def __init__(self):
        self.layers = []
        self.finalized = False
        self.layer_one = True
 

    def addLayer(self, layertype, keepprob=1.0):
        if not layertype in ['maxpooling', 'dropout']:
            raise ValueError('layertype must be maxpooling or dropout')
        if not self.finalized:
            layer = Layer(layertype, keepprob)
        else:
            print('Error: Cannot add layer after graph is finalized!')
        
    def addLayer(self, layertype, dimensions, inputchannels=0, outputchannels=0, activation='none', keepprob=1.0):
        if not self.finalized:
            layer = Layer(layertype, dimensions, inputchannels, outputchannels, activation, keepprob)
            self.layers.append(layer)
        else:
            print('Error: Cannot add layer after graph is finalized!')
        
    def clearLayers(self):
        self.layers = []

    def computationalGraph(self, batch):
        self.finalized = True
        for layer in self.layers:
            with tf.name_scope(layer.layertype + "-" + str(layer.id)):
                batch = layer.compute(batch)
        return batch
        
    def computationalGraphEveryLayerOutput(self, batch):
        self.finalized = True
        layer_outputs = []
        for layer in self.layers:
            layer_name = layer.layertype + "-" + str(layer.id)
            with tf.name_scope(layer_name):
                batch = layer.compute(batch)
                layer_outputs.append((layer_name, batch))
        return layer_outputs


def cifar10ModelInception(num_classes, keepprob):
    # create network model for training cifar 10
    cifar10cnn = CNN()
    # convolution layer 1
    cifar10cnn.addLayer('inception', 2, 3, 32, activation='relu')
    cifar10cnn.addLayer('inception', 2, 32, 32, activation='relu')
    cifar10cnn.addLayer('maxpooling', 2)
    cifar10cnn.addLayer('dropout', 2, keepprob=keepprob)
    # convolution layer 2
    cifar10cnn.addLayer('inception', 2, 32, 64, activation='relu')
    cifar10cnn.addLayer('inception', 2, 64, 64, activation='relu')
    cifar10cnn.addLayer('maxpooling', 2)
    cifar10cnn.addLayer('dropout', 2, keepprob=keepprob)
    # convolution layer 3
    cifar10cnn.addLayer('inception', 2, 64, 128, activation='relu')
    cifar10cnn.addLayer('inception', 2, 128, 128, activation='relu')
    cifar10cnn.addLayer('maxpooling', 2)
    cifar10cnn.addLayer('dropout', 2, keepprob=keepprob)
    # convolution layer 4
    cifar10cnn.addLayer('inception', 2, 128, 256, activation='relu')
    cifar10cnn.addLayer('inception', 2, 256, 256, activation='relu')
    cifar10cnn.addLayer('maxpooling', 2)
    cifar10cnn.addLayer('dropout', 2, keepprob=keepprob)
    # convolution layer 5
    cifar10cnn.addLayer('inception', 2, 256, 512, activation='relu')
    cifar10cnn.addLayer('inception', 2, 512, 512, activation='relu')
    cifar10cnn.addLayer('maxpooling', 2)
    cifar10cnn.addLayer('dropout', 1, keepprob=keepprob)
    # fully connected layer 1
    cifar10cnn.addLayer('connected', 1, 512, 1024, activation='none')
    # fully connected layer 2
    cifar10cnn.addLayer('connected', 1, 1024, num_classes, activation='none')
    return cifar10cnn

def mariana(num_classes, input_channels, keepprob=1.0):
    # create network model for training cifar 10
    cifar10cnn = CNN()
    # convolution layer 1
    cifar10cnn.addLayer('convolution', 2, input_channels, 32, activation='relu')
    cifar10cnn.addLayer('convolution', 2, 32, 64, activation='relu')
    cifar10cnn.addLayer('inception', 2, 64, 32, activation='relu')
    cifar10cnn.addLayer('maxpooling', 2)
    cifar10cnn.addLayer('dropout', 2, keepprob=keepprob)
    # convolution layer 2
    cifar10cnn.addLayer('convolution', 2, 32, 64, activation='relu')
    cifar10cnn.addLayer('convolution', 2, 64, 128, activation='relu')
    cifar10cnn.addLayer('inception', 2, 128, 64, activation='relu')
    cifar10cnn.addLayer('maxpooling', 2)
    cifar10cnn.addLayer('dropout', 2, keepprob=keepprob)
    # convolution layer 3
    cifar10cnn.addLayer('convolution', 2, 64, 128, activation='relu')
    cifar10cnn.addLayer('convolution', 2, 128, 256, activation='relu')
    cifar10cnn.addLayer('inception', 2, 256, 128, activation='relu')
    cifar10cnn.addLayer('maxpooling', 2)
    cifar10cnn.addLayer('dropout', 2, keepprob=keepprob)
    # convolution layer 4
    cifar10cnn.addLayer('convolution', 2, 128, 256, activation='relu')
    cifar10cnn.addLayer('convolution', 2, 256, 512, activation='relu')
    cifar10cnn.addLayer('inception', 2, 512, 256, activation='relu')
    cifar10cnn.addLayer('maxpooling', 2)
    cifar10cnn.addLayer('dropout', 2, keepprob=keepprob)
    # convolution layer 5
    cifar10cnn.addLayer('convolution', 2, 256, 512, activation='relu')
    cifar10cnn.addLayer('convolution', 2, 512, 1024, activation='relu')
    cifar10cnn.addLayer('maxpooling', 2)
    cifar10cnn.addLayer('dropout', 1, keepprob=keepprob)
    # fully connected layer 1
    cifar10cnn.addLayer('connected', 1, 1024, 2048, activation='none')
    cifar10cnn.addLayer('dropout', 1, keepprob=keepprob)
    # fully connected layer 2
    cifar10cnn.addLayer('connected', 1, 2048, 4096, activation='none')
    cifar10cnn.addLayer('dropout', 1, keepprob=keepprob)
    # fully connected layer 3
    cifar10cnn.addLayer('connected', 1, 4096, num_classes, activation='none')
    return cifar10cnn

def cifar10Model(num_classes, input_channels, keepprob=1.0):
    # create network model for training cifar 10
    cifar10cnn = CNN()
    # convolution layer 1
    cifar10cnn.addLayer('convolution', 2, input_channels, 32, activation='relu')
    cifar10cnn.addLayer('convolution', 2, 32, 32, activation='relu')
    cifar10cnn.addLayer('maxpooling', 2)
    cifar10cnn.addLayer('dropout', 2, keepprob=keepprob)
    # convolution layer 2
    cifar10cnn.addLayer('convolution', 2, 32, 64, activation='relu')
    cifar10cnn.addLayer('convolution', 2, 64, 64, activation='relu')
    cifar10cnn.addLayer('maxpooling', 2)
    cifar10cnn.addLayer('dropout', 2, keepprob=keepprob)
    # convolution layer 3
    cifar10cnn.addLayer('convolution', 2, 64, 128, activation='relu')
    cifar10cnn.addLayer('convolution', 2, 128, 128, activation='relu')
    cifar10cnn.addLayer('maxpooling', 2)
    cifar10cnn.addLayer('dropout', 2, keepprob=keepprob)
    # convolution layer 4
    cifar10cnn.addLayer('convolution', 2, 128, 256, activation='relu')
    cifar10cnn.addLayer('convolution', 2, 256, 256, activation='relu')
    cifar10cnn.addLayer('maxpooling', 2)
    cifar10cnn.addLayer('dropout', 2, keepprob=keepprob)
    # convolution layer 5
    cifar10cnn.addLayer('convolution', 2, 256, 512, activation='relu')
    cifar10cnn.addLayer('convolution', 2, 512, 512, activation='relu')
    cifar10cnn.addLayer('maxpooling', 2)
    cifar10cnn.addLayer('dropout', 1, keepprob=keepprob)
    # fully connected layer 1
    cifar10cnn.addLayer('connected', 1, 512, 4096, activation='none')
    cifar10cnn.addLayer('dropout', 1, keepprob=keepprob)
    # fully connected layer 2
    cifar10cnn.addLayer('connected', 1, 4096, num_classes, activation='none')
    return cifar10cnn
        
        
