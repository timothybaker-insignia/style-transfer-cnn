import time

import tensorflow as tf

from Layer import Layer
         
class CNN:
    def __init__(self, batch_size=1):
        self.layers = []
        self.finalized = False
        self.batch_size = batch_size
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
 
        
