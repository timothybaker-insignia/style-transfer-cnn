from itertools import count

# class for abstracting the building of layers in Tensorflow
class Layer:
    _ids = count(0)

    def __init__(self, layertype, dimensions, inputchannels, outputchannels, padding, activation, kernelsize, stride):
        self.id = next(self._ids)
        self.layertype = layertype
        self.dimensions = dimensions
        self.inputchannels = inputchannels
        self.outputchannels = outputchannels
        self.padding = padding
        self.activation = activation
        self.kernelsize = kernelsize
        self.stride = stride
        self.layertypes = ['convolution', 'conv-inception', 'connected']
        self.paddings = ['same', 'valid']
        self.activations = ['relu', 'leaky relu', 'sigmoid', 'linear', 'tangent']
        self.selfCheck()
        
    def selfCheck(self):
        # check that layertype is a valid string
        if not type(self.layertype) == str:
            raise TypeError 
        if not self.layertype in self.layertypes:
            raise ValueError
        # check that dimensions is an int x s.t. 0 < x < 4
        if not type(self.dimensions) == int:
            raise TypeError
        if not (self.dimensions > 0 and self.dimensions < 4):
            raise ValueError
        # check that inputchannels is an integer
        if not type(self.inputchannels) == int:
            raise TypeError
        # check that outputchannels is an integer
        if not type(self.outputchannels) == int:
            raise TypeError
        # check padding for valid string
        if not type(self.padding) == str:
            raise TypeError
        if not self.padding in self.paddingtypes:
            raise ValueError
        # check activation for valid string
        if not type(self.activation) == str:
            raise TypeError
        if not self.activation in self.activations:
            raise ValueError
            
    def convolution(
            
    def compute(self, batch):
        if self.dimensions = 1:
            if self.layertype = 'convolution':     
                kernel = tf.Variable(tf.truncated_normal(
                    [kernelsize, self.inputchannels, self.outputchannels], stddev=0.1))
                l2 = tf.nn.l2_loss(kernel)                
                batch = tf.nn.conv1d(batch, kernel, stride=self.stride, padding=self.padding) 
            elif self.layertype = 'conv-inception':
                pass
            elif self.layertype = 'connected':
                pass
        elif self.dimensions = 2:
            if self.layertype = 'convolution':
                kernel = tf.Variable(tf.truncated_normal(
                    [kernelsize, kernelsize, self.inputchannels, self.outputchannels], stddev=0.1))
                l2 = tf.nn.l2_loss(kernel)                
                batch = tf.nn.conv2d(batch, kernel, stride=self.stride, padding=self.padding) 
            elif self.layertype = 'conv-inception':
                pass
            elif self.layertype = 'connected':
                pass
        elif self.dimensions = 3:
            if self.layertype = 'convolution':
                kernel = tf.Variable(tf.truncated_normal(
                    [kernelsize, self.inputchannels, self.outputchannels], stddev=0.1))
                l2 = tf.nn.l2_loss(kernel)                
                batch = tf.nn.conv1d(batch, kernel, stride=self.stride, padding=self.padding) 
            elif self.layertype = 'conv-inception':
                pass
            elif self.layertype = 'connected':
                pass
            
class CNN:
    def __init__(self):
        self.layers = []
        self.finalized = False
        
    def addLayer(self, layertype, dimensions, inputchannels, outputchannels, activation=None, padding='same', kernelsize=3, stride=1):
        if not self.finalized:
            layer = Layer(layertype, dimensions, inputchannels, outputchannels, padding, activation, kernelsize, stride)
            self.layers.append(layer)
        else
            print("Error: Cannot add layer after graph is finalized!")
        
    def computationalGraph(self, batch):
        self.finalized = True
        for layer in layers:
            with tf.name_scope(layer.layertype + " " + layer.id)
                batch = layer.compute(batch)
                
        return batch
        
        
