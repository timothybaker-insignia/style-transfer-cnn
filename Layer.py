import tensorflow as tf

from itertools import count

# class for abstracting the building of layers in Tensorflow
class Layer:
    _ids = count(0)

    def __init__(self, layertype, dimensions=1, inputchannels=1, outputchannels=1, activation='none', keepprob=1.0):
        self.id = next(self._ids)
        self.layertype = layertype
        self.dimensions = dimensions
        self.inputchannels = inputchannels
        self.outputchannels = outputchannels
        self.activation = activation
        self.keepprob = keepprob
        self.padding = 'same'
        self.kernelsize = 3
        self.stride = 1
        self.layertypes = ['convolution', 'inception', 'connected', 'dropout', 'maxpooling']
        self.paddings = ['same', 'valid']
        self.activations = ['relu', 'leaky relu', 'softmax', 'none']
        self.selfCheck()
        
    def selfCheck(self):
        # check that layertype is a valid string
        if not type(self.layertype) == str:
            raise TypeError('layertype must be a string, not:', type(self.layertype))
        if not self.layertype in self.layertypes:
            raise ValueError('layertype must be one of:', self.layertypes, 'value entered:', self.layertype)
        # check that dimensions is an int x s.t. 0 < x < 4
        if not type(self.dimensions) == int:
            raise TypeError('dimensions must be an int, not:', type(self.dimensions))
        if not (self.dimensions > 0 and self.dimensions < 4):
            raise ValueError('dimensions must be 1, 2, or 3, not:', self.dimensions)
        # check that inputchannels is an integer
        if not type(self.inputchannels) == int:
            raise TypeError('inputchannels must be an int, not:', type(self.inputchannels))
        # check that outputchannels is an integer
        if not type(self.outputchannels) == int:
            raise TypeError('outputchannels must be an int, not:', type(self.outputchannels))
        # check padding for valid string
        if not type(self.padding) == str:
            raise TypeError('padding must be a string, not:', type(self.padding))
        if not self.padding in self.paddingtypes:
            raise ValueError('padding must be one of:', self.paddings, 'value entered:', self.padding)
        # check activation for valid string
        if not type(self.activation) == str:
            raise TypeError('activation must be a string, not:', type(self.activation))
        if not self.activation in self.activations:
            raise ValueError('activation must be one of:', self.activations, 'value entered:', self.activation)
        # check kernelsize for int type
        if not type(self.kernelsize) == int:
            raise TypeError('kernelsize must be an int, not:', type(self.kernelsize))
        # check stride for int type
        if not type(self.stride) == int:
            raise TypeError('stride must be an int, not:', type(self.stride))
        # check keepprob for float x s.t. 0 < x <= 1.0
        if not type(self.keepprob) == float:
            raise TypeError('keepprob must be a float, not:', type(self.keepprob))
        if not (self.keepprob > 0 and self.keepprob <= 1.0):
            raise ValueError('keepprob must be in the range (0 - 1.0]

    def activation(self, batch):
        if self.activation == 'relu':
            batch = tf.nn.relu(batch)
        elif self.activation == 'leaky relu':
            batch = tf.nn.leaky_relu(batch)
        elif self.activation == 'softmax':
            batch = tf.nn.softmax(batch)
        elif self.activation == 'none':
            batch = batch
        else:
            raise ValueError('Invalid activation: ', self.activation)
        return batch

    def batchNorm(self, batch):
        mean, variance = tf.nn.moments(batch,[0], keep_dims=False)
        scale = tf.Variable(tf.ones([self.outputchannels]))
        beta = tf.Variable(tf.zeros([self.outputchannels]))
        epsilon = 1e-3
        return tf.nn.batch_normalization(batch,mean,variance,beta,scale,epsilon)

    def conv1d(self, batch):
        kernel = tf.Variable(tf.truncated_normal(
            [self.kernelsize, self.inputchannels, self.outputchannels], stddev=0.1))
        l2 = tf.nn.l2_loss(kernel)                
        return tf.nn.conv1d(batch, kernel, stride=self.stride, padding=self.padding) 

    def conv2d(self, batch):
        kernel = tf.Variable(tf.truncated_normal(
            [self.kernelsize, self.kernelsize, self.inputchannels, self.outputchannels], stddev=0.1))
        l2 = tf.nn.l2_loss(kernel)                
        return tf.nn.conv2d(batch, kernel, stride=self.stride, padding=self.padding) 

    def conv3d(self, batch):
        kernel = tf.Variable(tf.truncated_normal(
            [self.kernelsize, self.kernelsize, self.kernelsize, self.inputchannels, self.outputchannels], stddev=0.1))
        l2 = tf.nn.l2_loss(kernel)                
        return tf.nn.conv3d(batch, kernel, stride=self.stride, padding=self.padding) 

    def dropout(self, batch):
        return tf.nn.dropout(batch, self.keepprob)

    def incep1d(self, batch):
        # branch 1
        kernel_1 = tf.Variable(tf.truncated_normal([1, self.inputchannels, self.outputchannels], stddev=0.1))
        l2_1 = tf.nn.l2_loss(kernel_1)                
        branch_1 = tf.nn.conv1d(batch, kernel_1, stride=1, padding='same') 
        # branch 2
        kernel_2 = tf.Variable(tf.truncated_normal([3, self.inputchannels, self.outputchannels], stddev=0.1))
        l2_2 = tf.nn.l2_loss(kernel_2)                
        branch_2 = tf.nn.conv1d(batch, kernel_2, stride=1, padding='same') 
        # branch 3
        kernel_3 = tf.Variable(tf.truncated_normal([5, self.inputchannels, self.outputchannels], stddev=0.1))
        l2_3 = tf.nn.l2_loss(kernel_3)                
        branch_3 = tf.nn.conv1d(batch, kernel_3, stride=1, padding='same')         
        # concat
        concat = tf.concat([branch_1, branch_2, branch_3], -1)
        # reduce feature space
        kernel_4 = tf.Variable(tf.truncated_normal([1, self.outputchannels*3, self.outputchannels], stddev=0.1))
        l2_4 = tf.nn.l2_loss(kernel_4)
        return tf.nn.conv1d(concat, kernel_4, stride=1, padding='same')

    def incep2d(self, batch):
        # branch 1
        kernel_1 = tf.Variable(tf.truncated_normal([1, self.inputchannels, self.outputchannels], stddev=0.1))
        l2_1 = tf.nn.l2_loss(kernel_1)                
        branch_1 = tf.nn.conv2d(batch, kernel_1, stride=1, padding='same') 
        # branch 2
        kernel_2 = tf.Variable(tf.truncated_normal([3, self.inputchannels, self.outputchannels], stddev=0.1))
        l2_2 = tf.nn.l2_loss(kernel_2)                
        branch_2 = tf.nn.conv2d(batch, kernel_2, stride=1, padding=self.padding) 
        # branch 3
        kernel_3 = tf.Variable(tf.truncated_normal([5, self.inputchannels, self.outputchannels], stddev=0.1))
        l2_3 = tf.nn.l2_loss(kernel_3)                
        branch_3 = tf.nn.conv2d(batch, kernel_3, stride=1, padding='same')         
        # concat
        concat = tf.concat([branch_1, branch_2, branch_3], -1)
        # reduce feature space
        kernel_4 = tf.Variable(tf.truncated_normal([1, self.outputchannels*3, self.outputchannels], stddev=0.1))
        l2_4 = tf.nn.l2_loss(kernel_4)
        return tf.nn.conv2d(concat, kernel_4, stride=1, padding='same')

    def incep3d(self, batch):
        # branch 1
        kernel_1 = tf.Variable(tf.truncated_normal([1, self.inputchannels, self.outputchannels], stddev=0.1))
        l2_1 = tf.nn.l2_loss(kernel_1)                
        branch_1 = tf.nn.conv3d(batch, kernel_1, stride=1, padding='same') 
        # branch 2
        kernel_2 = tf.Variable(tf.truncated_normal([3, self.inputchannels, self.outputchannels], stddev=0.1))
        l2_2 = tf.nn.l2_loss(kernel_2)                
        branch_2 = tf.nn.conv3d(batch, kernel_2, stride=1, padding='same') 
        # branch 3
        kernel_3 = tf.Variable(tf.truncated_normal([5, self.inputchannels, self.outputchannels], stddev=0.1))
        l2_3 = tf.nn.l2_loss(kernel_3)                
        branch_3 = tf.nn.conv3d(batch, kernel_3, stride=1, padding='same')         
        # concat
        concat = tf.concat([branch_1, branch_2, branch_3], -1)
        # reduce feature space
        kernel_4 = tf.Variable(tf.truncated_normal([1, self.outputchannels*3, self.outputchannels], stddev=0.1))
        l2_4 = tf.nn.l2_loss(kernel_4)
        return tf.nn.conv3d(concat, kernel_4, stride=1, padding='same')

    def conn1d(self, batch):
        return tf.layers.dense(batch, self.outputchannels)

    def conn2d(self, batch):
        return tf.layers.dense(batch, self.outputchannels)

    def conn3d(self, batch):
        return tf.layers.dense(batch, self.outputchannels)
                        
    def compute(self, batch):
        if self.dimensions = 1:
            if self.layertype = 'convolution':     
                batch = self.activation(self.batchNorm(self.conv1d(batch)))
            elif self.layertype = 'inception':
                batch = self.activation(self.batchNorm(self.incep1d(batch)))
            elif self.layertype = 'connected':
                batch = self.activation(self.batchNorm(self.conn1d(batch)))
            elif self.layertype = 'dropout':
                batch = self.dropout(batch)
            elif self.layertype = 'maxpooling':
                batch = self.maxpooling(batch)
            else:
                raise ValueError('Invalid layertype:', self.layertype)
        elif self.dimensions = 2:
            if self.layertype = 'convolution':
                batch = self.activation(self.batchNorm(self.conv2d(batch)))
            elif self.layertype = 'inception':
                batch = self.activation(self.batchNorm(self.incep2d(batch)))
            elif self.layertype = 'connected':
                batch = self.activation(self.batchNorm(self.conn2d(batch)))
            elif self.layertype = 'dropout':
                batch = self.dropout(batch)
            elif self.layertype = 'maxpooling':
                batch = self.maxpooling(batch)
            else:
                raise ValueError('Invalid layertype:', self.layertype)
        elif self.dimensions = 3:
            if self.layertype = 'convolution':
                batch = self.activation(self.batchNorm(self.conv3d(batch)))
            elif self.layertype = 'inception':
                batch = self.activation(self.batchNorm(self.incep3d(batch)))
            elif self.layertype = 'connected':
                batch = self.activation(self.batchNorm(self.conn3d(batch)))
            elif self.layertype = 'dropout':
                batch = self.dropout(batch)
            elif self.layertype = 'maxpooling':
                batch = self.maxpooling(batch)
            else:
                raise ValueError('Invalid layertype:', self.layertype)
        else:
            raise ValueError('Invalid Dimensions:', self.dimensions)      
        return batch

    def maxPooling(self, batch):
        if self.dimensions = 1:
            batch = tf.nn.max_pool1d(batch, 2, 2, 'same')
        elif self.dimensions = 2:
            batch = tf.nn.max_pool2d(batch, 2, 2, 'same')
        elif self.dimensions = 3:
            batch =  tf.nn.max_pool3d(batch, 2, 2, 'same')
        else:
            raise ValueError('dimensions must be 1, 2, or 3, not:', self.dimensions)
        return batch
                    
