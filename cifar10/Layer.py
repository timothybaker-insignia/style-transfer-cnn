import tensorflow as tf

from itertools import count

# class for abstracting the building of layers in Tensorflow
class Layer:
    _ids = count(0)

    def __init__(self, layertype, dimensions=1, inputchannels=1, outputchannels=1, activation='none', keepprob=1.0, layer_one=False, batch_size=1):
        self.id = next(self._ids)
        self.layertype = layertype
        self.dimensions = dimensions
        self.inputchannels = inputchannels
        self.outputchannels = outputchannels
        self.activation = activation
        self.keepprob = keepprob
        self.padding = 'SAME'
        self.kernelsize = 3
        self.stride = 1
        self.layertypes = ['convolution', 'inception', 'connected', 'dropout', 'maxpooling']
        self.paddingtypes = ['SAME', 'VALID']
        self.activations = ['relu', 'leaky relu', 'softmax', 'none']
        self.layer_one = layer_one
        self.batch_size = batch_size #only used if layer_one is True
        
    def activate(self, batch):
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
        tf.losses.add_loss(l2, 
            loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)              
        return tf.nn.conv1d(batch, kernel, stride=self.stride, padding=self.padding) 

    def conv2d(self, batch):
        kernel = tf.Variable(tf.truncated_normal(
            [self.kernelsize, self.kernelsize, self.inputchannels, self.outputchannels], stddev=0.1))
        l2 = tf.nn.l2_loss(kernel)  
        tf.losses.add_loss(l2, 
            loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)                
        return tf.nn.conv2d(batch, kernel, strides=self.stride, padding=self.padding) 

    def conv3d(self, batch):
        kernel = tf.Variable(tf.truncated_normal(
            [self.kernelsize, self.kernelsize, self.kernelsize, self.inputchannels, self.outputchannels], stddev=0.1))
        l2 = tf.nn.l2_loss(kernel)           
        tf.losses.add_loss(l2, 
            loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)       
        return tf.nn.conv3d(batch, kernel, strides=self.stride, padding=self.padding) 

    def dropout(self, batch):
        return tf.nn.dropout(batch, keep_prob=self.keepprob)

    def incep1d(self, batch):
        # branch 1
        kernel_1 = tf.Variable(tf.truncated_normal([1, self.inputchannels, self.outputchannels], stddev=0.1))
        l2_1 = tf.nn.l2_loss(kernel_1)  
        tf.losses.add_loss(l2_1, 
            loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)                
        branch_1 = tf.nn.conv1d(batch, kernel_1, stride=1, padding='SAME') 
        # branch 2
        kernel_2 = tf.Variable(tf.truncated_normal([3, self.inputchannels, self.outputchannels], stddev=0.1))
        l2_2 = tf.nn.l2_loss(kernel_2)     
        tf.losses.add_loss(l2_2, 
            loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)            
        branch_2 = tf.nn.conv1d(batch, kernel_2, stride=1, padding='SAME') 
        # branch 3
        kernel_3 = tf.Variable(tf.truncated_normal([5, self.inputchannels, self.outputchannels], stddev=0.1))
        l2_3 = tf.nn.l2_loss(kernel_3)  
        tf.losses.add_loss(l2_3, 
            loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)               
        branch_3 = tf.nn.conv1d(batch, kernel_3, stride=1, padding='SAME')         
        # concat
        concat = tf.concat([branch_1, branch_2, branch_3], -1)
        # reduce feature space
        kernel_4 = tf.Variable(tf.truncated_normal([1, self.outputchannels*3, self.outputchannels], stddev=0.1))
        l2_4 = tf.nn.l2_loss(kernel_4)
        tf.losses.add_loss(l2_4, 
            loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES) 
        return tf.nn.conv1d(concat, kernel_4, stride=1, padding='SAME')

    def incep2d(self, batch):
        # branch 1
        kernel_1 = tf.Variable(tf.truncated_normal([1, self.inputchannels, self.outputchannels], stddev=0.1))
        l2_1 = tf.nn.l2_loss(kernel_1)           
        tf.losses.add_loss(l2_1, 
            loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)      
        branch_1 = tf.nn.conv2d(batch, kernel_1, strides=1, padding='SAME') 
        # branch 2
        kernel_2 = tf.Variable(tf.truncated_normal([3, self.inputchannels, self.outputchannels], stddev=0.1))
        l2_2 = tf.nn.l2_loss(kernel_2)       
        tf.losses.add_loss(l2_2, 
            loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)          
        branch_2 = tf.nn.conv2d(batch, kernel_2, strides=1, padding=self.padding) 
        # branch 3
        kernel_3 = tf.Variable(tf.truncated_normal([5, self.inputchannels, self.outputchannels], stddev=0.1))
        l2_3 = tf.nn.l2_loss(kernel_3)       
        tf.losses.add_loss(l2_3, 
            loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)          
        branch_3 = tf.nn.conv2d(batch, kernel_3, strides=1, padding='SAME')         
        # concat
        concat = tf.concat([branch_1, branch_2, branch_3], -1)
        # reduce feature space
        kernel_4 = tf.Variable(tf.truncated_normal([1, self.outputchannels*3, self.outputchannels], stddev=0.1))
        l2_4 = tf.nn.l2_loss(kernel_4)
        tf.losses.add_loss(l2_4, 
            loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES) 
        return tf.nn.conv2d(concat, kernel_4, strides=1, padding='SAME')

    def incep3d(self, batch):
        # branch 1
        kernel_1 = tf.Variable(tf.truncated_normal([1, self.inputchannels, self.outputchannels], stddev=0.1))
        l2_1 = tf.nn.l2_loss(kernel_1)                
        branch_1 = tf.nn.conv3d(batch, kernel_1, strides=1, padding='SAME') 
        # branch 2
        kernel_2 = tf.Variable(tf.truncated_normal([3, self.inputchannels, self.outputchannels], stddev=0.1))
        l2_2 = tf.nn.l2_loss(kernel_2)                
        branch_2 = tf.nn.conv3d(batch, kernel_2, strides=1, padding='SAME') 
        # branch 3
        kernel_3 = tf.Variable(tf.truncated_normal([5, self.inputchannels, self.outputchannels], stddev=0.1))
        l2_3 = tf.nn.l2_loss(kernel_3)                
        branch_3 = tf.nn.conv3d(batch, kernel_3, strides=1, padding='SAME')         
        # concat
        concat = tf.concat([branch_1, branch_2, branch_3], -1)
        # reduce feature space
        kernel_4 = tf.Variable(tf.truncated_normal([1, self.outputchannels*3, self.outputchannels], stddev=0.1))
        l2_4 = tf.nn.l2_loss(kernel_4)
        return tf.nn.conv3d(concat, kernel_4, strides=1, padding='SAME')

    def conn1d(self, batch):
        return tf.layers.dense(batch, self.outputchannels)

    def conn2d(self, batch):
        return tf.layers.dense(batch, self.outputchannels)

    def conn3d(self, batch):
        return tf.layers.dense(batch, self.outputchannels)
                        
    def compute(self, batch):
        if self.dimensions == 1:
            if self.layertype == 'convolution':     
                batch = self.activate(self.batchNorm(self.conv1d(batch)))
            elif self.layertype == 'inception':
                batch = self.activate(self.batchNorm(self.incep1d(batch)))
            elif self.layertype == 'connected':
                batch = self.activate(self.batchNorm(self.conn1d(batch)))
            elif self.layertype == 'dropout':
                batch = self.dropout(batch)
            elif self.layertype == 'maxpooling':
                batch = self.maxPooling(batch)
            else:
                raise ValueError('Invalid layertype:', self.layertype)
        elif self.dimensions == 2:
            if self.layertype == 'convolution':
                batch = self.activate(self.batchNorm(self.conv2d(batch)))
            elif self.layertype == 'inception':
                batch = self.activate(self.batchNorm(self.incep2d(batch)))
            elif self.layertype == 'connected':
                batch = self.activate(self.batchNorm(self.conn2d(batch)))
            elif self.layertype == 'dropout':
                batch = self.dropout(batch)
            elif self.layertype == 'maxpooling':
                batch = self.maxPooling(batch)
            else:
                raise ValueError('Invalid layertype:', self.layertype)
        elif self.dimensions == 3:
            if self.layertype == 'convolution':
                batch = self.activate(self.batchNorm(self.conv3d(batch)))
            elif self.layertype == 'inception':
                batch = self.activate(self.batchNorm(self.incep3d(batch)))
            elif self.layertype == 'connected':
                batch = self.activate(self.batchNorm(self.conn3d(batch)))
            elif self.layertype == 'dropout':
                batch = self.dropout(batch)
            elif self.layertype == 'maxpooling':
                batch = self.maxPooling(batch)
            else:
                raise ValueError('Invalid layertype:', self.layertype)
        else:
            raise ValueError('Invalid Dimensions:', self.dimensions)      
        return batch

    def maxPooling(self, batch):
        if self.dimensions == 1:
            batch = tf.nn.max_pool1d(batch, 2, 2, 'SAME')
        elif self.dimensions == 2:
            batch = tf.nn.max_pool2d(batch, 2, 2, 'SAME')
        elif self.dimensions == 3:
            batch =  tf.nn.max_pool3d(batch, 2, 2, 'SAME')
        else:
            raise ValueError('dimensions must be 1, 2, or 3, not:', self.dimensions)
        return batch
                    
