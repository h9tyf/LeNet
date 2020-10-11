import tensorflow as tf

class neuralNetwork:

    def __init__(self):
        '''
        Define some basic parameters here
        '''

        pass

    def Net(self, input):
        '''
        Define network.
        You can use init_weight() and init_bias() function to init weight matrix,
        for example:
            conv1_W = self.init_weight((3, 3, 1, 6))
            conv1_b = self.init_bias(6)
        '''

        logits = (None, 10)   # logits.shape = (batch_size, 10)

        return logits
        
    def forward(self, input):
        '''
        Forward the network
        '''
        return self.Net(input)

    def init_weight(self, shape):
        '''
        Init weight parameter.
        '''
        w = tf.truncated_normal(shape=shape, mean = 0, stddev = 0.1)
        return tf.Variable(w)
        
    def init_bias(self, shape):
        '''
        Init bias parameter.
        '''
        b = tf.zeros(shape)
        return tf.Variable(b)