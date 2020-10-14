import tensorflow as tf
from tensorflow_core.python.layers.core import flatten


class neuralNetwork:

    def __init__(self):
        '''
        Define some basic parameters here
        '''
        self.stride = [1, 1, 1, 1]
        self.image_shape = (28, 28)

        self.c1_kernel_size = 5
        self.c1_kernel_num = 6

        self.c3_kernel_size = 5
        self.c3_kernel_num = 16

        self.c5_kernel_size = 4
        self.c5_kernel_num = 120

        self.f6_inodes = self.c5_kernel_num
        self.w_56 = None
        self.f6_onodes = 100

        self.f7_inodes = self.f6_onodes
        self.w_67 = None
        self.f7_onodes = 10

        pass

    def Net(self, input):
        '''
        Define network.
        You can use init_weight() and init_bias() function to init weight matrix,
        for example:
            conv1_W = self.init_weight((3, 3, 1, 6))
            conv1_b = self.init_bias(6)
        '''

        '''
        c1 input 28 * 28
        kernel 5 * 5 * 6
        output 24 * 24 * 6
        '''
        conv1_w = self.init_weight((self.c1_kernel_size, self.c1_kernel_size, self.c1_kernel_num))
        conv1_b = self.init_bias(self.c1_kernel_num)
        filterinput = tf.reshape(conv1_w, [self.c1_kernel_size, self.c1_kernel_size, 1, self.c1_kernel_num])
        c1_output = tf.nn.conv2d(input, filterinput, self.stride, 'VALID') + conv1_b
        # 非线性变换（激活函数）
        c1_output = tf.nn.relu(c1_output)
        '''
        s2 input 6 * 24 * 24
        kernel 1 * 2 * 2
        output 6 * 12 * 12
        '''
        s2_output = tf.nn.max_pool(c1_output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        '''
        c3 input 6 * 12 * 12
        kernel 16 * 5 * 5
        output 16 * 8 * 8
        '''

        conv3_w = self.init_weight((self.c3_kernel_size, self.c3_kernel_size, self.c1_kernel_num, self.c3_kernel_num))
        conv3_b = self.init_bias(self.c3_kernel_num)
        #filterinput = tf.reshape(conv3_w, [self.c3_kernel_size, self.c3_kernel_size, 1, self.c3_kernel_num])
        c3_output = tf.nn.conv2d(s2_output, conv3_w, strides = [1,1,1,1], padding='VALID') + conv3_b
        # 非线性变换（激活函数）
        c3_output = tf.nn.relu(c3_output)

        '''
        s4 input 16 * 8 * 8
        kernel 1 * 2 * 2
        output 16 * 4 * 4
        '''
        s4_output_temp = tf.nn.max_pool(c3_output,ksize=[1, 2, 2, 1],strides=[1, 2, 2,1], padding='VALID')

        # 压平，把输出排成一列，得到一个向量，输入4x4x16,输出 256
        s4_output = flatten(s4_output_temp)
        '''
        c5 input 16 * 4 * 4
        output 120
        '''
        c5_w = self.init_weight((256, 120))
        c5_b = self.init_bias(120)
        c5_output = tf.matmul(s4_output, c5_w)+ c5_b

        # 非线性变换（激活函数）
        c5_output = tf.nn.relu(c5_output)
        # dropout
        #c5_output = tf.nn.dropout(c5_output, keep_prob)
        '''
        f6 input 120
        param num: 120 * 84 + 84
        output 84//???
        '''
        f6_w = self.init_weight((120, 84))
        f6_b = self.init_bias(84)
        f6_output = tf.matmul(c5_output, f6_w)+ f6_b
        # 非线性变换(激活函数)
        f6_output = tf.nn.relu(f6_output)
        #  dropout2
        #fc2 = tf.nn.dropout(fc2, keep_prob)
        '''
        f7 input 84
        param num 84 * 10 + 10
        output 10
        '''
        f7_w = self.init_weight((84, 10))
        f7_b = self.init_bias(10)
        #这些输出被称为对数几率（logits）
        logits = tf.matmul(f6_output, f7_w)+ f7_b

        #logits = (None, 10)   # logits.shape = (batch_size, 10)
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