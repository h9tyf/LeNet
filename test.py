import tensorflow as tf
def init_weight(shape):
    w = tf.truncated_normal(shape=shape, mean = 0, stddev = 0.1)
    return tf.Variable(w)

def init_bias(shape):
    b = tf.zeros(shape)
    return tf.Variable(b)

a=tf.constant([
    [1.0,2.0,3.0,4.0],
     [5.0,6.0,7.0,8.0],
     [8.0,7.0,6.0,5.0],
     [4.0,3.0,2.0,1.0]
])

a = tf.reshape(a, [1, 4, 4, 1])
conv1_w = init_weight((3, 3, 1))
conv1_b = init_bias(1)
filterinput = tf.reshape(conv1_w, [3, 3, 1, 1])
c1_output = tf.nn.conv2d(a, filterinput, [1, 1, 1, 1], 'VALID') + conv1_b

with tf.Session() as sess:
    print(sess.run(a))
    print("_________________________________")
    print(sess.run(conv1_w))
    print("_______________________________")
    print(sess.run(conv1_b))
    print("__________________________________________")
    print(sess.run(c1_output))