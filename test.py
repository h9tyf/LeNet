import tensorflow as tf;
import numpy as np;
import matplotlib.pyplot as plt;

shape = (3,2,4,5)
c = tf.truncated_normal(shape, mean=0, stddev=1)

with tf.Session() as sess:
    print (sess.run(c))

