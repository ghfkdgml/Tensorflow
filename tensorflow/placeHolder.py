#!/bin/python3.5

import tensorflow as tf

input = [1,2,3,4,5]
x=tf.placeholder(dtype=tf.float32)
y=x+5

sess=tf.Session()
print(sess.run(y,feed_dict={x:input}))
