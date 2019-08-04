#!/bin/python3.5

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/",one_hot=True)

X = tf.placeholder(tf.float32,[None,784])
Y = tf.placeholder(tf.float32,[None,10])

W1 = tf.Variable(tf.random_normal([784,256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X,W1))


W2 = tf.Variable(tf.random_normal([256,64], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1,W2))

W3 = tf.Variable(tf.random_normal([64,10], stddev=0.01))
model = tf.matmul(L2,W3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model,labels=Y))

optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(10):
    total_cost = 0
    for i in range(total_batch):
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer,cost], feed_dict={X:batch_x,Y:batch_y})
        total_cost += cost_val
    print('반복:','%04d' %(epoch+1),'평균 손실값:','{:.4f}'.format(total_cost/total_batch))
print('학습 완료!')

is_correct = tf.equal(tf.argmax(model,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))
print('정확도:',sess.run(accuracy,feed_dict={X:mnist.test.images,Y:mnist.test.labels}))
